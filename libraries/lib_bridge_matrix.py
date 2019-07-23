import pandas as pd
import numpy as np
from libraries.lib_country_params import *

pd.set_option('display.width', 220)

def get_bridge_matrix_sf():

    if True:
        TM = pd.read_csv('Bridge_matrix_consumption_items_to_GTAP_power_sectors_v1.csv').dropna(how='all').drop(['Unnamed: 70','Unnamed: 71'],axis=1)
        #
        TM_dict_to_gtap = TM.loc[0].dropna().to_dict()
        TM_dict_to_english = {}
        for ikey in TM_dict_to_gtap:
            TM_dict_to_english[TM_dict_to_gtap[ikey]] = ikey
        #
        TM = TM.rename(columns=TM_dict_to_gtap)
        TM = TM.rename(columns={'Unnamed: 0':'hies_desc','Unnamed: 1':'hies_code'}).dropna(subset=['hies_code']) 
        TM = TM.reset_index().set_index('hies_code').loc[TM.index!=0].drop('index',axis=1)
        #
        TM_dict_to_hies = TM['hies_desc'].to_dict()
        TM_dict_to_hies_desc = {}
        for jkey in TM_dict_to_hies:
            TM_dict_to_hies_desc[TM_dict_to_hies[jkey]] = jkey
        #
        TM = TM.drop('hies_desc',axis=1).fillna(0)
        #
        res_gtap_to_hies = pd.DataFrame({'english':[TM_dict_to_hies[i] for i in TM.index],'gtap_categories':''},index=TM.index)
        res_hies_to_gtap = pd.DataFrame({'english':[TM_dict_to_english[i] for i in TM.columns],'hies_categories':''},index=TM.columns)
        res_hies_to_gtap_ccost =  pd.DataFrame({'english':[TM_dict_to_english[i] for i in TM.columns]},index=TM.columns)

    todos_paises = ['arg','bol','bra','cri','mex','per','pan','slv']
    for pais in todos_paises:

        # Country-specific bridge matrices
        bridge_str = 'consumption_and_household_surveys/2017-10-13/Bridge_matrix_consumption_items_to_GTAP_power_sectors.xlsx'
        try: ctryTM = pd.read_excel(bridge_str,sheet_name=pais).dropna(how='all')
        except: ctryTM = pd.read_excel(bridge_str,sheet_name='nae_of_default_tab').dropna(how='all')
        ctryTM = ctryTM.rename(columns={'Item':'hies_code','Item_english':'hies_desc'}).drop('hies_desc',axis=1).fillna(0).set_index('hies_code')

        #load HIES
        _hies = 'consumption_and_household_surveys/2017-10-13/Household_survey_with_new_file_name/'+pais+'_household_expenditure_survey.dta'
        df = pd.read_stata(_hies,index_col='cod_hogar')[['pais','factor_expansion']+[i for i in TM_dict_to_hies]]

        ano = int(pd.read_stata(_hies,columns=['anio']).mean())
        #new_single_fac = get_lcu_to_2011usd_ppp(pais,ano)
        #to_intd_2011 = get_2011usd(pais,ano)/get_fx(pais,ano)
        to_intd_2010 = get_lcu_to_2010intd(pais,ano)

        # Load GTAP FD vector
        FD = pd.read_excel('GTAP_power_IO_tables/'+pais+'IOT.xlsx','Final_Demand',index_col='sector')['Hou']

        # express gtap as hies with bridge matrix
        for ihies in ctryTM.index:
            hies_sum = to_intd_2010*float(df[[ihies,'factor_expansion']].prod(axis=1).sum())
            gtap_sum = 0.

            for igtap in ctryTM.columns:
                gtap_sum += float(ctryTM.loc[ihies,igtap])*float(FD.loc[igtap])*1.E6
                if float(ctryTM.loc[ihies,igtap])==1. and pais == todos_paises[0]: res_gtap_to_hies.ix[ihies,'gtap_categories'] += TM_dict_to_english[igtap]+'; '

            if gtap_sum != 0: res_gtap_to_hies.ix[ihies,pais+'_h/g'] = hies_sum/gtap_sum
            if hies_sum != 0: res_gtap_to_hies.ix[ihies,pais+'_g/h'] = gtap_sum/hies_sum

        # express hies as gtap with bridge matrix
        for igtap in ctryTM.columns:
            gtap_sum = float(FD.loc[igtap])*1.E6
            hies_sum = 0.

            for ihies in ctryTM.index:
                hies_sum += to_intd_2010*float(df[[ihies,'factor_expansion']].prod(axis=1).sum())*float(ctryTM.loc[ihies,igtap])
                if float(ctryTM.loc[ihies,igtap])==1. and pais == todos_paises[0]: res_hies_to_gtap.ix[igtap,'hies_categories'] += TM_dict_to_hies[ihies]+'; '

            if gtap_sum != 0: res_hies_to_gtap.ix[igtap,pais+'_h/g'] = hies_sum/gtap_sum
            if hies_sum != 0: res_hies_to_gtap.ix[igtap,pais+'_g/h'] = gtap_sum/hies_sum

        res_hies_to_gtap[pais+'_h/(gsum)'] = res_hies_to_gtap.groupby('hies_categories')[pais+'_g/h'].transform(lambda x: 1/(x.replace([np.inf,-np.inf],0).sum()))
        # ^ this is interesting if multiple GTAP categories fall under a single HIES category.
        # --> Example: HH exp. on food & bev (in home) = sum(rice,wheat, beef, etc...)

        # Multiply sectors by carbon costs
        res_hies_to_gtap_ccost['hies_categories'] = res_hies_to_gtap['hies_categories'].copy()
        res_hies_to_gtap_ccost[pais+'_h/(gsum)'] = res_hies_to_gtap[pais+'_h/(gsum)'].copy()

        GHG_hh_cost = pd.read_csv('hh_output/carbon_cost_all_'+pais+'.csv')
        #GHG_industry = pd.read_csv('emission_data/GTAP_GHG_emissions_by_industry_sector.csv',index_col = [0])[todos_paises] # industry emissions
        #res_hies_to_gtap_ccost['hies_categories'] = res_hies_to_gtap['hies_categories'].copy()
        res_hies_to_gtap_ccost['ghg'] = GHG_hh_cost.sum().transpose()/1.E6
        res_hies_to_gtap_ccost['gtap_fd'] = FD.copy().astype('float')

        #res_hies_to_gtap_ccost[pais+'_g/(CCxh)'] = (res_hies_to_gtap[pais+'_g/h'].copy()/res_hies_to_gtap_ccost['ghg']).fillna(0)
        #res_hies_to_gtap_ccost[pais+'_CCxh/(gsum)'] = res_hies_to_gtap_ccost.groupby('hies_categories'[pais+'_g/(CCxh)'].transform(lambda x: 1/(x.replace([np.inf,-np.inf],0).sum()))
        # ^ not right!
    
        res_hies_to_gtap_ccost[pais+'_total_ghg'] = res_hies_to_gtap_ccost.groupby('hies_categories')['ghg'].transform('sum')
        res_hies_to_gtap_ccost[pais+'_CCxh/(gsum)'] = res_hies_to_gtap_ccost[[pais+'_total_ghg',pais+'_h/(gsum)']].prod(axis=1)
        
        res_hies_to_gtap_ccost = res_hies_to_gtap_ccost.drop(['gtap_fd','ghg',pais+'_total_ghg'],axis=1)
        # ^ drop the intermediates

    exw = pd.ExcelWriter('sp_output/hies_gtap_comparison.xlsx')
    res_gtap_to_hies.to_excel(exw,'hies_cats')
    res_hies_to_gtap.to_excel(exw,'gtap_cats')
    res_hies_to_gtap_ccost.to_excel(exw,'gtap_cats_with_ccosts')
    GHG_hh_cost.sum().transpose().to_excel(exw,'gtap_ghg')
    exw.save()

    return res_hies_to_gtap_ccost


#_ = get_bridge_matrix_sf()
#print(_.head())
