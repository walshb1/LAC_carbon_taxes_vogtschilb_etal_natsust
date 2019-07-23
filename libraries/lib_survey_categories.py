import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, pinv

from libraries.lib_country_params import get_lcu_to_2010intd
from libraries.lib_common_plotting_functions import sns_pal
import seaborn as sns

def get_structure(level='all'):
    if level not in ['all','L0','L1','L2']: level = 'all'
    struct = {}
    
    # 0th level
    if level=='L0' or level=='all': 
        struct['gct'] = ['gasto_ali','gasto_veca','gasto_viv','gasto_ens','gasto_sal','gasto_com','gasto_trans','gasto_edre','gasto_otros']

    # 1st level
    if level=='L1' or level=='all':     
        struct['gasto_ali'] = ['gasto_alihogar','gasto_alifuera','gasto_alta']
        struct['gasto_viv'] = ['gasto_vcon','gasto_vag','gasto_vele','gasto_vgn','gasto_vpgk','gasto_vlp','gasto_vleca','gasto_vdi','gasto_vot']
        struct['gasto_trans'] = ['gasto_tserv','gasto_tcomb','gasto_tman','gasto_tadq','gasto_totros']
    # 2nd level
    if level=='L2' or level=='all': 
        struct['gasto_vpgk'] = ['gasto_vp','gasto_vgas','gasto_vk']
        struct['gasto_vleca'] = ['gasto_vca','gasto_vle']
        struct['gasto_tcomb'] = ['gasto_tga','gasto_tlp','gasto_tdie','gasto_tgnc','gasto_talc','gasto_totcomb']

    return struct

def check_gasto_consistency(df_exp,skip_gct=False,tolerance=1.,level='all'):
    struct = get_structure(level)
    fail_crit = '((calc_sum > exp_sum+@tolerance)|(calc_sum < exp_sum-@tolerance))&(calc_sum != 0)'
    rpt = []

    for icat in struct: 
        if skip_gct and icat == 'gct': continue

        if not (df_exp[icat].round(0) == (df_exp[struct[icat]].sum(axis=1)).round(0)).all():

            fail = pd.DataFrame({'exp_sum':df_exp[icat],'calc_sum':df_exp[struct[icat]].sum(axis=1)},index=df_exp.index)
            
            # crude -- maybe should be weighted
            if fail['exp_sum'].mean() < fail['calc_sum'].mean(): dir = 'upstream'
            else: dir = 'downstream'
            
            if fail.loc[fail.eval(fail_crit)].shape[0] != 0:
                print(icat,':',struct[icat])
                #print(fail.loc[fail.eval(fail_crit)])
                rpt.append([icat,dir])

    return rpt,struct

def get_dict_gtap_to_final(verbose=False):
    
    # load color palette
    pal3 = sns.color_palette('Set3', n_colors=12, desat=1.)

    return {'frac_vehicles'    :[['mvh','otn'], 'Vehicles',pal3[0]],
            'frac_health'      :[['ros','ofi','isr','obs','osg','dwe','ppp'],'Health, education, & recreation',pal3[5]],
            'frac_construction':[['cns','i_s','nmm','omn','nfm','crp','frs','lum','fmp'],'Construction incl. materials',pal3[0]],
            # includes construction costs, iron & steel, minerals, plastics, metals & rubber, lumber, wood, sheet metals
            'frac_water'       :[['wtr'],'Water',None],
            'frac_electricity' :[['TnD','Nucl','Coal','GasB','Wind','HydB',
                                  'OilB','OthB','GasP','HydP','OilP','SolP'],'Electricity',pal3[6]],
            'frac_fuels'   :[['p_c','oil',],'Petroleum, gasoline & diesel',pal3[11]],
            'frac_gas'     :[['gas','gdt'],'Natural gas',pal3[6]],
            'frac_trd'     :[['trd'],'Other retail & trade',None],
            'frac_gizmos'  :[['ele','ome','omf'],'Manufacturing, electronics & machinery',None],
            'frac_tex'     :[['pfb','tex','wol','wap','lea'],'Fibres, textiles, & clothing',pal3[8]],
            'frac_char'    :[['coa'],'Coal',None],
            'frac_pubtrans':[['otp','wtp','atp'],'Public transport',pal3[2]],
            'frac_comms'   :[['cmn'],'Communication',pal3[7]],
            'frac_food'    :[['pdr','wht','gro','v_f','osd','c_b','ocr','ctl','oap',
                              'rmk','fsh','cmt','omt','vol','mil','pcr','sgr','ofd','b_t'],'Food',pal3[4]]
            }

    


def get_expenditures_sf(pais,df_exp):

    _hies = 'consumption_and_household_surveys/2017-10-13/Household_survey_with_new_file_name/'+pais+'_household_expenditure_survey.dta'
    ano = int(pd.read_stata(_hies,columns=['anio']).mean())
    to_intd_2010 = get_lcu_to_2010intd(pais,ano)

    # Load bridge matrix
    bridge = pd.read_excel('consumption_and_household_surveys/2017-10-13/Bridge_matrix_consumption_items_to_GTAP_power_sectors.xlsx',sheet_name=pais).dropna(how='all')
    bridge = bridge.rename(columns={'Item':'hies_code','Item_english':'hies_desc'}).drop('hies_desc',axis=1).fillna(0).set_index('hies_code')
    
    # Invert bridge matrix to give unbridge matrix
    unbridge = pd.DataFrame(pinv(bridge),index=bridge.T.index,columns=bridge.T.columns)

    # Load final demand vector
    FD = pd.read_excel('GTAP_power_IO_tables/'+pais+'IOT.xlsx','Final_Demand',index_col=[0])['Hou']

    # Combine final demand matrix with unbridge matrix
    # --> describes GTAP final demand in terms of HIES categories
    _FD_to_hies = FD.dot(unbridge)
    _FD_to_hies.to_csv('~/Desktop/tmp/FD_to_hies.csv')

    # Make a copy of df_exp
    # --> already in per cap weighting
    _df_exp = df_exp[[i for i in df_exp.columns if i != 'gct']].copy()
    # df_exp.drop('gct',axis=1)

    cols_to_drop = []
    for i in _df_exp.columns:
        if _df_exp[i].sum() == 0.: cols_to_drop.append(i)
    _df_exp = _df_exp.drop(columns=cols_to_drop,axis=1)

    # Load hh weights
    _df_wgt = pd.read_stata(_hies,index_col='cod_hogar')[['factor_expansion','miembros_hogar']]
    _df_wgt['pcwgt'] = _df_wgt[['factor_expansion','miembros_hogar']].prod(axis=1)

    # Calculate each household's share of consumption, in hies categories
    _df_hh_share = pd.DataFrame(index=_df_exp.index)
    for i in _df_exp.columns:
        _df_hh_share[i] = (_df_exp[i]*_df_wgt['pcwgt'])/(_df_exp[i]*_df_wgt['pcwgt']).sum()

    _df_hies_FD = (_df_hh_share*_FD_to_hies).fillna(0)
    _df_hies_FD.to_csv('~/Desktop/tmp/df_hies_FD.csv')

    final_sf = pd.DataFrame(index=_df_hies_FD.index)
    mydict = bridge_report(pais)
    for i in mydict:
        final_sf[i] = _df_hies_FD[mydict[i]].sum(axis=1)*1.E6
    
    final_sf['gtap_gct'] = final_sf.sum(axis=1)
    final_sf['hies_gct'] = pd.read_stata(_hies,index_col='cod_hogar')[['factor_expansion','gct']].prod(axis=1)*to_intd_2010
    final_sf['sf'] = (final_sf['gtap_gct']/final_sf['hies_gct']).clip(lower=0)
    final_sf['pcwgt'] = _df_wgt['pcwgt']

    final_sf = final_sf[['gtap_gct','hies_gct','sf','pcwgt']].fillna(0)

    ax = plt.gca()
    _h, _b = np.histogram(final_sf['sf'],bins=50,weights=final_sf['pcwgt'])
    ax.bar((_b[1]-_b[0])/2+_b[:-1], _h, width=(_b[1]-_b[0]), label='SF', facecolor=sns_pal[0],edgecolor=sns_pal[0],alpha=0.45,log=True)
    plt.gcf().savefig('output/sp/'+pais+'_sf_hist.pdf',format='pdf',bbox_inches='tight')

    df_carbon_cost = pd.read_csv('output/carbon_cost/gtap_hies_sf_'+pais+'.csv',index_col='cod_hogar').fillna(0)
    #df_carbon_cost_ind = pd.read_csv('output/carbon_cost/gtap_hies_'+pais+'_indirect_tot.csv',index_col='cod_hogar').rename(columns={'gtap':'carbon_cost'}).fillna(0)

    final_sf['carbon_cost'] = df_carbon_cost['gtap']
    
    #final_sf.to_csv('~/Desktop/tmp/final_sf.csv')
    return final_sf
