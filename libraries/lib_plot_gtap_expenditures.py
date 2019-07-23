import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob

import subprocess

from libraries.lib_percentiles import *
from libraries.lib_gtap_to_final import gtap_to_final
from libraries.lib_common_plotting_functions import greys, quint_colors, quint_labels
from libraries.lib_country_params import get_FD_scale_fac,iso_to_name
from libraries.lib_get_hh_survey import get_hh_survey#, get_miembros_hogar
from libraries.lib_survey_categories import get_dict_gtap_to_final
from libraries.lib_results_to_excel import save_to_results_file
from matplotlib.ticker import FormatStrFormatter

import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.2    

import seaborn as sns
div_pal = sns.color_palette('BrBG', n_colors=11)

def plot_expenditures_by_category(pais,hies_FD,hies_FD_tot):
    out_dir = 'output/'
    if pais == 'brb': out_dir = '/Users/brian/Desktop/Dropbox/IDB/Barbados/output/'

    ####################
    # Plot expenditures by category
    # --> as fraction of total expenditures
    hies_FD     = hies_FD.reset_index().set_index(['cod_hogar','quintile'])
    hies_FD_tot = hies_FD_tot.reset_index().set_index(['cod_hogar','quintile'])

    final_FD_quints = pd.DataFrame(index=hies_FD_tot.sum(level='quintile').index).sort_index()
    # Reset df

    do_not_plot = []

    plt.figure(figsize=(6,6))

    fdict = get_dict_gtap_to_final()
    for _h in fdict: 
        hies_FD_tot[_h] = hies_FD[[fdict[_h][1]]].sum(axis=1)
        final_FD_quints[_h] = 100.*(hies_FD_tot[['hhwgt',_h]].prod(axis=1)/hies_FD_tot['totex_hh']).sum(level='quintile')/hies_FD_tot['hhwgt'].sum(level='quintile')

    _ = final_FD_quints.T.copy()
    _.columns = ['Q1','Q2','Q3','Q4','Q5']

    ##########################################################################################
    # Record sample (all countries) stats in out_dir+'all_countries/hh_expenditures_table.csv'
    try: hhexp = pd.read_csv(out_dir+'all_countries/hh_expenditures_table.csv').set_index('category')
    except: hhexp = pd.DataFrame({pais.upper():0,'category':[fdict[i][1] for i in fdict]},index=None).set_index('category')
    for _ex in fdict:
        hhexp.loc[fdict[_ex][1],pais.upper()] = _.loc[_ex].mean()
    try: hhexp.to_csv(out_dir+'all_countries/hh_expenditures_table.csv')
    except: pass
    ##########################################################################################

    ##########################################################################################
    # Record sample (all countries) stats in out_dir+'all_countries/hh_regressivity_table.csv'
    for _q in ['Q1','Q2','Q3','Q4']:
        try: hhreg = pd.read_csv(out_dir+'all_countries/hh_regressivity_table_'+_q+'.csv').set_index('category')
        except: hhreg = pd.DataFrame({pais.upper():0,'category':[fdict[i][1] for i in fdict]},index=None).set_index('category')
        for _ex in fdict:
            hhreg.loc[fdict[_ex][1],pais.upper()] = _.loc[_ex,'Q1']/_.loc[_ex,'Q5']
        try: hhreg.to_csv(out_dir+'all_countries/hh_regressivity_table_'+_q+'.csv')
        except: pass
    ##########################################################################################

    _ = _[['Q1','Q5']].T.sort_values(by='Q1',axis=1)

    null_col = []
    for _c in _:
        if round(_[_c].mean(),1)==0: null_col.append(_c)
        if _[_c].mean()<0.1: do_not_plot.append(_c)
    _ = _.drop(null_col,axis=1)

    final_FD_quints.to_csv(out_dir+'expenditures/'+pais+'_gasto_by_cat_and_quint.csv')

    col_wid=_.shape[1]/2

    ax = plt.barh(np.arange(0,_.shape[1],1)*col_wid,_.iloc[0],color=sns.color_palette('BrBG', n_colors=11)[2],height=2.5)
    plt.barh(np.arange(0,_.shape[1],1)*col_wid+2.5,_.iloc[1],color=sns.color_palette('BrBG', n_colors=11)[8],height=2.5)
    plt.gca().grid(False)
    sns.despine(bottom=True)
    
    plt.gca().set_yticks(np.arange(0,_.shape[1],1)*col_wid+1)
    plt.gca().set_yticklabels([fdict[_h][1] for _h in _.columns],ha='right',fontsize=10,weight='light',color=greys[7])
    plt.gca().set_xticklabels([])

    ax = plt.gca()

    _y = [0.,0.]
    rects = ax.patches
    for rect in rects:
        
        if (rect.get_y()+rect.get_height()/2.) > _y[0]:
            _y.append(rect.get_y()+rect.get_height()/2.);_y.sort();_y.pop(0)
              
    for rect in rects:
            
        _w = rect.get_width()

        pct = ''
        if (rect.get_y()+rect.get_height()/2.) in _y: pct = '%'

        ax.annotate(str(round(_w,1))+pct,xy=(rect.get_x()+rect.get_width()+0.5, rect.get_y()+rect.get_height()/2.-0.1),
                    ha='left', va='center',color=greys[7],fontsize=7,zorder=100,clip_on=False,style='italic')

    ax.annotate('Wealthiest quintile',xy=(0.8,_y[1]),ha='left',va='center',color=greys[0],fontsize=7,zorder=100,style='italic')
    ax.annotate('Poorest quintile',xy=(0.8,_y[0]),ha='left',va='center',color=greys[7],fontsize=7,zorder=100,style='italic')
    plt.title('Household expenditures in '+iso_to_name[pais],weight='bold',color=greys[7],fontsize=12,loc='right')
    plt.draw()
    try: 
        plt.gcf().savefig(out_dir+'expenditures/'+pais+'_gastos_all_categories.pdf',format='pdf',bbox_inches='tight')
        plt.gcf().savefig(out_dir+'expenditures/'+pais+'_gastos_all_categories.png',format='png',bbox_inches='tight')
    except: pass
    plt.cla(); plt.close('all')

    return hies_FD,hies_FD_tot,null_col

def plot_gtap_exp(pais,do_tax_food=True,verbose=False):

    out_dir = 'output/'
    if pais == 'brb': out_dir = '/Users/brian/Desktop/Dropbox/IDB/Barbados/output/'

    ############################
    # Kuishuang's code (mostly):

    # load household survey data
    hh_hhsector = get_hh_survey(pais)
    hh_hhsector = hh_hhsector.drop([i for i in hh_hhsector.columns if 'ing' in i or 'ict' in i],axis=1)
    #hh_hhsector = hh_hhsector.fillna(1E5)#flag

    if verbose: print(hh_hhsector.shape)

    # load bridge matrix
    xl = pd.ExcelFile('consumption_and_household_surveys/2017-10-13/Bridge_matrix_consumption_items_to_GTAP_power_sectors.xlsx')
    if pais in  xl.sheet_names:  # all sheet names
        print('using '+pais+' tab')
        bridge_to_use = xl.parse(pais).fillna(0).drop(['Item_english'],axis = 1).set_index('Item')  # read the specific sheet

    else:
        if verbose: print('using default tab')
        bridge_to_use = xl.parse('nae_of_default_tab').fillna(0).drop(['Item_english'],axis = 1).set_index('Item')

    cols_to_drop = []
    for i in bridge_to_use.columns:
        if verbose: print(i,bridge_to_use[i].sum())
        if bridge_to_use[i].sum(axis=0)==0:
            cols_to_drop.append(i)
    bridge_to_use = bridge_to_use.drop(cols_to_drop,axis=1)

    # household survey in GTAP sectors
    hh_gtap_sector = hh_hhsector[bridge_to_use.index].fillna(0).dot(bridge_to_use)
    hh_gtap_sector = hh_gtap_sector.reset_index()
    try: hh_gtap_sector['cod_hogar'] = hh_gtap_sector['cod_hogar'].astype('int')
    except: hh_gtap_sector['cod_hogar'] = hh_gtap_sector['cod_hogar'].astype('str')
    hh_gtap_sector = hh_gtap_sector.reset_index().set_index('cod_hogar')

    ## Run test.
    #print(hh_hhsector.columns)
    #print(hh_hhsector.head())
    #_hh_hhsector = hh_hhsector.copy()
    #for _c in _hh_hhsector.columns:
    #    if _c != 'gasto_ali':#and _c != 'gasto_alihogar':
    #        _hh_hhsector[_c] = 0
    #_hh_gtap_sector = _hh_hhsector[bridge_to_use.index].fillna(0).dot(bridge_to_use)

    if verbose: print(hh_gtap_sector.head(8))

    # calcuate each household's share of national consumption, by category
    hh_share = (hh_gtap_sector.mul(hh_hhsector.factor_expansion, axis=0).fillna(0))/(hh_gtap_sector.mul(hh_hhsector.factor_expansion, axis=0).fillna(0).sum())

    # Read household consumption vector from GTAP
    _iot_code = pais if pais != 'brb' else 'xcb'
    try:
        hh_fd_file = 'GTAP_power_IO_tables_with_imports/Household_consumption_both_domestic_import.xlsx'
        household_FD = get_FD_scale_fac(pais)*pd.read_excel(hh_fd_file,index_col=[0])[_iot_code].squeeze()
    except: 
        if pais == 'brb': household_FD = get_FD_scale_fac(pais)*pd.read_excel('GTAP_power_IO_tables/xcbIOT.xlsx',sheet_name='Final_Demand',index_col=[0])['Hou'].squeeze()
        else: assert(False)
    #              ^ get_FD_scale_fac(pais) != 1. ONLY IF pais == 'brb'

    # Final demand matrix
    hh_FD = household_FD*hh_share.fillna(0)
    for i in hh_FD.columns: hh_FD[i]/=hh_hhsector['factor_expansion']

    if verbose: 
        print(household_FD.head())
        print(hh_FD.head(5))

    ####################
    # Use gtap_to_final script to translate both expenditures & cc into HIES cats
    hies_FD, hies_FD_tot, hies_sf = gtap_to_final(hh_hhsector,hh_FD,pais,verbose=True)

    # Now, this df should be consistent with the FD vector
    if verbose:
        print((hh_FD.sum(axis=1)*hh_hhsector['factor_expansion']).sum())
        print(hies_FD_tot[['totex_hh','hhwgt']].prod(axis=1).sum())
        print('FD:',round(hies_FD_tot[['totex_hh','hhwgt']].prod(axis=1).sum(),3),round((hh_FD.sum(axis=1)*hh_hhsector['factor_expansion']).sum(),3))

    assert(hies_FD_tot[['totex_hh','hhwgt']].prod(axis=1).sum()/(hh_FD.sum(axis=1)*hh_hhsector['factor_expansion']).sum()>0.999)
    assert(hies_FD_tot[['totex_hh','hhwgt']].prod(axis=1).sum()/(hh_FD.sum(axis=1)*hh_hhsector['factor_expansion']).sum()<1.001)    
    ####################    

    ####################
    if pais == 'brb':
        energy_tax_total = get_FD_scale_fac(pais)*pd.read_csv('/Users/brian/Desktop/Dropbox/IDB/Barbados/output/tax_cost_to_hh_in_gtap_cats.csv').set_index('cod_hogar')
        final_CC,wgts,_ = gtap_to_final(hh_hhsector,energy_tax_total,pais)
        hhwgts = wgts[['pcwgt','hhwgt','hhsize']].copy().dropna()

        final_CC_ind = final_CC.copy()
        final_CC_CO2 = final_CC.copy()
        final_CC_nonCO2 = final_CC.copy()
        for col in final_CC_nonCO2.columns: final_CC_nonCO2[col].values[:] = 0        
        final_CC_dir = final_CC.copy()
        for col in final_CC_dir.columns: final_CC_dir[col].values[:] = 0        
        #print(hhwgts.shape[0],hhwgts.dropna().shape[0])
        # HACK: ^ should be no NAs in this df
        
    else:
        # Indirect carbon costs - CO2
        ccdf_ind_CO2 = get_FD_scale_fac(pais)*pd.read_csv(out_dir+'carbon_cost/CC_per_hh_indirect_'+pais+'_CO2.csv').set_index('cod_hogar')
        
        # Indirect carbon costs - non-CO2
        ccdf_ind_nonCO2 = get_FD_scale_fac(pais)*pd.read_csv(out_dir+'carbon_cost/CC_per_hh_indirect_'+pais+'_nonCO2.csv').set_index('cod_hogar')

        # Indirect carbon costs (allGHG)
        ccdf_ind = get_FD_scale_fac(pais)*pd.read_csv(out_dir+'carbon_cost/CC_per_hh_indirect_'+pais+'_allGHG.csv').set_index('cod_hogar')

        # Direct carbon costs (allGHG)
        ccdf_dir = get_FD_scale_fac(pais)*pd.read_csv(out_dir+'carbon_cost/CC_per_hh_direct_'+pais+'_allGHG.csv').set_index('cod_hogar')
        # ^ these files are per household (multiply by factor_expansion for total)

        # HACK
        _bypass = pd.DataFrame(index=ccdf_ind.index.copy())
        hacker_dict = {'col':['frac_gas'],
                       'gtm':['frac_gas'],
                       'pan':['frac_gas'],
                       'hnd':['frac_gas'],
                       'nic':['frac_gas','frac_water'],
                       'pry':['frac_gas','frac_electricity']}

        if pais in hacker_dict:
            for _set in hacker_dict[pais]:

                _gtap_cols = get_dict_gtap_to_final()[_set][0]
                _i = [i for i in _gtap_cols if i in ccdf_ind.columns]
                _d = [d for d in _gtap_cols if d in ccdf_dir.columns]

                _bypass[_set] = ccdf_ind[_i].sum(axis=1) + ccdf_dir[_d].sum(axis=1)
                _bypass[_set] *= hh_hhsector['factor_expansion']
            
                try: 
                    ccdf_ind_CO2[_i] = [0,0]
                    ccdf_ind_nonCO2[_i] = [0,0]
                    ccdf_ind[_i] = [0,0]
                except: ccdf_ind_CO2[_i],ccdf_ind_nonCO2[_i],ccdf_ind[_i] = 0,0,0

                try: ccdf_dir[_d] = [0,0]
                except: ccdf_dir[_d] = 0

        _bypass = _bypass.sum()*1E-6*get_FD_scale_fac(pais)

        if not do_tax_food:
            ccdf_ind_CO2[['pdr','wht','gro','v_f','osd','c_b','ocr','ctl','oap','rmk','fsh','cmt','omt','vol','mil','pcr','sgr','ofd','b_t']] = 0
            ccdf_ind_nonCO2[['pdr','wht','gro','v_f','osd','c_b','ocr','ctl','oap','rmk','fsh','cmt','omt','vol','mil','pcr','sgr','ofd','b_t']] = 0
            ccdf_ind[['pdr','wht','gro','v_f','osd','c_b','ocr','ctl','oap','rmk','fsh','cmt','omt','vol','mil','pcr','sgr','ofd','b_t']] = 0
            # No food categories in ccdf_dir

        final_CC_ind,wgts,_ = gtap_to_final(hh_hhsector,ccdf_ind,pais)
        final_CC_dir,wgts,_ = gtap_to_final(hh_hhsector,ccdf_dir,pais)
        final_CC = final_CC_ind + final_CC_dir
        #final_CC_tot = final_CC_ind_tot + final_CC_dir_tot

        final_CC_ind_CO2,wgts,_ = gtap_to_final(hh_hhsector,ccdf_ind_CO2,pais)
        final_CC_CO2 = final_CC_ind_CO2 + final_CC_dir
        #final_CC_tot_CO2 = final_CC_ind_tot_CO2 + final_CC_dir_tot
    
        final_CC_nonCO2,wgts,_ = gtap_to_final(hh_hhsector,ccdf_ind_nonCO2,pais) 
        hhwgts = wgts[['pcwgt','hhwgt','hhsize']].copy()
        
        if verbose:
            #print('FD:',round(hhwgts[['totex_hh','hhwgt']].prod(axis=1).sum(),1),round((hh_FD.sum(axis=1)*hh_hhsector['factor_expansion']).sum(),3))
            print('Direct costs:',round((final_CC_dir.sum(axis=1)*hh_hhsector['factor_expansion']).sum(),1),
                  round((ccdf_dir.sum(axis=1)*hh_hhsector['factor_expansion']).sum(),1))
            print('Indirect cost:',round((final_CC_ind.sum(axis=1)*hh_hhsector['factor_expansion']).sum(),1),
                  round((ccdf_ind.sum(axis=1)*hh_hhsector['factor_expansion']).sum(),1))

        assert((final_CC_dir.sum(axis=1)*hhwgts['hhwgt']).sum()/(ccdf_dir.sum(axis=1)*hh_hhsector['factor_expansion']).sum()>0.99)
        assert((final_CC_dir.sum(axis=1)*hhwgts['hhwgt']).sum()/(ccdf_dir.sum(axis=1)*hh_hhsector['factor_expansion']).sum()<1.01)
        
        assert((final_CC_ind.sum(axis=1)*hhwgts['hhwgt']).sum()/(ccdf_ind.sum(axis=1)*hh_hhsector['factor_expansion']).sum()>0.99)
        assert((final_CC_ind.sum(axis=1)*hhwgts['hhwgt']).sum()/(ccdf_ind.sum(axis=1)*hh_hhsector['factor_expansion']).sum()<1.01)
    
    # 5 dataframes with results in them
    # --> final_CC
    # --> final_CC_CO2 & final_CC_nonCO2
    # --> final_CC_ind & final_CC_dir
    #hhwgts = wgts[['pcwgt','hhwgt','hhsize']].copy()
    # ^ plus this, with necessary weights


    #########################
    # Assign decile based on totex (household expenditures, mapped to gtap)
    hies_FD_tot['pais'] = pais

    if 'quintile' not in hies_FD_tot.columns:

        _deciles=np.arange(0.10, 1.01, 0.10)
        _quintiles=np.arange(0.20, 1.01, 0.20)

        hies_FD_tot = hies_FD_tot.groupby('pais',sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.totex_pc),reshape_data(x.pcwgt),_deciles),'decile','totex_pc'))
        hies_FD_tot = hies_FD_tot.groupby('pais',sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.totex_pc),reshape_data(x.pcwgt),_quintiles),'quintile','totex_pc'))
 
        hies_FD_tot = hies_FD_tot.drop(['pais'],axis=1)
            
        hies_FD['decile'] = hies_FD_tot['decile'].copy()
        hies_FD['quintile'] = hies_FD_tot['quintile'].copy()
            

    ###################################    
    # Price hikes in all goods due to gasoline increase (% of current price)
    fdict = get_dict_gtap_to_final()

    try: df = pd.read_csv(out_dir+'all_countries/price_increase_full.csv').set_index('category')
    except: df = pd.DataFrame({pais.upper():0,'category':[fdict[i][1] for i in fdict]},index=None).set_index('category')
    
    for i in fdict:
        
        table_value = None
        
        gtap_cat_array = get_dict_gtap_to_final()[i][0]
                
        #table_value_n = (final_CC_ind_tot['hhwgt']*(final_CC_ind[fdict[i][0]].sum(axis=1)+final_CC_dir[fdict[i][0]].sum(axis=1))/1E6).sum()
        # ^ this is already zero when there's no data in the survey
        
        if pais == 'brb':
            table_value_n = energy_tax_total[[_g for _g in gtap_cat_array if _g in energy_tax_total.columns]].sum(axis=1).sum()
            table_value_d = get_FD_scale_fac(pais)*float(pd.read_excel('GTAP_power_IO_tables/xcbIOT.xlsx',sheet_name='Final_Demand',index_col=[0])['Hou'].squeeze()[gtap_cat_array].sum())
            #             ^ get_FD_scale_fac(pais) != 1. ONLY IF pais == 'brb'

        else:
            table_value_n = ((ccdf_ind[[_g for _g in gtap_cat_array if _g in ccdf_ind.columns]].sum(axis=1)
                              +ccdf_dir[[_g for _g in gtap_cat_array if _g in ccdf_dir.columns]].sum(axis=1))*hh_hhsector['factor_expansion']).sum()/1E6
            #table_value_d = get_FD_scale_fac(pais)*float(pd.read_excel('GTAP_power_IO_tables/'
            #                                                            +_iot_code+'IOT.xlsx','Final_Demand',index_col=[0])['Hou'].squeeze()[gtap_cat_array].sum())
            _fname = 'GTAP_power_IO_tables_with_imports/Household_consumption_both_domestic_import.xlsx'
            table_value_d  = get_FD_scale_fac(pais)*float(pd.read_excel(_fname,index_col=[0])[pais].squeeze()[gtap_cat_array].sum())
            #              ^ get_FD_scale_fac(pais) != 1. ONLY IF pais == 'brb'. so this should be deleted

            if table_value_n == 0 and table_value_d != 0:
                print('BYPASS:',pais,_bypass)
                try: table_value_n = float(_bypass[i])
                except: pass

            # throw results...look how clever we are!
            if verbose:
                print(i,table_value_n,table_value_d)
                print('ind:',(ccdf_ind[[_g for _g in gtap_cat_array if _g in ccdf_ind.columns]].sum(axis=1)*hh_hhsector['factor_expansion']).sum()/1E6)
                print('dir:',(ccdf_dir[[_g for _g in gtap_cat_array if _g in ccdf_dir.columns]].sum(axis=1)*hh_hhsector['factor_expansion']).sum()/1E6)
        
        table_value = round(100*table_value_n/table_value_d,1)                        
        df.loc[fdict[i][1],pais.upper()] = table_value

    if pais == 'brb':
        df['BRB']/=1000.
        df.loc['Petroleum, gasoline & diesel'] = 6.2
        # HACK: don't understand why *=1/1000. would be justified; haven't checked units
        # HACK: not sure why 'Petroleum, gasoline & diesel' doesn't come through analysis

    _df = df.sort_values(pais.upper(),ascending=False).drop([fdict[i][1] for i in cols_to_drop])[pais.upper()]
    _df.name = '[%]'
    _df.index.name = 'Relative increase'

    _df.round(1).to_latex(out_dir+'latex/pct_change_'+pais.lower()+'.tex')
    
    with open(out_dir+'latex/pct_change_'+pais.lower()+'.tex', 'r') as f:
        with open(out_dir+'latex/out_pct_change_'+pais.lower()+'.tex', 'w') as f2:
            
            f2.write(r'\documentclass[10pt]{article}'+'\n')
            f2.write(r'\usepackage{amssymb} %maths'+'\n')
            f2.write(r'\usepackage{amsmath} %maths'+'\n')
            f2.write(r'\usepackage{booktabs}'+'\n')
            f2.write(r'\begin{document}'+'\n')
            
            f2.write(f.read())
            
            f2.write(r'\end{document}')
            f2.close()
            
    subprocess.call('cd '+out_dir+'latex/; pdflatex out_pct_change_'+pais.lower()+'.tex',shell=True)
    for f in glob.glob(out_dir+'latex/*.aux'): os.remove(f)
    for f in glob.glob(out_dir+'latex/*.log'): os.remove(f)
    for f in glob.glob(out_dir+'latex/out_*.tex'): os.remove(f)
    if pais != 'brb': df.to_csv('output/all_countries/price_increase_full.csv')

    hies_FD,hies_FD_tot,cols_to_drop = plot_expenditures_by_category(pais,hies_FD,hies_FD_tot)

    ###################################
    # Current spending on all energy (electricity, petroleum, gasoline, diesel, natural gas, & coal), as % of totex
    energy_categories = [fdict['frac_fuels'][1],fdict['frac_gas'][1],fdict['frac_char'][1]]
    # ^ includes: gasto_tcomb = Household expenditure on transportation fuels
    # ^           gasto_vpgk  = Household expenditure on petroleum, gasoline and kerosene for domestic use
    # ^           gasto_vlp   = Household expenditure on liquified petroleum gas for domestic use
    # ^           gasto_vdi   = Household expenditure on diesel for domestic use"
    
    final_FD_quints = pd.DataFrame(index=hies_FD.reset_index().set_index('quintile').sum(level='quintile').index).sort_index()

    final_FD_quints['Direct fuel consumption'] = 100.*((hies_FD_tot['hhwgt']*hies_FD[energy_categories].sum(axis=1)/hies_FD_tot['totex_hh']).sum(level='quintile')
                                                  /hies_FD_tot['hhwgt'].sum(level='quintile'))
    _hack = final_CC_dir.copy()
    _hack['quintile'] = hies_FD_tot.reset_index('quintile')['quintile'].copy()
    _hack = _hack.reset_index().set_index(['cod_hogar','quintile'])

    final_FD_quints['Direct fuel consumption tax'] = (100./1E6*(_hack.sum(axis=1)*hhwgts['hhwgt']).sum(level='quintile')
                                                      /hies_FD_tot[['totex_pc','pcwgt']].prod(axis=1).sum(level='quintile'))
    
    final_FD_quints.plot(final_FD_quints.index,'Direct fuel consumption',kind='bar',color=quint_colors,legend=False)
    plt.gca().set_xticklabels(quint_labels,ha='center',rotation=0)
    plt.ylabel('Direct fuel consumption [% of total expenditures]',fontsize=11,weight='bold',labelpad=8)
    plt.xlabel('')
    
    plt.ylim([0,final_FD_quints[['Direct fuel consumption','Direct fuel consumption tax']].sum(axis=1).max()*1.05])

    rects = plt.gca().patches
    for rect in rects:
        _w = rect.get_height()
        plt.gca().annotate(str(round(_w,1))+'%',xy=(rect.get_x()+rect.get_width()/2, rect.get_y()+rect.get_height()+0.025),
                           ha='center', va='bottom',color='black',fontsize=8,weight='bold',clip_on=False)

    plt.gca().grid(False)
    sns.despine()

    plt.draw()
    plt.gcf().savefig(out_dir+'expenditures/'+pais+'_gasoline_as_pct_by_quintile.pdf',format='pdf',bbox_inches='tight')
    plt.gcf().savefig(out_dir+'expenditures/'+pais+'_gasoline_as_pct_by_quintile.png',format='png',bbox_inches='tight')

    ############################
    # Try to plot tax on top of expenditures
    #ax = plt.gca()
    plt.cla()
    final_FD_quints.plot(final_FD_quints.index,'Direct fuel consumption',kind='bar',color=quint_colors,legend=False)
    
    # Below labels the total cost, etc, by quintile
    if False:
        rects = plt.gca().patches
        for rect in rects:
            _w = rect.get_height()
            plt.gca().annotate(str(round(_w,1))+'%',xy=(rect.get_x()+rect.get_width()-0.025, rect.get_y()+rect.get_height()/2.),
                               ha='right', va='center',color='black',fontsize=8,weight='bold',clip_on=False)

    final_FD_quints.plot(final_FD_quints.index,'Direct fuel consumption tax',kind='bar',color=sns.color_palette('Set1', n_colors=9)[5],legend=False,bottom=final_FD_quints['Direct fuel consumption'],ax=plt.gca())
    plt.ylim([0,final_FD_quints[['Direct fuel consumption','Direct fuel consumption tax']].sum(axis=1).max()*1.05])

    plt.gca().grid(False)
    sns.despine()   

    plt.gca().set_xticklabels(quint_labels,ha='center',rotation=0)
    plt.ylabel('Direct fuel consumption [% of total expenditures]',fontsize=11,weight='bold',labelpad=8)
    plt.xlabel('') 

    plt.gcf().savefig(out_dir+'expenditures/'+pais+'_gasoline_as_pct_by_quintile_with_tax.pdf',format='pdf',bbox_inches='tight')
    plt.gcf().savefig(out_dir+'expenditures/'+pais+'_gasoline_as_pct_by_quintile_with_tax.png',format='png',bbox_inches='tight')

    plt.cla()

    ###################################
    # Put quintile info into final_CC_tot, final_CC_tot_CO2, final_CC_tot_nonCO2
    hies_FD_tot = hies_FD_tot.reset_index().set_index('cod_hogar')
    try: hies_FD_tot['quintile'] = hies_FD_tot['quintile'].astype('int')
    except: hies_FD_tot['quintile'] = hies_FD_tot['quintile'].astype('str')
    #
    hhwgts['quintile'] = hies_FD_tot['quintile'].copy()
    hhwgts = hhwgts.reset_index().set_index(['cod_hogar','quintile'])
    #
    final_CC['quintile'] = hies_FD_tot['quintile'].copy()
    final_CC = final_CC.reset_index().set_index(['cod_hogar','quintile'])
    #

    try: 
        final_CC_ind['quintile'] = hies_FD_tot['quintile'].copy()
        final_CC_ind = final_CC_ind.reset_index().set_index(['cod_hogar','quintile'])
        #
        final_CC_dir['quintile'] = hies_FD_tot['quintile'].copy()
        final_CC_dir = final_CC_dir.reset_index().set_index(['cod_hogar','quintile'])
        #
        final_CC_CO2['quintile'] = hies_FD_tot['quintile'].copy()
        final_CC_CO2 = final_CC_CO2.reset_index().set_index(['cod_hogar','quintile'])
        #
        final_CC_nonCO2['quintile'] = hies_FD_tot['quintile'].copy()
        final_CC_nonCO2 = final_CC_nonCO2.reset_index().set_index(['cod_hogar','quintile'])
        #
    except:  pass
    # ^ this (t/e) pair is for pais != 'brb'
    hies_FD_tot = hies_FD_tot.reset_index().set_index(['cod_hogar','quintile'])
    
    ##########################################################################################
    # Record sample (all countries) stats in hh_tax_cost_table.csv

    # total cost
    try: hhcost_t = pd.read_csv('output/all_countries/hh_tax_cost_table.csv').set_index('quintile')
    except: hhcost_t = pd.DataFrame({pais.upper():0,'quintile':['Q1','Q2','Q3','Q4','Q5']},index=None).set_index('quintile')

    # Direct
    try: hhcost_d = pd.read_csv('output/all_countries/hh_direct_tax_cost_table.csv').set_index('quintile')
    except: hhcost_d = pd.DataFrame({pais.upper():0,'quintile':['Q1','Q2','Q3','Q4','Q5']},index=None).set_index('quintile')

    # Indirect
    try: hhcost_i = pd.read_csv('output/all_countries/hh_indirect_tax_cost_table.csv').set_index('quintile')
    except: hhcost_i = pd.DataFrame({pais.upper():0,'quintile':['Q1','Q2','Q3','Q4','Q5']},index=None).set_index('quintile')

    # Direct frac of tax
    try: taxfrac_d = pd.read_csv('output/all_countries/hh_direct_tax_frac_table.csv').set_index('quintile')
    except: taxfrac_d = pd.DataFrame({pais.upper():0,'quintile':['Q1','Q2','Q3','Q4','Q5']},index=None).set_index('quintile')

    # Indirect frac of tax
    try: taxfrac_i = pd.read_csv('output/all_countries/hh_indirect_tax_frac_table.csv').set_index('quintile')
    except: taxfrac_i = pd.DataFrame({pais.upper():0,'quintile':['Q1','Q2','Q3','Q4','Q5']},index=None).set_index('quintile')

    # Indirect frac of tax - FOOD, ELEC, and PUBTRANS
    try: taxfrac_if = pd.read_csv('output/all_countries/hh_indirect_tax_foodnonCO2_frac_table.csv').set_index('quintile')
    except: taxfrac_if = pd.DataFrame({pais.upper():0,'quintile':['Q1','Q2','Q3','Q4','Q5']},index=None).set_index('quintile')

    try: taxfrac_ie = pd.read_csv('output/all_countries/hh_indirect_tax_elecCO2_frac_table.csv').set_index('quintile')
    except: taxfrac_ie = pd.DataFrame({pais.upper():0,'quintile':['Q1','Q2','Q3','Q4','Q5']},index=None).set_index('quintile')   
    try: taxfrac_ipt = pd.read_csv('output/all_countries/hh_indirect_tax_pubtransCO2_frac_table.csv').set_index('quintile')
    except: taxfrac_ipt = pd.DataFrame({pais.upper():0,'quintile':['Q1','Q2','Q3','Q4','Q5']},index=None).set_index('quintile')   


    _ = (100./1E6)*(final_CC.sum(axis=1)*hhwgts['hhwgt']).sum(level='quintile')/hies_FD_tot[['totex_pc','pcwgt']].prod(axis=1).sum(level='quintile')    
    for _nq in [1,2,3,4,5]: hhcost_t.loc['Q'+str(_nq),pais.upper()] = _.loc[_nq]
    
    if pais != 'brb':
        _ = (100./1E6)*(final_CC_dir.sum(axis=1)*hhwgts['hhwgt']).sum(level='quintile')/hies_FD_tot[['totex_pc','pcwgt']].prod(axis=1).sum(level='quintile')  
        for _nq in [1,2,3,4,5]: hhcost_d.loc['Q'+str(_nq),pais.upper()] = _.loc[_nq]

        _ = (100./1E6)*(final_CC_ind.sum(axis=1)*hhwgts['hhwgt']).sum(level='quintile')/hies_FD_tot[['totex_pc','pcwgt']].prod(axis=1).sum(level='quintile')  
        for _nq in [1,2,3,4,5]: hhcost_i.loc['Q'+str(_nq),pais.upper()] = _.loc[_nq]
        #
        #
        _ = (100.)*(final_CC_dir.sum(axis=1)*hhwgts['hhwgt']).sum(level='quintile')/(final_CC.sum(axis=1)*hhwgts['hhwgt']).sum(level='quintile')  
        for _nq in [1,2,3,4,5]: taxfrac_d.loc['Q'+str(_nq),pais.upper()] = _.loc[_nq]

        _ = (100.)*(final_CC_ind.sum(axis=1)*hhwgts['hhwgt']).sum(level='quintile')/(final_CC.sum(axis=1)*hhwgts['hhwgt']).sum(level='quintile')
        for _nq in [1,2,3,4,5]: taxfrac_i.loc['Q'+str(_nq),pais.upper()] = _.loc[_nq]
        #    
        _ = (100.)*(final_CC_nonCO2[fdict['frac_food'][1]]*hhwgts['hhwgt']).sum(level='quintile')/(final_CC.sum(axis=1)*hhwgts['hhwgt']).sum(level='quintile')  
        for _nq in [1,2,3,4,5]: taxfrac_if.loc['Q'+str(_nq),pais.upper()] = _.loc[_nq]


        _ = (100.)*(final_CC_CO2[fdict['frac_electricity'][1]]*hhwgts['hhwgt']).sum(level='quintile')/(final_CC.sum(axis=1)*hhwgts['hhwgt']).sum(level='quintile')  
        for _nq in [1,2,3,4,5]: taxfrac_ie.loc['Q'+str(_nq),pais.upper()] = _.loc[_nq]
        
        _ = (100.)*(final_CC_CO2[fdict['frac_pubtrans'][1]]*hhwgts['hhwgt']).sum(level='quintile')/(final_CC.sum(axis=1)*hhwgts['hhwgt']).sum(level='quintile')  
        for _nq in [1,2,3,4,5]: taxfrac_ipt.loc['Q'+str(_nq),pais.upper()] = _.loc[_nq]


        hhcost_t.to_csv(out_dir+'all_countries/hh_tax_cost_table.csv')
        hhcost_d.to_csv(out_dir+'all_countries/hh_direct_tax_cost_table.csv')
        hhcost_i.to_csv(out_dir+'all_countries/hh_indirect_tax_cost_table.csv')
        taxfrac_d.to_csv(out_dir+'all_countries/hh_direct_tax_frac_table.csv')
        taxfrac_i.to_csv(out_dir+'all_countries/hh_indirect_tax_frac_table.csv')
        taxfrac_if.to_csv(out_dir+'all_countries/hh_indirect_tax_foodnonCO2_frac_table.csv')
        taxfrac_ie.to_csv(out_dir+'all_countries/hh_indirect_tax_elecCO2_frac_table.csv')
        taxfrac_ipt.to_csv(out_dir+'all_countries/hh_indirect_tax_pubtransCO2_frac_table.csv')
    ########################################################################################## 


    ###################################
    # Cost of indirect carbon price increase (in $)

    final_FD_quints = pd.DataFrame(index=hies_FD.reset_index().set_index('quintile').sum(level='quintile').index).sort_index()
    
    final_FD_quints['indirect USD'] = (final_CC_ind.sum(axis=1)*hhwgts['hhwgt']).sum(level='quintile')/hhwgts['pcwgt'].sum(level='quintile')  
    
    final_FD_quints.plot(final_FD_quints.index,'indirect USD',kind='bar',color=quint_colors,legend=False)
    plt.gca().set_xticklabels(quint_labels,ha='right')
        
    plt.ylabel('Indirect carbon cost [INT$ per capita]',fontsize=11,labelpad=8)
    plt.xlabel('')
    plt.title(iso_to_name[pais],fontsize=14,weight='bold')
        
    rects = plt.gca().patches
    for rect in rects:
        _w = rect.get_width()
        plt.gca().annotate('$'+str(int(round(_w,0))),xy=(rect.get_x()+rect.get_width()/2,rect.get_y()+rect.get_height()+0.05),
                           ha='left', va='center',color='black',fontsize=8,weight='bold',clip_on=False)
        
    plt.gca().grid(False)
    sns.despine(left=True)

    plt.draw()
    plt.gcf().savefig(out_dir+'expenditures/'+pais+'_indirect_tax_total_USD_by_quintile.pdf',format='pdf',bbox_inches='tight')
    plt.gcf().savefig(out_dir+'expenditures/'+pais+'_indirect_tax_total_USD_by_quintile.png',format='png',bbox_inches='tight')

    # Plot total cost (stacked) in INT$
    plt.cla()
    final_FD_quints['direct USD'] = (final_CC_dir.sum(axis=1)*hhwgts['hhwgt']).sum(level='quintile')/hhwgts['pcwgt'].sum(level='quintile')  

    final_FD_quints.plot(final_FD_quints.index,'direct USD',kind='bar',color=quint_colors,legend=False)    
    final_FD_quints.plot(final_FD_quints.index,'indirect USD',kind='bar',color=quint_colors,legend=False,alpha=0.5,ax=plt.gca(),bottom=final_FD_quints['direct USD'])
        
    plt.gca().set_xticklabels(quint_labels,ha='right')
    plt.ylabel('Total carbon tax burden [INT$ per capita]',fontsize=11,labelpad=8)
    plt.xlabel('')

    sns.despine(left=True)
    plt.gca().grid(False)
    
    plt.draw()
    plt.gcf().savefig(out_dir+'expenditures/'+pais+'_tax_total_USD_by_quintile.pdf',format='pdf',bbox_inches='tight')
    plt.gcf().savefig(out_dir+'expenditures/'+pais+'_tax_total_USD_by_quintile.png',format='png',bbox_inches='tight')

    plt.cla()    


    ###################################
    # Cost of indirect carbon price increase (% of totex)
    final_FD_quints = pd.DataFrame(index=hies_FD.reset_index().set_index('quintile').sum(level='quintile').index).sort_index()
    
    final_FD_quints['pct of expenditures'] = (100./1E6)*(final_CC_ind.sum(axis=1)*hhwgts['hhwgt']).sum(level='quintile')/hies_FD_tot[['pcwgt','totex_pc']].prod(axis=1).sum(level='quintile')
        
    final_FD_quints.plot(final_FD_quints.index,'pct of expenditures',kind='bar',color=quint_colors,legend=False)
    plt.gca().set_xticklabels(quint_labels,ha='right')
    
    plt.ylabel('Indirect carbon cost relative to expenditures [%]',fontsize=11,weight='bold',labelpad=8)
    plt.xlabel('')
    plt.title(iso_to_name[pais],fontsize=14,weight='bold')
    
    rects = plt.gca().patches
    for rect in rects:
        _w = rect.get_width()
        plt.gca().annotate(str(round(_w,1))+'%',xy=(rect.get_x()+1.025*rect.get_width(),rect.get_y()+rect.get_height()/2.),
                           ha='left', va='center',color='black',fontsize=8,weight='bold',clip_on=False)
        
    plt.gca().grid(False)
    sns.despine(left=True)
    
    plt.draw()
    plt.gcf().savefig(out_dir+'expenditures/'+pais+'_indirect_tax_as_pct_of_gastos_by_quintile.pdf',format='pdf',bbox_inches='tight')
    plt.gcf().savefig(out_dir+'expenditures/'+pais+'_indirect_tax_as_pct_of_gastos_by_quintile.png',format='png',bbox_inches='tight')
    plt.cla()    


    ###################################
    # Cost of direct carbon price increase (in $)
    final_FD_quints = pd.DataFrame(index=hies_FD.reset_index().set_index('quintile').sum(level='quintile').index).sort_index()
    
    final_FD_quints['total USD'] = (final_CC_dir.sum(axis=1)*hhwgts['hhwgt']).sum(level='quintile')/hhwgts['pcwgt'].sum(level='quintile')
    
    final_FD_quints.plot(final_FD_quints.index,'total USD',kind='bar',color=quint_colors,legend=False)
    plt.gca().set_xticklabels(quint_labels,ha='right')

    plt.ylabel('Carbon tax on fuels [INT$ per capita]',fontsize=11,weight='bold',labelpad=8)
    plt.xlabel('')
    plt.title(iso_to_name[pais],fontsize=14,weight='bold')
    
    rects = plt.gca().patches
    for rect in rects:
        _w = rect.get_width()
        plt.gca().annotate('$'+str(int(round(_w,0))),xy=(rect.get_x()+1.025*rect.get_width(),rect.get_y()+rect.get_height()/2.),
                           ha='left', va='center',color='black',fontsize=8,weight='bold',clip_on=False)
        
    plt.gca().grid(False)
    sns.despine(left=True)
    
    plt.draw()
    plt.gcf().savefig(out_dir+'expenditures/'+pais+'_direct_tax_total_USD_by_quintile.pdf',format='pdf',bbox_inches='tight')  
    plt.cla()    
    

    ###################################
    # Cost of direct carbon price increase (% of tot_exp)
    final_FD_quints = pd.DataFrame(index=hies_FD.reset_index().set_index('quintile').sum(level='quintile').index).sort_index()
        
    final_FD_quints['pct of expenditures'] = 100./1E6*(final_CC_dir.sum(axis=1)*hhwgts['hhwgt']).sum(level='quintile')/hies_FD_tot[['totex_pc','pcwgt']].prod(axis=1).sum(level='quintile')

    final_FD_quints.plot(final_FD_quints.index,'pct of expenditures',kind='bar',color=quint_colors,legend=False)
    plt.gca().set_xticklabels(quint_labels,ha='right')

    #_x_ticks = plt.gca().get_xticks()
    #plt.gca().set_xticklabels([str(round(_x,1)) for _x in _x_ticks[::2]]) 
 
    plt.ylabel('Carbon tax on direct fuel consumption [% of total expenditures]',fontsize=11,weight='bold',labelpad=8)
    plt.xlabel('')
    plt.title(iso_to_name[pais],fontsize=14,weight='bold')
    
    rects = plt.gca().patches
    for rect in rects:
        _w = rect.get_width()
        plt.gca().annotate(str(round(_w,1))+'%',xy=(rect.get_x()+rect.get_width()+0.002,rect.get_y()+rect.get_height()/2.),
                           ha='left', va='center',color='black',fontsize=8,clip_on=False,weight='bold')

    plt.gca().grid(False)
    sns.despine()
    
    plt.draw()
    plt.gcf().savefig(out_dir+'expenditures/'+pais+'_direct_tax_as_pct_of_gastos_by_quintile.pdf',format='pdf',bbox_inches='tight')  
    plt.cla()


    ###################################
    # Cost of direct & indirect carbon price increase (% of totex)
    do_column_annotations = False
    plt.figure(figsize=(6,6))  

    final_FD_quints = pd.DataFrame(index=hies_FD.reset_index().set_index('quintile').sum(level='quintile').index).sort_index()

    ##########
    # All CO2-related costs
    final_FD_quints['CO2 expenditures'] = (100./1E6)*(final_CC_CO2.sum(axis=1)*hhwgts['hhwgt']).sum(level='quintile')/hies_FD_tot[['pcwgt','totex_pc']].prod(axis=1).sum(level='quintile')

    ##########
    # All nonCO2-related costs
    final_FD_quints['nonCO2 expenditures'] = (100./1E6)*(final_CC_nonCO2.sum(axis=1)*hhwgts['hhwgt']).sum(level='quintile')/hies_FD_tot[['pcwgt','totex_pc']].prod(axis=1).sum(level='quintile')
    
    orig_columns = final_FD_quints.columns

    ##########
    # This grabs the largest category endogenously
    find_max_CO2 = pd.DataFrame({'abs':[]})
    find_max_nonCO2 = pd.DataFrame({'abs':[]})
    
    for _ex in fdict:

        try: find_max_CO2.loc[_ex] = 100./1E6*(hhwgts['hhwgt']*final_CC_CO2[fdict[_ex][1]]).sum()/hies_FD_tot[['pcwgt','totex_pc']].prod(axis=1).sum()
        except: find_max_CO2.loc[_ex] = 0.

        try: find_max_nonCO2.loc[_ex] = 100./1E6*(hhwgts['hhwgt']*final_CC_nonCO2[fdict[_ex][1]]).sum()/hies_FD_tot[['pcwgt','totex_pc']].prod(axis=1).sum()
        except: find_max_nonCO2.loc[_ex] = 0.

    for _plot in ['','CO2','nonCO2']:

        # Reset the dataframe
        final_FD_quints = final_FD_quints.drop([_c for _c in final_FD_quints.columns if _c not in orig_columns],axis=1).sort_index()

        # SORT
        find_max_CO2 = find_max_CO2.sort_values('abs',ascending=False)
        find_max_nonCO2 = find_max_nonCO2.sort_values('abs',ascending=False)

        _floor = 0.75
        other_CO2_min, other_nonCO2_min = _floor*find_max_CO2['abs'].sum(), _floor*find_max_nonCO2['abs'].sum()

        # Grab the categories by descending magnitude until we reach at least other_CO2_min: 
        _, _ix = 0, 0
        while _ < other_CO2_min:
            if float(find_max_CO2.iloc[_ix]) > 0.1*find_max_CO2['abs'].sum():
                _ += float(find_max_CO2.iloc[_ix])
                _ix += 1
            else: break
        find_max_CO2 = find_max_CO2.iloc[:_ix]

        # Grab the categories by descending magnitude until we reach at least other_nonCO2_min: 
        _, _ix = 0, 0
        while _ < other_nonCO2_min:
            if float(find_max_nonCO2.iloc[_ix]) > 0.1*find_max_nonCO2['abs'].sum():
                _ += float(find_max_nonCO2.iloc[_ix])
                _ix += 1
            else: break
        find_max_nonCO2 = find_max_nonCO2.iloc[:_ix]
        
        # Get the quintile info for the categories that remain:
        if _plot == '': my_merge_ix = pd.concat([find_max_CO2,find_max_nonCO2]).index
        elif _plot == 'CO2': my_merge_ix = find_max_CO2.index
        elif _plot == 'nonCO2': my_merge_ix = find_max_nonCO2.index

        for _ix in my_merge_ix:

            if _plot != 'nonCO2':
                final_FD_quints['CO2 '+_ix] = 100./1E6*((hhwgts['hhwgt']*final_CC_CO2[fdict[_ix][1]]).sum(level='quintile')
                                                        /hies_FD_tot[['hhwgt','totex_hh']].prod(axis=1).sum(level='quintile'))
            if _plot != 'CO2':
                final_FD_quints['nonCO2 '+_ix] = 100./1E6*((hhwgts['hhwgt']*final_CC_nonCO2[fdict[_ix][1]]).sum(level='quintile')
                                                           /hies_FD_tot[['hhwgt','totex_hh']].prod(axis=1).sum(level='quintile'))

        _other_CO2 = final_FD_quints['CO2 expenditures'].copy()
        _other_nonCO2 = final_FD_quints['nonCO2 expenditures'].copy()    
        
        my_labels = []

        _bot = [0,0,0,0,0]
        for _col in final_FD_quints.columns:
            # Loop over/plot CO2 & non-CO2 emissions
            if (_plot != 'nonCO2' and _col[:4] == 'CO2 ') or (_plot == 'nonCO2' and _col[:7] == 'nonCO2 '):

                _col_name = _col.replace('nonCO2 ','').replace('CO2 ','')
                if _col_name == 'expenditures': continue

                _alp = [0.85,0.40]

                my_labels.append(fdict[_col_name][1].replace('Public transport','Public\ntransport')
                                 .replace('Liquid fuels','Liquid\nfuels')
                                 .replace('Household goods','Household\ngoods')
                                 .replace('Petroleum, gasoline & diesel','Petroleum,\ngasoline & diesel'))

                if _plot != 'nonCO2':
                    try: _p = plt.bar(final_FD_quints.index, final_FD_quints['CO2 '+_col_name],color=fdict[_col_name][2],bottom=_bot,alpha=_alp[0],width=0.5)
                    except: _p = plt.bar(final_FD_quints.index, final_FD_quints['CO2 '+_col_name],color=greys[4],bottom=_bot,alpha=_alp[0],width=0.5)
                    _bot += final_FD_quints['CO2 '+_col_name]
                    _other_CO2 -= final_FD_quints['CO2 '+_col_name]

                if _plot != 'CO2':
                    try: _p = plt.bar(final_FD_quints.index, final_FD_quints['nonCO2 '+_col_name],color=fdict[_col_name][2],bottom=_bot,alpha=_alp[1],width=0.5)
                    except: _p = plt.bar(final_FD_quints.index, final_FD_quints['nonCO2 '+_col_name],color=greys[4],bottom=_bot,alpha=_alp[1],width=0.5)
                    _bot += final_FD_quints['nonCO2 '+_col_name]
                    _other_nonCO2 -= final_FD_quints['nonCO2 '+_col_name]

                try: save_to_results_file(iso_to_name[pais],'Increase in household costs due to CO2 emissions from '+fdict[_col_name][1]+', mean',
                                          np.array(final_FD_quints['CO2 '+_col_name].round(3).squeeze()),units='% of total expenditures')
                except: pass
                try: save_to_results_file(iso_to_name[pais],'Increase in household costs due to non-CO2 emissions from '+fdict[_col_name][1]+', mean',
                                          np.array(final_FD_quints['nonCO2 '+_col_name].round(3).squeeze()),units='% of total expenditures')
                except: pass

        if _plot != 'nonCO2':
            _p = plt.bar(final_FD_quints.index, _other_CO2,color=greys[4],bottom=_bot,alpha=0.80,width=0.5)
            _bot += _other_CO2
        if _plot != 'CO2':
            _p = plt.bar(final_FD_quints.index, _other_nonCO2,color=greys[4],bottom=_bot,alpha=0.50,width=0.5)
            _bot += _other_nonCO2
        my_labels.append('Other')

        try: save_to_results_file(iso_to_name[pais],'Increase in household costs due to CO2 emissions from other consumption, mean ',
                                  np.array(_other_CO2.round(3).squeeze()),units='% of total expenditures')
        except: pass
        try: save_to_results_file(iso_to_name[pais],'Increase in household costs due to CO2 emissions from other consumption, mean',
                                  np.array(_other_nonCO2.round(3).squeeze()),units='% of total expenditures')
        except: pass

        plt.gca().set_xticklabels(['','']+quint_labels,ha='center')
        plt.xlim(-0.6,5.5)
    
        __ = ''
        if _plot != '': __ = '('+_plot.replace('non','non-').replace('CO2','CO$_2$')+')'

        if pais == 'brb': _ylabel = 'Energy tax impact as % of expenditures' 
        else: _ylabel = (r'Carbon '+__+' tax as % of expenditures').replace('  ',' ')
        plt.ylabel((_ylabel),fontsize=11,labelpad=8)
        #plt.xlabel('Quintile',fontsize=11,labelpad=8)

        y_top, x_left = 0.,0.
        y_bottom = []
        x_bottom = 1E3

        # Gather info about stacks
        rects = plt.gca().patches
        for rect in rects: 
            _x,_y = rect.xy

            if y_top < _y+rect.get_height(): 
                y_top = _y+rect.get_height()
            
            if x_left < _x:
                x_left = rect.get_x()

            if _x <= x_bottom: 
                x_bottom = _x
                y_bottom.append(_y+rect.get_height())

        # Label the individual height of each block in each stack
        for rect in rects:
            _col = greys[6]
            _pct = ''
            if rect.get_y() == 0:
                _col = greys[7]
                _pct = '%'
            #if rect.get_y() == y_bottom[0]: 
            #    _col = greys[7]
            # ^ only catches Q1
            
            _w = rect.get_width()

            hack_y = rect.get_y()+rect.get_height()-0.6*y_top/100
            if hack_y < 0: hack_y = 0
        
            _l = round(rect.get_height(),2)
            if _plot == '': _l = round(rect.get_height(),2)
            
            if _l < 0.05: _l = r'$<$0.05'

            if do_column_annotations:
                if rect.get_height() > 2.*y_top/100:
                    plt.gca().annotate(str(_l)+_pct,xy=(rect.get_x()+_w/2,hack_y),ha='center', va='top',color=_col,fontsize=4.5,weight=40,clip_on=False)
                #elif rect.get_height() > 0.5*y_top/100:
                #    plt.gca().annotate(str(_l)+_pct,xy=(rect.get_x()+_w/2,rect.get_y()+rect.get_height()-0.3*y_top/100),ha='center', va='top',color=_col,fontsize=3.5,weight=40,clip_on=False)
               
        # Label direct emissions
        #if _plot == '':
        #    plt.plot([x_bottom-0.55,x_bottom+0.5],[y_bottom[0],y_bottom[0]],linewidth=1.25,color=greys[6])
        #    plt.annotate('Direct',xy=(x_bottom-0.60,y_bottom[0]-0.6*y_top/100),va='top',ha='right',rotation=0,fontsize=6.5,color=greys[6])
        #    if do_column_annotations:
        #        plt.annotate(r'CO$_2$',xy=(x_bottom-0.01,y_bottom[0]-0.6*y_top/100),va='top',ha='right',rotation=0,fontsize=6.5,color=greys[6])

        #elif _plot == 'CO2': 
        #    plt.plot([x_bottom-0.50,x_bottom+0.5],[y_bottom[0],y_bottom[0]],linewidth=1.25,color=greys[6])
        #    # This line does the labeling without GHG label
        #    plt.annotate('Direct',xy=(x_bottom-0.03,y_bottom[0]-y_top/100),va='top',ha='right',rotation=0,fontsize=6.5,color=greys[6])

        # Put the categorical & GHG labels in
        if _plot == '':

            for _n, _lab in enumerate(my_labels):
                __n = 2*_n+1

                plt.plot([x_bottom-0.55,x_bottom+0.5],[y_bottom[__n],y_bottom[__n]],linewidth=1.25,color=greys[6])
                plt.annotate(my_labels[_n].replace(' &','\n&'),
                             xy=(x_bottom-0.60,y_bottom[__n]-0.6*y_top/100),va='top',ha='right',linespacing=0.9,fontsize=6.5,color=greys[6])
                if do_column_annotations or (not do_column_annotations and my_labels[_n] == 'Other'):
                    plt.annotate(r'CO$_2$',xy=(x_bottom-0.01,y_bottom[__n-1]-0.6*y_top/100),va='top',ha='right',rotation=0,fontsize=6.5,color=greys[6])
                    if y_bottom[__n] >= y_bottom[__n-1]+2.5*y_top/100:
                        plt.annotate(r'non-CO$_2$',xy=(x_bottom-0.01,y_bottom[__n]-0.6*y_top/100),va='top',ha='right',rotation=0,fontsize=6.5,color=greys[6])
                    elif y_bottom[__n] >= y_bottom[__n-1]+0.5*y_top/100: 
                        plt.annotate(r'non-CO$_2$',xy=(x_bottom-0.01,y_bottom[__n]-0.4*y_top/100),va='top',ha='right',rotation=0,fontsize=6.5,color=greys[6])
        
        elif _plot == 'CO2':
            for _n, _lab in enumerate(my_labels):
                __n = _n
                plt.plot([x_bottom-0.50,x_bottom+0.5],[y_bottom[__n],y_bottom[__n]],linewidth=1.25,color=greys[6])

                # These 2 lines do the labeling as in the all-GHG plot
                #plt.annotate(my_labels[_n],xy=(x_bottom-0.60,y_bottom[__n]-0.6*y_top/100),va='top',ha='right',linespacing=0.9,fontsize=6.5,color=greys[6])
                #plt.annotate(r'CO$_2$',xy=(x_bottom-0.01,y_bottom[__n]-0.6*y_top/100),va='top',ha='right',rotation=0,fontsize=5.5,color=greys[6])

                # This line does the labeling without GHG label
                plt.annotate(my_labels[_n],xy=(x_bottom-0.03,y_bottom[__n]-y_top/100),va='top',ha='right',rotation=0,fontsize=6.5,color=greys[6])

        elif _plot == 'nonCO2':
            for _n, _lab in enumerate(my_labels):
                __n = _n
                plt.plot([x_bottom-0.40,x_bottom+0.5],[y_bottom[__n],y_bottom[__n]],linewidth=1.25,color=greys[6])

                # These 2 lines do the labeling as in the all-GHG plot
                #plt.annotate(my_labels[_n],xy=(x_bottom-0.60,y_bottom[__n]-0.6*y_top/100),va='top',ha='right',linespacing=0.9,fontsize=6.5,color=greys[6])
                #plt.annotate(r'nonCO$_2$',xy=(x_bottom-0.01,y_bottom[__n-1]-0.6*y_top/100),va='top',ha='right',rotation=0,fontsize=5.5,color=greys[6])

                # This line does the labeling without GHG label
                plt.annotate(my_labels[_n],xy=(x_bottom-0.03,y_bottom[__n]-y_top/100),va='top',ha='right',rotation=0,fontsize=6.5,color=greys[6])
    
        plt.gca().grid(False)
        sns.despine(left=True)

        if pais == 'mex': plt.yticks([0,0.5,1.0,1.5,2.0])
    
        plt.draw()
        plt.gcf().savefig(out_dir+'expenditures/'+pais+'_total_tax_as_pct_of_gastos_by_quintile'+_plot+'.pdf',format='pdf',bbox_inches='tight')
        plt.cla(); plt.close('all')

        _bot = 0

    ###################################
    # Cumulative cost vs. share of consumers
    plt.figure(figsize=(6,6))   
    
    # Forget about quintiles
    hhwgts = hhwgts.reset_index('quintile')
    final_CC = final_CC.reset_index('quintile')
    hies_FD_tot = hies_FD_tot.reset_index('quintile')
    try: hies_FD_tot.index = hies_FD_tot.index.astype('int')
    except: hies_FD_tot.index = hies_FD_tot.index.astype('str')

    _df = pd.DataFrame({'pc_expenditures':hies_FD_tot.totex_pc.copy(),
                        'quintile':final_CC.quintile.copy()},
                       index=final_CC.index.copy())
    _df['tot_tax'] = final_CC.sum(axis=1)*hhwgts['hhwgt']
    _df['tot_ex'] = hies_FD_tot[['totex_pc','pcwgt']].prod(axis=1)
    _df['pcwgt'] = hhwgts['pcwgt'].copy()

    _df = _df.dropna()    
    _df = _df.reset_index().set_index('cod_hogar')
    _df = _df.sort_values('pc_expenditures',ascending=True)

    _df['cumpct_pcwgt'] = 100.*_df['pcwgt'].cumsum()/_df['pcwgt'].sum()
    _df['cumpct_tot_tax'] = 100.*_df['tot_tax'].cumsum()/_df['tot_tax'].sum()

    _df = _df.reset_index().set_index(['cod_hogar','quintile']) 
    quint_ix_x = np.array(_df['cumpct_pcwgt'].max(level='quintile').squeeze())
    quint_ix_y = np.array(_df['cumpct_tot_tax'].max(level='quintile').squeeze())
        
    plt.xlabel('Share of consumers [%]',fontsize=11,weight='bold',labelpad=8)
    plt.ylabel('Cumulative cost [%]',fontsize=11,weight='bold',labelpad=8)

    ax = plt.plot([0,100],[0,100],color=greys[4],linestyle='--',linewidth=2.0)
    plt.plot(_df['cumpct_pcwgt'],_df['cumpct_tot_tax'],linestyle='-',linewidth=2.0)

    for _nq,_q in enumerate(quint_ix_x):
        plt.plot([_q,_q],[0,quint_ix_y[_nq]],color=greys[3],linewidth=1.25,linestyle=':')
        # ^ vertical
        plt.plot([0,_q],[quint_ix_y[_nq],quint_ix_y[_nq]],color=greys[3],linewidth=1.25,linestyle=':')
        # ^ horizontal
        plt.scatter(_q,quint_ix_y[_nq],color=greys[5],zorder=100,alpha=1,s=15)
        # ^ marker on curve
        lab = ''
        if _nq == 0: lab = '% of\nrevenue'
        if _nq != 4: plt.annotate(str(round(quint_ix_y[_nq],2))+lab,xy=(_q,quint_ix_y[_nq]),
                                  xytext=(_q+3,quint_ix_y[_nq]-2),arrowprops=dict(arrowstyle="-",connectionstyle="arc3,rad=0.0"),
                                  va='center',ha='left',weight='bold',fontsize=7,color=greys[6])
        # ^ label: % of cost

    sns.despine()
    plt.gca().grid(False)
    plt.xlim(0,101)
    plt.ylim(0,101)
     
    plt.gcf().savefig(out_dir+'expenditures/'+pais+'_ranked_redistribution.pdf',format='pdf',bbox_inches='tight')
    plt.gcf().savefig(out_dir+'expenditures/'+pais+'_ranked_redistribution.png',format='png',bbox_inches='tight')

    plt.close('all')

    return hies_FD, final_CC_ind.sum(axis=1).to_frame(name='totex_hh'), final_CC_dir.sum(axis=1).to_frame(name='totex_hh')
