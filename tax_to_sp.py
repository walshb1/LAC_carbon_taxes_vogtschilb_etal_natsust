#python
import matplotlib
matplotlib.use('AGG')

import os
import sys, glob
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt,lines

from multiprocessing import Pool
from itertools import repeat
from itertools import product

#local
from libraries.lib_all_countries import *
from libraries.lib_get_hh_survey import get_hh_survey
from libraries.lib_enrollment import targeting_exercise
from libraries.lib_country_params import get_uclip,iso_to_name
from libraries.lib_plot_gtap_expenditures import plot_gtap_exp
from libraries.lib_results_to_excel import save_to_results_file
#from libraries.lib_latex import write_latex_header, write_latex_footer
from libraries.lib_pol_dict import get_policies,get_pol_dict#,update_pol_dict 
from libraries.lib_reverse_engineer_spending_from_net_beneficiaries import reverse_engineer
from libraries.lib_common_plotting_functions import sns_pal, greys, title_legend_labels, get_order, prettyf

sns.set_style('whitegrid')
pd.set_option('display.width', 220)
do_scatter = False
global_fontsize = 10

#########################
# look at command line to choose country
todos = ['arg','bol','bra','chl','col','cri','ecu','mex','gtm','pan','hnd','per','slv','nic','pry','ury']
# -->  NOT USING 'brb' because GTAP only has as part of Caribbean IOT (xcb)
correo = []

if len(sys.argv) > 1: miPais = sys.argv[1]
else: miPais = 'all'

if '[' in miPais: correo = np.array(miPais.replace('[','').replace(']','').split(','))
else: correo.append(miPais)

if miPais == 'all': correo = todos
# Colombia, Ecuador, Paraguay, Dominican Rep
print(correo)

# arg / Argentina
# bol / Bolivia
# bra / Brazil
# brb / Barbados
# chl / Chile
# col / Colombia
# cri / Costa Rica
# ecu / Ecuador
# gtm / Guatemala
# mex / Mexico
# pan / Panama
# pry / Paraguay
# per / Peru
# slv / El Salvador

#########################
#########################
def run_tax_to_sp(miPais,verbose=False):

    _ss = get_hh_survey(miPais)
    try: summary_stats = pd.read_csv('output/all_countries/survey_summary_stats.csv').set_index('pais')
    except: 
        summary_stats = pd.DataFrame(index=todos)
        summary_stats.index.name = 'pais'
    summary_stats.loc[miPais,'n_households_in_sample'] = _ss.shape[0]
    summary_stats.loc[miPais,'population'] = _ss[['miembros_hogar','factor_expansion']].prod(axis=1).sum()
    summary_stats.loc[miPais,'survey_name'] = _ss['encuesta'].iloc[0]
    summary_stats.loc[miPais,'year'] = _ss['anio'].iloc[0]
    summary_stats.to_csv('output/all_countries/survey_summary_stats.csv')

    out_dir = 'output/'
    if miPais == 'brb': out_dir = '/Users/brian/Desktop/Dropbox/IDB/Barbados/output/'

    do_verbose=False
    do_optimized_SP = False
    do_trsgob = False
    do_tax_food = True

    do_better = 25
    policies = get_policies(do_better)
    pdict = get_pol_dict(miPais,policies)

    #########################
    # Run expenditures plots, return post-bridge matrix in HIES categories.. 
    # --> inside this, we use gtap_to_hies script to translate both expenditures & cc into HIES cats
    _ = plot_gtap_exp(miPais,do_tax_food,verbose=do_verbose)
    final_FD, final_CC_ind, final_CC_dir = _

    final_FD = final_FD.reset_index('quintile')
    final_CC_ind = final_CC_ind.reset_index('quintile')
    final_CC_dir = final_CC_dir.reset_index('quintile')

    final_FD = final_FD.drop(['decile','quintile'],axis=1)

    #########################
    # GET EXPENDITURES
    # --> Loaded directly from hies file
    final_meta = get_hh_survey(miPais)[['pais','miembros_hogar','factor_expansion']]
    final_meta['pcwgt'] = final_meta[['miembros_hogar','factor_expansion']].prod(axis=1).astype(float)
    print('This must equal the total FD vector:',(final_FD.sum(axis=1)*final_meta['factor_expansion']).sum())

    final_CC_ind['totex_pc'] = final_CC_ind['totex_hh']/final_meta['miembros_hogar']
    final_CC_dir['totex_pc'] = final_CC_dir['totex_hh']/final_meta['miembros_hogar']
    final_meta[['dir_pc','dir_hh']] = final_CC_dir[['totex_pc','totex_hh']]
    final_meta[['ind_pc','ind_hh']] = final_CC_ind[['totex_pc','totex_hh']]
    # ^ per cap weighting

    hies_income = get_hh_survey(miPais)[[#'ict',              # HH INCOME
                                         #'ing_lab',          # -- hh income from labor
                                         #'ing_ren',          # -- hh income from rent
                                         #'ing_tpriv',        # -- hh income from private transfers
                                         #'ing_otro',         # -- hh income from other sources (extraordinary)
                                         'ing_tpub',         # -- hh income from government transfers
                                         'ing_ct',           # -- -- hh income from conditional (CCT) & unconditional (UCT) transfers
                                         #'ing_ct_mon',       # -- -- -- hh income (monetary) from CCT/UCT transfers
                                         #'ing_ct_nomon',     # -- -- -- hh income  (non-monetary) from CCT/UCT transfers
                                         #'ing_trsgob',       # -- -- hh income from other government transfers
                                         #'ing_trsgob_mon',   # -- -- -- hh income (monetary) from other government transfers
                                         #'ing_trsgob_nomon', # -- -- -- hh income (non-monetary) from other government transfers
                                         'gct'               # HH EXPENDITURES
                                         ]]
    hies_income['gct'] = hies_income['gct'].clip(lower=1E-5)

    # Set flag if there's no info on CT
    hay_CT = True
    if (miPais == 'arg' or miPais == 'chl'
        or miPais == 'pan' or miPais == 'pry'
        or miPais == 'XXX'
        or hies_income['ing_ct'].isnull().values.all()): 
        hay_CT = False
        hies_income['ing_ct'] = hies_income['ing_tpub'].copy()

    pdict = get_pol_dict(miPais,policies,hay_CT)
    hies_income = hies_income.fillna(0)

    try:ano = int(get_hh_survey(miPais)['anio'].mean())
    except: ano = 2018

    # Sanity check
    if final_meta.shape[0] != final_FD.shape[0]:
        print('\n'+miPais+': unable to load carbon cost info for',(final_meta.shape[0]-final_FD.dropna().shape[0]),'hh!')
        assert(False)

    #########################
    # Use final_FD to scale hies_income:
    # --> could first scale using to_intd_2010, but that is redundant
    mean_sf = (final_meta['factor_expansion']*(1.E6*final_FD.sum(axis=1)/hies_income['gct'])).sum()/final_meta['factor_expansion'].sum()

    #########################
    # PLOT: scale factors
    try:
        ax = plt.gca()

        heights, bins = np.histogram((1.E6*final_FD.sum(axis=1)/hies_income['gct']).clip(upper=3*mean_sf),bins=50,weights=final_meta['factor_expansion'])
    
        ax.bar((bins[1]-bins[0])/2+bins[:-1], heights, width=(bins[1]-bins[0]), label='LCL to INTD = '+str(round(get_lcu_to_2010intd(miPais,ano),3)), 
               facecolor=sns_pal[0],edgecolor=sns_pal[0],alpha=0.45)
    
        lgd = title_legend_labels(ax,iso_to_name[miPais],lab_x='Household scale factor (HIES to FD)',lab_y='Number of households')
        plt.gcf().savefig(out_dir+'sp/'+miPais+'_hies_to_gtap_scale_factors.pdf',format='pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')
        #plt.gcf().savefig(out_dir+'sp/'+miPais+'_hies_to_gtap_scale_factors.png',format='png',bbox_extra_artists=(lgd,), bbox_inches='tight')

        plt.cla(); plt.close('all')
    except: pass

    for _hc in [_ for _ in hies_income.columns if _ !='gct']:
        hies_income[_hc] *= (1E6*final_FD.sum(axis=1).clip(lower=0)/hies_income['gct'].clip(lower=0)).fillna(0)
    hies_income['gct'].update(1E6*final_FD.sum(axis=1))

    # FLAG ^ above: should that 1E-10 be there? is that OK? NO
    if hies_income.loc[hies_income.ing_ct<0].shape[0]!=0:
        assert(hies_income.loc[hies_income['ing_ct']<0].shape[0]==0)

    ########################################
    # Deprecated currency conversions
    #
    ##fx_fac = 1./get_fx(miPais,ano) # inverse b/c returns USD to local  
    ##ppp_fac = get_ppp(miPais,ano)
    ##usd_2011_fac = get_2011usd(miPais,ano)
    #to_intd_2010 = get_lcu_to_2010intd(miPais,ano)
    #print('\n\n\n'+miPais+': using currency sf =',round(to_intd_2010,3))
    
    # Convert all columns in hies_income to international USD using to_ind_2010
    #for ic in hies_income.columns: hies_income[ic] = hies_income[ic].astype(float)*to_intd_2010
    #for mc in ['dir_pc','dir_hh','ind_pc','ind_hh']: final_meta[mc] = final_meta[mc].astype(float)*to_intd_2010


    #########################
    # Choose per cap vs. hh weighting
    #
    wgt,dec_size = 'factor_expansion','nHH'

    run_percap = True
    # ^ set to true to look at individuals instead of hh
    if run_percap:
        wgt, dec_size = 'pcwgt','pop'

        for invar in hies_income.columns: hies_income[invar] = (hies_income[invar]/final_meta['miembros_hogar']).astype(float)
        # Don't scale carbon cost, because I have those columns already in per cap units

        final_meta = final_meta.rename(columns={'dir_pc':'cc_dir','ind_pc':'cc_ind'})        
        final_meta = final_meta.drop(['dir_hh','ind_hh'],axis=1)

        print('\n'+miPais+': running at per cap level!\n')
    else: 
        final_meta = final_meta.drop(['dir_pc','ind_pc'],axis=1)
        final_meta = final_meta.rename(columns={'dir_hh':'cc_dir','ind_hh':'cc_ind'})

        print('\n'+miPais+': Running at household level!\n')

    # Merge final_meta and hies_income
    df = pd.merge(final_meta.reset_index(),hies_income.reset_index(),on='cod_hogar').set_index('cod_hogar').dropna()
    df['carbon_cost'] = df[['cc_dir','cc_ind']].sum(axis=1)
    df = df.drop(['cc_dir','cc_ind'],axis=1)

    # hh income after carbon taxes
    df['new_expenditure'] = df['gct']-df['carbon_cost']

    mean_gct = (final_meta['factor_expansion']*df['gct']).sum()/final_meta['factor_expansion'].sum()
    mean_cc = (final_meta['factor_expansion']*df['carbon_cost']).sum()/final_meta['factor_expansion'].sum()
    print('\nmean expenditures:',mean_gct)
    print('mean carbon cost:',mean_cc)

    pdict['c_rev'] = float(df[[wgt,'carbon_cost']].prod(axis=1).sum())
    dist_frac = 1.0 # what fraction of the revenue is going to be redistributed through SP?
    print(miPais+': total carbon revenue = '+str(round(pdict['c_rev']/1e6,1))+'M')
    print(miPais+': ... of which: '+str(round(pdict['c_rev']*dist_frac/1e6,1))+' ('+str(100.*dist_frac)+'%) is distributed')

    #########################
    ## Set flags based on hh enrollment in existing programs?
    # (maybe useful if the inclusion criteria get more complex)
    #df['in_CT'] = False; df.loc[df.ing_ct>0,'in_CT'] = True
    

    #########################
    # Assign quintile based on ict (household income) or GCT?
    #_quintiles=np.arange(0.20, 1.01, 0.20)
    #df = df.groupby('pais',sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.gct),reshape_data(x.pcwgt),_quintiles),'quintile','gct'))
    try: df['quintile'] = final_CC_dir['quintile'].copy()
    except: df['quintile'] = final_CC['quintile'].copy()
    #

    if df.shape[0] != df.dropna(how='any').shape[0]:
        print(df.shape[0],df.dropna().shape[0])
        df['quintile'] = df['quintile'].fillna(-1)
        df.to_csv('~/Desktop/wrong.csv')
        print('Need to figure out why quintile isnt categorizing some hh')
        assert(False)

    if False:
        df = df.groupby('pais',sort=True).apply(lambda x:match_percentiles(x,perc_with_spline(reshape_data(x.ict),reshape_data(x.pcwgt),_quintiles),'quintile_ict','ict'))
        
    df = df.reset_index().set_index('quintile').drop([i for i in ['index'] if i in df.columns],axis=1)
    
    #########################
    # Collect average info about each quintile
    df_dec = pd.DataFrame(index=df.sum(level='quintile').index).sort_index()
    df_dec['pop'] = df['pcwgt'].sum(level='quintile')
    df_dec['nHH'] = df['factor_expansion'].sum(level='quintile')

    pdict['tot_cost_ct'] = df[[wgt,'ing_ct']].prod(axis=1).sum()

    df_dec['mean_val_tpub'] = df.loc[df.ing_tpub>0,[wgt,'ing_tpub']].prod(axis=1).sum(level='quintile')/df.loc[df.ing_tpub>0,wgt].sum(level='quintile')
    df_dec['mean_val_ct'] = df.loc[df.ing_ct>0,[wgt,'ing_ct']].prod(axis=1).sum(level='quintile')/df.loc[df.ing_ct>0,wgt].sum(level='quintile')
    #df_dec['mean_val_trsgob'] = df.loc[df.ing_trsgob>0,[wgt,'ing_trsgob']].prod(axis=1).sum(level='quintile')/df.loc[df.ing_trsgob>0,wgt].sum(level='quintile')   

    df_dec['pct_receiving_tpub'] = 100.*df.loc[df.ing_tpub>0,wgt].sum(level='quintile')/df_dec[dec_size] 
    df_dec['pct_receiving_ct'] = 100.*df.loc[df.ing_ct>0,wgt].sum(level='quintile')/df_dec[dec_size]
   
    #########################
    ## Find optimal combination of scale out & up (before targeting shift)
    #if do_optimized_SP:
    #
    #    opt_ct = class_opt_sp(miPais,'ct',df,df_dec,pdict,tshift=None)
    #
    #    df = df.reset_index().set_index(['quintile','cod_hogar'])
    #    df['optimum_ct'] = opt_ct.optimal_sp_value
    #    df = df.reset_index().set_index('quintile')
    
    ##########################
    ## Shift targeting, if appropriate
    #df = df.reset_index().set_index('quintile').drop([i for i in ['index'] if i in df.columns],axis=1)

    #########################
    # Collect more average info about each quintile
    df = df.drop([i for i in ['factor_expansion','pcwgt'] if i != wgt],axis=1)
    
    df_dec['tot_inc'] = df[[wgt,'gct']].prod(axis=1).sum(level='quintile')
    df_dec['mean_inc'] = df_dec['tot_inc']/df_dec[dec_size]
    
    save_to_results_file(iso_to_name[miPais],'Population',df_dec['pop'].astype('int').squeeze(),units='count')
    save_to_results_file(iso_to_name[miPais],'Number of households',df_dec['nHH'].astype('int').squeeze(),units='count')
    save_to_results_file(iso_to_name[miPais],'Mean value of initial transfers, per cap, enrollees only',df_dec['mean_val_ct'].round(2).squeeze(),units='I$')
    save_to_results_file(iso_to_name[miPais],'Population fraction enrolled in cash transfers',df_dec['pct_receiving_ct'].round(3).squeeze(),units='% of individuals')
    save_to_results_file(iso_to_name[miPais],'Population fraction enrolled in cash transfers',df_dec['pct_receiving_ct'].round(3).squeeze(),units='% of individuals')
    save_to_results_file(iso_to_name[miPais],'Total expenditures, by quintile',df_dec['tot_inc'].round(2).squeeze(),units='I$')
    save_to_results_file(iso_to_name[miPais],'Mean expenditures, per cap',df_dec['mean_inc'].round(2).squeeze(),units='I$')

    #########################
    # Calculate impacts (abs_impact = average cost & win_frac = beneficiaries) for each quintile
    rel_impact_group_by_dec = True
    abs_impact_group_by_dec = True    

    # SCENARIO: no policy
    df_dec['abs_impact_no_pol'] = -1*df[[wgt,'carbon_cost']].prod(axis=1).sum(level='quintile')/df[wgt].sum(level='quintile')
    if not rel_impact_group_by_dec: 
        df_dec['rel_impact_no_pol'] = (100.*df.loc[df.gct!=0].eval(wgt+'*-1.*carbon_cost/gct')).sum(level='quintile')/(df.loc[df.gct!=0,wgt]).sum(level='quintile')
    else: df_dec['rel_impact_no_pol'] = (100.*df.loc[df.gct!=0].eval(wgt+'*-1.*carbon_cost')).sum(level='quintile')/(df.loc[df.gct!=0,[wgt,'gct']].prod(axis=1)).sum(level='quintile')

    if verbose:
        print((100.*(df.loc[df.gct!=0].eval(wgt+'*-1.*carbon_cost/gct')).sum(level='quintile')/(df.loc[df.gct!=0,wgt]).sum(level='quintile')).sort_index())
        print((100.*(df.loc[df.gct!=0].eval(wgt+'*-1.*carbon_cost')).sum(level='quintile')/(df.loc[df.gct!=0,[wgt,'gct']].prod(axis=1)).sum(level='quintile')).sort_index())

        print((100.*(df.loc[df.gct!=0].eval(wgt+'*-1.*carbon_cost/gct')).sum(level='quintile')/(df.loc[df.gct!=0,wgt]).sum(level='quintile')).sort_index())
        print((100.*(df.loc[df.gct!=0].eval(wgt+'*-1.*carbon_cost')).sum(level='quintile')/(df.loc[df.gct!=0,[wgt,'gct']].prod(axis=1)).sum(level='quintile')).sort_index())

    df['rel'] = -100
    df.loc[df.gct!=0,'rel'] = 100.*df.loc[df.gct!=0].eval('(-1.)*carbon_cost/gct')
    
    df_dec['win_frac_no_pol'] = (100.*df.loc[-1*df.carbon_cost>0,wgt].sum(level='quintile')/df[wgt].sum(level='quintile')).fillna(0)
    pdict['no_polcost'] = 0

    assert(round(-1*pdict['c_rev'],1) == round(df_dec[[dec_size,'abs_impact_no_pol']].prod(axis=1).sum(),1))
    # ^ sanity check: average impact should sum to total revenue (will redo for every scenario)

    #########################
    # SCENARIO: scaleup -- only households currently receiving monetary cash transfers receive carbon money (scale up)
    # Create secondary loop: absolute scaleup (constant payout to all enrollees) & relative (constant % increase to all enrollees)

    for _scaleup_type in ['scaleup_ct_rel','scaleup_ct_abs']:
        bool_scaleup_abs = True if 'abs' in _scaleup_type else False


        #####################################################
        # STEP 1: assign scaleup distros to enrolled households
        df = df.reset_index().drop([i for i in ['index','level_0'] if i in df.columns],axis=1)

        df['di_'+_scaleup_type] = pdict['c_rev']*targeting_exercise(df.copy(),'ct',0.00,scaleup_abs=bool_scaleup_abs)
        # ^  scaleup wth no targeting improvement

        for _te in [10,20,25]:
            df['di_'+_scaleup_type+'_shift'+str(_te)] = pdict['c_rev']*targeting_exercise(df.copy(),'ct',_te/100,scaleup_abs=bool_scaleup_abs)
        # ^  scaleup WITH targeting improvement    

        df = df.reset_index().set_index('quintile').drop([i for i in ['index','level_0'] if i in df.columns],axis=1)


        #####################################################
        # STEP 2: record enrolled percentage
        df_dec['pct_receiving_'+_scaleup_type] = (100.*(df.loc[df['di_'+_scaleup_type]>0,'pcwgt'].sum(level='quintile').fillna(0))/df_dec[dec_size]).fillna(0)
        for _te in [10,20,25]:
            df_dec['pct_receiving_'+_scaleup_type+'_shift'+str(_te)] = 100.*(df.loc[df['di_'+_scaleup_type+'_shift'+str(_te)]>0,'pcwgt'].sum(level='quintile').fillna(0))/df_dec[dec_size]

        #####################################################
        # STEP 3: sanity check        
        #print(df_dec['pct_receiving_'+_scaleup_type])
        #print(100.*(df.loc[df['di_'+_scaleup_type]>0,'pcwgt'].sum(level='quintile'))/df_dec[dec_size])
        df_dec = df_dec.fillna(0)
        #for _t in ['','_shift10','_shift20','_shift25']:
        #    print(df_dec['pct_receiving_'+_scaleup_type+_t],100.*(df.loc[df['di_'+_scaleup_type+_t]>0,'pcwgt'].sum(level='quintile'))/df_dec[dec_size])
        #    assert(df_dec['pct_receiving_'+_scaleup_type+_t].round(2) == ((100.*df.loc[df['di_'+_scaleup_type+_t]>0,'pcwgt'].sum(level='quintile'))/df_dec[dec_size]).fillna(0.).round(2)).fillna(0).all()
        # ^ Sanity check
        #assert(False)


        #####################################################
        # STEP 4: record absolute & relative impact of tax, net of scaleup & by quintile
        for _t in ['','_shift10','_shift20','_shift25']:
            df_dec['abs_impact_'+_scaleup_type+_t] = (df[wgt]*df.eval('di_'+_scaleup_type+_t+'-carbon_cost')).sum(level='quintile')/df[wgt].sum(level='quintile')
            df_dec['rel_impact_'+_scaleup_type+_t] = 100.*df.eval(wgt+'*(di_'+_scaleup_type+_t+'-carbon_cost)').sum(level='quintile')/df[[wgt,'gct']].prod(axis=1).sum(level='quintile')

            df_dec['win_frac_'+_scaleup_type+_t] = 100.*df.loc[df.eval('(di_'+_scaleup_type+_t+'-carbon_cost)>0'),wgt].sum(level='quintile')/df[wgt].sum(level='quintile')
            pdict[_scaleup_type+_t+'cost'] = df[[wgt,'di_'+_scaleup_type+_t]].prod(axis=1).sum()

            #if do_optimized_SP:
            #    try: df_dec['abs_impact_optimum_ct'+_t] = df.eval(wgt+'*(optimum_ct'+_t+'-ing_ct-carbon_cost)').sum(level='quintile')/df[wgt].sum(level='quintile')
            #    except: pass
            
        #####################################################
        # STEP 5: Check the math
        print(pdict['c_rev'],int(df[['di_'+_scaleup_type,wgt]].prod(axis=1).sum()))
        assert((pdict['c_rev']/df[['di_'+_scaleup_type,wgt]].prod(axis=1).sum() > 0.99) 
               and (pdict['c_rev']/df[['di_'+_scaleup_type,wgt]].prod(axis=1).sum() < 1.01)
               and (round(df_dec['abs_impact_'+_scaleup_type].sum(),0)/pdict['c_rev'] < 0.01)
               and (round(df_dec['abs_impact_'+_scaleup_type].sum(),0)/pdict['c_rev'] > -0.01))

        try: cross_country = pd.read_csv('~/Desktop/tmp/cross_country_frac_aid_to_poor.csv').set_index('country')
        except: cross_country = pd.DataFrame(columns={'country':None,'frac_aid_to_poor':None,'cost_per_dollar_to_poor':None}).set_index('country')
        df = df.reset_index()
        cross_country.loc[miPais,'frac_aid_to_poor'] = round(float(df.loc[df.quintile<=2,['pcwgt','di_'+_scaleup_type]].prod(axis=1).sum()/df[['pcwgt','di_'+_scaleup_type]].prod(axis=1).sum()),3)
        cross_country.loc[miPais,'cost_per_dollar_to_poor'] = round(1/cross_country.loc[miPais,'frac_aid_to_poor'],2)
        df = df.reset_index().set_index('quintile')
        #cross_country.to_csv('~/Desktop/tmp/cross_country_frac_aid_to_poor.csv')

    #########################
    # SCENARIO: new_sp -- give cash (UBI), with a new magic cash transfer
    df['di_new_sp'] = (dist_frac*pdict['c_rev'])/df[wgt].sum()
    df_dec['abs_impact_new_sp'] = (df[wgt]*df.eval('di_new_sp-carbon_cost')).sum(level='quintile')/df[wgt].sum(level='quintile')

    if not rel_impact_group_by_dec: 
        df_dec['rel_impact_new_sp'] = 100.*df.loc[df.gct!=0].eval(wgt+'*(di_new_sp-carbon_cost)/gct').sum(level='quintile')/df.loc[df.gct!=0,wgt].sum(level='quintile')
    else: 
        df_dec['rel_impact_new_sp'] = 100.*df.loc[df.gct!=0].eval(wgt+'*(di_new_sp-carbon_cost)').sum(level='quintile')/df.loc[df.gct!=0,[wgt,'gct']].prod(axis=1).sum(level='quintile')

    df_dec['pct_receiving_new_sp'] = (100.*df.loc[(df.di_new_sp)>0,wgt].sum(level='quintile')/df[wgt].sum(level='quintile')).fillna(0)    
    df_dec['win_frac_new_sp'] = (100.*df.loc[(df.di_new_sp-df.carbon_cost)>0,wgt].sum(level='quintile')/df[wgt].sum(level='quintile')).fillna(0)
    pdict['new_spcost'] = df[[wgt,'di_new_sp']].prod(axis=1).sum()
  
    # Check the math  
    assert((round(-1.*(1.-dist_frac)*pdict['c_rev'],1) == round(df_dec.eval(dec_size+'*abs_impact_new_sp').sum(),1)) or (df_dec.eval(dec_size+'*abs_impact_new_sp').sum() == 0))

    #########################
    # SCENARIO: scaleout10_ct -- simulate higher fractions of household in a given quintile receiving cash transfer, 
    # --> 1) average per capita transfer among beneficiary in that quintile does not change
    
    df = df.reset_index()
    df = df.drop([_c for _c in ['level_0','index'] if _c in df.columns],axis=1)
    df['di_scaleout00_ct_abs'] = targeting_exercise(df.copy(),'ct',0.00,scaleup_abs=True)
    df.loc[df['di_scaleout00_ct_abs']!=0,'di_scaleout00_ct_abs'] = 1
    # _abs is normed so everyone gets '1'
    df['di_scaleout00_ct_rel'] = targeting_exercise(df.copy(),'ct',0.00,scaleup_abs=False)
    df['di_scaleout00_ct_rel'] *= pdict['c_rev']
    # _rel is normed to total revenue spent
    df = df.reset_index().set_index(['cod_hogar','quintile'])

    # Get distribution (fraction) from these scenarios:
    quintile_frac = pd.DataFrame({'population':df_dec['pop'],
                                  #'check_mean_val_ct':df_dec['mean_val_ct'].copy(),
                                  #'mean_val_abs':pdict['tot_cost_ct']*(df[['pcwgt','di_scaleout00_ct_abs']].prod(axis=1).sum(level='quintile')/
                                  #                                     df.loc[df['di_scaleout00_ct_abs']!=0,'pcwgt'].sum(level='quintile'))
                                  'mean_val_abs':1.,
                                  'mean_val_rel':(df[['pcwgt','ing_ct']].prod(axis=1).sum(level='quintile')/
                                                  df.loc[df['ing_ct']!=0,'pcwgt'].sum(level='quintile'))
                                  },index=df_dec.index)
                                  
    #quintile_frac['cash_frac_to_quint'] = df[['di_scaleout00_ct_abs','pcwgt']].prod(axis=1).sum(level='quintile')/df[['di_scaleout00_ct_abs','pcwgt']].prod(axis=1).sum()
    quintile_frac['enrolled_frac_quint_abs'] = df.loc[df['di_scaleout00_ct_abs']!=0,'pcwgt'].sum(level='quintile')/df['pcwgt'].sum(level='quintile')
    quintile_frac['enrolled_frac_quint_rel'] = df.loc[df['di_scaleout00_ct_rel']!=0,'pcwgt'].sum(level='quintile')/df['pcwgt'].sum(level='quintile')
    
    ### Clean up df
    df = df.drop([_c for _c in ['index','level_0'] if _c in df.columns],axis=1)

    #########################    
    # Scale out
    for _so in [25]:
        _sso = str(_so)

        df['di_scaleout'+_sso+'_ct_abs'] = df['di_scaleout00_ct_abs'].copy()
        # _abs is normed so everyone gets '1'
        df['di_scaleout'+_sso+'_ct_rel'] = df['di_scaleout00_ct_rel'].copy()
        # _rel is normed to total revenue spent

        # grab list of people not enrolled in CT
        not_enrolled = df.loc[df['di_scaleout00_ct_abs']==0].copy().sort_values('gct',ascending=True)
        # also works for scaleout00_ct_rel here; will be zero for both, if for one
        not_enrolled = pd.merge(not_enrolled.reset_index(),quintile_frac.reset_index(),on=['quintile']).set_index(['cod_hogar'])
    
        # Enroll new people
        not_enrolled['cumsum_pcwgt'] = not_enrolled.groupby('quintile')['pcwgt'].transform('cumsum')/not_enrolled['population']
        not_enrolled.loc[not_enrolled.cumsum_pcwgt<=_so/100,'di_scaleout'+_sso+'_ct_abs']=not_enrolled.loc[not_enrolled.cumsum_pcwgt<=(_so/100*not_enrolled.enrolled_frac_quint_abs),'mean_val_abs'].copy()
        not_enrolled.loc[not_enrolled.cumsum_pcwgt<=_so/100,'di_scaleout'+_sso+'_ct_rel']=not_enrolled.loc[not_enrolled.cumsum_pcwgt<=(_so/100*not_enrolled.enrolled_frac_quint_rel),'mean_val_rel'].copy()
        assert(_so == 0 or not_enrolled.loc[not_enrolled['di_scaleout'+_sso+'_ct_abs'] != not_enrolled['di_scaleout00_ct_abs']].shape[0] != 0)
        assert(_so == 0 or not_enrolled.loc[not_enrolled['di_scaleout'+_sso+'_ct_rel'] != not_enrolled['di_scaleout00_ct_rel']].shape[0] != 0)
        # ^ Check that something changed

        # re-merge with full df
        not_enrolled = not_enrolled.reset_index().set_index(['cod_hogar','quintile'])
        df['di_scaleout'+_sso+'_ct_abs'].update(not_enrolled['di_scaleout'+_sso+'_ct_abs'])
        df['di_scaleout'+_sso+'_ct_rel'].update(not_enrolled['di_scaleout'+_sso+'_ct_rel'])
        assert(_so == 0 or df.loc[df['di_scaleout'+_sso+'_ct_abs'] != df['di_scaleout00_ct_abs']].shape[0] != 0)
        assert(_so == 0 or df.loc[df['di_scaleout'+_sso+'_ct_rel'] != df['di_scaleout00_ct_rel']].shape[0] != 0)
        # ^ Check that something changed

        # Scale so that sum(i_scaleoutXX_ct) = c_rev + tot_cost_ct
        df['di_scaleout'+_sso+'_ct_abs'] *= pdict['c_rev']/df[['pcwgt','di_scaleout'+_sso+'_ct_abs']].prod(axis=1).sum()
        df['di_scaleout'+_sso+'_ct_rel'] *= pdict['c_rev']/df[['pcwgt','di_scaleout'+_sso+'_ct_rel']].prod(axis=1).sum()
    
        # Calculate metrics
        df_dec['win_frac_scaleout'+_sso+'_ct_abs'] = 100.*df.loc[(df['di_scaleout'+_sso+'_ct_abs']-df.carbon_cost)>0,wgt].sum(level='quintile')/df[wgt].sum(level='quintile')
        df_dec['abs_impact_scaleout'+_sso+'_ct_abs'] = df.eval(wgt+'*(di_scaleout'+_sso+'_ct_abs-carbon_cost)').sum(level='quintile')/df[wgt].sum(level='quintile')
        df_dec['win_frac_scaleout'+_sso+'_ct_rel'] = 100.*df.loc[(df['di_scaleout'+_sso+'_ct_rel']-df.carbon_cost)>0,wgt].sum(level='quintile')/df[wgt].sum(level='quintile')
        df_dec['abs_impact_scaleout'+_sso+'_ct_rel'] = df.eval(wgt+'*(di_scaleout'+_sso+'_ct_rel-carbon_cost)').sum(level='quintile')/df[wgt].sum(level='quintile')
        #
        df_dec['rel_impact_scaleout'+_sso+'_ct_abs'] = 100.*(df.loc[df.gct!=0].eval(wgt+'*(di_scaleout'+_sso+'_ct_abs-carbon_cost)').sum(level='quintile')
                                                             /df.loc[df.gct!=0,[wgt,'gct']].prod(axis=1).sum(level='quintile'))
        df_dec['rel_impact_scaleout'+_sso+'_ct_rel'] = 100.*(df.loc[df.gct!=0].eval(wgt+'*(di_scaleout'+_sso+'_ct_rel-carbon_cost)').sum(level='quintile')
                                                             /df.loc[df.gct!=0,[wgt,'gct']].prod(axis=1).sum(level='quintile'))
        #
        df_dec['pct_receiving_scaleout'+_sso+'_ct_abs'] = 100.*(df.loc[df['di_scaleout'+_sso+'_ct_abs']>0,'pcwgt'].sum(level='quintile'))/df_dec[dec_size]  
        df_dec['pct_receiving_scaleout'+_sso+'_ct_rel'] = 100.*(df.loc[df['di_scaleout'+_sso+'_ct_rel']>0,'pcwgt'].sum(level='quintile'))/df_dec[dec_size]  
        #
        pdict['scaleout'+_sso+'_ct_abscost'] = df[['di_scaleout'+_sso+'_ct_abs','pcwgt']].prod(axis=1).sum()
        pdict['scaleout'+_sso+'_ct_relcost'] = df[['di_scaleout'+_sso+'_ct_rel','pcwgt']].prod(axis=1).sum()  
  
    df_dec.to_csv(out_dir+'sp/'+miPais+'_quintile_info.csv')
    # hack out a few results for paper
    try: _results_hack = pd.read_csv(out_dir+'sp/_results_hacks.csv').set_index('pais')
    except: 
        _results_hack = pd.DataFrame(columns={'pais':None,
                                              'net_gain_frac_of_enrollees':None,
                                              'ct_enrollment_base':None,
                                              'ct_enrollment_25':None,
                                              'q1_rel_impact_scaleup_ct_abs':None,
                                              'q1_rel_impact_scaleout25_ct_abs':None,
                                              'q1_rel_impact_scaleup_ct_abs_shift25':None,
                                              'q1_win_frac_scaleup_ct_abs':None,
                                              'q1_win_frac_scaleout25_ct_abs':None,
                                              'q1_win_frac_scaleup_ct_abs_shift25':None}).set_index('pais')

    _results_hack.loc[miPais,'net_gain_frac_of_enrollees'] = float(df_dec.loc[1,'win_frac_scaleup_ct_abs']/df_dec.loc[1,'pct_receiving_scaleup_ct_abs'])
    _results_hack.loc[miPais,'ct_enrollment'] = float(df_dec.loc[1,'pct_receiving_ct'])
    _results_hack.loc[miPais,'ct_enrollment_25'] = float(df_dec.loc[1,'pct_receiving_scaleup_ct_abs_shift25'])
    _results_hack.loc[miPais,'q1_rel_impact_scaleup_ct_abs'] = float(df_dec.loc[1,'rel_impact_scaleup_ct_abs'])
    _results_hack.loc[miPais,'q1_rel_impact_scaleup_ct_abs_shift25'] = float(df_dec.loc[1,'rel_impact_scaleup_ct_abs_shift25'])
    _results_hack.loc[miPais,'q1_rel_impact_scaleout25_ct_abs'] = float(df_dec.loc[1,'rel_impact_scaleout25_ct_abs'])
    _results_hack.loc[miPais,'q1_win_frac_scaleup_ct_abs'] = float(df_dec.loc[1,'win_frac_scaleup_ct_abs'])
    _results_hack.loc[miPais,'q1_win_frac_scaleup_ct_abs_shift25'] = float(df_dec.loc[1,'win_frac_scaleup_ct_abs_shift25'])
    _results_hack.loc[miPais,'q1_win_frac_scaleout25_ct_abs'] = float(df_dec.loc[1,'win_frac_scaleout25_ct_abs'])
    _results_hack.to_csv(out_dir+'sp/_results_hacks.csv')
    
    for result_out in ['win_frac','rel_impact','pct_receiving']:
        _len_ro = len(result_out)+1
        try: 
            df_to_table = pd.read_csv(out_dir+'latex/'+result_out+'.csv').set_index('policy')
            df_to_table[miPais] = (df_dec[[_c for _c in df_dec if result_out in _c and _c[_len_ro:] in policies]].mean().T).to_frame()    
        except: 
            df_to_table = (df_dec[[_c for _c in df_dec if result_out in _c and _c[_len_ro:] in policies]].mean().T).to_frame(name=miPais)
            df_to_table.index.name = 'policy'
        try: df_to_table = df_to_table.drop('Median',axis=1)
        except: pass
        df_to_table['Median'] = df_to_table.median(axis=1)
        df_to_table.to_csv(out_dir+'latex/'+result_out+'.csv')
    
        
        for _q in [1,2,3,4,5]:
            try: 
                df_to_table = pd.read_csv(out_dir+'latex/'+result_out+'_q'+str(_q)+'.csv').set_index('policy')
                df_to_table[miPais] = (df_dec.loc[_q][[_c for _c in df_dec if result_out in _c and _c[_len_ro:] in policies]].T).to_frame()    
            except: 
                df_to_table = (df_dec.loc[_q][[_c for _c in df_dec if result_out in _c and _c[_len_ro:] in policies]].T).to_frame(name=miPais)
                df_to_table.index.name = 'policy'
            try: df_to_table = df_to_table.drop('Median',axis=1)
            except: pass
            df_to_table['Median'] = df_to_table.median(axis=1)
            df_to_table.to_csv(out_dir+'latex/'+result_out+'_q'+str(_q)+'.csv')

    ##################################################    
    ##################################################
    # PLOT: hh income (histogram)
    my_index = df.index.names
    df = df.reset_index('quintile')

    ax = plt.gca()
    
    hhctax_heights, bins = np.histogram(df['new_expenditure'].clip(lower=0,upper=get_uclip(miPais)),bins=50,weights=df[wgt])
    hhinc_heights, _ = np.histogram(df['gct'].clip(lower=0,upper=get_uclip(miPais)),bins=bins,weights=df[wgt])
    
    ax.bar((bins[1]-bins[0])/2+bins[:-1], hhinc_heights, width=(bins[1]-bins[0]), label='survey says', facecolor=sns_pal[0],edgecolor=sns_pal[0],alpha=0.45)
    ax.bar((bins[1]-bins[0])/2+bins[:-1], hhctax_heights, width=(bins[1]-bins[0]), label='with carbon tax', facecolor=sns_pal[1],edgecolor=sns_pal[1],alpha=0.45)

    _y_lim = ax.get_ylim()
    last_q = 0
    for _q in range(1,6):
        plt.plot([df.loc[df.quintile==_q,'gct'].max(),df.loc[df.quintile==_q,'gct'].max()],_y_lim,color=greys[6])
        plt.annotate('Q'+str(_q),xy=((last_q+df.loc[df.quintile==_q,'gct'].clip(upper=get_uclip(miPais)).max())/2,_y_lim[1]),color=greys[6],ha='center')
        last_q = df.loc[df.quintile==_q,'gct'].max()

    lgd = title_legend_labels(ax,iso_to_name[miPais],lab_x='Per cap. expenditures [GTAP$]',lab_y='Number of individuals',lim_x=[bins[0],get_uclip(miPais)],lim_y=[0])
    plt.gcf().savefig(out_dir+'sp/'+miPais+'_hh_expenditures.pdf',format='pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.gcf().savefig(out_dir+'sp/'+miPais+'_hh_expenditures.png',format='png',bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.cla(); plt.close('all')

    
    df = df.reset_index().set_index(my_index)
    try: df = df.drop(['new_expenditure'],axis=1)
    except: pass
    
    ##################################################    
    ##################################################
    # LATEX: cost of each SP system, as fraction of total revenue, by quintile

    new_df = pd.DataFrame({'pais':miPais},index=df_dec.index.copy())
    
    new_df['tot_tax_by_Q'] = df_dec[['abs_impact_no_pol','pop']].prod(axis=1)

    new_df = new_df.reset_index().set_index('pais')
    new_df['tot_tax_all_Q'] = new_df['tot_tax_by_Q'].groupby('pais').transform('sum')
    new_df = new_df.reset_index().set_index('quintile') 

    for _c in df_dec.columns:

        if _c[:11] == 'abs_impact_' and 'optimum' not in _c and 'shift' not in _c:
            new_df['net_impact_'+_c[11:]] = df_dec[[_c,'pop']].prod(axis=1)
            new_df['abs_cost_'+_c[11:]] = new_df['net_impact_'+_c[11:]]-new_df['tot_tax_by_Q']
            # FLAG: going to check above

            new_df[pdict[_c[11:]+'desc']] = (100.*new_df['abs_cost_'+_c[11:]].abs()/new_df['tot_tax_all_Q'].abs()).round(1)
            new_df = new_df.drop(['net_impact_'+_c[11:],'abs_cost_'+_c[11:]],axis=1)
    new_df = new_df.drop(['pais','tot_tax_by_Q','tot_tax_all_Q',pdict['no_poldesc']],axis=1)

    new_df = new_df.reset_index()
    new_df['quintile'] = new_df['quintile'].astype(int)
    new_df = new_df.set_index('quintile').drop([i for i in ['index','level_0'] if i in new_df.columns])

    new_df = new_df.T.sort_values(1,ascending=False)
    new_df = new_df.rename(columns={1:'Poorest',2:'Second',3:'Third',4:'Fourth',5:'Wealthiest'})
    new_df.columns.name = 'Quintile'
    new_df.to_latex(out_dir+'latex/sp_expenditures_by_quint_'+miPais+'.tex')
    new_df.to_csv(out_dir+'latex/sp_expenditures_by_quint_'+miPais+'.csv')

    ##################################################    
    ##################################################
    # LATEX: cost, as fraction of total revenue, of making each quintile whole, by SP system

    new_df = pd.DataFrame({'pais':miPais},index=df_dec.index.copy())
    
    new_df['tot_tax_by_Q'] = df_dec[['abs_impact_no_pol','pop']].prod(axis=1)
    new_df['cum_tot_tax_by_Q'] = new_df['tot_tax_by_Q'].cumsum()

    new_df = new_df.reset_index().set_index('pais')
    new_df['tot_tax_all_Q'] = new_df['tot_tax_by_Q'].groupby('pais').transform('sum')
    new_df = new_df.reset_index().set_index('quintile').sort_index()

    for _c in df_dec.columns:

        if _c[:11] == 'abs_impact_' and 'optimum' not in _c and ('shift' not in _c or 'shift25' in _c) and ('scaleout' not in _c or 'scaleout25' in _c) and 'rel' not in _c:
            
            # This code calculates cost to reimburse each quintile, separately.
            # That means, eg, how much does it cost to reimburse everyone, under each SP, to the level such that Q[1-5] is made whole?  
            if False: 
                new_df['net_impact_'+_c[11:]] = df_dec[[_c,'pop']].prod(axis=1)
                
                new_df['abs_cost_'+_c[11:]] = new_df['net_impact_'+_c[11:]]-new_df['tot_tax_by_Q']
                new_df['frac_receipt'+_c[11:]] = (new_df['abs_cost_'+_c[11:]]/new_df['tot_tax_all_Q']).abs()
                new_df[(pdict[_c[11:]+'desc']).replace('\n',' ')] = (100.*new_df['tot_tax_by_Q']/new_df[['frac_receipt'+_c[11:],'tot_tax_all_Q']].prod(axis=1)).round(1)
                
            # This code calculates cost to reimburse each quintile, cumulatively.
            # That means, eg, how much does it cost to reimburse everyone, under each SP, to the level such that Q[1, 12, 123, 1234, 12345] is made whole?  
            if True: 
                new_df['net_impact_'+_c[11:]] = df_dec[[_c,'pop']].prod(axis=1).cumsum()
    
                new_df['abs_cost_'+_c[11:]] = new_df['net_impact_'+_c[11:]]-new_df['cum_tot_tax_by_Q']
                new_df['frac_receipt'+_c[11:]] = (new_df['abs_cost_'+_c[11:]]/new_df['tot_tax_all_Q']).abs()
                new_df[(pdict[_c[11:]+'desc']).replace('\n',' ')] = (100.*new_df['cum_tot_tax_by_Q']/new_df[['frac_receipt'+_c[11:],'tot_tax_all_Q']].prod(axis=1)).round(1)

            new_df = new_df.drop([_c for _c in ['net_impact_'+_c[11:],'abs_cost_'+_c[11:],'frac_receipt'+_c[11:],
                                                'cum_net_impact_'+_c[11:],'cum_abs_cost_'+_c[11:],'cum_frac_receipt'+_c[11:]] if _c in new_df.columns],axis=1)

    new_df['Perfect compensation'] = round(100.*(new_df['tot_tax_by_Q'].cumsum()/new_df['tot_tax_all_Q']).abs(),1)

    new_df = new_df.drop(['pais','tot_tax_by_Q','cum_tot_tax_by_Q','tot_tax_all_Q',pdict['no_poldesc']],axis=1)

    new_df = new_df.reset_index()
    new_df['quintile'] = new_df['quintile'].astype(int)
    new_df = new_df.set_index('quintile').drop([i for i in ['index','level_0'] if i in new_df.columns])

    new_df = new_df.T.sort_values(1,ascending=True)
    
    new_df = new_df.rename(columns={1:'Poorest',2:'Second',3:'Third',4:'Fourth',5:'Wealthiest'})
    new_df.columns.name = ''

    new_df.to_latex(out_dir+'latex/redistribution_expenditures_by_sp_and_quint_'+miPais+'.tex')
    new_df.to_csv(out_dir+'latex/redistribution_expenditures_by_sp_and_quint_'+miPais+'.csv')    
    # Leave these files here, pick up in refresh_country_comparison() in lib_country_comparison.py

    cummax_df = new_df.cummax(axis=1)
    cummax_df.to_latex(out_dir+'latex/redistribution_expenditures_by_sp_and_quint_'+miPais+'_cummax.tex')
    cummax_df.to_csv(out_dir+'latex/redistribution_expenditures_by_sp_and_quint_'+miPais+'_cummax.csv')
    # Leave these files here, pick up in refresh_country_comparison() in lib_country_comparison.py


    # Try in a hackish way to get the numbers adrien wants.
    hack_df = df[['pcwgt','carbon_cost','di_scaleup_ct_abs','di_scaleup_ct_abs_shift25','di_new_sp','di_scaleout25_ct_abs']].copy()
    hack_df = hack_df.rename(columns={'di_scaleup_ct_abs':'scaleup_ct_absdesc',
                                      'di_scaleup_ct_abs_shift25':'scaleup_ct_abs_shift25desc',
                                      'di_new_sp':'new_spdesc',
                                      'di_scaleout25_ct_abs':'scaleout25_ct_absdesc'}).rename(columns=pdict)
    if True: reverse_engineer(miPais,hack_df.copy(),.60)

    try: hack_out = pd.read_csv('output/all_countries_sp/total_pop_benefit_from_Q12_redistribution.csv').set_index('Policy')
    except:
        hack_out = pd.DataFrame(index=new_df.index)
        hack_out.index.name = 'Policy'

    other_hack = new_df['Second'].to_frame().T
    ax = plt.gca()


    for _c in other_hack.columns:

        if _c == 'Perfect compensation': continue
        hack_df[_c] *= float(other_hack[_c])/100.
        hack_out.loc[_c,miPais] = round(100.*hack_df.loc[hack_df[_c]>=hack_df['carbon_cost'],'pcwgt'].sum()/hack_df['pcwgt'].sum(),1)
        
        #print(miPais,_c,100.*hack_df.loc[hack_df[_c]>=hack_df['carbon_cost'],'pcwgt'].sum(level='quintile')/hack_df['pcwgt'].sum(level='quintile'))
        plt.plot([1,2,3,4,5],100.*(hack_df.loc[hack_df[_c]>=hack_df['carbon_cost'],'pcwgt'].sum(level='quintile')/hack_df['pcwgt'].sum(level='quintile')).fillna(0),
                 color=pdict[{pdict[_pol]:_pol for _pol in pdict if('desc' in _pol and 'xx' not in _pol)}[_c].replace('desc','col')],
                 label= str(int(round(float(other_hack[_c]),0)))+'% carbon revenue into '+_c.lower(),lw=2.5)

    sns.despine(bottom=True)
    plt.gca().grid(False)
    plt.ylim(-1,101)
    
    plt.xticks(np.linspace(1,5,5),size=global_fontsize)
    plt.gca().xaxis.set_ticks_position('top') 
    plt.yticks(size=global_fontsize)
    plt.plot([0.5,5.5],[50,50],linewidth=0.75,color=greys[1],zorder=10,linestyle=':')
    plt.gca().tick_params(axis='x', which='major', pad=10)
    plt.gca().set_xticklabels(['Poorest\nquintile','Second','Third','Fourth','Wealthiest\nquintile'],size=8,color=greys[7])
    lgd = title_legend_labels(plt.gca(),iso_to_name[miPais],
                              lab_x='',lab_y='Individuals benefitting from reform in '+iso_to_name[miPais]+' [% of quintile]',
                              lim_x=[0.5,5.5],global_fs=global_fontsize)
    ax.legend(fontsize=global_fontsize,fancybox=True,frameon=False,facecolor='white')    

    plt.gcf().savefig('output/all_countries_sp/total_pop_benefit_from_Q12_redistribution/'+miPais+'.pdf')

    # These two below must be equal for bottow 2 quintiles
    #print(hack_df[[_c,'pcwgt']].prod(axis=1).sum(level='quintile').sort_index())
    #print(hack_df[['carbon_cost','pcwgt']].prod(axis=1).sum(level='quintile').sort_index())
    
    hack_out.loc['Perfect compensation',miPais] = 40
    hack_out.to_csv('output/all_countries_sp/total_pop_benefit_from_Q12_redistribution.csv')

    ##################################################    
    ##################################################
    # PLOT: relative & absolute impacts of SP distribution systems (% of hh expenditures)
    df_dec = df_dec.reset_index()
    if verbose: print(policies)

    all_subsets = [('no_redistribution',['no_pol']),
                   #('sp_vs_scaleup',['new_sp','scaleup_ct_rel']),
                   #('set_ct',['no_pol','new_sp','scaleup_ct_rel','scaleout25_ct']),
                   ('set_ct_w_targeting',['no_pol','scaleup_ct_abs','new_sp','scaleup_ct_abs_shift25','scaleout25_ct_abs'])
                   #('all',['no_pol','new_sp','scaleup_ct_rel','scaleout25_ct_abs']),
                   #('all_redist',['new_sp','scaleup_ct_rel','scaleout25_ct_abs'])
                    ]# <-- create additional subsets here

    lab_y_dict = {'rel':'Net impact [% of expenditures]',
                  'abs':'Net impact [INT$]'}

    for _plot in lab_y_dict:

        # Loop over policy subsets
        for set_desc, pol_subset in all_subsets:

            #if not hay_CT:  pol_subset = [_ps for _ps in pol_subset if ('scaleup_ct' not in _ps and 'scaleout10_ct_abs' not in _ps)]
        
            # Loop over the policies in each subset
            # --> we'll highlight each one, one by one
            for select_pol in pol_subset+['all','legend']:
                if select_pol == 'legend' and not hay_CT: continue
        
                ax = plt.gca()
                _ct, _dct = 0, 0.1
                _wid = 1.

                ordered = get_order(pol_subset,df_dec,_plot+'_impact_',desc=True)
                #ordered = ['no_pol','scaleup_ct_abs','scaleout25_ct_abs','scaleup_ct_abs_shift25','new_sp']
                for _ip in ordered:

                    if _ip != None:

                        if _ip != 'no_pol' and prettyf(pdict[_ip+'cost']) != prettyf(pdict['c_rev']):
                            print(_ip,prettyf(pdict[_ip+'cost']), prettyf(pdict['c_rev']))

                        color_hack = greys[4]
                        if _ip == select_pol or select_pol == 'all' or select_pol == 'legend':
                            color_hack = pdict[_ip+'col']

                            y_val = max(0,1.05*df_dec[_plot+'_impact_'+_ip][0])
                            try: 
                                for __eep in ordered: y_val = max(y_val,1.05*df_dec[_plot+'_impact_'+__eep][1])
                                #                                                                           ^ [0] =  1st quintile
                            except: pass #                                                                  ^ [1] =  2nd quintile
                            
                            
                            # annotate total revenue/cost of each program
                            if select_pol != 'all' and select_pol != 'legend':
                                plt.annotate(('Redistribution mechanism:\n'+pdict[_ip+'desc']),#+'\nTotal cost: '+prettyf(pdict['c_rev'])),
                                             xy=(_ct,max(0,df_dec[_plot+'_impact_'+_ip][0])),va='bottom',ha='left',size=7.5,color=greys[6],weight='light',
                                             xytext=(_ct+_wid,y_val),arrowprops=dict(arrowstyle="-",connectionstyle="angle,angleA=0,angleB=90,rad=5"),clip_on=False)

                        if select_pol == 'all' or select_pol == 'legend':
                            if select_pol == 'legend' and _ip == ordered[0]:
                                for _xx in ['xx1','xx2']:
                                    p1 = plt.bar(-2,0,color=pdict[_xx+'col'],alpha=0.90,width=_wid,label=pdict[_xx+'desc'])
                            p1 = plt.bar(df_dec.index*(len(pol_subset)+1.5)*(_wid+_dct)+_ct, df_dec[_plot+'_impact_'+_ip],color=color_hack,alpha=0.90,width=_wid,label=pdict[_ip+'desc'])

                        else: p1 = plt.bar(df_dec.index*(len(pol_subset)+1.5)*(_wid+_dct)+_ct, df_dec[_plot+'_impact_'+_ip],color=color_hack,alpha=0.90,width=_wid)
                        _ct+=_wid+_dct

                #for iy in df_dec.index:
                #    _ix = iy*(len(pol_subset)+1.5)*(_wid+_dct)-_wid/2
                #    plt.plot([_ix,_ix+_ct],[df_dec.loc[iy][_plot+'_impact_no_pol'],df_dec.loc[iy][_plot+'_impact_no_pol']],color=greys[5])

                # labels for each quintile
                quint_L_edge = np.array(df_dec.index*(len(pol_subset)+1.5)*(_wid+_dct)-_wid/2)
                quint_R_edge = np.array(df_dec.index*(len(pol_subset)+1.5)*(_wid+_dct)+_ct-_wid/2)
                _qlab = ['Poorest\nquintile','Second','Third','Fourth','Wealthiest\nquintile']

                _y_min, _y_max = [0.], [0.]
                rects = ax.patches
                for rect in rects:
                    if (rect.get_y()+rect.get_height()) < _y_min[0]: _y_min.append(rect.get_y()+rect.get_height());_y_min.sort();_y_min.pop(-1)
                    if (rect.get_y()+rect.get_height()) > _y_max[0]: _y_max.append(rect.get_y()+rect.get_height());_y_max.sort();_y_max.pop(0)
                _y = _y_min[0]; _y_max = _y_max[0]
                _dy = -(_y_max-_y)/20.

                for _n, _lr in enumerate(zip(quint_L_edge,quint_R_edge)):
                    _l, _r = _lr
                    plt.plot([_l,_r],[_y+_dy/2,_y+_dy/2],color=greys[5],linewidth=1.0)
                    plt.plot([_l,_l],[_y+_dy/2,_y+_dy*(1/2-1/10)],color=greys[5],linewidth=1.0)
                    plt.plot([_r,_r],[_y+_dy/2,_y+_dy*(1/2-1/10)],color=greys[5],linewidth=1.0)
                    plt.annotate(_qlab[_n],xy=((_l+_r)/2,_y+_dy),va='top',ha='center',size=global_fontsize,color=greys[7],clip_on=False)

                # label floor
                #plt.annotate('Gross carbon tax\nrelative to expenditures',xy=(_r,df_dec[_plot+'_impact_no_pol'][4]),color=greys[6],fontsize=global_fontsize,va='center',ha='left',
                #             xytext=(_r+_wid,df_dec[_plot+'_impact_no_pol'][4]),arrowprops=dict(arrowstyle="-",connectionstyle="angle,angleA=0,angleB=90,rad=0"),clip_on=False)

                # set x-axis limits of plot
                _x_lim = [df_dec.index[0]-_wid,df_dec.index[-1]*(len(pol_subset)+1.5)*(_wid+_dct)+(_ct-(_wid+_dct))+_wid]
                plt.plot(_x_lim,[0,0],color=greys[6],zorder=1,linewidth=1.0)

                # formatting function
                if select_pol == 'legend':
                    lgd = title_legend_labels(ax,iso_to_name[miPais],lab_x='',lab_y=lab_y_dict[_plot],lim_x=_x_lim,global_fs=global_fontsize)
                elif select_pol == 'all':
                    lgd = title_legend_labels(ax,iso_to_name[miPais],lab_x='',lab_y=lab_y_dict[_plot],lim_x=_x_lim,global_fs=global_fontsize,do_leg=False)
                    _leg = plt.legend(loc='upper right',labelspacing=0.90,fontsize=global_fontsize)
                    plt.setp(_leg.get_texts(),color=greys[7])
                    plt.gca().tick_params(axis='y',labelsize=global_fontsize,labelcolor=greys[7])
                else:
                    lgd = title_legend_labels(ax,'',lab_x='',lab_y=lab_y_dict[_plot],lim_x=_x_lim,global_fs=global_fontsize,do_leg=False)
                

                plt.xticks([],size=global_fontsize)
                plt.yticks(size=global_fontsize)
                plt.gca().grid(False)
                sns.despine(bottom=True,left=True)
                plt.draw()

                # save out
                _out_dir = out_dir+'sp/tax_net_sp_'+_plot+'/'+set_desc
                if not os.path.exists(_out_dir): os.makedirs(_out_dir)    

                if select_pol == 'legend':
                    plt.gcf().savefig(_out_dir+'/'+select_pol+'.pdf',format='pdf',bbox_inches='tight',bbox_extra_artists=(lgd,))                    
                elif select_pol == 'all': 
                    plt.gcf().savefig(_out_dir+'/'+miPais+'_'+select_pol+'.pdf',format='pdf',bbox_inches='tight')
                    #plt.gcf().savefig(out_dir+'/'+miPais+'_'+select_pol+'.png',format='png',bbox_inches='tight')
                else: 
                    plt.gcf().savefig(_out_dir+'/'+miPais+'_'+select_pol+'.pdf',format='pdf',bbox_inches='tight')
                    #plt.gcf().savefig(_out_dir+'/'+miPais+'_'+select_pol+'.png',format='png',bbox_inches='tight')
                plt.cla(); plt.close('all')
                
    ##################################################    
    ##################################################
    # PLOT: fraction who win under various SP distribution systems
    label_dict = {'win_frac_':'Individuals benefitting from reform in '+iso_to_name[miPais]+'\n[% of quintile]',
                  'pct_receiving_':'Cash transfer enrollment in '+iso_to_name[miPais]+'\n[% of population]'}
    if not hay_CT: label_dict.update({'pct_receiving_':'Public transfer enrollment in '+iso_to_name[miPais]+'\n[% of population]'})

    for _polmax in range(1,len(policies)):

        for result_set in ['pct_receiving_','win_frac_']:
            df_dec[result_set+'no_pol'] = 0

            ax = plt.gca()
            _zorder_descending = 91

            _npol = 0
            for _ip in get_order(policies,df_dec,result_set,desc=True):
                if result_set == 'pct_receiving_' and (_ip == 'scaleup_ct_rel' or _ip == 'scaleup_ct_rel_shift25' or _ip == 'scaleout25_ct_rel'): continue
                if _ip == 'no_pol': continue

                _npol+=1
                if _npol > _polmax: continue
                #if _ip == 'new_sp': continue
                #if _ip == 'scaleup_ct_abs_shift25': continue
                #if _ip == 'scaleout25_ct_abs': continue
                #['no_pol', 'new_sp', 'scaleup_ct_abs', 'scaleup_ct_abs_shift25', 'scaleout25_ct_abs']

                if _ip != None:

                    if (prettyf(pdict[_ip+'cost']) != prettyf(pdict['c_rev'])): 
                        print(_ip,prettyf(pdict[_ip+'cost']),prettyf(pdict['c_rev']))

                    df_dec.plot('quintile',result_set+_ip,
                                label=(pdict[_ip+'desc'].replace('\n',' ') if result_set!='pct_receiving_' or ('scaleup_ct_rel' not in _ip and 'scaleout25_ct_rel' not in _ip) else ''),
                                ax=ax,color=pdict[_ip+'col'],linewidth=2.5,zorder=_zorder_descending)
                    if do_scatter: df_dec.plot.scatter('quintile',result_set+_ip,label=None,ax=ax,color=pdict[_ip+'col'],zorder=50)

                    save_to_results_file(iso_to_name[miPais],(pdict[_ip+'desc']+result_set.replace('pct_receiving_',': enrollment').replace('win_frac_',': individuals who are net beneficiaries')),
                                         np.array(df_dec[result_set+_ip].round(3).squeeze()),units='% of individuals')

                    if result_set == 'pct_receiving_':
                        if _ip == 'scaleup_ct_rel':
                            df_dec.plot('quintile',result_set+'scaleup_ct_abs',linestyle='--',color=pdict['scaleup_ct_abscol'],dashes=(5,5),
                                        linewidth = 2.5,ax=ax,label='',zorder=_zorder_descending+0.1)
                            dotted_line1 = lines.Line2D([], [], linewidth=2.5, linestyle="--", dashes=(10, 1), color=pdict[_ip+'col'])
                            dotted_line2 = lines.Line2D([], [], linewidth=2.5, linestyle="--", dashes=(5, 4), color=pdict['scaleup_ct_abscol'])
                        if 'scaleup_ct_rel_shift' in _ip:
                            df_dec.plot('quintile',result_set+'scaleup_ct_abs_shift25',linestyle='--',color=pdict['scaleup_ct_abs_shift25col'],dashes=(5,5),
                                        linewidth = 2.5,ax=ax,label='',zorder=_zorder_descending+0.1)
                            dotted_line3 = lines.Line2D([], [], linewidth=2.5, linestyle="--", dashes=(10, 1), color=pdict[_ip+'col'])
                            dotted_line4 = lines.Line2D([], [], linewidth=2.5, linestyle="--", dashes=(5, 4), color=pdict['scaleup_ct_abs_shift25col'])
                        if 'scaleout25_ct_rel' in _ip:
                            df_dec.plot('quintile',result_set+'scaleout25_ct_abs',linestyle='--',color=pdict['scaleout25_ct_abscol'],dashes=(5,5),
                                        linewidth = 2.5,ax=ax,label='',zorder=_zorder_descending+0.1)
                            dotted_line5 = lines.Line2D([], [], linewidth=2.5, linestyle="--", dashes=(10, 1), color=pdict[_ip+'col'])
                            dotted_line6 = lines.Line2D([], [], linewidth=2.5, linestyle="--", dashes=(5, 4), color=pdict['scaleout25_ct_abscol'])      
                _zorder_descending-=1
            # End of SP policy loop

            # 
            plt.gca().grid(False)
            sns.despine(bottom=True)

            _xlim = plt.gca().get_xlim()
            plt.plot([-100,100],[50,50],linewidth=0.75,color=greys[1],zorder=10,linestyle=':')
            plt.gca().set_xlim(_xlim)

            plt.xticks(np.linspace(1,5,5),size=global_fontsize)
            plt.gca().xaxis.set_ticks_position('top') 
            plt.gca().tick_params(axis='x', which='major', pad=10)
            plt.gca().set_xticklabels(['Poorest\nquintile','Second','Third','Fourth','Wealthiest\nquintile'],size=global_fontsize,color=greys[7])

            plt.yticks(size=global_fontsize)

            lgd = title_legend_labels(ax,iso_to_name[miPais],lab_x='',lab_y=label_dict[result_set],lim_x=[0.5,5.5],lim_y=[0,101],global_fs=global_fontsize,do_leg=True)
            if result_set == 'pct_receiving_':
                handles, labels = ax.get_legend_handles_labels()             
                plt.setp(plt.legend().get_texts(), color=greys[7])
                try: plt.legend(handles+[(dotted_line1, dotted_line2),(dotted_line3,dotted_line4),(dotted_line5,dotted_line6)],
                                labels+['Scale up (uniformly or proportionally)',
                                        'Improve targeting + scale up (uniformly or proportionally)',
                                        'Scaleout + scale up (uniformly or proportionally)'],fontsize=global_fontsize,labelspacing=0.90)
                except: pass 
            #lgd = title_legend_labels(ax,iso_to_name[miPais],lab_x='',lab_y=label_dict[result_set],lim_x=[0.5,5.5],lim_y=[0,101],global_fs=global_fontsize,do_leg=True)
            #except: pass

            #plt.gcf().savefig(out_dir+'sp/'+miPais+'_'+result_set[:-1]+'_offset_legend.pdf',format='pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')
            #plt.gcf().savefig(out_dir+'sp/'+miPais+'_'+result_set[:-1]+'_offset_legend.png',format='png',bbox_extra_artists=(lgd,), bbox_inches='tight')

            lgd = title_legend_labels(ax,iso_to_name[miPais],lab_x='',lab_y=label_dict[result_set],lim_x=[0.75,6.0],lim_y=[0,101],global_fs=global_fontsize,do_leg=False)
            legend = ax.legend()
            legend.remove()

            plt.gcf().savefig(out_dir+'sp/'+miPais+'_'+result_set[:-1]+'_'+str(_polmax)+'.pdf',format='pdf', bbox_inches='tight')
            #plt.gcf().savefig(out_dir+'sp/'+miPais+'_'+result_set[:-1]+'_'+str(_polmax)+'.png',format='png', bbox_inches='tight')

            legend.remove()        
            plt.setp(plt.legend().get_texts(),color=greys[7],fontsize=global_fontsize)
            if result_set == 'pct_receiving_': 
                try:
                    plt.legend(handles+[(dotted_line1, dotted_line2),(dotted_line3,dotted_line4),(dotted_line5,dotted_line6)],
                               labels+['Scale up (uniformly or proportionally)',
                                       'Improve targeting + scale up (uniformly or proportionally)',
                                       'Scaleout + scale up (uniformly or proportionally)'],fontsize=global_fontsize,labelspacing=0.40,loc='best')
                except: plt.legend(fontsize=global_fontsize,labelspacing=0.40,loc='best')
            else: plt.legend(fontsize=global_fontsize,labelspacing=0.40,loc='best')
            # ^ this is a hack to get the legend into the bottom left corner of the plot; the legend has 1 more entry for win_fractions

            plt.gcf().savefig(out_dir+'sp/'+miPais+'_'+result_set[:-1]+'_inset_legend_'+str(_polmax)+'.pdf',format='pdf', bbox_inches='tight')
            #plt.gcf().savefig(out_dir+'sp/'+miPais+'_'+result_set[:-1]+'_inset_legend'+str(_polmax)+'.png',format='png', bbox_inches='tight')            

            if hay_CT:
                #for _xx in ['xx1','xx2']:
                #    df_dec.plot('quintile',result_set+'no_pol',label=pdict[_xx+'desc'],ax=ax,color=pdict[_xx+'col'],linewidth=3.0)
                lgd = title_legend_labels(ax,iso_to_name[miPais],lab_x='',lab_y='Cash transfer recipients [%]',lim_x=[0.5,5.5],lim_y=[0,101],global_fs=global_fontsize)
                plt.gcf().savefig(out_dir+'sp/'+result_set+'legend.pdf',format='pdf',bbox_extra_artists=(lgd,), bbox_inches='tight') 

            plt.cla(); plt.close('all')


####################################################
####################################################
# Everything below runs when the file loads

####################################################

# Call main function
if len(correo) == 1: 
    for _p in correo: 
        run_tax_to_sp(_p)
        #study_inclusion_exclusion_error(_p)

else: 
    with Pool(processes=1) as pool:
        print('LAUNCHING',correo)
        pool.starmap(run_tax_to_sp,list(product(correo)))

#####################################
# Latex all the files into a pdf
if False and miPais != 'brb':
    for _ in ['rel','abs']:
        with open('output/latex/all_countries_sp_impact_'+_+'.tex', 'w') as f2:
            f2 = write_latex_header(f2)

            for _p in todos:
                f2.write(r'\subfloat{\includegraphics[trim={0 2.4cm 0 0},clip,width=0.32\columnwidth]{../sp/tax_net_sp_'+_+'/all/'+_p+'_all.pdf}}\\'+'\n')
            f2.write(r'\subfloat{\includegraphics[trim={0 0 0 11cm},clip,width=0.64\columnwidth]{../sp/tax_net_sp_'+_+'/all/legend.pdf}}'+'\n') 
            
            f2 = write_latex_footer(f2);f2.close()

        subprocess.call('cd ~/Box\ Sync/stranded\ jobs\ and\ distributional\ impacts/output/latex/; pdflatex all_countries_sp_impact_'+_+'.tex',shell=True)

    for _ in ['direct','indirect']:
        with open('output/latex/all_countries_'+_+'_tax.tex', 'w') as f2:
            f2 = write_latex_header(f2)
            
            for _p in todos:
                f2.write(r'\subfloat{\includegraphics[trim={0 0 0 0},clip,width=0.32\columnwidth]{../expenditures/'+_p+'_'+_+'_tax_as_pct_of_gastos_by_quintile.pdf}}\\'+'\n')

            f2 = write_latex_footer(f2); f2.close()

        subprocess.call('cd ~/Box\ Sync/stranded\ jobs\ and\ distributional\ impacts/output/latex/; pdflatex all_countries_'+_+'_tax.tex',shell=True)


    for _ in ['winning_fraction','hh_fraction_receive_SP']:
        with open('output/latex/all_countries_'+_+'.tex', 'w') as f2:
            f2 = write_latex_header(f2)
            
            for _p in todos:
                f2.write(r'\subfloat{\includegraphics[trim={0 2.4cm 0 0},clip,width=0.32\columnwidth]{../sp/'+_p+'_'+_+'.pdf}}\\'+'\n')
            f2.write(r'\subfloat{\includegraphics[trim={0 0 0 11cm},clip,width=0.64\columnwidth]{../sp/legend_'+_+'.pdf}}'+'\n') 

            f2 = write_latex_footer(f2); f2.close()

        #try: subprocess.call('cd ~/Box\ Sync/stranded\ jobs\ and\ distributional\ impacts/output/latex/; pdflatex all_countries_'+_+'.tex',shell=True)
        #except: pass

success = False
_ctr = 0
while not success and _ctr < 2:
    try: 
        q12_reimbursement_costs()
        success=True
    except: _ctr+=1

q12_reimbursement_costs()
q12_reimbursement_beneficiaries()

hh_expenditure_table()
hh_regressivity_table()
cost_increase_table()
hh_costs_table()
pop_frac_reimbursement_costs()

for f in glob.glob('output/latex/*.aux'): os.remove(f)
for f in glob.glob('output/latex/*.log'): os.remove(f)
for f in glob.glob('output/latex/out_*.tex'): os.remove(f)
