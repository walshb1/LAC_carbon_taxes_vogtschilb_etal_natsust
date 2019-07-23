import pandas as pd
import seaborn as sns

from libraries.lib_common_plotting_functions import _12col_paired_pal,greys

sns.set_style('whitegrid')
sns_pal = sns.color_palette('Set1', n_colors=8, desat=.5)
sns_set2 = sns.color_palette('Set2', n_colors=8)

def get_policies(do_better = 0):
    # do better = percent change in coverage (scaleout) and inclusion (targeting)
    # these are totally orthogonal, but it makes comms sense to compare, EG: 25% scaleout to 25% targeting improvement

    if do_better == 0:
        return ['no_pol','new_sp','scaleup_ct_rel','scaleup_ct_abs']
    elif do_better == 'all':
        return ['no_pol','scaleup_ct_rel','scaleup_ct_abs',
                'new_sp',
                'scaleup_ct_shift10','scaleup_ct_abs_shift10','scaleup_ct_rel_shift10',
                'scaleup_ct_shift20','scaleup_ct_abs_shift20','scaleup_ct_rel_shift20',
                'scaleout10_ct_abs',
                'scaleout20_ct_abs',
                'scaleout30_ct_abs',
                'scaleout40_ct_abs',
                'scaleout100_ct_abs',
                'scaleout200_ct_abs']
    else: 
        _sdb = str(do_better)
        return ['no_pol','new_sp',
                #'scaleup_ct_rel',
                'scaleup_ct_abs',
                #'scaleup_ct_rel_shift'+_sdb,
                'scaleup_ct_abs_shift'+_sdb,
                #'scaleout'+_sdb+'_ct_rel',
                'scaleout'+_sdb+'_ct_abs']
           
def get_pol_dict(miPais,pols,hay_CT=True):

    pdict = {}

    # assign a color to each policy
    pdict['no_polcol'] = greys[4]
    pdict['new_spcol'] = _12col_paired_pal[9]

    pdict['scaleup_ct_relcol'] = _12col_paired_pal[6]
    pdict['scaleup_ct_abscol'] = _12col_paired_pal[7]
    #
    pdict['scaleup_ct_rel_shift10col'] = _12col_paired_pal[0]
    pdict['scaleup_ct_rel_shift20col'] = _12col_paired_pal[0]
    pdict['scaleup_ct_rel_shift25col'] = _12col_paired_pal[0]
    pdict['scaleup_ct_abs_shift10col'] = _12col_paired_pal[1]
    pdict['scaleup_ct_abs_shift20col'] = _12col_paired_pal[1]
    pdict['scaleup_ct_abs_shift25col'] = _12col_paired_pal[1]

    #pdict['scaleout10_ct_relcol'] = _12col_paired_pal[2]
    #pdict['scaleout20_ct_relcol'] = _12col_paired_pal[2]
    pdict['scaleout25_ct_relcol'] = _12col_paired_pal[2]
    #pdict['scaleout30_ct_relcol'] = _12col_paired_pal[2]
    #pdict['scaleout40_ct_relcol'] = _12col_paired_pal[2]        
    #pdict['scaleout10_ct_abscol'] = _12col_paired_pal[3]
    #pdict['scaleout20_ct_abscol'] = _12col_paired_pal[3]
    pdict['scaleout25_ct_abscol'] = _12col_paired_pal[3]
    #pdict['scaleout30_ct_abscol'] = _12col_paired_pal[3]
    #pdict['scaleout40_ct_abscol'] = _12col_paired_pal[3]

    pdict['xx1col'] = _12col_paired_pal[7]
    pdict['xx2col'] = _12col_paired_pal[8]

    # assign a descriptor to each policy
    pdict['no_poldesc'] = 'No redistribution'
    pdict['new_spdesc'] = 'Universal rebate'
    
    ttype = 'cash'
    if not hay_CT: ttype = 'all'

    pdict['scaleup_ct_absdesc'] = 'Current-enrollees rebate'
    pdict['scaleup_ct_reldesc'] = 'Current-enrollees rebate (proportional)'

    for _t in ['_shift25']:#['_shift10','_shift20','_shift25','_shift30','_shift40','_shift50']:

        _tstr = ''
        if _t != '': _tstr = _t[-2:]+'%'
        pdict['scaleup_ct_abs'+_t+'desc'] = 'Poverty-targeted rebate'
        pdict['scaleup_ct_rel'+_t+'desc'] = 'Poverty-targeted rebate (proportional)'

    for _so in [25]:#[10,20,25,30,40,50,100,200]:
        pdict['scaleout'+str(_so)+'_ct_absdesc'] = 'Expanded-enrollees rebate'
        pdict['scaleout'+str(_so)+'_ct_reldesc'] = 'Expanded-enrollees rebate (proportional)'
 
    pdict['xx1desc'] = 'Current-enrollees rebate'
    pdict['xx2desc'] = 'Expanded-enrollees rebate'

    return pdict

def update_pol_dict(df_dec,pdict):

    print(df_dec.columns)

    try:pdict.update({'scaleout10_ct_absdmaxdesc':'Scale out cash transfers to '+str(int(df_dec.pct_receiving_ct.max()))+'%',
                      'scaleout10_trsgob_dmax_absdesc':'Scale out other transfers to '+str(int(df_dec.pct_receiving_trsgob.max()))+'%'})
    except:pass

    return pdict


def study_inclusion_exclusion_error(miPais,df=pd.DataFrame()):

    pdict = get_pol_dict()

    if not df.empty: print('df with hh info passed directly')
    else: 
        df = pd.read_csv('~/Desktop/tmp/study.csv')
        print('df with hh info loaded')

    df_dec = pd.read_csv('output/sp/'+miPais+'_quintile_info.csv').set_index('quintile').sort_index()

    # Exercise 1: what fraction of the bottom 40% (in abs % points) can be covered for each 1% (abs) decrease in inclusion error (transfers to top 40%)?
    
    # -- Need 2 data points for each SP system:
    # -- 1) Current exclusion/inclusion error
    # -- 2) E/I err if I is reduced 10%

    ie = {}
    # -- 1) Current E/I err
    ie['ct_i'] = [100.-df_dec.loc[:2,'pct_receiving_ct'].mean(),df_dec.loc[4:,'pct_receiving_ct'].mean()]
    ie['ct_f'] = [ie['ct_i'][0]-0.1*ie['ct_i'][1]*df_dec.loc[4:,'mean_val_ct'].mean()/df_dec.loc[:2,'mean_val_ct'].mean(), 0.9*ie['ct_i'][1]]
    plt.annotate('Cash transfer',xy=(100.-df_dec.loc[:2,'pct_receiving_ct'].mean(),df_dec.loc[4:,'pct_receiving_ct'].mean()))

    plt.plot([ie['ct_i'][0],ie['ct_f'][0]],[ie['ct_i'][1],ie['ct_f'][1]],color=pdict['scaleup_ctcol'])
    plt.scatter(ie['ct_i'][0],ie['ct_i'][1],color=pdict['scaleup_ct_relcol'],label='Cash transfers')
    plt.scatter(ie['ct_f'][0],ie['ct_f'][1],color=pdict['scaleup_ct_relcol'],label='')

    if do_trsgob:
        ie['trsgob_i'] = [100.-df_dec.loc[:2,'pct_receiving_trsgob'].mean(),df_dec.loc[4:,'pct_receiving_trsgob'].mean()]
        ie['trsgob_f'] = [ie['trsgob_i'][0]-0.1*ie['trsgob_i'][1]*df_dec.loc[4:,'mean_val_trsgob'].mean()/df_dec.loc[:2,'mean_val_trsgob'].mean(), 0.9*ie['trsgob_i'][1]]
        plt.annotate('Non-cash\ntransfer',xy=(100.-df_dec.loc[:2,'pct_receiving_trsgob'].mean(),df_dec.loc[4:,'pct_receiving_trsgob'].mean()))

        plt.plot([ie['trsgob_i'][0],ie['trsgob_f'][0]],[ie['trsgob_i'][1],ie['trsgob_f'][1]],color=pdict['scaleup_trsgobcol'])
        plt.scatter(ie['trsgob_i'][0],ie['trsgob_i'][1],color=pdict['scaleup_trsgobcol'],label='Non-cash transfers')
        plt.scatter(ie['trsgob_f'][0],ie['trsgob_f'][1],color=pdict['scaleup_trsgobcol'],label='')
        print(ie['trsgob_f'][0],ie['trsgob_f'][1])

    ax = plt.gca()
    lgd = title_legend_labels(ax,iso_to_name[miPais],
                              lab_x='Exclusion error\nfraction of bottom 40% not receiving transfers',
                              lab_y='Inclusion error\nfraction of top 40% receiving transfers',lim_x=[0,100],lim_y=[0,50],leg_fs=8,do_leg=False)

    plt.gcf().savefig('output/exclusion_inclusion/'+miPais+'_exclusion_inclusion_10pct.pdf',format='pdf', bbox_inches='tight')

    return True
