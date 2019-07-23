import pandas as pd
import numpy as np

def get_enrollment(pais,wgt,df,df_dec,tot_rev,inc_src,increase=None):

    # INPUTS:
    # - pais    = country
    # - wgt     = household weights
    # - df      = dataframe with household info
    # - df_dec  = dataframe decile-level averages 
    # - tot_rev = total revenue
    # - inc_src = income source

    df = df.reset_index('quintile')

    # create a df that will get returned
    df_out = pd.DataFrame({wgt:df[wgt],'quintile':df['quintile'],'enrolled':0},index=df.index)

    # Get string to slice df (exclude hh already enrolled)
    crit = 'ing_'+inc_src+'==0'
    _df = df.loc[df.eval(crit)].copy().sort_values('gct')

    #####
    # Figure out what % is getting enrolled
    _df_dec = df_dec.copy()

    if increase == 'dmax':
        _df_dec['increase'] = float(df_dec['pct_receiving_'+inc_src].max()) - df_dec['pct_receiving_'+inc_src]
    elif increase == '100':

        #print(_df_dec.head())

        _df_dec['init_cost'] = _df_dec[['mean_val_'+inc_src,'pct_receiving_'+inc_src,'pop']].prod(axis=1)
        cost_scale_fac = tot_rev/_df_dec['init_cost'].sum()

        _df_dec['increase'] = 100*cost_scale_fac*_df_dec['pct_receiving_'+inc_src]

        #print(_df_dec['increase'])
        #assert(False)

        #print(tot_rev,_df_dec['init_cost'].sum(),cost_scale_fac)
        #_df_dec['increase'] = 0#float(df_dec['pct_receiving_'+inc_src].max()) - df_dec['pct_receiving_'+inc_src]
        #assert(False)

    # Run through from poorest to wealthiest, and stop when we hit 'increase'
    _df['enrolled'] = 0.
    for dec, idec in _df.groupby('quintile'):
        
        dec_denom = df_out.loc[df_out.quintile==dec,wgt].sum()
        # Do the loop, find the poorest <increase>% of hh
        hhno = 0
        while hhno < idec.shape[0] and idec.iloc[:hhno][wgt].sum()/dec_denom < _df_dec.loc[dec,'increase']/100.:
            hhno+=1

        # list these hh as enrolled
        _df.loc[idec.iloc[:hhno].index.tolist(),'enrolled'] = 1.
            
    df_out.loc[_df.index.tolist(),'enrolled'] = _df['enrolled']

    df_out = df_out.reset_index().set_index(['cod_hogar','quintile'])
    return df_out['enrolled']


def targeting_exercise(hh_df,transfer_type,corr_inclusion=0.,corr_exclusion=None,scaleup_abs=False):

    #INPUTS: 
    # - hh_df          = household-level dataframe
    # - transfer_type  = ct OR trsgob
    # - corr_inclusion = correction to inclusion error can have value [0-1]
    # - corr_exclusion = JUST A PLACEHOLDER; doesn't do anything

    if corr_inclusion < 0. or corr_inclusion > 1.: 
        print('corr_inclusion =',corr_inclusion,'; should be less than 1.')
        assert(False)
    if corr_exclusion is not None and corr_exclusion > 1.: print('corr_exclusion =',corr_exclusion,'; should be less than 1.')

    _transfer = 'ing_'+transfer_type
    _transfer_total_value = float(hh_df[['pcwgt',_transfer]].prod(axis=1).sum())
    _total_pop   = float(hh_df['pcwgt'].sum())

    # Get list of wealthy hh included in CT
    enrolled = hh_df.loc[(hh_df.quintile>=4)&(hh_df[_transfer]!=0)].sort_values('gct',ascending=False).copy()
    enrolled['cumsum_pcwgt'] = enrolled['pcwgt'].transform('cumsum')/enrolled['pcwgt'].sum()

    # Get list of poor hh excluded from CT
    excluded = hh_df.loc[(hh_df[_transfer]==0)].sort_values('gct',ascending=True).copy()
    excluded['cumsum_pcwgt'] = excluded['pcwgt'].transform('cumsum')/hh_df['pcwgt'].sum()

    # The idea here is:
    # 1) decrease inclusion error (nominally, by 10%), in order of decreasing income
    # 2) calculate the fraction of revenue saved by excluding these hh 
    
    # Run the whole routine separately for scaleup_abs, not scaleup_abs:
    if not scaleup_abs:

        enrolled['new_'+_transfer] = enrolled[_transfer]
        # ^ What fraction of each dollar spent in new program will households in Q4 & Q5 get?
     
        for _q in [1,2,3,4,5]:
            excluded.loc[excluded.quintile==_q,'new_'+_transfer] = (hh_df.loc[hh_df['quintile']==_q,['pcwgt',_transfer]].prod(axis=1).sum()
                                                                    /(hh_df.loc[(hh_df['quintile']==_q)&(hh_df[_transfer]!=0),'pcwgt'].sum()))
        # ^ How much (in dollar) spent in new program will households in Q1 & Q2 get?

        #ie_revenue = float(enrolled.loc[enrolled['cumsum_pcwgt']<=corr_inclusion,['pcwgt','new_'+_transfer]].prod(axis=1).sum())
        # ^ This is what we'll save from improving targeting error (fraction of total expenditures)
        ie_pop_unenrolled = float(enrolled.loc[enrolled['cumsum_pcwgt']<=corr_inclusion,'pcwgt'].sum()/hh_df['pcwgt'].sum())
        # ^ This is the total number of people unenrolled
        
        enrolled.loc[enrolled['cumsum_pcwgt']<corr_inclusion,'new_'+_transfer] = 0.
        # ^ What fraction of expenditures can be saved if we reduce inclusion error by <corr_inclusion>%?

        #excluded['total_distribution'] = excluded[['pcwgt','new_'+_transfer]].prod(axis=1)
        #excluded['cumsum_distribution'] = excluded['total_distribution'].transform('cumsum')

        excluded.loc[excluded['cumsum_pcwgt']>ie_pop_unenrolled,'new_'+_transfer] = 0
        total_distribution = excluded[['pcwgt','new_'+_transfer]].prod(axis=1).sum()

        # THIS IS NOT REVENUE NEUTRAL!
        #if (abs(total_distribution/ie_revenue-1.) > 0.1): 
        #    print('scaleup_abs = ',scaleup_abs,'ie_revenue = ',ie_revenue)
        #    print(total_distribution/ie_revenue)
        #    print(total_distribution,ie_revenue)
        #    assert(False)
        #else: print('OK! Spent all the money!')

    else: 
        
        enrolled['new_'+_transfer] = 1.
        excluded['new_'+_transfer] = 1.
        other_enrollees = hh_df[~(hh_df.index.isin(enrolled.index) | hh_df.index.isin(excluded.index))].copy()
        
        enrolled.loc[enrolled['cumsum_pcwgt']<=corr_inclusion,'new_'+_transfer] = 0.
        # ^ What fraction of expenditures can be saved if we reduce inclusion error by <corr_inclusion>%?
        
        frac_pop_disenrolled = enrolled.loc[enrolled['new_'+_transfer]==0.,'pcwgt'].sum()/hh_df['pcwgt'].sum()
        excluded.loc[excluded['pcwgt'].cumsum()/hh_df['pcwgt'].sum()>frac_pop_disenrolled,'new_'+_transfer] = 0.

        total_enrollees = (enrolled.loc[enrolled['new_'+_transfer]!=0,'pcwgt'].sum()
                           +excluded.loc[excluded['new_'+_transfer]!=0,'pcwgt'].sum()
                           +other_enrollees.loc[other_enrollees[_transfer]!=0,'pcwgt'].sum())
        enrolled.loc[enrolled['new_'+_transfer]!=0,'new_'+_transfer] = 1./total_enrollees
        excluded.loc[excluded['new_'+_transfer]!=0,'new_'+_transfer] = 1./total_enrollees
    

    # 5) put it into the original dataframe
    if scaleup_abs: hh_df['new_'+_transfer] = 1./total_enrollees
    else: 
        hh_df['new_'+_transfer] = hh_df[_transfer].copy()
    

    hh_df.loc[enrolled.index.tolist(),'new_'+_transfer] = enrolled['new_'+_transfer]
    hh_df.loc[excluded.index.tolist(),'new_'+_transfer] = excluded['new_'+_transfer]

    if not scaleup_abs: hh_df['new_'+_transfer]/=hh_df[['new_'+_transfer,'pcwgt']].prod(axis=1).sum()
    
    try: assert(round(hh_df[['pcwgt','new_'+_transfer]].prod(axis=1).sum(),2) == 1.00)
    except: 
        print('\n\nCHECK!!!!  should equal unity (',scaleup_abs,corr_inclusion,') =',round(hh_df[['pcwgt','new_'+_transfer]].prod(axis=1).sum(),2))
        print('NOTE: returning fractional hh distributions (combine with pcwgt, adds up to 1). Need to multiply this result by revenue\n\n')

    return(hh_df['new_'+_transfer])
