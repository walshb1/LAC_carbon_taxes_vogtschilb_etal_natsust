def methodA():

    df.loc[df['ing_ct']!=0,['di_scaleout_ct','di_scaleout_ct_shift10','di_scaleout_ct_shift20']] = 0,0,0
    # ^ This is scale out--money will only go to hh that weren't receiving transfers

    for _t in ['','_shift10','_shift20']:

        not_enrolled = df.loc[df['ing_ct']==0].copy().sort_values('gct',ascending=True)

        not_enrolled = pd.merge(not_enrolled.reset_index(),quintile_frac.reset_index(),on=['quintile']).set_index(['cod_hogar'])
        not_enrolled['quint_total_expenditure'+_t] = pdict['c_rev']*not_enrolled['cash_frac_to_quint'+_t]

        not_enrolled['di_scaleout_ct'+_t] = not_enrolled['mean_val'+_t].copy()

        not_enrolled['tot_cost_scaleout_ct'+_t] = not_enrolled[['pcwgt','di_scaleout_ct'+_t]].prod(axis=1)
        not_enrolled['cum_cost_scaleout_ct'+_t] = not_enrolled.groupby('quintile')['tot_cost_scaleout_ct'+_t].transform('cumsum')

        not_enrolled.loc[not_enrolled['cum_cost_scaleout_ct'+_t]>=not_enrolled['quint_total_expenditure'+_t],'di_scaleout_ct'+_t] = 0.
        not_enrolled = not_enrolled.reset_index().set_index(['cod_hogar','quintile'])

        # Finally, scale out from bottom to top if money left
        not_enrolled['tot_cost_scaleout_ct'+_t] = not_enrolled[['pcwgt','di_scaleout_ct'+_t]].prod(axis=1)
        additional_rev = pdict['c_rev'] - not_enrolled['tot_cost_scaleout_ct'+_t].sum()

        if additional_rev > 1:

            still_not_enrolled = not_enrolled.loc[not_enrolled['di_scaleout_ct'+_t]==0].sort_values('gct').copy()

            still_not_enrolled['di_scaleout_ct'+_t] = still_not_enrolled['mean_val'+_t].copy()
            still_not_enrolled['cum2_cost_scaleout_ct'+_t] = (still_not_enrolled[['di_scaleout_ct'+_t,'pcwgt']].prod(axis=1)).cumsum()

            still_not_enrolled.loc[still_not_enrolled['cum2_cost_scaleout_ct'+_t]>additional_rev,'di_scaleout_ct'+_t] = 0

            not_enrolled.loc[still_not_enrolled.index.tolist(),'di_scaleout_ct'+_t] += still_not_enrolled['di_scaleout_ct'+_t]

            # last check
            additional_rev = pdict['c_rev'] - not_enrolled[['pcwgt','di_scaleout_ct'+_t]].prod(axis=1).sum()
            if additional_rev > 1:
                not_enrolled['di_scaleout_ct'+_t] *= pdict['c_rev']/not_enrolled[['pcwgt','di_scaleout_ct'+_t]].prod(axis=1).sum()

        df['di_scaleout_ct'+_t].update(not_enrolled['di_scaleout_ct'+_t])

        # Check that we spent all the money
        if (df[['di_scaleout_ct'+_t,'pcwgt']].prod(axis=1).sum()/pdict['c_rev'] > 1.01 or 
            df[['di_scaleout_ct'+_t,'pcwgt']].prod(axis=1).sum()/pdict['c_rev'] < 0.99):
            print(df[['di_scaleout_ct'+_t,'pcwgt']].prod(axis=1).sum(),pdict['c_rev'])
            df['tot_cost_scaleout_ct'+_t] = df[['di_scaleout_ct'+_t,'pcwgt']].prod(axis=1)
            assert(False)

        df_dec['pct_receiving_scaleout_ct'+_t] = 100.*(df.loc[df['di_scaleout_ct'+_t]>0,'pcwgt'].sum(level='quintile'))/df_dec[dec_size]    

def methodB():
    ####################################
    # Get poorest hh in each quintile, and enroll them in scaleouts
    df['enrolled_scaleout_ct'] = get_enrollment(miPais,wgt,df,df_dec,pdict['c_rev'],'ct','100')

    #################################
    # POLICY: scale out CT to 100% of pop
    # --> Distributing $ to all hh with no CT income, 
    # --> then scaling those people's income up or down to match pdict['c_rev']

    # Grab a slice of the df
    df_pol = df.loc[df.ing_ct==0].copy()
    df_pol = pd.merge(df_pol.reset_index(),df_dec['mean_val_ct'].reset_index(),on='quintile').set_index('cod_hogar')

    # Calculate group average income
    df_pol['hh_tot_inc'] = df_pol[['gct',wgt]].prod(axis=1)
    df_pol['avg_inc'] = df_pol.groupby('quintile')['hh_tot_inc'].transform('sum')/df_pol.groupby('quintile')[wgt].transform('sum')

    ##############
    # The NEW new way: use enrollment script
    df_pol = df_pol.reset_index().set_index(['cod_hogar','quintile'])

    df_pol.loc[(df_pol.ing_ct==0)&(df_pol.enrolled_scaleout_ct==1),'di_scaleout_ct'] = df_pol.loc[(df_pol.ing_ct==0)&(df_pol.enrolled_scaleout_ct==1),'mean_val_ct']
    df_dec['di_scaleout_ct'] = df_pol[['di_scaleout_ct',wgt]].prod(axis=1).sum(level='quintile')
    df_pol = df_pol.reset_index('quintile')

    print(pdict['c_rev'])
    print(df_pol[[wgt,'di_scaleout_ct']].prod(axis=1).sum())
    print('Scaling ratio:',pdict['c_rev']/df_dec['di_scaleout_ct'].sum())

    ##if my_revenue > 0:
    ## Scale up to match pdict['c_rev']
    ## --> avg payout for new enrollees != avg payout for existing program 
    df_pol['di_scaleout_ct'] *= pdict['c_rev']/df_pol[['di_scaleout_ct',wgt]].prod(axis=1).sum()
    df_dec['di_scaleout_ct'] *= pdict['c_rev']/df_dec['di_scaleout_ct'].sum()    
    ##############

    # Reset index and re-merge
    df_pol = df_pol.reset_index().set_index(['cod_hogar','quintile'])
    df.loc[df.ing_ct==0,'di_scaleout_ct'] = df_pol['di_scaleout_ct']
    df['di_scaleout_ct'] = df['di_scaleout_ct'].fillna(0)
