import pandas as pd
import numpy as np

def reverse_engineer(pais,df,net_ben=.60):
    #
    total_revenue = df[['carbon_cost','pcwgt']].prod(axis=1).sum()
    #
    policies = [_c for _c in df if _c not in ['pcwgt','carbon_cost']]
    df[policies] = df[policies].div(df[['carbon_cost','pcwgt']].prod(axis=1).sum())
    
    total_pop = df['pcwgt'].sum()
    for _pol in policies:

        try: 
            df_out = pd.read_csv('output/reverse_engineering/'+_pol.replace(' ','').replace('-','')+'.csv',index_col=0)
        except: 
            df_out = pd.DataFrame(index={'20','40','60'},columns={pais})
            #df_out.index.name = 'country'
        
        beneficiaries = -1.0
        for _th in df_out.index:
            _th_float = float(_th)*1E-2

            for spend_frac in np.linspace(0.0,2.0,1000):
                _crit = (spend_frac*total_revenue)*df[_pol]>df['carbon_cost']
                beneficiaries = df.loc[_crit,'pcwgt'].sum()/total_pop
                if beneficiaries >= _th_float: 
                    df_out.loc[_th,pais] = 100*round(spend_frac,2)
                    print('Spend',100*round(spend_frac,1),'% of revenue for',1E2*beneficiaries,'% of population to experience benefit from '+_pol)  
                    break
                
            print('Spend',100*round(spend_frac,1),'% of revenue for',1E2*beneficiaries,'% of population to experience benefit from '+_pol)  
        
        df_out.fillna('-').sort_index().to_csv('output/reverse_engineering/'+_pol.replace(' ','').replace('-','')+'.csv')


    df_all = None
    for _pol in policies:
        df_out = pd.read_csv('output/reverse_engineering/'+_pol.replace(' ','').replace('-','')+'.csv',index_col=0)
        df_out.index.name = 'Net beneficiaries'
        df_out['Policy'] = _pol

        if df_all is not None: df_all = pd.concat([df_all,df_out])
        else: df_all = df_out.copy()

    df_all.reset_index().set_index(['Policy','Net beneficiaries']).to_csv('output/reverse_engineering/all_policies.csv')
