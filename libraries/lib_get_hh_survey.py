import pandas as pd
import os, glob

def get_negative_cols(pais,hh_df):

    try: negative_dict = pd.read_csv('output/hh_survey_negative_values.csv').set_index('pais')
    except: negative_dict = pd.DataFrame(columns=['negative_values'])

    negative_cols = [_c for _c in hh_df.columns if ((hh_df[_c].dtype == 'float32' or hh_df[_c].dtype == 'float64') 
                                                    and ('ict' not in _c) and ('ing' not in _c or 'ct' in _c or 'trsgob' in _c) 
                                                    and (hh_df[_c].min() < 0))]

    out_str = ''
    if len(negative_cols) == 0: out_str = '--, '
    else:
        for i in negative_cols: out_str += i+', '

    negative_dict.loc[pais,'negative_values'] = out_str[:-2]
    negative_dict.index.name = 'pais'
    negative_dict.sort_index().to_csv('output/hh_survey_negative_values.csv')

    if len(negative_cols)==0: return None
    return negative_cols

def get_hh_survey(pais):
    hh_survey = None

    if pais == 'chl': pais = 'chi'

    try: 
        file_name = 'consumption_and_household_surveys/2017-10-13/Household_survey_with_new_file_name/'+pais+'_household_expenditure_survey.dta'
        hh_survey = pd.read_stata(file_name).set_index('cod_hogar')

    except:
        file_name = 'consumption_and_household_surveys/Expansion_Countries/'
        for f in glob.glob(file_name+pais.upper()+'*'):
            if 'PERSONA' not in f: 
                hh_survey = pd.read_stata(f)
                try: hh_survey['cod_hogar'] = hh_survey['cod_hogar'].astype('int')
                except: pass
                hh_survey = hh_survey.reset_index().set_index('cod_hogar')
                hh_survey = hh_survey.drop([i for i in ['index'] if i in hh_survey.columns],axis=1)
                break

        if 'miembros_hogar' not in hh_survey.columns: 
            hh_survey['miembros_hogar'] = get_miembros_hogar(pais)
            if hh_survey['miembros_hogar'].shape[0] != hh_survey['miembros_hogar'].dropna().shape[0]:
                n_fail = hh_survey['miembros_hogar'].shape[0] - hh_survey['miembros_hogar'].dropna().shape[0]
                print('Finding',n_fail,'hh with no info on miembros hogar! ('+str(int(100.*n_fail/hh_survey['miembros_hogar'].shape[0]))+'% of hh)')
                assert(False)
            print('\nLOADED miembros hogar')

    if pais == 'ury': 
        hh_survey['gasto_vca'] = hh_survey['gasto_vca'].fillna(0)
        hh_survey['gasto_viv'] -= hh_survey['gasto_vca']
        hh_survey['gasto_vleca'] -= hh_survey['gasto_vca']
        hh_survey['gasto_vca'] = 0
    
    hh_survey = hh_survey.loc[hh_survey['gct']>0]
    # Check whether there are any hh that don't return CT info, but do show a difference between total receipts & other transfers
    #print(hh_survey.loc[(hh_survey.ing_tpub!=0)&(hh_survey.ing_tpub!=hh_survey.ing_trsgob),['ing_tpub','ing_ct','ing_trsgob']].head())

    if (pais == 'col' 
        or pais == 'gtm' 
        or pais == 'pan' 
        or pais == 'nic' 
        or pais == 'pry'
        or pais == 'hnd'): hh_survey['gasto_vgn'] = hh_survey['gasto_vgn'].fillna(1E-6)
    if pais == 'nic': hh_survey['gasto_vag'] = hh_survey['gasto_vag'].fillna(1E-6)
    if pais == 'pry': hh_survey['gasto_vele'] = hh_survey['gasto_vele'].fillna(1E-6)

    hh_survey = hh_survey.rename(columns={'factor_expansion_1':'factor_expansion'}).fillna(0)
    
    n_hh = hh_survey.shape[0]
    negative_cols = get_negative_cols(pais, hh_survey)

    if negative_cols is not None:
        for _n in negative_cols:

            #if pais == 'arg':
            # -> This code would reduce % of hh dropped from 4.9% to 0.1%
            #    hh_survey.loc[hh_survey['gasto_totros']<0,'gct'] -= hh_survey.loc[hh_survey['gasto_totros']<0,'gasto_totros']
            #    hh_survey.loc[hh_survey['gasto_totros']<0,'gasto_trans'] -= hh_survey.loc[hh_survey['gasto_totros']<0,'gasto_totros']
            #    hh_survey.loc[hh_survey['gasto_totros']<0,'gasto_totros'] -= hh_survey.loc[hh_survey['gasto_totros']<0,'gasto_totros']

            if 'ing' in _n: hh_survey[_n] = hh_survey[_n].clip(lower=0.)
            else:
                hh_survey.loc[(hh_survey[_n]>=-1E-2)&(hh_survey[_n]<0),_n] = 0.
                hh_survey = hh_survey.loc[hh_survey[_n]>=0]
    percent_dropped = str(round(100.*(1-hh_survey.shape[0]/n_hh),1))

    print('Dropping '+percent_dropped+'% of surveyed hh in',pais)
    try: dropped_record = pd.read_csv('./output/percent_of_survey_dropped_negative_values.csv').set_index('pais')
    except: dropped_record = pd.DataFrame(columns={'pct_dropped':None,'pais':pais}).set_index('pais')
    dropped_record.loc[pais,'pct_dropped'] = float(percent_dropped)
    dropped_record.sort_values('pct_dropped',ascending=False).to_csv('./output/percent_of_survey_dropped_negative_values.csv')

    if pais == 'pry': 
        _ = hh_survey.shape[0]
        hh_survey = hh_survey.loc[hh_survey['gct']<=1E10]
        print('dropping ',_/hh_survey.shape[0],'% of rows in PRY survey')

    return hh_survey
            
def get_miembros_hogar(pais):

    file_name = 'consumption_and_household_surveys/Expansion_Countries/'
    for f in glob.glob(file_name+pais.upper()+'*'):
        print('\n\n',f)
        if 'PERSONA' in f: 
            hh_personas = pd.read_stata(f)

            try: hh_personas['cod_hogar'] = hh_personas['cod_hogar'].astype('int')
            except: pass
            hh_personas = hh_personas.set_index('cod_hogar')
            return hh_personas['miembros_hogar'].mean(level='cod_hogar')
            
    print('FATAL: can not get miembros_hogar!')
    assert(False)
