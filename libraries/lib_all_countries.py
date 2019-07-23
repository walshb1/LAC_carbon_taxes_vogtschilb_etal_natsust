import pandas as pd
import subprocess
import glob, os
from libraries.lib_country_params import iso_to_name
from libraries.lib_country_comparison import refresh_country_comparison


col_form = '@{}lr|rrrrrrrrrrrrrrrr@{}'

def hh_costs_table():

    for _ii in ['cost','frac']:
        for _id in ['_tax','_direct_tax','_indirect_tax',
                    '_indirect_tax_foodnonCO2','_indirect_tax_elecCO2','_indirect_tax_pubtransCO2']:
            if _id == '_tax' and _ii == 'frac': continue
            if _ii == 'cost' and _id not in ['_tax','_indirect_tax','_direct_tax']: continue
            
            df = pd.read_csv('output/all_countries/hh'+_id+'_'+_ii+'_table.csv').set_index('quintile')
            try: df = df.drop('Median')
            except: pass
            try: df = df.drop('Median',axis=1)
            except: pass

            #if _id != '_tax': df.columns.name = _id[1:2].upper()+_id[2:-4]+' cost by quintile'
            #else: df.columns.name = 'Tax cost by quintile'
            df.columns.name = (_id[1:2].upper()+_id[2:].replace('_',' ')
                               .replace('tax','')
                               .replace('foodnonCO2','Food (non-CO2)')
                               .replace('elecCO2','Electricity (CO2)')
                               .replace('pubtransCO2','Pub. trans. (CO2)'))+' cost by quintile'

            if _ii == 'cost': df.index.name = 'as % of total budget'
            else: df.index.name = 'as % of total tax'
    
            for _c in df.columns:
                df = df.rename(columns={_c:iso_to_name[_c.lower()]})
            df = df.reindex(sorted(df.columns), axis=1)
            df = df.sort_values(df.columns[0],ascending=False,na_position='last')
            df = df.fillna('-')

            __quints = {'Q1':'Poorest','Q2':'Second','Q3':'Third','Q4':'Fourth','Q5':'Wealthiest'}

            #df = df.dropna(how='all')
            df.insert(0,'Median',df.median(axis=1,skipna=True).round(1).fillna('-'))
            df = df.sort_index().round(1)
            df.to_latex('output/all_countries/_hh'+_id+'_'+_ii+'_table.tex',column_format=col_form)
        
            with open('output/all_countries/_hh'+_id+'_'+_ii+'_table.tex', 'r') as f:
                with open('output/all_countries/hh'+_id+'_'+_ii+'_table.tex', 'w') as f2:

                    f2.write(r'\documentclass[preview=true]{standalone}'+'\n')
                    f2.write(r'\usepackage{amssymb} %maths'+'\n')
                    f2.write(r'\usepackage{amsmath} %maths'+'\n')
                    f2.write(r'\usepackage{booktabs}'+'\n')
                    f2.write(r'\usepackage{rotating}'+'\n')
                    f2.write(r'\begin{document}'+'\n')
                    
                    reading_is_fundamental = f.read()
                    reading_is_fundamental = reading_is_fundamental.replace('Median',r'\rotatebox{90}{Median}')
                    for _n in iso_to_name:
                        if _n == 'chi': continue
                        reading_is_fundamental = reading_is_fundamental.replace(iso_to_name[_n],r'\rotatebox{90}{'+iso_to_name[_n]+'}')
                    for _q in __quints:
                        reading_is_fundamental = reading_is_fundamental.replace(_q,__quints[_q])
                    f2.write(reading_is_fundamental)

                    f2.write(r'\end{document}')
                    f2.close()

            subprocess.call('cd ~/Box\ Sync/stranded\ jobs\ and\ distributional\ impacts/output/all_countries/; pdflatex hh'+_id+'_'+_ii+'_table.tex',shell=True)
        
            for f in glob.glob('output/all_countries/*.aux'): os.remove(f)
            for f in glob.glob('output/all_countries/*.log'): os.remove(f)
            for f in glob.glob('output/all_countries/_*.tex'): os.remove(f)




def hh_expenditure_table():
    
    df = pd.read_csv('output/all_countries/hh_expenditures_table.csv').set_index('category')#.fillna('-')
    try: df = df.drop('Median')
    except: pass
    try: df = df.drop('Median',axis=1)
    except: pass

    try: df = df.drop(['Coal'])
    except: pass

    df.columns.name = 'Household expenditures by'
    df.index.name = 'category, as % of total budget'
    
    for _c in df.columns:
        df = df.rename(columns={_c:iso_to_name[_c.lower()]})
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.sort_values(df.columns[0],ascending=False,na_position='last')
    df = df.fillna('-')

    #df = df.dropna(how='all')
    df.insert(0,'Median',df.median(axis=1,skipna=True).round(1).fillna('-'))
    df = df.sort_values('Median',ascending=False).round(1)
    df.to_latex('output/all_countries/_hh_expenditures_full.tex',column_format=col_form)

    df.loc['Other expenditures'] = df.loc[('Other retail & trade',
                                           'Fibres, textiles, & clothing',
                                           'Vehicles',
                                           'Health, education, & recreation',
                                           'Communication'),:].mean(skipna=True).round(1)
    df = df.fillna('-')
    try: df = df.drop(['Other retail & trade',
                                           'Fibres, textiles, & clothing',
                                           'Vehicles',
                                           'Health, education, & recreation',
                                           'Communication'])
    except: pass
    df.to_latex('output/all_countries/_hh_expenditures_abridged.tex',column_format=col_form)
    
    for _tab in ['full','abridged']:

        with open('output/all_countries/_hh_expenditures_'+_tab+'.tex', 'r') as f:
            with open('output/all_countries/hh_expenditures_'+_tab+'.tex', 'w') as f2:

                f2.write(r'\documentclass[preview=true]{standalone}'+'\n')
                f2.write(r'\usepackage{amssymb} %maths'+'\n')
                f2.write(r'\usepackage{amsmath} %maths'+'\n')
                f2.write(r'\usepackage{booktabs}'+'\n')
                f2.write(r'\usepackage{rotating}'+'\n')
                f2.write(r'\begin{document}'+'\n')
                
                reading_is_fundamental = f.read()
                reading_is_fundamental = reading_is_fundamental.replace('Median',r'\rotatebox{90}{Median}')
                for _n in iso_to_name:
                    if _n == 'chi': continue
                    reading_is_fundamental = reading_is_fundamental.replace(iso_to_name[_n],r'\rotatebox{90}{'+iso_to_name[_n]+'}')
                f2.write(reading_is_fundamental)

                f2.write(r'\end{document}')
                f2.close()

        subprocess.call('cd ~/Box\ Sync/stranded\ jobs\ and\ distributional\ impacts/output/all_countries/; pdflatex hh_expenditures_'+_tab+'.tex',shell=True)
        
    for f in glob.glob('output/all_countries/*.aux'): os.remove(f)
    for f in glob.glob('output/all_countries/*.log'): os.remove(f)
    for f in glob.glob('output/all_countries/_*.tex'): os.remove(f)


def q12_reimbursement_beneficiaries():
    refresh_country_comparison()
    
    df = pd.read_csv('output/all_countries_sp/total_pop_benefit_from_Q12_redistribution.csv').set_index('Policy')#.fillna('-')
     
    try: df = df.drop('Median')
    except: pass
    try: df = df.drop('Median',axis=1)
    except: pass

    df.columns.name = 'Net beneficiaries of spending as in'
    df.index.name = 'above (as % of total population)'
    df = df.fillna(-1)
    #df = df.astype('int')

    for _c in df.columns:
        df = df.rename(columns={_c:iso_to_name[_c.lower()]})
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.sort_values(df.columns[0],ascending=False,na_position='last')
    df = df.fillna('-')
    df = df.round(1)

    df.insert(0,'Median',df.median(axis=1,skipna=True).round(0).astype('int').fillna('-'))
    df = df.round(0).astype('int')

    df = df.sort_values('Median',ascending=True)
    df.to_latex('output/all_countries_sp/_redist_ben.tex',column_format=col_form)
    
    with open('output/all_countries_sp/_redist_ben.tex', 'r') as f:
        with open('output/all_countries_sp/redist_ben.tex', 'w') as f2:

            f2.write(r'\documentclass[preview=true]{standalone}'+'\n')
            f2.write(r'\usepackage{amssymb} %maths'+'\n')
            f2.write(r'\usepackage{amsmath} %maths'+'\n')
            f2.write(r'\usepackage{booktabs}'+'\n')
            f2.write(r'\usepackage{rotating}'+'\n')
            f2.write(r'\begin{document}'+'\n')
                
            reading_is_fundamental = f.read()
            reading_is_fundamental = reading_is_fundamental.replace('Median',r'\rotatebox{90}{Median}')
            for _n in iso_to_name:
                if _n == 'chi': continue
                reading_is_fundamental = reading_is_fundamental.replace(iso_to_name[_n],r'\rotatebox{90}{'+iso_to_name[_n]+'}')
            f2.write(reading_is_fundamental)

            f2.write(r'\end{document}')
            f2.close()

    subprocess.call('cd ~/Box\ Sync/stranded\ jobs\ and\ distributional\ impacts/output/all_countries_sp/; pdflatex redist_ben.tex',shell=True)
        
    for f in glob.glob('output/all_countries_sp/*.aux'): os.remove(f)
    for f in glob.glob('output/all_countries_sp/*.log'): os.remove(f)
    for f in glob.glob('output/all_countries_sp/_*.tex'): os.remove(f)



def q12_reimbursement_costs():
    refresh_country_comparison()
    
    df = pd.read_csv('output/all_countries_sp/redistribution_expenditures_by_country.csv').set_index('sp')#.fillna('-')
    try: df = df.drop('Median')
    except: pass
    try: df = df.drop('Median',axis=1)
    except: pass

    #df.columns.name = 'Cost to reimburse bottom 40%'
    #df.index.name = '(as % of total carbon revenue)'
    df.index.name = 'Rebate program'
    
    
    df = df.fillna(-1)
    #df = df.astype('int')

    for _c in df.columns:
        df = df.rename(columns={_c:iso_to_name[_c.lower()]})
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.sort_values(df.columns[0],ascending=False,na_position='last')
    df = df.fillna('-')
    df = df.round(1)

    df.insert(0,'Median',df.median(axis=1,skipna=True).round(0).astype('int').fillna('-'))
    df = df.round(0).astype('int')

    df = df.sort_values('Median',ascending=True)
    df.to_latex('output/all_countries_sp/_redist_exp_fractional_spending_Q12_compensated.tex',column_format=col_form)
    
    with open('output/all_countries_sp/_redist_exp_fractional_spending_Q12_compensated.tex', 'r') as f:
        with open('output/all_countries_sp/redist_exp_fractional_spending_Q12_compensated.tex', 'w') as f2:

            f2.write(r'\documentclass[preview=true]{standalone}'+'\n')
            f2.write(r'\usepackage{amssymb} %maths'+'\n')
            f2.write(r'\usepackage{amsmath} %maths'+'\n')
            f2.write(r'\usepackage{booktabs}'+'\n')
            f2.write(r'\usepackage{rotating}'+'\n')
            f2.write(r'\begin{document}'+'\n')
                
            reading_is_fundamental = f.read()
            reading_is_fundamental = reading_is_fundamental.replace('Median',r'\rotatebox{90}{Median}')
            for _n in iso_to_name:
                if _n == 'chi': continue
                reading_is_fundamental = reading_is_fundamental.replace(iso_to_name[_n],r'\rotatebox{90}{'+iso_to_name[_n]+'}')
            f2.write(reading_is_fundamental)

            f2.write(r'\end{document}')
            f2.close()

    subprocess.call('cd ~/Box\ Sync/stranded\ jobs\ and\ distributional\ impacts/output/all_countries_sp/; pdflatex redist_exp_fractional_spending_Q12_compensated.tex',shell=True)
        
    for f in glob.glob('output/all_countries_sp/*.aux'): os.remove(f)
    for f in glob.glob('output/all_countries_sp/*.log'): os.remove(f)
    for f in glob.glob('output/all_countries_sp/_*.tex'): os.remove(f)








def hh_regressivity_table(_q='Q1'):
    
    df = pd.read_csv('output/all_countries/hh_regressivity_table_'+_q+'.csv').set_index('category')#.fillna('-')
    #df = df.drop(['Coal'])
    df.columns.name = 'Cost to '+_q+' vs Q5'
    df.index.name = '(as % of household budgets)'
    df = df.round(1)
    try: df = df.drop('Median')
    except: pass
    try: df = df.drop('Median',axis=1)
    except: pass


    for _c in df.columns:
        df = df.rename(columns={_c:iso_to_name[_c.lower()]})
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.sort_values(df.columns[0],ascending=False,na_position='last')
    df = df.fillna('-')

    df.insert(0,'Median',df.median(axis=1,skipna=True).round(1).fillna('-'))
    df = df.sort_values('Median',ascending=False)
    df.to_latex('output/all_countries/_hh_regressivity_'+_q+'_full.tex',column_format=col_form)


    df.loc['Other expenditures'] = df.loc[('Other retail & trade',
                                           'Fibres, textiles, & clothing',
                                           'Vehicles',
                                           'Health, education, & recreation',
                                           'Communication'),:].mean(skipna=True).round(1)
    df = df.fillna('-')
    try: df = df.drop(['Other retail & trade',
                                           'Fibres, textiles, & clothing',
                                           'Vehicles',
                                           'Health, education, & recreation',
                                           'Communication'])
    except: pass
    df.to_latex('output/all_countries/_hh_regressivity_'+_q+'_abridged.tex',column_format=col_form)
    
    for _tab in ['full','abridged']:

        with open('output/all_countries/_hh_regressivity_'+_q+'_'+_tab+'.tex', 'r') as f:
            with open('output/all_countries/hh_regressivity_'+_q+'_'+_tab+'.tex', 'w') as f2:

                f2.write(r'\documentclass[preview=true]{standalone}'+'\n')
                f2.write(r'\usepackage{amssymb} %maths'+'\n')
                f2.write(r'\usepackage{amsmath} %maths'+'\n')
                f2.write(r'\usepackage{booktabs}'+'\n')
                f2.write(r'\usepackage{rotating}'+'\n')
                f2.write(r'\begin{document}'+'\n')
                
                reading_is_fundamental = f.read()
                reading_is_fundamental = reading_is_fundamental.replace('Median',r'\rotatebox{90}{Median}')
                for _n in iso_to_name:
                    if _n == 'chi': continue
                    reading_is_fundamental = reading_is_fundamental.replace(iso_to_name[_n],r'\rotatebox{90}{'+iso_to_name[_n]+'}')
                f2.write(reading_is_fundamental)

                f2.write(r'\end{document}')
                f2.close()

        subprocess.call('cd ~/Box\ Sync/stranded\ jobs\ and\ distributional\ impacts/output/all_countries/; pdflatex hh_regressivity_'+_q+'_'+_tab+'.tex',shell=True)
        
    for f in glob.glob('output/all_countries/*.aux'): os.remove(f)
    for f in glob.glob('output/all_countries/*.log'): os.remove(f)
    for f in glob.glob('output/all_countries/_*.tex'): os.remove(f)











def cost_increase_table():
    
    df = pd.read_csv('output/all_countries/price_increase_full.csv').set_index('category')#.fillna('-')
    df = df.drop(['Coal'])
    df.columns.name = 'Price increase from carbon'
    df.index.name = 'tax, as % of current price'    
    try: df = df.drop('Median')
    except: pass
    try: df = df.drop('Median',axis=1)
    except: pass    

    for _c in df.columns:
        df = df.rename(columns={_c:iso_to_name[_c.lower()]})
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.sort_values(df.columns[0],ascending=False,na_position='last')
    df = df.fillna('-')

    df.insert(0,'Median',df.median(axis=1,skipna=False).round(1).fillna('-'))
    df = df.sort_values('Median',ascending=False)

    df.to_latex('output/all_countries/_price_increase_full.tex',column_format=col_form)

    df.loc['Other expenditures'] = df.loc[('Other retail & trade',
                                           'Fibres, textiles, & clothing',
                                           'Vehicles',
                                           'Health, education, & recreation',
                                           'Communication'),:].mean(skipna=True).round(1)
    df = df.fillna('-')
    try: df = df.drop(['Other retail & trade',
                                           'Fibres, textiles, & clothing',
                                           'Vehicles',
                                           'Health, education, & recreation',
                                           'Communication'])
    except: pass
    df.to_latex('output/all_countries/_price_increase_abridged.tex',column_format=col_form)
    
    for _tab in ['full','abridged']:

        with open('output/all_countries/_price_increase_'+_tab+'.tex', 'r') as f:
            with open('output/all_countries/price_increase_'+_tab+'.tex', 'w') as f2:

                f2.write(r'\documentclass[preview=true]{standalone}'+'\n')
                f2.write(r'\usepackage{amssymb} %maths'+'\n')
                f2.write(r'\usepackage{amsmath} %maths'+'\n')
                f2.write(r'\usepackage{booktabs}'+'\n')
                f2.write(r'\usepackage{rotating}'+'\n')
                f2.write(r'\begin{document}'+'\n')
                
                reading_is_fundamental = f.read()
                reading_is_fundamental = reading_is_fundamental.replace('Median',r'\rotatebox{90}{Median}')
                for _n in iso_to_name:
                    if _n == 'chi': continue
                    reading_is_fundamental = reading_is_fundamental.replace(iso_to_name[_n],r'\rotatebox{90}{'+iso_to_name[_n]+'}')
                f2.write(reading_is_fundamental)

                f2.write(r'\end{document}')
                f2.close()

        subprocess.call('cd ~/Box\ Sync/stranded\ jobs\ and\ distributional\ impacts/output/all_countries/; pdflatex price_increase_'+_tab+'.tex',shell=True)
        
    for f in glob.glob('output/all_countries/*.aux'): os.remove(f)
    for f in glob.glob('output/all_countries/*.log'): os.remove(f)
    for f in glob.glob('output/all_countries/_*.tex'): os.remove(f)


def force_round(z):
    try: return int(round(float(z),0))
    except: return z

def pop_frac_reimbursement_costs():
    refresh_country_comparison()
    
    df = pd.read_csv('output/reverse_engineering/all_policies.csv').set_index('Policy').sort_index()

    #df = pd.read_csv('output/reverse_engineering/all_policies.csv').set_index(['Policy','Net beneficiaries']).sort_index()
    try: df = df.drop('Median')
    except: pass
    try: df = df.drop('Median',axis=1)
    except: pass

    df.columns.name = 'Social spending to offset carbon tax'
    df.index.name = '(as % of total carbon revenue)'
    df = df.fillna('-')
    #df = df.astype('int')

    for _c in df.columns:
        if _c == 'Net beneficiaries': continue
        df = df.rename(columns={_c:iso_to_name[_c.lower()]})


    df = df[['Net beneficiaries']+sorted([_c for _c in df.columns if _c != 'Net beneficiaries'])]
    df = df.sort_index(ascending=True)

    #df = df.sort_values(df.columns[0],ascending=False,na_position='last')
    df = df.fillna('-')
    df = df.round(1)

    df.insert(1,'Median',df.median(axis=1,skipna=True).round(0).astype('int').fillna('-'))

    for _c in df.columns:
        if _c == 'Net beneficiaries' or _c =='Median': continue
        df[_c] = df[_c].apply(lambda x:force_round(x))
        #for _c in df.columns:
        #    df.loc[df[_c].values.dtype=='float',_c].dtype = 'int'

    #df = df.sort_values('Median',ascending=True)
    df.to_latex('output/reverse_engineering/_all_policies.tex',column_format='@{}l'+col_form[3:])
    
    with open('output/reverse_engineering/_all_policies.tex', 'r') as f:
        with open('output/reverse_engineering/all_policies.tex', 'w') as f2:

            f2.write(r'\documentclass[preview=true]{standalone}'+'\n')
            f2.write(r'\usepackage{amssymb} %maths'+'\n')
            f2.write(r'\usepackage{amsmath} %maths'+'\n')
            f2.write(r'\usepackage{booktabs}'+'\n')
            f2.write(r'\usepackage{rotating}'+'\n')
            f2.write(r'\begin{document}'+'\n')
                
            reading_is_fundamental = f.read()

            reading_is_fundamental = reading_is_fundamental.replace('Median',r'\rotatebox{90}{Median}').replace('(as \% of total carbon revenue) &','(as \% of total carbon revenue) & (\% of population)')
            for _n in iso_to_name:
                if _n == 'chi': continue
                reading_is_fundamental = reading_is_fundamental.replace(iso_to_name[_n],r'\rotatebox{90}{'+iso_to_name[_n]+'}')
            f2.write(reading_is_fundamental)

            f2.write(r'\end{document}')
            f2.close()

    subprocess.call('cd ~/Box\ Sync/stranded\ jobs\ and\ distributional\ impacts/output/reverse_engineering/; pdflatex all_policies.tex',shell=True)
        
    for f in glob.glob('output/reverse_engineering/*.aux'): os.remove(f)
    for f in glob.glob('output/reverse_engineering/*.log'): os.remove(f)
    for f in glob.glob('output/reverse_engineering/_*.tex'): os.remove(f)

#pop_frac_reimbursement_costs()


def pop_frac_reimbursement_costs():
    refresh_country_comparison()
    
    df = pd.read_csv('output/reverse_engineering/all_policies.csv').set_index('Policy').sort_index()

    #df = pd.read_csv('output/reverse_engineering/all_policies.csv').set_index(['Policy','Net beneficiaries']).sort_index()
    try: df = df.drop('Median')
    except: pass
    try: df = df.drop('Median',axis=1)
    except: pass

    df.columns.name = 'Social spending to offset carbon tax'
    df.index.name = '(as % of total carbon revenue)'
    df = df.fillna('-')
    #df = df.astype('int')

    for _c in df.columns:
        if _c == 'Net beneficiaries': continue
        df = df.rename(columns={_c:iso_to_name[_c.lower()]})


    df = df[['Net beneficiaries']+sorted([_c for _c in df.columns if _c != 'Net beneficiaries'])]
    df = df.sort_index(ascending=True)

    #df = df.sort_values(df.columns[0],ascending=False,na_position='last')
    df = df.fillna('-')
    df = df.round(1)

    df.insert(1,'Median',df.median(axis=1,skipna=True).round(0).astype('int').fillna('-'))

    for _c in df.columns:
        if _c == 'Net beneficiaries' or _c =='Median': continue
        df[_c] = df[_c].apply(lambda x:force_round(x))
        #for _c in df.columns:
        #    df.loc[df[_c].values.dtype=='float',_c].dtype = 'int'

    #df = df.sort_values('Median',ascending=True)
    df.to_latex('output/reverse_engineering/_all_policies.tex',column_format='@{}l'+col_form[3:])
    
    with open('output/reverse_engineering/_all_policies.tex', 'r') as f:
        with open('output/reverse_engineering/all_policies.tex', 'w') as f2:

            f2.write(r'\documentclass[preview=true]{standalone}'+'\n')
            f2.write(r'\usepackage{amssymb} %maths'+'\n')
            f2.write(r'\usepackage{amsmath} %maths'+'\n')
            f2.write(r'\usepackage{booktabs}'+'\n')
            f2.write(r'\usepackage{rotating}'+'\n')
            f2.write(r'\begin{document}'+'\n')
                
            reading_is_fundamental = f.read()

            reading_is_fundamental = reading_is_fundamental.replace('Median',r'\rotatebox{90}{Median}').replace('(as \% of total carbon revenue) &','(as \% of total carbon revenue) & (\% of population)')
            for _n in iso_to_name:
                if _n == 'chi': continue
                reading_is_fundamental = reading_is_fundamental.replace(iso_to_name[_n],r'\rotatebox{90}{'+iso_to_name[_n]+'}')
            f2.write(reading_is_fundamental)

            f2.write(r'\end{document}')
            f2.close()

    subprocess.call('cd ~/Box\ Sync/stranded\ jobs\ and\ distributional\ impacts/output/reverse_engineering/; pdflatex all_policies.tex',shell=True)
        
    for f in glob.glob('output/reverse_engineering/*.aux'): os.remove(f)
    for f in glob.glob('output/reverse_engineering/*.log'): os.remove(f)
    for f in glob.glob('output/reverse_engineering/_*.tex'): os.remove(f)


def pop_frac_reimbursement_costs():
    refresh_country_comparison()
    
    df = pd.read_csv('output/reverse_engineering/all_policies.csv').set_index('Policy').sort_index()

    #df = pd.read_csv('output/reverse_engineering/all_policies.csv').set_index(['Policy','Net beneficiaries']).sort_index()
    try: df = df.drop('Median')
    except: pass
    try: df = df.drop('Median',axis=1)
    except: pass

    df.columns.name = 'Social spending to offset carbon tax'
    df.index.name = '(as % of total carbon revenue)'
    df = df.fillna('-')
    #df = df.astype('int')

    for _c in df.columns:
        if _c == 'Net beneficiaries': continue
        df = df.rename(columns={_c:iso_to_name[_c.lower()]})


    df = df[['Net beneficiaries']+sorted([_c for _c in df.columns if _c != 'Net beneficiaries'])]
    df = df.sort_index(ascending=True)

    #df = df.sort_values(df.columns[0],ascending=False,na_position='last')
    df = df.fillna('-')
    df = df.round(1)

    df.insert(1,'Median',df.median(axis=1,skipna=True).round(0).astype('int').fillna('-'))

    for _c in df.columns:
        if _c == 'Net beneficiaries' or _c =='Median': continue
        df[_c] = df[_c].apply(lambda x:force_round(x))
        #for _c in df.columns:
        #    df.loc[df[_c].values.dtype=='float',_c].dtype = 'int'

    #df = df.sort_values('Median',ascending=True)
    df.to_latex('output/reverse_engineering/_all_policies.tex',column_format='@{}l'+col_form[3:])
    
    with open('output/reverse_engineering/_all_policies.tex', 'r') as f:
        with open('output/reverse_engineering/all_policies.tex', 'w') as f2:

            f2.write(r'\documentclass[preview=true]{standalone}'+'\n')
            f2.write(r'\usepackage{amssymb} %maths'+'\n')
            f2.write(r'\usepackage{amsmath} %maths'+'\n')
            f2.write(r'\usepackage{booktabs}'+'\n')
            f2.write(r'\usepackage{rotating}'+'\n')
            f2.write(r'\begin{document}'+'\n')
                
            reading_is_fundamental = f.read()

            reading_is_fundamental = reading_is_fundamental.replace('Median',r'\rotatebox{90}{Median}').replace('(as \% of total carbon revenue) &','(as \% of total carbon revenue) & (\% of population)')
            for _n in iso_to_name:
                if _n == 'chi': continue
                reading_is_fundamental = reading_is_fundamental.replace(iso_to_name[_n],r'\rotatebox{90}{'+iso_to_name[_n]+'}')
            f2.write(reading_is_fundamental)

            f2.write(r'\end{document}')
            f2.close()

    subprocess.call('cd ~/Box\ Sync/stranded\ jobs\ and\ distributional\ impacts/output/reverse_engineering/; pdflatex all_policies.tex',shell=True)
        
    for f in glob.glob('output/reverse_engineering/*.aux'): os.remove(f)
    for f in glob.glob('output/reverse_engineering/*.log'): os.remove(f)
    for f in glob.glob('output/reverse_engineering/_*.tex'): os.remove(f)


def float_with_commas(x):
    return '{0:,.0f}'.format(x)

def survey_summary_stats():

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_colwidth', -1)

    df = pd.read_excel('output/all_countries/survey_summary_stats_avs.xlsx',index_col=0).sort_index()
    #df = pd.read_csv('output/all_countries/survey_summary_stats_avs.csv',index_col=0).sort_index()
    df = df.rename(columns={'url':'URL','availability of microdata':'Microdata'})


    df['Individuals'] = df['Individuals'].apply(lambda x:float_with_commas(x))
    df['Households'] = df['Households'].apply(lambda x:float_with_commas(x))

    df.to_latex('output/all_countries/_survey_summary_stats.tex',column_format='@{}lrrrrrl@{}')
    
    with open('output/all_countries/_survey_summary_stats.tex', 'r') as f:
        with open('output/all_countries/survey_summary_stats.tex', 'w') as f2:

            f2.write(r'\documentclass[preview=true]{standalone}'+'\n')
            f2.write(r'\usepackage{amssymb} %maths'+'\n')
            f2.write(r'\usepackage{amsmath} %maths'+'\n')
            f2.write(r'\usepackage{booktabs}'+'\n')
            f2.write(r'\usepackage{rotating}'+'\n')
            f2.write(r'\begin{document}'+'\n')
            f2.write(r'\def\x{%'+'\n')
            f2.write(r'\footnotesize'+'\n')                
            reading_is_fundamental = f.read()
            f2.write(reading_is_fundamental)

            f2.write(r'}'+'\n')
            #f2.write(r'\x\par'+'\n')
            f2.write(r'\scalebox{.80}{\x}'+'\n')
            f2.write(r'\end{document}')
            f2.close()

    subprocess.call('cd ~/Box\ Sync/stranded\ jobs\ and\ distributional\ impacts/output/all_countries/; pdflatex survey_summary_stats.tex',shell=True)
        
    for f in glob.glob('output/all_countries/*.aux'): os.remove(f)
    for f in glob.glob('output/all_countries/*.log'): os.remove(f)
    for f in glob.glob('output/all_countries/_*.tex'): os.remove(f)


#survey_summary_stats()
q12_reimbursement_costs()
