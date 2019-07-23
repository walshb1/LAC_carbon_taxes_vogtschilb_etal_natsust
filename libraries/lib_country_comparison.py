import pandas as pd
import os, glob


def refresh_country_comparison():

    idir = '/Users/brian/Box Sync/stranded jobs and distributional impacts/output/latex/'
    qfile = 'redistribution_expenditures_by_sp_and_quint_*.csv'

    ofile_str = '/Users/brian/Box Sync/stranded jobs and distributional impacts/output/all_countries_sp/redistribution_expenditures_by_country.csv'
    ofile_tex = '/Users/brian/Box Sync/stranded jobs and distributional impacts/output/all_countries_sp/redistribution_expenditures_by_country.tex'
    try: ofile = pd.read_csv(ofile_str).set_index('sp')
    except: ofile = pd.DataFrame(columns={'sp':None}).set_index('sp')

    for f in glob.glob(idir+qfile):

        if 'cummax' in f or 'optimal' in f: continue
        cty = f.replace(idir,'').replace(qfile[:-5],'').replace('.csv','')
    
        df = pd.read_csv(f,index_col=0)
        df.index.name = ''

        for _i in df.index.get_values():
            _j = _i.replace('cash ','').replace('all ','')
            
            ofile.loc[_j,cty] = df.loc[_i,'Second']
            # This is correct, because lines 616-636 in tax_to_sp.py are set to calculate the cost of reimbursing Q1&2, cumulatively
            # so the level of payout isn't set s.t. Q2 is made individually whole (Q1: payout > tax & Q2: payout = tax & Q3-5: payout < tax), 
            # but instead that (and this is a slightly lower standard) The union of Q1 & Q2 is made whole ((Q1 U Q2): payout = tax & Q3-5: payout < tax)
    
    ofile['Median'] = ofile.median(axis=1)
    #
    ofile.to_csv(ofile_str)

    ofile.index.name = ''
    ofile.to_latex(ofile_tex)
    ###
    #with open('output/all_countries_sp/redistribution_expenditures_by_country.tex', 'w') as f2:
    #    f2 = write_latex_header(f2)            
    #
    #    f2 = write_latex_footer(f2); f2.close()

    #subprocess.call('cd ~/Box\ Sync/stranded\ jobs\ and\ distributional\ impacts/output/latex/; pdflatex all_countries_'+_+'_tax.tex',shell=True)    



    idir = '/Users/brian/Box Sync/stranded jobs and distributional impacts/output/sp/'
    qfile = '*_quintile_info.csv'

    ofile_str = '/Users/brian/Box Sync/stranded jobs and distributional impacts/output/all_countries_sp/enrolled_ct.csv'
    try: ofile = pd.read_csv(ofile_str).set_index('country')
    except: ofile = pd.DataFrame(columns={'country':None,
                                          'ct_enrollment_Q1':None,'ct_enrollment_Q5':None,
                                          'rel_impact_Q1_scaleup_ct_rel':None,'rel_impact_Q3_scaleup_ct_rel':None,'rel_impact_Q5_scaleup_ct_rel':None,
                                          'frac_winners_Q1_scaleup_ct_rel':None,'frac_winners_Q5_scaleup_ct_rel':None,
                                          'frac_winners_Q1_ubi':None,'frac_winners_Q5_ubi':None,
                                          'rel_impact_Q1_ubi':None,'rel_impact_Q3_ubi':None,'rel_impact_Q5_ubi':None}).set_index('country')

    for f in glob.glob(idir+qfile):

        cty = f.replace(idir,'').replace(qfile[1:],'')

        df = pd.read_csv(f)
        print(df.columns)

        #ofile.loc[cty,'ct_enrollment_Q1'] = float(df.loc[df.quintile==1,'pct_receiving_ct'])
        #ofile.loc[cty,'ct_enrollment_Q5'] = float(df.loc[df.quintile==5,'pct_receiving_ct'])    
        ##
        #ofile.loc[cty,'rel_impact_Q1_scaleup_ct_rel'] = float(df.loc[df.quintile==1,'rel_impact_scaleup_ct_rel']) 
        #ofile.loc[cty,'rel_impact_Q3_scaleup_ct_rel'] = float(df.loc[df.quintile==3,'rel_impact_scaleup_ct_rel']) 
        #ofile.loc[cty,'rel_impact_Q5_scaleup_ct_rel'] = float(df.loc[df.quintile==5,'rel_impact_scaleup_ct_rel'])
        #ofile.loc[cty,'frac_winners_Q1_scaleup_ct_rel'] = float(df.loc[df.quintile==1,'win_frac_scaleup_ct_rel']) 
        #ofile.loc[cty,'frac_winners_Q5_scaleup_ct_rel'] = float(df.loc[df.quintile==5,'win_frac_scaleup_ct_rel'])
        ##
        ##
        #ofile.loc[cty,'enrollment_Q1_scaleout25_ct'] = float(df.loc[df.quintile==1,'pct_receiving_scaleout25_ct'])
        #ofile.loc[cty,'frac_winners_Q1_scaleout25_ct'] = float(df.loc[df.quintile==1,'win_frac_scaleout25_ct']) 
        #ofile.loc[cty,'frac_winners_Q5_scaleout25_ct'] = float(df.loc[df.quintile==5,'win_frac_scaleout25_ct'])
        ##
        ofile.loc[cty,'enrollment_Q12_scaleup_ct_rel_shift25'] = float(df.loc[df.quintile<=2,'pct_receiving_scaleup_ct_rel_shift25'].mean())
        ofile.loc[cty,'frac_winners_Q12_scaleup_ct_rel_shift25'] = float(df.loc[df.quintile<=2,'win_frac_scaleup_ct_rel_shift25'].mean()) 
        #ofile.loc[cty,'enrollment_Q1_scaleup_ct_shift25'] = float(df.loc[df.quintile==1,'pct_receiving_scaleup_ct_shift25'])
        #ofile.loc[cty,'frac_winners_Q1_scaleup_ct_shift25'] = float(df.loc[df.quintile==1,'win_frac_scaleup_ct_shift25']) 
        #ofile.loc[cty,'frac_winners_Q5_scaleout25_ct'] = float(df.loc[df.quintile==5,'win_frac_scaleout25_ct'])
        ##
        #ofile.loc[cty,'frac_winners_Q1_ubi'] = float(df.loc[df.quintile==1,'win_frac_new_sp']) 
        #ofile.loc[cty,'frac_winners_Q5_ubi'] = float(df.loc[df.quintile==5,'win_frac_new_sp'])
        #ofile.loc[cty,'rel_impact_Q1_ubi'] = float(df.loc[df.quintile==1,'rel_impact_new_sp']) 
        #ofile.loc[cty,'rel_impact_Q3_ubi'] = float(df.loc[df.quintile==3,'rel_impact_new_sp']) 
        #ofile.loc[cty,'rel_impact_Q5_ubi'] = float(df.loc[df.quintile==5,'rel_impact_new_sp'])

    #
    for _c in ofile.columns:
        ofile.loc['Median',_c] = ofile[_c].median()
    #
    ofile.to_csv(ofile_str)
    return True
