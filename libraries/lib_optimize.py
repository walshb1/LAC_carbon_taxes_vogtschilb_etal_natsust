import numpy as np
import pandas as py
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from libraries.lib_country_params import iso_to_name
from libraries.lib_common_plotting_functions import title_legend_labels, greys

class class_opt_sp:

    def __init__(self, pais, sp_type, df_hh, df_dec, pdict, col_to_use=None, tshift=None):

        self.sp = sp_type
        self.col_to_use = col_to_use
        self.incl_targeting_corr = tshift

        self.pais = pais
        self.df_hh = df_hh
        self.df_dec = df_dec

        self.means, self.fracs, self.pops = {},{},{}

        self.pdict = pdict
        self.rev = pdict['c_rev']

        self.n_steps = 250

        self.optimal_sp_value = pd.DataFrame(index=df_hh.index.copy())

        self.scale_out, self.scale_outq12avg, self.scale_up, self.impact_winners, self.impact_rel, self.impact_rel_by_hh, self.impact_abs = self.optimal_sp()
        
        self.optimum_fom = max(self.impact_winners)
        self.optimum_ix = self.impact_winners.index(self.optimum_fom)
        self.optimum_out = round(self.scale_out[self.optimum_ix],3)
        self.optimum_up = round(self.scale_up[self.optimum_ix],3)
        print('Optimum: scale out',self.optimum_out,'and up',self.optimum_up,':',self.optimum_fom,'% of bottom 40% wins')

    def optimal_sp(self):
    
        ing_sp = 'ing_'+self.sp
        if self.col_to_use is not None: ing_sp = self.col_to_use

        self.df_hh = self.df_hh.reset_index().set_index(['cod_hogar','quintile'])
    
        # Want to define an objective function for the optimal combination of scale up & out
        # Definition of optimum: cheapest combination of up & out that maximizes the fraction of bottom 40% that wins
        
        # run with step =  1% increase (relative or absolute?)
        
        # 1st step: increase coverage from [0 - 100 (or until money runs out)], 
        # 2nd step: distribute the remainder of the money according to average in each decile
        
        self.df_hh = self.df_hh.sort_values('gct',ascending=True).fillna(0)
        
        self.means = (self.df_hh.loc[self.df_hh.eval(ing_sp+'!=0'),['pcwgt',ing_sp]].prod(axis=1).sum(level='quintile').fillna(0)
                 /self.df_hh.loc[self.df_hh.eval(ing_sp+'!=0'),'pcwgt'].sum(level='quintile').fillna(0)).fillna(0).to_dict()
        self.fracs = (self.df_hh.loc[self.df_hh.eval(ing_sp+'!=0'),'pcwgt'].sum(level='quintile').fillna(0)/self.df_hh['pcwgt'].sum(level='quintile').fillna(0)).to_dict()
        self.pops = (self.df_hh['pcwgt'].sum(level='quintile').fillna(0)).to_dict()
        
        # Scaling out 1% point(absolute)
        coverage_scaleout, coverage_outq12avg,coverage_scaleup, coverage_impact_winners = [], [], [], []
        coverage_impact_rel, coverage_impact_rel_by_hh, coverage_impact_abs = [], [], []

        for _out in np.linspace(0.0,0.50,self.n_steps):
                    
            self.df_hh['enrolled'] = False
            # Find total cost of scale_out step:
            # --> need to plan for when one quintile reaches 100% enrollment

            _out_cost = 0.
            for _q in [1.,2.,3.,4.,5.]:
                _out_cost += min(1.-self.fracs[_q],_out)*self.means[_q]*self.pops[_q]

            # skip everything below if it's already too expensive
            if _out_cost > self.rev: 
                print('Scaling out '+str(_out)+'% is too expensive!')
                break #continue

            # enroll the new hh
            self.df_hh['optimal_'+self.sp] = self.df_hh[ing_sp]

            for _q in [1,2,3,4,5]:

                _qslice = '(quintile=='+str(_q)+')'
                _slice = '(optimal_'+self.sp+'!=0)&'+_qslice
                _notslice = '(optimal_'+self.sp+'==0)&'+_qslice

                df_q = self.df_hh.loc[self.df_hh.eval(_notslice)].copy().sort_values('gct',ascending=True)
                _frac = self.df_hh.loc[self.df_hh.eval(_slice),'pcwgt'].sum()/self.pops[_q]

                df_q['pcwgt_cumfrac'] = df_q['pcwgt'].cumsum()/self.pops[_q]
                df_q = df_q.loc[df_q.pcwgt_cumfrac<min(1.-self.fracs[_q],_out)]

                self.df_hh.loc[df_q.index.tolist(),'optimal_'+self.sp] = self.means[_q]

            _out_cost_actual = self.df_hh[['optimal_'+self.sp,'pcwgt']].prod(axis=1).sum()-self.df_hh[['ing_'+self.sp,'pcwgt']].prod(axis=1).sum()

            print('Cost of scaling out '+self.sp+' '+str(round(100.*_out,1))+'%:',_out_cost_actual,'(',round(100.*_out_cost_actual/_out_cost,1),'% of expected)\n')
              
            # calc fraction of revenue going to each quintile
            _frac_rev = self.df_hh[['optimal_'+self.sp,'pcwgt']].prod(axis=1).sum(level='quintile')/self.df_hh[['optimal_'+self.sp,'pcwgt']].prod(axis=1).sum()

            # work with scale up
            _up_cost = self.rev - _out_cost_actual
            
            # Scaling up proportionally with whatever money is left 
            self.df_hh= self.df_hh.reset_index('cod_hogar')

            #df['di_scaleup_ct'+_t] = (dist_frac*pdict['c_rev'])/df[[wgt,'ing_ct'+_t]].prod(axis=1).sum()*df['ing_ct'+_t]

            self.df_hh['optimal_'+self.sp] += ((_frac_rev*_up_cost)*self.df_hh['optimal_'+self.sp]
                                               /(self.df_hh[['pcwgt','optimal_'+self.sp]].prod(axis=1).groupby('quintile')).transform('sum'))

            _tot_cost_actual = self.df_hh[['optimal_'+self.sp,'pcwgt']].prod(axis=1).sum()-self.df_hh[['ing_'+self.sp,'pcwgt']].prod(axis=1).sum()
        
            print('Total cost:\n',_tot_cost_actual,self.rev,'\n')
            #_frac_rev_B = self.df_hh[['optimal_'+self.sp,'pcwgt']].prod(axis=1).sum(level='quintile')/self.df_hh[['optimal_'+self.sp,'pcwgt']].prod(axis=1).sum()


            #########################
            # Measure outcomes
            # 0) Enrollment
            coverage_scaleout.append(100.*_out)
            coverage_outq12avg.append(100.*(min(self.fracs[1]+_out,1.0)+min(self.fracs[2]+_out,1.0))/2.)
            
            coverage_scaleup.append(100.*_up_cost/self.df_hh[['ing_'+self.sp,'pcwgt']].prod(axis=1).sum())

            # 1) FOM = fraction of bottom 40% who win
            fom_slice = '(quintile<=2)&((optimal_'+self.sp+'-ing_'+self.sp+')>carbon_cost)'
            myfom = self.df_hh.loc[self.df_hh.eval(fom_slice),'pcwgt'].sum()/self.df_hh['pcwgt'].sum()
            coverage_impact_winners.append(100.*myfom)
            #--
            print('Finding optimum: scale out '+str(round(100.*_out,1))+'% --> '+str(round(100.*myfom,2))+'% (of bottom 2 quintiles) benefits\n')

            # 2) FOM = average impact (rel)
            fom_slice = '(quintile<=2)'
            fom = (self.df_hh.loc[self.df_hh.eval(fom_slice)].eval('pcwgt*(optimal_'+self.sp+'-ing_'+self.sp+'-carbon_cost)').sum()
                   /self.df_hh.loc[self.df_hh.eval(fom_slice),['pcwgt','gct']].prod(axis=1).sum())
            coverage_impact_rel.append(100.*fom)

            # 2) FOM = average impact (rel, by hh)
            fom_slice = '(quintile<=2)'
            fom = self.df_hh.loc[self.df_hh.eval(fom_slice)].eval('pcwgt*(optimal_'+self.sp+'-ing_'+self.sp+'-carbon_cost)/gct').sum()/self.df_hh.loc[self.df_hh.eval(fom_slice),['pcwgt']].sum()
            coverage_impact_rel_by_hh.append(100.*fom)

            # 3) FOM = average impact (abs)
            fom_slice = '(quintile<=2)'
            fom = self.df_hh.loc[self.df_hh.eval(fom_slice)].eval('pcwgt*(optimal_'+self.sp+'-ing_'+self.sp+'-carbon_cost)').sum()/self.df_hh.loc[self.df_hh.eval(fom_slice),['pcwgt']].sum()
            coverage_impact_abs.append(fom)

            print('\n\n')
            #
            self.df_hh = self.df_hh.reset_index().set_index(['quintile','cod_hogar'])

            # This is the FOM I'm planning to use, so copy the SP payout info if it seems to be the optimum
            if 100.*myfom == max(coverage_impact_winners): self.optimal_sp_value = self.df_hh['optimal_'+self.sp].copy()

        return coverage_scaleout, coverage_outq12avg, coverage_scaleup, coverage_impact_winners, coverage_impact_rel, coverage_impact_rel_by_hh, coverage_impact_abs
    
    def plot(self,c1=None,c2=None,c3=None,c4=None):
        
        x_shift = 2.

        # Plot the fraction of winners
        plt.plot([0,100],[0,0],linewidth=1.5,color=greys[5],zorder=1)

        for _c in [c1,c2,c3,c4]:
            if _c is not None:

                _ls = '-'
                if _c.incl_targeting_corr is not None: _ls = '--'

                plt.plot(_c.scale_out,_c.impact_winners,color=_c.pdict['scaleout_'+_c.sp+'_100col'],linestyle=_ls)

                #####
                # Find/plot/annotate max
                #plt.scatter(_c.scale_out[_c.impact_winners.index(max(_c.impact_winners))],max(_c.impact_winners),color=_c.pdict['scaleout_'+_c.sp+'_100col'])
                _opty = max(_c.impact_winners)
                _optx = _c.scale_out[_c.impact_winners.index(_opty)]
                plt.plot([_optx,_optx],[_opty,_opty+0.5],color=greys[4],linewidth=1.2)
                
                _init_enroll = 100.*(_c.fracs[1]+_c.fracs[2])/2.
                plt.annotate('Increase by\n'+str(round(_optx-_init_enroll,1))+'% in all quintiles',xy=(_optx,_opty+0.75),size=5,va='bottom',ha='center')
                #####
                
                if _c.incl_targeting_corr is None:
                    _anno = _c.sp.replace('ct','Cash\ntransfer').replace('trsgob','Non-cash\ntransfer')
                    plt.annotate(_anno,xy=(_c.scale_out[0]-x_shift,_c.impact_winners[0]),weight='bold',color=greys[6],ha='right',va='center',size=5.5)
                else: 
                    _anno = 'Reduce inclusion\nerror by 10%'
                    plt.annotate(_anno,xy=(_c.scale_out[0]+x_shift,_c.impact_winners[0]),weight='bold',color=greys[5],ha='left',va='center',size=5)
        
        plt.xlim(0,100)
        lgd = title_legend_labels(plt.gca(),iso_to_name[self.pais],
                                  lab_x='Scale out (% of each quintile enrolled)',
                                  lab_y='Percent of poorest 40% that benefits',do_leg=False)
        sns.despine(bottom=True)
        plt.gcf().savefig('output/sp/'+self.pais+'_optimal_sp_winners.pdf',format='pdf', bbox_inches='tight')        
        plt.cla()

        # Plot relative impact (summed at quintile-level)
        plt.plot([0,100],[0,0],linewidth=1.5,color=greys[5],zorder=1)

        for _c in [c1,c2,c3,c4]:
            if _c is not None:

                _ls = '-'
                if _c.incl_targeting_corr is not None: _ls = '--'

                plt.plot(_c.scale_out,_c.impact_rel,color=_c.pdict['scaleout_'+_c.sp+'_100col'],linestyle=_ls)

                if _c.incl_targeting_corr is None:
                    _anno = _c.sp.replace('ct','Cash\ntransfer').replace('trsgob','Non-cash\ntransfer')
                    plt.annotate(_anno,xy=(_c.scale_out[0]-x_shift,_c.impact_rel[0]),weight='bold',color=greys[6],ha='right',va='center',size=5.5)
                else: 
                    _anno = 'Reduce inclusion\nerror by '+str(int(100*_c.incl_targeting_corr))+'%'
                    plt.annotate(_anno,xy=(_c.scale_out[0]+x_shift,_c.impact_rel[0]),weight='bold',color=greys[5],ha='left',va='center',size=5)                    

        plt.xlim(0,100)
        lgd = title_legend_labels(plt.gca(),iso_to_name[self.pais],
                                  lab_x='Scale out (% of bottom 40% enrolled)',
                                  lab_y='Net impact on bottom 40%\n(relative to quintile expenditures)',do_leg=False)
        sns.despine(bottom=True)
        plt.gcf().savefig('output/sp/'+self.pais+'_optimal_sp_rel_impact.pdf',format='pdf', bbox_inches='tight')        
        plt.cla()
        
        # Plot relative impact (calculated at household level)
        plt.plot([0,100],[0,0],linewidth=1.5,color=greys[5],zorder=1)

        for _c in [c1,c2,c3,c4]:
            if _c is not None:

                _ls = '-'
                if _c.incl_targeting_corr is not None: _ls = '--'

                plt.plot(_c.scale_out,_c.impact_rel_by_hh,color=_c.pdict['scaleout_'+_c.sp+'_100col'],linestyle=_ls)

                if _c.incl_targeting_corr is None:
                    _anno = _c.sp.replace('ct','Cash\ntransfer').replace('trsgob','Non-cash\ntransfer')
                    plt.annotate(_anno,xy=(_c.scale_out[0]-x_shift,_c.impact_rel_by_hh[0]),weight='bold',color=greys[6],ha='right',va='center',size=5.5)
                else: 
                    _anno = 'Reduce inclusion\nerror by '+str(int(100*_c.incl_targeting_corr))+'%'
                    plt.annotate(_anno,xy=(_c.scale_out[0]+x_shift,_c.impact_rel_by_hh[0]),weight='bold',color=greys[4],ha='left',va='center',size=5)     

        plt.xlim(0,100)
        lgd = title_legend_labels(plt.gca(),iso_to_name[self.pais],
                                  lab_x='Scale out (% of bottom 40% enrolled)',
                                  lab_y='Net impact on bottom 40%\n(relative to household expenditures)',do_leg=False)
        sns.despine(bottom=True)
        plt.gcf().savefig('output/sp/'+self.pais+'_optimal_sp_rel_impact_by_hh.pdf',format='pdf', bbox_inches='tight')        
        plt.cla()

        # Plot average impact, in $
        plt.plot([0,100],[0,0],linewidth=1.5,color=greys[5],zorder=1)
        
        for _c in [c1,c2,c3,c4]:
            if _c is not None:
                
                _ls = '-'
                if _c.incl_targeting_corr is not None: _ls = '--'

                plt.plot(_c.scale_out,_c.impact_abs,color=_c.pdict['scaleout_'+_c.sp+'_100col'],linestyle=_ls)

                if _c.incl_targeting_corr is None:
                    _anno = _c.sp.replace('ct','Cash\ntransfer').replace('trsgob','Non-cash\ntransfer')
                    plt.annotate(_anno,xy=(_c.scale_out[0]-x_shift,_c.impact_abs[0]),weight='bold',color=greys[6],ha='right',va='center',size=5.5)
                else: 
                    _anno = 'Reduce inclusion\nerror by '+str(int(100*_c.incl_targeting_corr))+'%'
                    plt.annotate(_anno,xy=(_c.scale_out[0]+x_shift,_c.impact_abs[0]),weight='bold',color=greys[4],ha='left',va='center',size=5)

        plt.xlim(0,100)
        lgd = title_legend_labels(plt.gca(),iso_to_name[self.pais],
                                  lab_x='Scale out (% of bottom 40% enrolled)',
                                  lab_y='Impact on bottom 40% [INT$]',do_leg=False)
        sns.despine(bottom=True)
        plt.gcf().savefig('output/sp/'+self.pais+'_optimal_sp_abs_impact.pdf',format='pdf', bbox_inches='tight')        
        plt.cla()
        plt.close('all')
