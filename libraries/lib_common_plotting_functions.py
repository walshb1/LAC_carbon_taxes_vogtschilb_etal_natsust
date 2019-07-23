import matplotlib.pyplot as plt

#python/aesthetics
import seaborn as sns
sns.set_style('white')
sns_pal = sns.color_palette('Set1', n_colors=8, desat=.5)
greys = sns.color_palette('Greys', n_colors=8, desat=.5)

quint_labels = ['Poorest\nquintile','Second','Third','Fourth','Wealthiest\nquintile']
quint_colors = [sns.color_palette('RdYlBu', n_colors=11)[7],
                sns.color_palette('RdBu', n_colors=11)[7],
                sns.color_palette('RdBu', n_colors=11)[8],
                sns.color_palette('RdBu', n_colors=11)[10],
                sns.color_palette('PuOr', n_colors=11)[10]
                ]

_12col_paired = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
_12col_paired_pal = sns.color_palette(_12col_paired,n_colors=12)


def title_legend_labels(ax,pais,lab_x=None,lab_y=None,lim_x=None,lim_y=None,global_fs=10.5,do_leg=True,leg_spacing=0.9):
    
    #try:plt.title(iso_to_name[pais],fontsize=10,weight='bold')
    #except:plt.title(pais,fontsize=14,weight='bold')

    try: plt.xlim(lim_x)
    except:
        try: plt.xlim(lim_x[0])
        except: pass

    try: plt.ylim(lim_y)
    except:
        try: plt.ylim(lim_y[0])
        except: pass

    plt.xlabel(lab_x,fontsize=global_fs,labelpad=8)
    plt.ylabel(lab_y,fontsize=global_fs,labelpad=8)

    if do_leg:
        _legend = ax.legend(bbox_to_anchor=(0., -0.42, 1., -.16), loc=8,
                            ncol=1, mode="expand", borderaxespad=0.75,borderpad=0.75,fontsize=global_fs,
                            fancybox=True,frameon=True,framealpha=0.9,facecolor='white',labelspacing=leg_spacing)

        for _nlbl, _lbl in enumerate(_legend.get_texts()):
            _legend.get_texts()[_nlbl].set_text(str(_lbl).replace(r'\n',' ').replace("Text(0,0,'","").replace("')","").replace('\\n',' '))

    #ax.legend(loc='best',labelspacing=0.75,ncol=1,fontsize=global_fs,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9,facecolor='white')

    return ax


def prettyf(_f):

    _u = 'INT$'

    if _f >= 1E3:
        _u = 'k INT$'
        _f /= 1E3

        if _f >= 1E3:
            _u = 'M INT$'
            _f /= 1E3

            if _f >=1E3:
                _u = 'B INT$'
                _f /= 1E3

    return str(round(_f,2))+_u

def get_order(policies,df_dec,crit,desc=True):

    ordering = []
    while len(ordering) != len(policies):

        if desc:
            __,_p = -1E20,''
            for _ in policies:
                if (_ not in ordering) and (df_dec.loc[0][crit+_] > __): 
                    __ = df_dec.loc[0][crit+_]
                    _p = _
            if _p != '': ordering.append(_p)
            else: ordering.append(None)

        else:
            __,_p = 1E20,''
            for _ in policies:
                if (_ not in ordering) and (df_dec.loc[0][crit+_] < __): 
                    __ = df_dec.loc[0][crit+_]
                    _p = _
            if _p != '': ordering.append(_p)
            else: ordering.append(None)        

    return ordering
