import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from libraries.lib_survey_categories import get_dict_gtap_to_final

def gtap_to_final(hh_hhsector,hh_FD,pais,verbose=False):

    gtap_to_final_dict = get_dict_gtap_to_final()

    final_FD = pd.DataFrame(index=hh_FD.index)
    print(hh_FD.head())

    for _final_cat in gtap_to_final_dict:
        final_FD[gtap_to_final_dict[_final_cat][1]] = hh_FD[[_f for _f in gtap_to_final_dict[_final_cat][0] if _f in hh_FD.columns]].sum(axis=1).copy()

    print('\nThese must be identical')
    print('GTAP cats:',(hh_FD.sum(axis=1)*hh_hhsector['factor_expansion']).sum())
    print('Final cats:',(final_FD.sum(axis=1)*hh_hhsector['factor_expansion']).sum(),'\n')
    if pais != 'brb':
        assert(round((final_FD.sum(axis=1)*hh_hhsector['factor_expansion']).sum(),0)==round((hh_FD.sum(axis=1)*hh_hhsector['factor_expansion']).sum(),0))

    ######################
    # summed dataframe
    final_FD_tot = final_FD.sum(axis=1).to_frame(name='totex_hh')
    # we have made all the categories & sub-cats consistent, so we can sum over the L0 cats in final

    final_FD_tot['hhwgt'] = hh_hhsector['factor_expansion']
    final_FD_tot['hhsize'] = hh_hhsector['miembros_hogar']

    final_FD_tot['pcwgt'] = final_FD_tot[['hhwgt','hhsize']].prod(axis=1)
    final_FD_tot['totex_pc'] = final_FD_tot['totex_hh']/final_FD_tot['hhsize']
    
    final_sf = pd.DataFrame(index=final_FD_tot.index)
    final_sf['scale_fac'] = final_FD_tot['totex_hh']/hh_hhsector['gct']
    
    return final_FD, final_FD_tot, final_sf
