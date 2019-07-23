import pandas as pd

iso_to_name = {'arg':'Argentina',
               'bol':'Bolivia',
               'bra':'Brazil',
               'brb':'Barbados',
               'chl':'Chile',
               'chi':'Chile',
               'col':'Colombia',
               'cri':'Costa Rica',
               'ecu':'Ecuador',
               'gtm':'Guatemala',
               'hnd':'Honduras',
               'mex':'Mexico',
               'nic':'Nicaragua',
               'pan':'Panama',
               'per':'Peru',
               'pry':'Paraguay',
               'slv':'El Salvador',
               'tt' :'Trinidad & Tobago',
               'ury':'Uruguay'}

iso_to_curr = {'arg':'ARS',
               'bol':'BOB',
               'bra':'BRL',
               'brb':'BBD',
               'chl':'CLP',
               'chi':'CLP',
               'col':'COP',
               'cri':'CRC',
               'ecu':'USD',
               'gtm':'GTQ',
               'hnd':'HNL',
               'mex':'MXN',
               'nic':'NIO',
               'pan':'PAB',
               'per':'PEN',
               'pry':'PYG',
               'slv':'SVC',
               'ury':'UYU'}

def get_uclip(miPais):

    _uclip = {'bra':2.5E4,'slv':8E3,'bol':4E3,'arg':2.5E4,'per':1.0E4,'cri':1.4E4,'mex':2.E4,
              'pan':2E4,'chl':4E4,'chi':4E4,'brb':3E4,'ecu':1E4,'col':2E5}
    
    try: return _uclip[miPais]
    except:return 50000

def get_FD_scale_fac(miPais):

    if miPais != 'brb': scale_fac = 1.
    
    if miPais == 'brb':
        scale_fac = 4.588E9/(108.9E6+1.449E9+2.584E9+9.047E9+4.588E9+839.7E6+3.207E9
                             +87.13E9+525.4E6+8.023E9+3.810E9+916.9E6+1.379E9+770.8E6+3.765E9)

        # Anguilla    = 108.9E6
        # Antigua     = 1.449E9
        # Aruba       = 2.584E9
        # Bahamas     = 9.047E9
        # Barbados    = 4.588E9
        # BVI         = 839.7E6
        # Cayman      = 3.207E9
        # Cuba        = 87.13E9
        # Dominica    = 525.4E6
        # Haiti       = 8.023E9
        # Montserrat  = N/A    
        # Neth. Ant.  = 3.810E9
        # St. Kitts   = 916.9E6
        # St. Lucia   = 1.379E9
        # St. Vincent = 770.8E6
        # USVI        = 3.765E9

    return scale_fac

def get_fx(miPais,ano):
    # oanda
    try: 
        _fx = pd.read_csv('FX/rates_data_1_06_17.csv',index_col=['Period'])
        return(_fx.loc[ano,'USD/'+iso_to_curr[miPais]])
    except:
        print('not finding USD/'+iso_to_curr[miPais])
        return -1.

def get_ppp(miPais,ano):
    # WB
    try:
        ppp_lib = pd.read_csv('libraries/ppp_global.csv',index_col=['LOCATION','TIME'])
        return(ppp_lib.loc[(miPais.upper(),ano),'Value'])

    except:
        
        try:
            ppp_lib = pd.read_csv('libraries/ppp_select.csv',index_col=['LOCATION','TIME'])
            return(ppp_lib.loc[(miPais.upper(),ano),'gdp_pc_ppp']/ppp_lib.loc[(miPais.upper(),ano),'gdp_pc'])
        except: return 1.

def get_2011usd(miPais,ano):
    # 
    try:
        ppp_lib = pd.read_csv('libraries/ppp_global.csv',index_col=['LOCATION','TIME'])
        return(ppp_lib.loc[(miPais.upper(),2011),'Value']/ppp_lib.loc[(miPais.upper(),ano),'Value'])
    except:
        try:
            ppp_lib = pd.read_csv('libraries/ppp_select.csv',index_col=['LOCATION','TIME'])
            return((ppp_lib.loc[(miPais.upper(),2011),'gdp_pc_ppp']/ppp_lib.loc[(miPais.upper(),2011),'gdp_pc'])/
                   (ppp_lib.loc[(miPais.upper(),ano),'gdp_pc_ppp']/ppp_lib.loc[(miPais.upper(),ano),'gdp_pc']))
        except: return -1.

def get_lcu_to_2011usd_ppp(miPais,ano):
    
    try: 
        gdp_ppp = pd.read_csv('FX/GDP_const2011USD_ppp/API_NY.GDP.MKTP.PP.KD_DS2_en_csv_v2.csv',
                              skiprows=4,index_col=['Country Code']).drop(['Country Name','Indicator Name','Indicator Code'],axis=1)
        gdp_lcu = pd.read_csv('FX/GDP_LCU/API_NY.GDP.MKTP.CN_DS2_en_csv_v2.csv',
                              skiprows=4,index_col=['Country Code']).drop(['Country Name','Indicator Name','Indicator Code'],axis=1)

        return gdp_ppp.loc[miPais.upper(),str(ano)]/gdp_lcu.loc[miPais.upper(),str(ano)]
        
    except:
        print('error in get_lcu_to_2011usd_ppp: Could not load fx files')
        return 0.

def get_lcu_to_2010intd(miPais,ano):

    try:
        gdp_usd = pd.read_csv('FX/GDP_2010USD/API_NY.GDP.MKTP.KD_DS2_en_csv_v2.csv',
                              skiprows=4,index_col=['Country Code']).drop(['Country Name','Indicator Name','Indicator Code'],axis=1)   
        
        gdp_lcu = pd.read_csv('FX/GDP_LCU/API_NY.GDP.MKTP.CN_DS2_en_csv_v2.csv',
                              skiprows=4,index_col=['Country Code']).drop(['Country Name','Indicator Name','Indicator Code'],axis=1)
        return gdp_usd.loc[miPais.upper(),str(ano)]/gdp_lcu.loc[miPais.upper(),str(ano)]

    except:
        print('error in get_lcu_to_2011intd: Could not load fx files')
        return 0.
