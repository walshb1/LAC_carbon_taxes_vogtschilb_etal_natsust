import pandas as pd
import numpy as np

def save_to_results_file(pais,result_desc,result_values,units=None):
    
    in_file = pd.ExcelFile('output/analysis_results.xlsx')
    writer = pd.ExcelWriter('output/analysis_results.xlsx')
    for _f in in_file.sheet_names:
        _df = in_file.parse(_f).set_index('Result')
        if _f == pais:
            _df.loc[result_desc,'units'] = units
            _df.loc[result_desc,1:] = result_values
            
        _df.to_excel(writer,_f)
    if pais not in in_file.sheet_names:
        _df = pd.DataFrame(columns={'units',1,2,3,4,5},index={result_desc})
        _df.index.name = 'Result'
        _df.loc[result_desc,'units'] = units
        _df.loc[result_desc,1:] = result_values

        _df.to_excel(writer,pais)
    writer.save()

#save_to_results_file('arg','test',[1,2,3,4,5])

