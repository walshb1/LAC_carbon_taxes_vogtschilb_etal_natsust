import numpy as np
from scipy.interpolate import UnivariateSpline,interp1d

def perc_with_spline(data, wt, percentiles):
	assert np.greater_equal(percentiles, 0.0).all(), 'Percentiles less than zero' 
	assert np.less_equal(percentiles, 1.0).all(), 'Percentiles greater than one' 
	data = np.asarray(data) 
	assert len(data.shape) == 1 
	if wt is None: 
		wt = np.ones(data.shape, np.float) 
	else: 
		wt = np.asarray(wt, np.float) 
		assert wt.shape == data.shape 
		assert np.greater_equal(wt, 0.0).all(), 'Not all weights are non-negative.' 
	assert len(wt.shape) == 1 
	i = np.argsort(data) 
	sd = np.take(data, i, axis=0)
	sw = np.take(wt, i, axis=0) 
	aw = np.add.accumulate(sw) 
	if not aw[-1] > 0: 
	 raise ValueError('Nonpositive weight sum' )
	w = (aw)/aw[-1] 
	# f = UnivariateSpline(w,sd,k=1)
	f = interp1d(np.append([0],w),np.append([0],sd))
	return f(percentiles)	 
	
def match_percentiles(hhdataframe,quintiles,col_label,sort_val):
    hhdataframe.loc[hhdataframe[sort_val]<=quintiles[0],col_label]=1

    for j in np.arange(1,len(quintiles)):
        hhdataframe.loc[(hhdataframe[sort_val]<=quintiles[j])&(hhdataframe[sort_val]>quintiles[j-1]),col_label]=j+1
        
    return hhdataframe
	
def match_quintiles_score(hhdataframe,quintiles):
    hhdataframe.loc[hhdataframe['score']<=quintiles[0],'quintile_score']=1
    for j in np.arange(1,len(quintiles)):
        hhdataframe.loc[(hhdataframe['score']<=quintiles[j])&(hhdataframe['score']>quintiles[j-1]),'quintile_score']=j+1
    return hhdataframe
	
	
def reshape_data(income):
	data = np.reshape(income.values,(len(income.values))) 
	return data
