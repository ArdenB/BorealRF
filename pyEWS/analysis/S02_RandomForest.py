"""
Script goal, 

To learn about the existing dataset through the use of Random Forest
	- Open the datasets
	- Perform a random forest regression and variable selection using both scipy and CUDA
	- Compare the results

Relevent existing R code
	./code/analysis/modeling/build_model/rf_class......
"""

# ==============================================================================

__title__ = "Random Forest Implementation"
__author__ = "Arden Burrell"
__version__ = "v1.0(30.04.2020)"
__email__ = "arden.burrell@gmail.com"

# ==============================================================================

# +++++ Check the paths and set ex path to Boreal folder +++++
import os
import sys
if not os.getcwd().endswith("Boreal"):
	if "Boreal" in os.getcwd():
		p1, p2, _ =  os.getcwd().partition("Boreal")
		os.chdir(p1+p2)
	else:
		raise OSError(
			"This script was called from an unknown path. CWD can not be set"
			)
sys.path.append(os.getcwd())

# ==============================================================================

# ========== Import packages ==========
import numpy as np
import pandas as pd
# import geopandas as gpd
import argparse
import datetime as dt
import warnings as warn
import xarray as xr
import bottleneck as bn
import scipy as sp
import glob
import shutil
import time
import ipdb
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
import seaborn as sns

# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf

# ========== Import packages for parellelisation ==========
# import multiprocessing as mp

# ========== Import ml packages ==========
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn import metrics as sklMet
from sklearn.utils import shuffle
from scipy.stats import spearmanr
from scipy.cluster import hierarchy

# ========== Import Cuda Package packages ==========
warn.warn("add a cuda state variable here")
# import cupy
# import cudf
# import cuml

# from cuml.dask.common import utils as dask_utils
# from dask.distributed import Client, wait
# from dask_cuda import LocalCUDACluster
# import dask_cudf

# # from cuml import RandomForestRegressor as cuRFR
# from cuml.dask.ensemble import RandomForestRegressor as cuRFR
# from cuml.preprocessing.model_selection import train_test_split as  CUDAtrain_test_split
# model_selection.train_test_split

# ==============================================================================
def main():
	# ========== These are all from Sol ===========
	region      = 'all' 
	num_quants  = 7 
	window      = 10
	ntree       = 500
	test_size   = 0.2
	tmpath      = "./pyEWS/experiments/1.FeatureSelection/tmp/"
	SelMethod   = "hierarchicalPermutation"
	VarImp      = "recursiveDrop"
	maxiter     = 40
	force       = False
	writenames  = []

	# ========== Make a folder to store the results  ==========
	folder = "./pyEWS/experiments/1.FeatureSelection/"
	cf.pymkdir(folder)

	for test in np.arange(10):
		# ========== Create a max number of branches ==========
		test_branchs = np.arange(maxiter)
		ColNm        = None #will be replaced as i keep adding new columns
		corr_linkage = None # will be replaced after the 0 itteration
		orig_clnm    = None

		# ========== Create a dictionary so i can store performance metrics ==========
		perf  = OrderedDict()

		warn.warn("Add a cuda test here")
		dftype="pandas"
		# ========== Make a file name =============
		fnout =  folder + "S02_RandomForest_Testing_%s_Exp%02d.csv" % (dftype, test)

		# ========== Check i the file already exists ===========
		if os.path.isfile(fnout) and not force:
			# The file already exists and i'm not focing the reload
			writenames.append(fnout)
			continue

		# ========== start a loop over the branches ==========
		for branch in test_branchs:
			print("starting Random forest with %s clustering level %d at:" % (SelMethod, branch), pd.Timestamp.now())
			# ========== open and import dataframes ===========
			X_train, X_test, y_train, y_test, col_nms, loadstats, corr = df_proccessing(
				window, tmpath, test_size, dftype=dftype, cols_keep=ColNm, recur=None )#"dask_cudf"

			# ========== perform the random forest ==========
			# cuda_rf_regression( X_train, X_test, y_train, y_test, col_nms, ntree, test_size)
			time,  r2, feature_imp  = skl_rf_regression(
				X_train, X_test, y_train, y_test, col_nms, ntree, test_size, verbose=False, perm=True)

			# ========== perform some zeo branch data storage ==========
			if branch == 0:
				# +++++ Calculate the ward hierarchy +++++
				corr_linkage = hierarchy.ward(corr)
				# +++++ Create a zero time deltat +++++
				cum_time     = pd.Timedelta(0)
				orig_clnm    = col_nms.copy()
			else:
				# ========== Pull out the existing total time ==========
				cum_time = perf["Branch%02d" % (branch-1)]["TimeCumulative"]


			ColNm = Variable_selection(corr_linkage, branch, feature_imp, X_test, col_nms, orig_clnm)
			print("this is where i need to store the results from a given itteration")

			perf["Branch%02d" % branch] = ({
				"testmethod":dftype, "dfloadtime":loadstats["loadtime"], 
				"RFtime":time, "TimeCumulative":cum_time + (loadstats["loadtime"]+time),  
				"R2":r2, "NumVar":loadstats["colcount"],  "SiteFraction":loadstats["fractrows"],
				"correlated":loadstats["covariate"], "meancorr":loadstats["meancorr"],
				})
		
		# ========== Pull out the features that i'm going to use on the next loop ==========
		perf_df = pd.DataFrame(perf).T
		for var in ["TimeCumulative", "RFtime", "dfloadtime"]:	
			perf_df[var] = perf_df[var].astype('timedelta64[s]') / 60
		perf_df["Iteration"] = test

		# ========== Save the df with the timing and performance ===========
		perf_df.to_csv(fnout)
		writenames.append(fnout)

		
	ipdb.set_trace()


def Variable_selection(corr_linkage, branch, feature_imp, X_train, col_nms, orig_clnm):
	"""
	Function uses the coorrelation linchage and the branch to select the best variables
	args:
		corr_linkage	
			output of hierarchy.ward(corr)
		branch:			int
			the branch of the correlation to select from
		feature_imp:	OrderedDict
			variable name and feature importance
		X_train:		df
			df to get the column names currently being used
	returns:
		ColNm:	list of column names to test on the next round of the itteration

	"""
	# ========== Performing Clustering based on the branch level ==========
	cluster_ids   = hierarchy.fcluster(corr_linkage, branch, criterion='distance')
	clusID_featID = defaultdict(list)

	# ========== Find what variable belong in each cluster ==========
	for idx, cluster_id in enumerate(cluster_ids): clusID_featID[cluster_id].append(idx)

	# ========== Find the most important variable in each cluster ==========
	sel_feat = [] #Container for the featture names

	for clus in  clusID_featID:
		try:
			# +++++ Get the IDs, the feature importance and the name +++++
			IDs = clusID_featID[clus]
			NMs = orig_clnm[IDs]

			# +++++ drop the names that are irrelevant +++++
			# This approach will look at the most recent feature performance
			FIs = []

			for fn in NMs:
				if (fn in col_nms) and (feature_imp[fn]>0):
					FIs.append(feature_imp[fn])
				else:
					FIs.append(np.NAN)
			
			try:
				sel_feat.append(NMs[bn.nanargmax(FIs)])
			except ValueError:
				pass
		except Exception as er:
			warn.warn("something went wrong here " )
			print(str(er))
			ipdb.set_trace()

	# ========== Pull out the features that i'm going to use on the next loop ==========
	ColNm = sel_feat

	# ========== readd lagged biomass ==========
	ColNm.append("lagged_biomass")

	return ColNm

# ==============================================================================
# ============================== Forest modeling ===============================
# ==============================================================================


def cuda_rf_regression( X_train, X_test, y_train, y_test, col_nms, ntree, test_size):
	"""
	This function is to test out the  speed of the random forest regressions using
	sklearn 
	args:
		X: 		numpy array
			data to be predicted
		y: 		numpy array 
			data to be used for prediction
		col_nms:	array
			name of the columns
	"""

	# This will use all GPUs on the local host by default
	cluster = LocalCUDACluster(threads_per_worker=1)
	c = Client(cluster)

	# Query the client for all connected workers
	workers = c.has_what().keys()
	n_workers = len(workers)
	n_streams = 8 # Performance optimization


	# ========== convert to a GPU dataframe ==========
	print ("Moving data to the GPU at: ", pd.Timestamp.now())
	Xgpu = cudf.DataFrame.from_pandas(Xin)
	ygpu = cudf.Series.from_pandas(yin) #.to_frame()
	# Xdask = dask_cudf.from_cudf(Xgpu, npartitions=5)
	# ydask = dask_cudf.from_cudf(ygpu, npartitions=5)

	# ipdb.set_trace()

	# ========== Split the data  ==========
	X_train, X_test, y_train, y_test = CUDAtrain_test_split(Xgpu, ygpu, test_size=test_size)
	# X_train, X_test, y_train, y_test = CUDAtrain_test_split(Xdask, ydask, test_size=test_size)
	Xdask_train = dask_cudf.from_cudf(X_train, npartitions=5)
	ydask_train = dask_cudf.from_cudf(y_train, npartitions=5)

	# Shard the data across all workers
	Xdask_train, ydask_trainf = dask_utils.persist_across_workers(c,[Xdask_train,ydask_train],workers=workers)

	# ========== Setup some skl params ==========
	# cu_rf_params = ({"n_estimators":ntree})
	cu_rf_params = {
	    "n_estimators": ntree,
	    "max_depth": 13,
	    "n_streams": n_streams}

	# ========== Start timing  ==========
	t0 = pd.Timestamp.now()
	print("starting CUDA random forest regression at:", t0)
	
	# ========== Do the RF regression training ==========
	regressor = cuRFR(**cu_rf_params)
	regressor.fit(Xdask_train, ydask_train)

	ipdb.set_trace()


def skl_rf_regression( X_train, X_test, y_train, y_test, col_nms, ntree, test_size, cores=-1, verbose=True, perm=False, insamp=False):
	"""
	This function is to test out the  speed of the random forest regressions using
	sklearn 
	args:
		Xin: 			ndarray or pd dataframe
			data to be used for prediction
		yin: 			ndarray or pd dataframe
			data to be predicted
		col_nms:	array
			name of the columns
		cores:		int
			the number of CPU cores to use, Defualt=-1 (the total number of threads)
		verbose: 	bool
			How much infomation to print, Defualt=True
		perm:		bool
			Use the permutation importance rather than feature importance
	"""


	# ========== Setup some skl params ==========
	skl_rf_params = ({
		'n_estimators': ntree,
		'n_jobs': cores })
		# 'max_depth': 13,

	# ========== Start timing  ==========
	t0 = pd.Timestamp.now()
	print("starting sklearn random forest regression at:", t0)

	# ========== Do the RF regression training ==========
	regressor = RandomForestRegressor(**skl_rf_params)
	regressor.fit(X_train, y_train)

	# ========== Testing out of prediction ==========
	print("starting sklearn random forest prediction at:", pd.Timestamp.now())
	y_pred = regressor.predict(X_test)
	
	# ========== make a list of names ==========
	clnames = X_train.columns.values
	
	# ========== print all the infomation if verbose ==========
	if verbose:
		print('r squared score:',         sklMet.r2_score(y_test, y_pred))
		print('Mean Absolute Error:',     sklMet.mean_absolute_error(y_test, y_pred))
		print('Mean Squared Error:',      sklMet.mean_squared_error(y_test, y_pred))
		print('Root Mean Squared Error:', np.sqrt(sklMet.mean_squared_error(y_test, y_pred)))
		
		# ========== print Variable importance ==========
		for var, imp in  zip(clnames, regressor.feature_importances_):
			print("Variable: %s Importance: %06f" % (var, imp))

	# ========== Convert Feature importance to a dictionary ==========
	FI = OrderedDict()
	if perm:
		# +++++ use permutation importance +++++
		print("starting sklearn permutation importance calculation at:", pd.Timestamp.now())
		if insamp:
			result = permutation_importance(regressor, X_train, y_train, n_repeats=3) #n_jobs=cores
			# result = permutation_importance(regressor, X_test, y_test, n_repeats=5) #n_jobs=cores
		else:
			result = permutation_importance(regressor, X_test, y_test, n_repeats=5) #n_jobs=cores
		
		for fname, f_imp in zip(clnames, result.importances_mean): 
			FI[fname] = f_imp
	else:
		# +++++ use standard feature importance +++++
		for fname, f_imp in zip(clnames, regressor.feature_importances_):
			FI[fname] = f_imp

	# ========== Print the time taken ==========
	tDif = pd.Timestamp.now()-t0
	print("The time taken to perform the random forest regression:", tDif)

	return tDif, sklMet.r2_score(y_test, y_pred), FI


# ==============================================================================
# ============================== Data processing ===============================
# ==============================================================================

def df_proccessing(window, tmpath, test_size, dftype="pandas", cols_keep=None, recur=None,  iteration = 0):
	"""
	This function opens and performs all preprocessing on the dataframes.
	These datasets were originally made using:
		./code/data_prep/create_VI_df.R
		./code/data_prep/vi_calcs/functions/modified......
		./code/data_prep/vi_calcs/full_df_calcs_loop......
	args:
		window:		int
			the max size of the values considered 
		tmpath:		str
			Location to save temp files
		dftype:		str
			How to open th dataset
		cols_keep:		list		
			A list of columns to subest the datasett
		recur:
			dataframe to be used in place of recursive dataset
		iteration: 	int
			the iteration number

	"""
	# ============ Set a timeer ============
	t0 = pd.Timestamp.now()

	# ============ Setup the file names ============
	# The level of correlation between datasets
	cor_fn = "./EWS_package/data/models/input_data/correlations_2019-10-09.csv"
	vi_fn  = "./EWS_package/data/models/input_data/vi_df_all_V2.csv"

	# ============ Open the variables and correlations ============
	# This may need to change when using cuda 
	print("loading the files using %s at:" % dftype, t0)
	if dftype == "pandas":
		vi_df  = pd.read_csv( vi_fn, index_col=0)
		cor_df = pd.read_csv(cor_fn, index_col=0)
	elif dftype == "cudf":
		ipdb.set_trace()
	elif dftype == "dask_cudf":
		ipdb.set_trace()
		vi_df  = dask_cudf.read_csv( vi_fn, npartitions=5).set_index('Unnamed: 0') #index_col=0, 
		cor_df = dask_cudf.read_csv(cor_fn, npartitions=5).set_index('Unnamed: 0') #index_col=0, 

	
	print("loading the files using %s took: " % dftype, pd.Timestamp.now()-t0)
	
	# ============ Filter the rows ============
	if cols_keep is None:
		# +++++ A container to hold the kept columns  +++++ 
		cols_keep = []
		clnm =  vi_df.columns.values
		for cn in clnm:
			# Test to see if the column is one of the ones i want to keep
			if cn.startswith("LANDSAT"):
				# The VI datasets, check the length of window considered
				if int(cn.split("_")[-1]) <= window:
					cols_keep.append(cn)
			elif cn in ['site']:
				pass
			else:
				cols_keep.append(cn)
	# else:
	# 	ipdb.set_trace()

	# ========== Fill in any NA rows ==========
	warn.warn("This is the simplified file, might be way faster to load in")
	data      = vi_df[cols_keep].dropna()

	# =========== drop rows with 0 variance ===========
	data.drop(data.columns[data.std() == 0], axis=1, inplace=True)

	# =========== Pull out the data used for prediction ===========
	X        = data.drop(["lagged_biomass"], axis = 1).astype("float32")
	cols_out = X.columns.values

	# =========== Pull out the data the is to be predicted ===========
	y = (data["lagged_biomass"]).astype("float32")

	# ========== Make some simple stats ===========
	def _simplestatus(vi_df, X, corr, threshold=0.5):
		statsOD = OrderedDict()
		statsOD["totalrows"] = vi_df.shape[0]
		statsOD["itterrows"] = X.shape[0]
		statsOD["fractrows"] = float(X.shape[0])/float(vi_df.shape[0])
		statsOD["colcount" ] = X.shape[1]

		# =========== work out how many things correlate ===========
		corr                 = spearmanr(X).correlation
		corr[corr == 1.]     = np.NaN
		statsOD["covariate"] = np.sum(abs(corr)>threshold)
		statsOD["meancorr"]  = bn.nanmean(abs(corr))
		
		statsOD["loadtime"]  = 0
		return statsOD

	# ========== Setup the inital clustering ==========
	# +++++ Build a spearman table +++++
	try:
		corr         = spearmanr(X).correlation
	except Exception as er:
		print(str(er))
		ipdb.set_trace()



	statsOD = _simplestatus(vi_df, X, corr, threshold=0.5)

	# ========== Split the data  ==========
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
	
	# =========== Fnction to quicky save dataframes ==========
	def _quicksave(dftype, tmpath, df, dfname, iteration):
		# ========== Quickly save the CSV files of the components
		# +++++ make a file name ++++++ 
		fnout = tmpath + "S02tmp_%s_%s_%03d.csv" % (dftype, dfname, iteration)
		# +++++ write the result ++++++ 
		df.to_csv(fnout)
		# +++++ return the fname ++++++
		return fnout

	# =========== save the split dataframe out so i can reload with dask as needed ============
	fnames = {dfname:_quicksave(dftype, tmpath, df, dfname, iteration) for df, dfname in zip(
		[X_train, X_test, y_train, y_test], ["X_train", "X_test", "y_train", "y_test"])}

	# =========== reload the data if needed ==========
	if dftype == "pandas":
		pass
	else:
		ipdb.set_trace()
	
	# return the split data and the time it takes
	statsOD["loadtime"] = pd.Timestamp.now() - t0
	# ============ Return the filtered data  ============
	return X_train, X_test, y_train, y_test, cols_out, statsOD, corr

# ==============================================================================
if __name__ == '__main__':
	main()