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
import cupy
import cudf
import cuml

from cuml.dask.common import utils as dask_utils
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
import dask_cudf

# from cuml import RandomForestRegressor as cuRFR
from cuml.dask.ensemble import RandomForestRegressor as cuRFR
from cuml.preprocessing.model_selection import train_test_split as  CUDAtrain_test_split
# model_selection.train_test_split

# ==============================================================================
def main():
	# ========== These are all from Sol ===========
	region      = 'all' 
	num_quants  = 7 
	window      = 10
	ntree       = 500
	test_size   = 0.2

	# ========== open and import dataframes ===========
	X, y, col_nms = df_proccessing(window, dftype="pandas")#"dask_cudf"

	ntree       = 500
	test_size   = 0.2

	cuda_rf_regression( X, y, col_nms, ntree, test_size)




def cuda_rf_regression( Xin, yin, col_nms, ntree, test_size):
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


def skl_rf_regression( Xin, yin, col_nms, ntree, test_size, cores=-1, verbose=True, perm=False, insamp=False):
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
	# ========== Split the data  ==========
	X_train, X_test, y_train, y_test = train_test_split(Xin, yin, test_size=test_size)

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
	# /// 	This is done because i may be shuffling dataframes 
	# 		so the Col_nms won't match ///
	if isinstance(Xin, np.ndarray):
		clnames = col_nms[:-1]
	else:
		clnames = Xin.columns.values
	
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

def df_proccessing(window, dftype="pandas"):
	"""
	This function opens and performs all preprocessing on the dataframes.
	These datasets were originally made using:
		./code/data_prep/create_VI_df.R
		./code/data_prep/vi_calcs/functions/modified......
		./code/data_prep/vi_calcs/full_df_calcs_loop......
	args:
		window:		int
			the max size of the values considered 
		dftype:		str
			specifies the type of dataframe 

	"""
	# ============ Setup the file names ============
	# The level of correlation between datasets
	cor_fn = "./EWS_package/data/models/input_data/correlations_2019-10-09.csv"
	vi_fn  = "./EWS_package/data/models/input_data/vi_df_all_2019-10-30.csv"

	# ============ Open the variables and correlations ============
	# This may need to change when using cuda 
	if dftype == "pandas":
		vi_df  = pd.read_csv( vi_fn, index_col=0)
		cor_df = pd.read_csv(cor_fn, index_col=0)
	elif dftype == "cudf":
		ipdb.set_trace()
	elif dftype == "dask_cudf":
		ipdb.set_trace()
		vi_df  = dask_cudf.read_csv( vi_fn, npartitions=5).set_index('Unnamed: 0') #index_col=0, 
		cor_df = dask_cudf.read_csv(cor_fn, npartitions=5).set_index('Unnamed: 0') #index_col=0, 


	# ============ Filter the rows ============

	# +++++ A container to hold the kept columns  +++++ 
	cols_keep = []
	clnm =  vi_df.columns.values
	for cn in clnm:
		# Test to see if the column is one of the ones i want to keep
		# if cn in ["biomass", "stem_density", "lagged_biomass"]:
		# 	cols_keep.append(cn)
		if cn.startswith("LANDSAT"):
			# The VI datasets, check the lengtu
			if int(cn.split("_")[-1]) <= window:
				cols_keep.append(cn)
		elif cn in ['site']:
			pass
		else:
			cols_keep.append(cn)

		# ========== Fill in any NA rows ==========
		# data      = vi_df[cols_keep].fillna(value=0)
		data      = vi_df[cols_keep].dropna()

	# =========== drop rows with 0 variance ===========
	data.drop(data.columns[data.std() == 0], axis=1, inplace=True)
	cols_out = data.columns.values

	# =========== Pull out the data used for prediction ===========
	# X = data.values[:, :-1]
	X = data.drop(["lagged_biomass"], axis = 1).astype("float32")

	# =========== Pull out the data the is predicted ===========
	# y = data.values[:, -1]
	y = (data["lagged_biomass"]).astype("float32")


	# ============ Return the filtered data  ============
	return X, y, cols_out

# ==============================================================================
if __name__ == '__main__':
	main()