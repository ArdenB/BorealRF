"""
Script goal, 

To learn about the existing dataset through the use of Random Forest
	- Open the datasets the Sol used in his classification 
	- Perform a basic random forest on them
	- Bonus Goal, test out some CUDA 

Relevent existing R code
	./code/analysis/modeling/build_model/rf_class......
"""

# ==============================================================================

__title__ = "Random Forest Testing"
__author__ = "Arden Burrell"
__version__ = "v1.0(18.03.2020)"
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

# ========== Import packages for parellelisation ==========
import multiprocessing as mp

# ========== Import ml packages ==========
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics as sklMet

# from sklearn.metrics import r2_score, mean_squared_error

# ==============================================================================
def main():
	# ========== These are all from Sol ===========
	region      = 'all' 
	num_quants  = 7 
	window      = 10
	predict_var = 'biomass'
	ntree       = 500
	test_size   = 0.2

	# ========== open and import dataframes ===========
	X, y, col_nms = df_proccessing(window)

	# ========== Random forest regression ===========
	skl_rf_regression( X, y,  col_nms, ntree, test_size, cores=-1)

# ==============================================================================

def skl_rf_regression( X, y,  col_nms, ntree, test_size, cores=-1):
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
		cores:	int
			the number of CPU cores to use, Defualt is the total number of threads
	"""
	# ========== Setup some skl params ==========
	skl_rf_params = ({
		'n_estimators': ntree,
		'n_jobs': cores })
		# 'max_depth': 13,
	
	# ========== Split the data  ==========
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

	# ========== Start timing  ==========
	t0 = pd.Timestamp.now()
	print("starting sklearn random forest regression at:", t0)

	# ========== Do the RF regression training ==========
	regressor = RandomForestRegressor(**skl_rf_params)
	regressor.fit(X_train, y_train)

	# ========== Testing out of prediction ==========
	print("starting sklearn random forest prediction at:", pd.Timestamp.now())
	y_pred = regressor.predict(X_test)

	print('r squared score:', sklMet.r2_score(y_test, y_pred))
	print('Mean Absolute Error:', sklMet.mean_absolute_error(y_test, y_pred))
	print('Mean Squared Error:', sklMet.mean_squared_error(y_test, y_pred))
	print('Root Mean Squared Error:', np.sqrt(sklMet.mean_squared_error(y_test, y_pred)))

	# ========== Print the time taken ==========
	tDif = pd.Timestamp.now()-t0
	print("The time taken to perform the random forest regression:", tDif)
	for var, imp in  zip(col_nms[:-1], regressor.feature_importances_):
		print("Variable: %s Importance: %06f" % (var, imp))

	return tDif


# ==============================================================================

def df_proccessing(window):
	"""
	This function opens and performs all preprocessing on the dataframes.
	These datasets were originally made using:
		./code/data_prep/create_VI_df.R
		./code/data_prep/vi_calcs/functions/modified......
		./code/data_prep/vi_calcs/full_df_calcs_loop......
	"""
	# ============ Setup the file names ============
	# The level of correlation between datasets
	cor_fn = "./EWS_package/data/models/input_data/correlations_2019-10-09.csv"
	vi_fn  = "./EWS_package/data/models/input_data/vi_df_all_2019-10-30.csv"

	# ============ Open the variables and correlations ============
	# This may need to change when using cuda 
	vi_df  = pd.read_csv( vi_fn, index_col=0)
	cor_df = pd.read_csv(cor_fn, index_col=0)

	# ============ Filter the rows ============
	warn.warn("This is a simplified implementation")
	cols_keep = np.hstack([
		["biomass", "stem_density" ], 
		vi_df.columns[np.where([cln.endswith("%d" % 10) for cln in vi_df.columns])],
		["lagged_biomass"]
		])
	data      = vi_df[cols_keep].dropna()

	# =========== Pull out the data used for prediction ===========
	X = data.values[:, :-1]

	# =========== Pull out the data the is predicted ===========
	y = data.values[:, -1]
	# ============ Return the filtered data  ============
	return X, y, cols_keep


# ==============================================================================
if __name__ == '__main__':
	main()