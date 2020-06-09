"""
Script goal, 

Test out the hyperperamterisation of my random forest approach

Relevent existing R code
	./code/analysis/modeling/build_model/rf_class......
"""

# ==============================================================================

__title__ = "Random Forest hyperperamterisation"
__author__ = "Arden Burrell"
__version__ = "v1.0(02.05.2020)"
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
import pickle
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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


# ========== define the main function ==========
def main():
	# ========== These are all from Sol ===========
	region      = 'all' 
	# num_quants  = 7 
	window      = 10
	ntree       = 500
	test_size   = 0.2
	tmpath      = "./pyEWS/experiments/1.FeatureSelection/tmp/"
	SelMethod   = "hierarchicalPermutation"
	VarImp      = "recursiveDrop"
	force       = False
	writenames  = []
	cores       = -1
	cores       = 12

	# ========== Make a folder to store the results  ==========
	folder = "./pyEWS/experiments/1.FeatureSelection/"
	cf.pymkdir(folder)

	# ========== Setup some skl params ==========
	skl_rf_params = ({
		'n_estimators': ntree,
		'n_jobs': cores,
		"random_state":42})
		# 'max_depth': 13,
		# These params are the basis for comparison
	force = False


	# ========== Loop over the experiments ==========
	for experiment, runs in zip([0, 1], [5, 10]):
		print(experiment)
		random_grid, gridsearch = param_storage(experiment)
		fnout = "./pyEWS/experiments/1.FeatureSelection/S04_Hyper_Experiment%02d.csv" % experiment
		if os.path.isfile(fnout) and not force:
			force = True
			continue
		else:
			params = OrderedDict()
			for exp in range(runs):
				X_train, X_test, y_train, y_test = open_data(exp)
				loopfn = tmpath +"S04_config%02d_exp%02d.pkl" % (experiment, exp)
				if os.path.isfile(loopfn):
					with open(loopfn, 'rb') as f:   
						params["exp%02d" % exp] = pickle.load(f)
				else:
					params["exp%02d" % exp] = skl_rf_regression( X_train, X_test, y_train, y_test, ntree, 
						test_size, exp, tmpath, skl_rf_params, random_grid=random_grid, gridsearch=gridsearch, cores=cores)

					f = open(loopfn,"wb")
					pickle.dump(params["exp%02d" % exp],f)
					f.close()

			dfparams = pd.DataFrame(params)
			dfparams.to_csv(fnout)
			breakpoint()
			ipdb.set_trace()

	breakpoint()
	ipdb.set_trace()


# ==============================================================================

def param_storage(experiment):

	# n_estimators = number of trees in the foreset
	# max_features = max number of features considered for splitting a node
	# max_depth = max number of levels in each decision tree
	# min_samples_split = min number of data points placed in a node before the node is split
	# min_samples_leaf = min number of data points allowed in a leaf node
	# bootstrap = method for sampling data points (with or without replacement)

	if experiment == 0:
		# Number of trees in random forest
		n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
		# Number of features to consider at every split
		max_features = ['auto', 'sqrt']
		# Maximum number of levels in tree
		max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
		max_depth.append(None)
		# Minimum number of samples required to split a node
		min_samples_split = [2, 5, 10]
		# Minimum number of samples required at each leaf node
		min_samples_leaf = [1, 2, 4]
		# Method of selecting samples for training each tree
		bootstrap = [True, False]

		gridsearch = False
	elif experiment == 1:
		# ========== This is based on looking at the results  from experiment 1 ==========
		# Number of trees in random forest
		n_estimators = [1300, 1500, 1600, 1700, 1800, 1900, 2000, 2200]
		# Number of features to consider at every split
		max_features = ['auto']
		# Maximum number of levels in tree
		max_depth = [70, 80, 80, None]
		# Minimum number of samples required to split a node
		min_samples_split = [2]
		# Minimum number of samples required at each leaf node
		min_samples_leaf = [2]
		# Method of selecting samples for training each tree
		bootstrap = [True]

		gridsearch = True

	random_grid = ({'n_estimators': n_estimators,
	               'max_features': max_features,
	               'max_depth': max_depth,
	               'min_samples_split': min_samples_split,
	               'min_samples_leaf': min_samples_leaf,
	               'bootstrap': bootstrap
	               })
	return random_grid, gridsearch



# ==============================================================================

def skl_rf_regression( X_train, X_test, y_train, y_test, ntree, 
	test_size, exp, tmpath, skl_rf_params, random_grid = None, gridsearch=False, cores=-1, verbose=True):
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

	# ========== Start timing  ==========
	t0 = pd.Timestamp.now()
	print("starting sklearn random forest regression at:", t0)

	# ========== Do the RF regression training ==========
	regressor = RandomForestRegressor(**skl_rf_params)
	regressor.fit(X_train, y_train.values.ravel())

	# ========== Testing out of prediction ==========
	print("starting sklearn random forest prediction at:", pd.Timestamp.now())
	y_pred = regressor.predict(X_test)
	
	# ========== make a list of names ==========
	clnames = X_train.columns.values
	

	if random_grid is None:
		breakpoint()
	elif not random_grid is None:
		# Use the random grid to search for best hyperparameters
		# First create the base model to tune
		rf = RandomForestRegressor()
		# Random search of parameters, using 3 fold cross validation, 
		# search across 100 different combinations, and use all available cores
		if gridsearch:
			print("using GridsearchCV")
			rf_random = GridSearchCV(
				estimator = rf, 
				param_grid = random_grid, 
				cv = 3, 
				verbose=2, #random_state=42, 
				n_jobs = cores)
		else:
			rf_random = RandomizedSearchCV(
				estimator = rf, 
				param_distributions = random_grid, 
				n_iter = 100, cv = 3, 
				verbose=2, random_state=42, 
				n_jobs = cores)
		# Fit the random search model
		rf_random.fit(X_train, y_train.values.ravel())
		best_params = rf_random.best_params_
		best_random = rf_random.best_estimator_
		y_predHyp   = best_random.predict(X_test)

		# ========== print all the infomation if verbose ==========
		if verbose:
			print('r squared score:',               sklMet.r2_score(y_test, y_pred))
			print('Hyper:r squared score:',         sklMet.r2_score(y_test, y_predHyp))
			print('Mean Absolute Error:',           sklMet.mean_absolute_error(y_test, y_pred))
			print('Hyper:Mean Absolute Error:',     sklMet.mean_absolute_error(y_test, y_predHyp))
			print('Mean Squared Error:',            sklMet.mean_squared_error(y_test, y_pred))
			print('Hyper:Mean Squared Error:',      sklMet.mean_squared_error(y_test, y_predHyp))
			print('Root Mean Squared Error:',       np.sqrt(sklMet.mean_squared_error(y_test, y_pred)))
			print('Hyper:Root Mean Squared Error:', np.sqrt(sklMet.mean_squared_error(y_test, y_predHyp)))
			
			# ========== print Variable importance ==========
			for var, imp in  zip(clnames, regressor.feature_importances_):
				print("Variable: %s Importance: %06f" % (var, imp))
		BP = best_params
		BP["R2_obs"] = sklMet.r2_score(y_test, y_pred)
		BP["R2_hyp"] = sklMet.r2_score(y_test, y_predHyp)
		print(BP)
		return BP



def open_data(exp):
	path = "./pyEWS/experiments/1.FeatureSelection/tmp/"
	X_train = pd.read_csv(path+"S02tmp_pandas_X_train_010_exp%d.csv" % exp, index_col=0)
	X_test  = pd.read_csv(path+"S02tmp_pandas_X_test_010_exp%d.csv" % exp, index_col=0)
	y_train = pd.read_csv(path+"S02tmp_pandas_y_train_010_exp%d.csv" % exp, index_col=0)
	y_test  = pd.read_csv(path+"S02tmp_pandas_y_test_010_exp%d.csv" % exp, index_col=0)
	return X_train, X_test, y_train, y_test

# ==============================================================================

if __name__ == '__main__':
	main()