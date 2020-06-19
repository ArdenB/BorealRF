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

__title__ = "One Stage Random Forest Implementation"
__author__ = "Arden Burrell"
__version__ = "v1.0(18.06.2020)"
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
# import xarray as xr
import bottleneck as bn
import scipy as sp
import glob
import shutil
import time
# import ipdb
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
import seaborn as sns

# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf
import myfunctions.benchmarkfunctions as bf

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

# ==============================================================================
def main():
	force       = False
	# ========== Load the experiments ==========
	exper = experiments()
	# ========== Loop over the Versions (10 per experiment) ==========
	for version in range(10):
		# ========== Loop over the experiments ==========
		for experiment in exper:
			# ========== Create the path ==========
			setup = exper[experiment].copy()
			path = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/%d/" % experiment
			cf.pymkdir(path)
			fn_br  = path + "Exp%03d_%s_vers%02d_BranchItteration.csv" % (experiment, setup["name"], version)
			fn_res = path + "Exp%03d_%s_vers%02d_Results.csv" % (experiment, setup["name"], version)
			fn_PI  = path + "Exp%03d_%s_vers%02d_PermutationImportance.csv" % (experiment, setup["name"], version)

			if all([os.path.isfile(fn) for fn in [fn_br, fn_res, fn_PI]]) and not force:
				print ("Experiment:", experiment, setup["name"], " version:", version, "complete")
				continue
			else:
				print ("\nExperiment:", experiment, setup["name"], " version:", version)
			# ========== Setup the loop specific variables ==========
			branch       = 0
			final        = False
			ColNm        = None #will be replaced as i keep adding new columns
			corr_linkage = None # will be replaced after the 0 itteration
			orig_clnm    = None
			t0           = pd.Timestamp.now()
			# ========== Create a dictionary so i can store performance metrics ==========
			perf  = OrderedDict()
			# ========== Loop over the branchs ===========
			while not final:
				print("branch:", branch, pd.Timestamp.now())
				
				# ========== Add a check to see if this is the final round ==========
				# Final round uses the full test train dataset 
				if not (ColNm is None) and (len(ColNm) <= 35):
					final = True
					# setup["BranchDepth"] = branch
				elif setup["maxitter"] is None:
					pass
				elif branch >= setup["maxitter"]:
					# ========== Catch to stop infinit looping ==========
					warn.warn("Branch reached max depth, setting final = True to stop ")
					final = True
					# setup["BranchDepth"] = branch
					breakpoint()


				# ========== load in the data ==========
				X_train, X_test, y_train, y_test, col_nms, loadstats, corr, df_site = bf.datasplit(
					experiment, version,  branch, setup, final=final,  cols_keep=ColNm)

				# ========== Perform the Regression ==========
				time,  r2, feature_imp  = skl_rf_regression(
					X_train, X_test, y_train, y_test, col_nms, experiment, 
					version, branch,  setup, verbose=False, final=final)

				# ========== perform some zeo branch data storage ==========
				if branch == 0:
					# +++++ Calculate the ward hierarchy +++++
					corr_linkage = hierarchy.ward(corr)
					# +++++ Create a zero time deltat +++++
					orig_clnm    = col_nms.copy()

					# ========== Add a catch to fill in the maxitter if it isn't known ==========
					if setup["maxitter"] is None:
						for brch in np.arange(50):
							# ========== Performing Clustering based on the branch level ==========
							cluster_ids   = hierarchy.fcluster(corr_linkage, brch, criterion='distance')
							if np.unique(cluster_ids).size <35:
								setup["maxitter"] = brch +1
								break
					print("Max Branch is:", setup["maxitter"])



				# ========== Add the results of the different itterations to OD ==========
				perf["Branch%02d" % branch] = ({"experiment":experiment, "version":version, 
					"RFtime":time, "TimeCumulative":pd.Timestamp.now() -t0,  
					"R2":r2, "NumVar":loadstats["colcount"],  "SiteFraction":loadstats["fractrows"]
					})
				
				# ========== Print out branch performance ==========
				print("Branch %02d had %d veriables and an R2 of " % (branch, len(col_nms)), r2)

				# ========== Print out branch performance ==========
				if not final:
					# ========== Perform Variable selection and get new column names ==========
					ColNm = Variable_selection(corr_linkage, branch, feature_imp, col_nms, orig_clnm)

					# ========== Move to next branch ==========
					branch += 1
					print("Branch %02d will test %d veriables" % (branch, len(ColNm)))

			# ========== Save the branch performance ==========
			df_perf = pd.DataFrame(perf).T
			df_perf.to_csv(fn_br)

			# ////////// SAVE THE RESULTS \\\\\\\\\\
			# ========== Save the results of the main ==========
			res = OrderedDict()
			res["experiment"]= experiment
			res["version"]   = version
			res["R2"]        = r2
			res["TotalTime"] = pd.to_timedelta(df_perf.TimeCumulative.values[-1])
			res["FBranch"]   = branch
			for van in loadstats:
				res[van] = loadstats[van]
			df_res = pd.DataFrame(pd.Series(res), columns=["Exp%03d.%02d" % (experiment, version)])
			df_res.to_csv(fn_res)

			# ========== Save the Permutation Importance ==========
			df_perm = pd.DataFrame(
				pd.Series(feature_imp), columns=["PermutationImportance"]).reset_index().rename(
				{"index":"Variable"}, axis=1)
			df_perm.to_csv(fn_PI)

			# ========== Write the metadata ==========
			meta_fn = path + "Metadata"
			if not os.path.isfile(meta_fn + ".txt"):
				maininfo = "Results in this folder is written from %s (%s):%s by %s, %s" % (__title__, __file__, 
					__version__, __author__, pd.Timestamp.now())
				gitinfo = cf.gitmetadata()
				cf.writemetadata(meta_fn, [maininfo, gitinfo])
				# ========== Write the setup info ==========
				pd.DataFrame(
					pd.Series(setup), columns=["Exp%03d" % (experiment)]).to_csv(path+"Exp%03d_setup.csv" % (experiment))



	breakpoint()

# ==============================================================================
def skl_rf_regression( 
	X_train, X_test, y_train, y_test, col_nms, 
	experiment, version, branch, setup, verbose=True, perm=True, final = False):
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
		'n_estimators'     :setup["ntree"],
		'n_jobs'           :setup["cores"],
		'max_depth'        :setup["max_depth"],
		"max_features"     :setup["max_features"],
		"max_depth"        :setup["max_depth"],
		"min_samples_split":setup["min_samples_split"],
		"min_samples_leaf" :setup["min_samples_leaf"],
		"bootstrap"        :setup["bootstrap"],
		})

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

	# +++++ use permutation importance +++++
	print("starting sklearn permutation importance calculation at:", pd.Timestamp.now())
	result = permutation_importance(regressor, X_test, y_test, n_repeats=5) #n_jobs=cores
	
	for fname, f_imp in zip(clnames, result.importances_mean): 
		FI[fname] = f_imp


	# ========== Print the time taken ==========
	tDif = pd.Timestamp.now()-t0
	print("The time taken to perform the random forest regression:", tDif)

	# =========== Save out the results if the branch is approaching the end ==========
	if final:
		# =========== save the predictions of the last branch ==========
		_predictedVSobserved(y_test, y_pred, experiment, version, branch, setup)
		return tDif, sklMet.r2_score(y_test, y_pred), FI

	else:
		return tDif, sklMet.r2_score(y_test, y_pred), FI

def _predictedVSobserved(y_test, y_pred, experiment, version, branch, setup):
	"""
	function to save out the predicted vs the observed values
	"""
	# ========== Create the path ==========
	path = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/%d/" % experiment
	cf.pymkdir(path)

	dfy  = pd.DataFrame(y_test).rename({"lagged_biomass":"Observed"}, axis=1)
	dfy["Estimated"] = y_pred
	
	fnameout = path + "Exp%03d_%s_vers%02d_OBSvsPREDICTED.csv" % (experiment, setup["name"], version)
	print(fnameout)
	dfy.to_csv(fnameout)

def Variable_selection(corr_linkage, branch, feature_imp, col_nms, orig_clnm):
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
	# ColNm.append("lagged_biomass")

	return ColNm
# ==============================================================================
def experiments(ncores = -1):
	""" Function contains all the infomation about what experiments i'm 
	performing """
	expr = OrderedDict()
	expr[100] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :100,
		"name"             :"OneStageRFRegression",
		"desc"             :"Baseline model. This is a simple one stage regression",
		"window"           :10,
		"Nstage"           :1, 
		"Model"            :"Scikit-learn RandomForestRegressor", 
		# +++++ The Model setup params +++++
		"ntree"            :500,
		"max_features"     :'auto',
		"max_depth"        :None,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ModVar"           :None, 
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"SKL Random Forest Regression", 
		"maxitter"         :10, 
		})
	
	# ==========  Different LS windows ==========
	expr[101] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :101,
		"name"             :"OneStageRFRegression_NoLS",
		"desc"             :"Baseline model but with no landsat data. This is a simple one stage regression",
		"window"           :4,
		"Nstage"           :1, 
		"Model"            :"Scikit-learn RandomForestRegressor", 
		# +++++ The Model setup params +++++
		"ntree"            :500,
		"max_features"     :'auto',
		"max_depth"        :None,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ModVar"           :"window", 
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"SKL Random Forest Regression", 
		"maxitter"         :None, 
		})
	expr[102] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :102,
		"name"             :"OneStageRFRegression_5yrLS",
		"desc"             :"Baseline model but with only 5yr landsat data. This is a simple one stage regression",
		"window"           :5,
		"Nstage"           :1, 
		"Model"            :"Scikit-learn RandomForestRegressor", 
		# +++++ The Model setup params +++++
		"ntree"            :500,
		"max_features"     :'auto',
		"max_depth"        :None,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ModVar"           :"window", 
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"SKL Random Forest Regression", 
		"maxitter"         :None, 
		})
	expr[103] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :103,
		"name"             :"OneStageRFRegression_15yrLS",
		"desc"             :"Baseline model but with only 15yr landsat data. This is a simple one stage regression",
		"window"           :15,
		"Nstage"           :1, 
		"Model"            :"Scikit-learn RandomForestRegressor", 
		# +++++ The Model setup params +++++
		"ntree"            :500,
		"max_features"     :'auto',
		"max_depth"        :None,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ModVar"           :"window", 
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"SKL Random Forest Regression", 
		"maxitter"         :None, 
		})
	expr[104] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :104,
		"name"             :"OneStageRFRegression_20yrLS",
		"desc"             :"Baseline model but with only 20yr landsat data. This is a simple one stage regression",
		"window"           :20,
		"Nstage"           :1, 
		"Model"            :"Scikit-learn RandomForestRegressor", 
		# +++++ The Model setup params +++++
		"ntree"            :500,
		"max_features"     :'auto',
		"max_depth"        :None,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ModVar"           :"window", 
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"SKL Random Forest Regression", 
		"maxitter"         :None, 
		})
	
	# ========== Mod the RF params ==========
	expr[110] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :110,
		"name"             :"OneStageRFRegression_HyperPrams",
		"desc"             :"Tweaked RF params. Changed the max_depth and the ntrees. This is a simple one stage regression",
		"window"           :10,
		"Nstage"           :1, 
		"Model"            :"Scikit-learn RandomForestRegressor", 
		# +++++ The Model setup params +++++
		"ntree"            :1900,
		"max_features"     :'auto',
		"max_depth"        :80,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ModVar"           :"ntree, max_depth", 
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"SKL Random Forest Regression", 
		"maxitter"         :10, 
		})

	""" Experiments to be implemented
	1. Different setup conditions worked out  
	"""
	return expr
# ==============================================================================

if __name__ == '__main__':
	main()