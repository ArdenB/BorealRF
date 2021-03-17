"""
Script goal, 

Work on dataset variatioaion. Experiments:
1. How far can we predict into the future 
2. Does including NaNs make a difference? 
3. Dataset normalisation 
4. Additional variables 
"""

# ==============================================================================

__title__ = "One Stage XGboost Dataset manipulations"
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
import xgboost as xgb


# ==============================================================================
def main(args):
	warn.warn("Add a method to save the model out at the end with package version numbers for all relevant packages")
	force = args.force
	fix   = args.fix
	# ========== Load the experiments ==========
	exper = experiments()
	# ========== Loop over the Versions (10 per experiment) ==========
	for version in range(10):
		# if version < 4:
		# 	continue
		# ========== Loop over the experiments ==========
		for experiment in exper:
			# ========== Create the path ==========
			setup = exper[experiment].copy()
			path = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/%d/" % experiment
			cf.pymkdir(path)
			fn_br  = path + "Exp%03d_%s_vers%02d_BranchItteration.csv" % (experiment, setup["name"], version)
			fn_res = path + "Exp%03d_%s_vers%02d_Results.csv" % (experiment, setup["name"], version)
			fn_PI  = path + "Exp%03d_%s_vers%02d_%sImportance.csv" % (experiment, setup["name"], version, setup["ImportanceMet"])
			
			if setup['predictwindow'] is None:
				fnamein  = f"./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/VI_df_AllSampleyears.csv"
				sfnamein = f"./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/SiteInfo_AllSampleyears.csv"
			else:
				fnamein  = f"./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/VI_df_{setup['predictwindow']}years.csv"
				sfnamein = f"./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/SiteInfo_{setup['predictwindow']}years.csv"
			# if experiment == 310:
			# 	breakpoint()
			# ========== load in the data ==========
			if all([os.path.isfile(fn) for fn in [fn_br, fn_res, fn_PI]]) and not force:
				print ("Experiment:", experiment, setup["name"], " version:", version, "complete")
				# ========== Fixing the broken site counts ==========
				if fix:
					Region_calculation(experiment, version, setup, path, fn_PI, fn_res)
				continue
			else:
				print ("\nExperiment:", experiment, setup["name"], " version:", version)
				T0 = pd.Timestamp.now()
			# ========== Setup the loop specific variables ==========
			branch       = 0
			final        = False
			ColNm        = None #will be replaced as i keep adding new columns
			corr_linkage = None # will be replaced after the 0 itteration
			orig_clnm    = None # Original Column names
			RequestFinal = False # a way to request final if i'm using REECV
			BackStepOD   = OrderedDict() # Container to allow stepback

			t0           = pd.Timestamp.now()
			# ////// To do, ad a way to record when a feature falls out \\\\\\
			# ========== Create a dictionary so i can store performance metrics ==========
			perf  = OrderedDict()
			# ========== Loop over the branchs ===========
			while not final:
				print("branch:", branch, pd.Timestamp.now())
				
				# ========== Add a check to see if this is the final round ==========
				# Final round uses the full test train dataset 
				if RequestFinal:
					final = True
				elif not (ColNm is None) and (len(ColNm) <= setup['StopPoint']):
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
					


				if not setup['predictwindow'] is None:
					bsestr = f"TTS_VI_df_{setup['predictwindow']}years"
				else:
					bsestr = f"TTS_VI_df_AllSampleyears" 
				# breakpoint()

				X_train, X_test, y_train, y_test, col_nms, loadstats, corr, df_site = bf.datasplit(
					experiment, version,  branch, setup, final=final,  cols_keep=ColNm, #force=True,
					vi_fn=fnamein, region_fn=sfnamein, basestr=bsestr)

				# ========== perform some zeo branch data storage ==========
				# breakpoint()
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

				# ========== Perform the Regression ==========
				time,  r2, feature_imp, ColNm  = ml_regression(
					X_train, X_test, y_train, y_test, col_nms, orig_clnm, experiment, 
					version, branch,  setup, corr_linkage,  verbose=False, final=final)


				# ========== Add the results of the different itterations to OD ==========
				perf["Branch%02d" % branch] = ({"experiment":experiment, "version":version, 
					"RFtime":time, "TimeCumulative":pd.Timestamp.now() -t0,  
					"R2":r2, "NumVar":loadstats["colcount"],  "SiteFraction":loadstats["fractrows"]
					})
				
				# ========== Print out branch performance ==========
				print("Branch %02d had %d veriables and an R2 of " % (branch, len(col_nms)), r2)

				# breakpoint()

				# ========== Print out branch performance ==========
				if not final:
					# ========== deal feature selection slow dowwn mode ==========
					if len(ColNm) <=  setup['SlowPoint']:
						# ========== Implement some fancy stopping here ==========
						if setup["AltMethod"] == "BackStep":
							# Check and see if the performance has degraded too much
							indx = len(BackStepOD)
							BackStepOD[indx] = {"R2":r2, "FI":feature_imp.copy(), "ColNm":ColNm.copy()}
							if len(BackStepOD) > 1:
								if r2 > BackStepOD[0]["R2"]:
									#  if the model is better store that, This removes the OD and puts the reslt in it place
									BackStepOD   = OrderedDict()
									BackStepOD[len(BackStepOD)] = {"R2":r2, "FI":feature_imp.copy(), "ColNm":col_nms.copy()}
								elif (BackStepOD[0]["R2"] - r2) > setup["maxR2drop"]:
									# its unacceptably worse backtrack, the threshold is 0.025

									feature_imp  = BackStepOD[indx-1]["FI"]
									ColNm        = BackStepOD[indx-1]["ColNm"]
									RequestFinal = True
								else:
									# if its acceptably worse store that
									pass
						elif setup["AltMethod"] == "RFECV":
							# +++++ Set of rules +++++
							RequestFinal = True
							print("This may get removed if i RFECV is super slow")
							breakpoint()

					# ========== Perform Variable selection and get new column names ==========
					# ColNm = Variable_selection(corr_linkage, branch, feature_imp, col_nms, orig_clnm)

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
			# for van in loadstats:
			# 	res[van] = loadstats[van]
			# df_res = pd.DataFrame(pd.Series(res), columns=["Exp%03d.%02d" % (experiment, version)])
			# df_res.to_csv(fn_res)

			# ========== Save the Permutation Importance ==========
			df_perm = pd.DataFrame(
				pd.Series(feature_imp), columns=["PermutationImportance"]).reset_index().rename(
				{"index":"Variable"}, axis=1)
			df_perm.to_csv(fn_PI)
			try:
				Region_calculation(experiment, version, setup, path, fn_PI, fn_res, fnamein, sfnamein, res=res)
			except  Exception as er:
				print(str(er))
				breakpoint()

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
			print("Total time taken:", pd.Timestamp.now()-T0)



	breakpoint()

# ==============================================================================


def ml_regression( 
	X_train, X_test, y_train, y_test, col_nms, orig_clnm, 
	experiment, version, branch, setup, corr_linkage, verbose=True, perm=True, final = False):
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
	print(f"starting {setup['model']} regression at:", t0)

	if setup["model"]  == "SKL Random Forest Regression":
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


		# ========== Do the RF regression training ==========
		regressor = RandomForestRegressor(**skl_rf_params)
		if setup["AltMethod"] == "RFECV" and final:
			#This should be the same as the 
			breakpoint()
		else:
			regressor.fit(X_train, y_train.values.ravel())


		# ========== Testing out of prediction ==========
		print("starting regression prediction at:", pd.Timestamp.now())
		y_pred = regressor.predict(X_test)

	
	elif setup["model"] == "XGBoost":

		# ========== convert the values ==========\
		regressor = xgb.XGBRegressor(objective ='reg:squarederror', tree_method='hist', colsample_bytree = 0.3, learning_rate = 0.1,
		                max_depth = setup['max_depth'], n_estimators =setup['nbranch'], 
		                num_parallel_tree=setup["ntree"])

		eval_set = [(X_test.values, y_test.values.ravel())]

		if setup["AltMethod"] == "RFECV" and final:
			breakpoint()
		else:
			regressor.fit(X_train.values, y_train.values.ravel(), early_stopping_rounds=15, verbose=True, eval_set=eval_set)
		# breakpoint()


		# ========== Testing out of prediction ==========
		print("starting regression prediction at:", pd.Timestamp.now())
		y_pred = regressor.predict(X_test.values)
	
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
	if setup["ImportanceMet"] == "Permutation":
		print("starting sklearn permutation importance calculation at:", pd.Timestamp.now())
		result = permutation_importance(regressor, X_test.values, y_test.values.ravel(), n_repeats=5) #n_jobs=cores
		impMet = result.importances_mean
	elif setup["ImportanceMet"] =="Feature":
		print("starting XGBoost Feature importance calculation at:", pd.Timestamp.now())
		impMet = regressor.feature_importances_
		# breakpoint()
	else:
		print("Not implemented yet")
		breakpoint()
	
	for fname, f_imp in zip(clnames, impMet): FI[fname] = f_imp


	# ========== Print the time taken ==========
	tDif = pd.Timestamp.now()-t0
	print(f"The time taken to perform {setup['model']} regression:", tDif)

	# =========== Save out the results if the branch is approaching the end ==========
	if final:
		# =========== save the predictions of the last branch ==========
		_predictedVSobserved(y_test, y_pred, experiment, version, branch, setup)
		return tDif, sklMet.r2_score(y_test, y_pred), FI, None

	else:
		ColNm = Variable_selection(corr_linkage, branch, FI, col_nms, orig_clnm)
		# if len(ColNm) <=  setup['SlowPoint']:
		# 	# ========== Implement some fof stopping here ==========
		# 	breakpoint()
		return tDif, sklMet.r2_score(y_test, y_pred), FI, ColNm


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

def Region_calculation(experiment, version, setup, path, fn_PI, fn_res,fnamein, sfnamein, res=None):
	"""
	This function exists so i can repair my regions, in future this should be 
	exported without calling this function
	"""
	# ========== read the permutation importance file to get the VA list ==========
	PI = pd.read_csv(fn_PI, index_col=0)
	ColNm = PI.Variable.values
	# y_names = ([
	# 	'./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/%d/Exp%d_TwoStageRF_vers%02d_OBSvsPREDICTEDClas_y_test.csv' % (experiment, experiment, version), 
	# 	'./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/%d/Exp%d_TwoStageRF_vers%02d_OBSvsPREDICTEDClas_y_train.csv' % (experiment, experiment, version)])
	
	# ========== load in the data ==========
	 # bf.datasplit(experiment, version,  0, setup, final=True, cols_keep=ColNm, 
	if not setup['predictwindow'] is None:
		bsestr = f"TTS_VI_df_{setup['predictwindow']}years"
	else:
		bsestr = f"TTS_VI_df_AllSampleyears" 
	loadstats = bf.datasplit(experiment, version,  0, setup, final=True,  cols_keep=ColNm, 
		RStage=True, sitefix=True, 	vi_fn=fnamein, region_fn=sfnamein, basestr=bsestr)

	# ========== Create a new data file ==========
	if res is None:
		# ========== Load the original results  ==========
		res = pd.read_csv(fn_res, index_col=0)
		print("Fixing the site counts for:", experiment, version)
		OD        = OrderedDict()

		OD["experiment"] = int(res.loc["experiment"].values[0])
		OD["version"]    = int(res.loc["version"].values[0])
		OD["R2"]         = float(res.loc["R2"].values[0])
		OD["TotalTime"]  = pd.Timedelta(res.loc["TotalTime"].values[0])
		OD["FBranch"]    = float(res.loc["FBranch"].values[0])
		# OD["totalrows"]  = int(res.loc["totalrows"].values[0])
		# OD["itterrows"]  = int(res.loc["itterrows"].values[0])
		# OD["fractrows"]  = float(res.loc["fractrows"].values[0])
		# OD["colcount"]   = int(res.loc["colcount"].values[0])

		for va in loadstats:
			OD[va] = loadstats[va]

		df_res = pd.DataFrame(pd.Series(OD), columns=["Exp%03d.%02d" % (experiment, version)])

	else:
		for va in loadstats:
			res[va] = loadstats[va]
		df_res = pd.DataFrame(pd.Series(res), columns=["Exp%03d.%02d" % (experiment, version)])
		# warn.warn("this has not been tested, ")

	df_res.to_csv(fn_res)

# ==============================================================================

def experiments(ncores = -1):
	""" Function contains all the infomation about what experiments i'm 
	performing """
	expr = OrderedDict()

	expr[300] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :300,
		"name"             :"OneStageXGBOOST_5ypred",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window",
		"window"           :5,
		"predictwindow"    :5,
		"Nstage"           :1, 
		"Model"            :"XGBoost", 
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"XGBoost", 
		"maxitter"         :10, 
		"DropNAN"          :0.0, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[301] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :301,
		"name"             :"OneStageXGBOOST_5ypred_10yrLS",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window",
		"window"           :10,
		"predictwindow"    :5,
		"Nstage"           :1, 
		"Model"            :"XGBoost", 
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"XGBoost", 
		"maxitter"         :10, 
		"DropNAN"          :0.0, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[302] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :302,
		"name"             :"OneStageXGBOOST_5ypred_15yrLS",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window",
		"window"           :15,
		"predictwindow"    :5,
		"Nstage"           :1, 
		"Model"            :"XGBoost", 
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"XGBoost", 
		"maxitter"         :10, 
		"DropNAN"          :0.0, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})

	expr[303] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :303,
		"name"             :"OneStageXGBOOST_10ypred_5yrLS",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window",
		"window"           :5,
		"predictwindow"    :10,
		"Nstage"           :1, 
		"Model"            :"XGBoost", 
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"XGBoost", 
		"maxitter"         :10, 
		"DropNAN"          :0.0, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[304] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :304,
		"name"             :"OneStageXGBOOST_10ypred_10yrLS",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window",
		"window"           :10,
		"predictwindow"    :10,
		"Nstage"           :1, 
		"Model"            :"XGBoost", 
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"XGBoost", 
		"maxitter"         :10, 
		"DropNAN"          :0.0, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[305] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :305,
		"name"             :"OneStageXGBOOST_10ypred_15yrLS",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window",
		"window"           :15,
		"predictwindow"    :10,
		"Nstage"           :1, 
		"Model"            :"XGBoost", 
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"XGBoost", 
		"maxitter"         :10, 
		"DropNAN"          :0.0, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[310] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :310,
		"name"             :"OneStageXGBOOST_AllGap",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window",
		"window"           :5,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"Model"            :"XGBoost", 
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"XGBoost", 
		"maxitter"         :10, 
		"DropNAN"          :0.0, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})

	expr[320] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :320,
		"name"             :"OneStageXGBOOST_5ypred_WithNA",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window and a nan fraction",
		"window"           :5,
		"predictwindow"    :5,
		"Nstage"           :1, 
		"Model"            :"XGBoost", 
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"XGBoost", 
		"maxitter"         :20, 
		"DropNAN"          :0.25, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[321] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :321,
		"name"             :"OneStageXGBOOST_5ypred_50perNA",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window and a nan fraction",
		"window"           :5,
		"predictwindow"    :5,
		"Nstage"           :1, 
		"Model"            :"XGBoost", 
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"XGBoost", 
		"maxitter"         :20, 
		"DropNAN"          :0.50, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[322] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :322,
		"name"             :"OneStageXGBOOST_5ypred_75perNA",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window and a nan fraction",
		"window"           :5,
		"predictwindow"    :5,
		"Nstage"           :1, 
		"Model"            :"XGBoost", 
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"XGBoost", 
		"maxitter"         :20, 
		"DropNAN"          :0.75, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[323] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :323,
		"name"             :"OneStageXGBOOST_5ypred_100perNA",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window 1and a nan fraction",
		"window"           :5,
		"predictwindow"    :5,
		"Nstage"           :1, 
		"Model"            :"XGBoost", 
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"XGBoost", 
		"maxitter"         :20, 
		"DropNAN"          :1.0, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[330] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :330,
		"name"             :"OneStageXGBOOST_AllGap_50perNA",
		"desc"             :"Gradient boosted regression in place of Random Forest with variable prediction window and a nan fraction",
		"window"           :5,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"Model"            :"XGBoost", 
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"XGBoost", 
		"maxitter"         :10, 
		"DropNAN"          :0.5, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[331] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :331,
		"name"             :"OneStageXGBOOST_AllGap_100perNA",
		"desc"             :"Gradient boosted regression in place of Random Forest with variable prediction window and a nan fraction",
		"window"           :5,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"Model"            :"XGBoost", 
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"XGBoost", 
		"maxitter"         :10, 
		"DropNAN"          :1.0, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[332] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :332,
		"name"             :"OneStageXGBOOST_AllGap_50perNA_Disturbance",
		"desc"             :"Gradient boosted regression in place of Random Forest with variable prediction window, a nan fraction and stand age",
		"window"           :5,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"Model"            :"XGBoost", 
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"XGBoost", 
		"maxitter"         :10, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[333] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :333,
		"name"             :"OneStageXGBOOST_AllGap_50perNA_FeatureImp",
		"desc"             :"Gradient boosted regression with variable prediction window, a nan fraction and Feature Importance",
		"window"           :5,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"Model"            :"XGBoost", 
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Feature",
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"XGBoost", 
		"maxitter"         :10, 
		"DropNAN"          :0.5, 
		"DropDist"         :True,
		"StopPoint"        :5,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	# expr[334] = ({
	# 	# +++++ The experiment name and summary +++++
	# 	"Code"             :334,
	# 	"name"             :"OneStageXGBOOST_AllGap_50perNA_Permutation_backstep",
	# 	"desc"             :"After the slow down point the model can backtrack if performance degrades too much",
	# 	"window"           :5,
	# 	"predictwindow"    :None,
	# 	"Nstage"           :1, 
	# 	"Model"            :"XGBoost", 
	# 	# +++++ The Model setup params +++++
	# 	"ntree"            :10,
	# 	"nbranch"          :2000,
	# 	"max_features"     :'auto',
	# 	"max_depth"        :5,
	# 	"min_samples_split":2,
	# 	"min_samples_leaf" :2,
	# 	"bootstrap"        :True,
	# 	# +++++ The experiment details +++++
	# 	"test_size"        :0.2, 
	# 	"SelMethod"        :"RecursiveHierarchicalPermutation",
	# 	"ImportanceMet"    :"Permutation",
	# 	"ModVar"           :"ntree, max_depth", "dataset"
	# 	"classifer"        :None, 
	# 	"cores"            :ncores,
	# 	"model"            :"XGBoost", 
	# 	"maxitter"         :10, 
	# 	"DropNAN"          :0.5, 
	# 	"DropDist"         :True,
	# 	"StopPoint"        :5,
	# 	"SlowPoint"        :150, # The point i start to slow down feature selection and allow a different method
	# "maxR2drop"        :0.025,
	# 	"AltMethod"        :"BackStep" # alternate method to use after slowdown point is reached
	# 	})
	# expr[335] = ({
	# 	# +++++ The experiment name and summary +++++
	# 	"Code"             :335,
	# 	"name"             :"OneStageXGBOOST_AllGap_50perNA_FeatureImp_backstep",
	# 	"desc"             :"After the slow down point the model can backtrack if performance degrades too much",
	# 	"window"           :5,
	# 	"predictwindow"    :None,
	# 	"Nstage"           :1, 
	# 	"Model"            :"XGBoost", 
	# 	# +++++ The Model setup params +++++
	# 	"ntree"            :10,
	# 	"nbranch"          :2000,
	# 	"max_features"     :'auto',
	# 	"max_depth"        :5,
	# 	"min_samples_split":2,
	# 	"min_samples_leaf" :2,
	# 	"bootstrap"        :True,
	# 	# +++++ The experiment details +++++
	# 	"test_size"        :0.2, 
	# 	"SelMethod"        :"RecursiveHierarchicalPermutation",
	# 	"ImportanceMet"    :"Feature",
	# 	"ModVar"           :"ntree, max_depth", "dataset"
	# 	"classifer"        :None, 
	# 	"cores"            :ncores,
	# 	"model"            :"XGBoost", 
	# 	"maxitter"         :10, 
	# 	"DropNAN"          :0.5, 
	# 	"DropDist"         :True,
	# 	"StopPoint"        :5,
	# 	"SlowPoint"        :150, # The point i start to slow down feature selection and allow a different method
	# 	"maxR2drop"        :0.025,
	# 	"AltMethod"        :"BackStep" # alternate method to use after slowdown point is reached
	# 	})
	# expr[336] = ({
	# 	# +++++ The experiment name and summary +++++
	# 	"Code"             :336,
	# 	"name"             :"OneStageXGBOOST_AllGap_50perNA_PermutationImp_RFECV",
	# 	"desc"             :"After the slow down point the model switches to RFECV to do feature selection",
	# 	"window"           :5,
	# 	"predictwindow"    :None,
	# 	"Nstage"           :1, 
	# 	"Model"            :"XGBoost", 
	# 	# +++++ The Model setup params +++++
	# 	"ntree"            :10,
	# 	"nbranch"          :2000,
	# 	"max_features"     :'auto',
	# 	"max_depth"        :5,
	# 	"min_samples_split":2,
	# 	"min_samples_leaf" :2,
	# 	"bootstrap"        :True,
	# 	# +++++ The experiment details +++++
	# 	"test_size"        :0.2, 
	# 	"SelMethod"        :"RecursiveHierarchicalPermutation",
	# 	"ImportanceMet"    :"Permutation",
	# 	"ModVar"           :"ntree, max_depth", "dataset"
	# 	"classifer"        :None, 
	# 	"cores"            :ncores,
	# 	"model"            :"XGBoost", 
	# 	"maxitter"         :10, 
	# 	"DropNAN"          :0.5, 
	# 	"DropDist"         :True,
	# 	"StopPoint"        :5,
	# 	"SlowPoint"        :150, # The point i start to slow down feature selection and allow a different method
	# 	"maxR2drop"        :0.01,
	# 	"AltMethod"        :"RFECV" # alternate method to use after slowdown point is reached
		# })
	expr[337] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :337,
		"name"             :"OneStageXGBOOST_AllGap_50perNA_FeatureImp_RFECV",
		"desc"             :"After the slow down point the model switches to RFECV to do feature selection",
		"window"           :5,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"Model"            :"XGBoost", 
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.2, 
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Feature",
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"model"            :"XGBoost", 
		"maxitter"         :10, 
		"DropNAN"          :0.5, 
		"DropDist"         :True,
		"StopPoint"        :5,
		"SlowPoint"        :150, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.01,
		"AltMethod"        :"RFECV" # alternate method to use after slowdown point is reached
		})
	return expr

# ==============================================================================


if __name__ == '__main__':
	# ========== Set the args Description ==========
	description='Calculate the One Stange Models'
	parser = argparse.ArgumentParser(description=description)
	
	# ========== Add additional arguments ==========
	parser.add_argument(
		"-x", "--fix", action="store_true",
		help="Fix the sites")
	parser.add_argument(
		"-f", "--force", action="store_true", 
		help="Force: redo existing models")
	args = parser.parse_args() 
		
	# ========== Call the main function ==========
	main(args)