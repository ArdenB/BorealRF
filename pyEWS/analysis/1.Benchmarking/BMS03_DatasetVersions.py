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
import pickle

# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf
import myfunctions.benchmarkfunctions as bf

# ========== Import packages for parellelisation ==========
# import multiprocessing as mp

# ========== Import ml packages ==========
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import QuantileTransformer
from sklearn import metrics as sklMet
from sklearn.utils import shuffle
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
import xgboost as xgb
import joblib
import optuna
from optuna.samplers import TPESampler
from optuna.integration import XGBoostPruningCallback
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

print("seaborn version : ", sns.__version__)
# print("xgb version : ", xgb.__version__)
# breakpoint()


# ==============================================================================
def main(args):
	force = args.force
	fix   = args.fix
	inheritrows = True # a way to match the rows 
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
			cf.pymkdir(path+"models/")
			fn_br  = path + "Exp%03d_%s_vers%02d_BranchItteration.csv" % (experiment, setup["name"], version)
			fn_res = path + "Exp%03d_%s_vers%02d_Results.csv" % (experiment, setup["name"], version)
			fn_PI  = path + "Exp%03d_%s_vers%02d_%sImportance.csv" % (experiment, setup["name"], version, setup["ImportanceMet"])
			fn_RFE =  path + "Exp%03d_%s_vers%02d_%sImportance_RFECVfeature.csv" % (experiment, setup["name"], version, setup["ImportanceMet"])
			fn_RCV =  path + "Exp%03d_%s_vers%02d_%sImportance_RFECVsteps.csv" % (experiment, setup["name"], version, setup["ImportanceMet"])
			
			if setup['predictwindow'] is None:
				if setup["predvar"] == "lagged_biomass":
					fnamein  = f"./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/VI_df_AllSampleyears.csv"
					sfnamein = f"./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/SiteInfo_AllSampleyears.csv"
				else:
					fnamein  = f"./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/VI_df_AllSampleyears_ObsBiomass.csv"
					sfnamein = f"./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/SiteInfo_AllSampleyears_ObsBiomass.csv"
			else:
				fnamein  = f"./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/VI_df_{setup['predictwindow']}years.csv"
				sfnamein = f"./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/SiteInfo_{setup['predictwindow']}years.csv"
			


			# ========== Allow for version skipping ==========
			if experiment < 420:
				warn.warn("Skipping this one so everything else can finish")
				continue
			
			# ========== create a base string ==========
			if not setup['predictwindow'] is None:
				basestr = f"TTS_VI_df_{setup['predictwindow']}years"
			else:
				if (setup["predvar"] == "lagged_biomass") or inheritrows :
					basestr = f"TTS_VI_df_AllSampleyears" 
				else:
					basestr = f"TTS_VI_df_AllSampleyears_{setup['predvar']}" 

				if not setup["FullTestSize"] is None:
					basestr += f"_{int(setup['FullTestSize']*100)}FWH"
					if setup["splitvar"] == ["site", "yrend"]:
						basestr += f"_siteyear{setup['splitmethod']}"
					elif setup["splitvar"] == "site":
						basestr += f"_site{setup['splitmethod']}"

			# ========== load in the data ==========
			if all([os.path.isfile(fn) for fn in [fn_br, fn_res, fn_PI]]) and not force:
				print ("Experiment:", experiment, setup["name"], " version:", version, "complete")
				# ========== Fixing the broken site counts ==========
				if fix:
					Region_calculation(basestr, experiment, version, setup, path, fn_PI, fn_res)
				continue
			else:
				print ("\nExperiment:", experiment, setup["name"], " version:", version)
				T0 = pd.Timestamp.now()
			# ========== Create a dictionary so i can store performance metrics ==========
			perf         = OrderedDict()
			BackStepOD   = OrderedDict() # Container to allow stepback

			# ========== Setup the loop specific variables ==========
			branch       = 0
			final        = False
			t0           = pd.Timestamp.now()
			inhRFECV     = False # can be used to skip RFECV
			pairdf       = None
			
			if setup["SelMethod"] is None:
				#Models with no feature selection
				ColNm        = None # will be replaced as i keep adding new columns
				RequestFinal = True # a way to request final if i'm using REECV or no feature selection
			elif setup['pariedRun'] is None:
				ColNm        = None # will be replaced as i keep adding new columns
				RequestFinal = False # a way to request final if i'm using REECV
			else:
				ColNm, RequestFinal, t0, perf, branch, inhRFECV, pairdf =  _pairFinder(exper, experiment, setup, setup['pariedRun'], version, perf, force, t0, branch)

			corr_linkage = None # will be replaced after the 0 itteration
			orig_clnm    = None # Original Column names


			# ////// To do, ad a way to record when a feature falls out \\\\\\
			# ========== Loop over the branchs ===========
			while not final:
				print("Exp:",experiment, "version:", version, "branch:", branch, pd.Timestamp.now())
				
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

						# breakpoint()

				# breakpoint()
				# X_train2, X_test2, y_train2, y_test2, col_nms2, loadstats2, corr2, df_site2 = bf.datasplit(setup["predvar"], experiment, version,  branch, setup, final=False,  cols_keep=ColNm, vi_fn=fnamein, region_fn=sfnamein, basestr=bsestr, dropvar=setup["dropvar"])
				X_train, X_test, y_train, y_test, col_nms, loadstats, corr, df_site, dbg = bf.datasplit(
					setup["predvar"], experiment, version,  branch, setup, final=final,  cols_keep=ColNm, #force=True,
					vi_fn=fnamein, region_fn=sfnamein, basestr=basestr, dropvar=setup["dropvar"])
				# if final:

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

				# ========== Perform the Regression ==========
				time,  r2, feature_imp, ColNm, score_debug  = ml_regression(
					X_train, X_test, y_train, y_test, path, col_nms, orig_clnm, experiment, 
					version, branch,  setup, corr_linkage,fn_RFE, fn_RCV, 
					dbg,  inhRFECV, pairdf, df_site,
					verbose=False, final=final)

				if (setup["AltMethod"] in ["BackStep", "RFECV", "RFECVBHYP"]) and final:
					NV = len(ColNm)
				else:
					NV = loadstats["colcount"]

				# ========== Add the results of the different itterations to OD ==========
				perf["Branch%02d" % branch] = ({"experiment":experiment, "version":version, 
					"RFtime":time, "TimeCumulative":pd.Timestamp.now() -t0,  
					"R2":r2, "NumVar":NV,  "SiteFraction":loadstats["fractrows"]
					})
				if setup["debug"]:
					perf["Branch%02d" % branch].update(score_debug)
				

				# ========== Print out branch performance ==========

				if not final:
					# ========== Print out branch performance ==========
					print("Exp %d Branch %02d had %d veriables and an R2 of " % (experiment, branch, len(col_nms)), r2)

					# ========== deal feature selection slow dowwn mode ==========
					if len(col_nms) <=  setup['SlowPoint']:
						# ========== Implement some fancy stopping here ==========
						if setup["AltMethod"] in ["BackStep", "RFECV", "RFECVBHYP"]:
							# Check and see if the performance has degraded too much
							indx = len(BackStepOD)
							BackStepOD[indx] = {"R2":r2, "FI":feature_imp.copy(), "ColNm":col_nms.copy()}
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
									# breakpoint()
								else:
									# if its acceptably worse store that
									pass

					# ========== Perform Variable selection and get new column names ==========
					# ColNm = Variable_selection(corr_linkage, branch, feature_imp, col_nms, orig_clnm)

					# ========== Move to next branch ==========
					branch += 1
					print("Exp %d Branch %02d will test %d veriables" % (experiment, branch, len(ColNm)))
				else:
					# ========== Print out branch performance ==========
					print("Exp %d Branch %02d had %d veriables and an R2 of " % (experiment, branch, len(ColNm)), r2)


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
			res["Computer"]  = os.uname()[1]
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
				Region_calculation(basestr, experiment, version, setup, path, fn_PI, fn_res, fnamein, sfnamein, res=res)
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

		# breakpoint()
	breakpoint()

# ==============================================================================
def Objective(trial, X_train, y_train, g_train, 
	random_state=42,
	n_splits=3,	n_repeats=2,
	# n_jobs=1,
	early_stopping_rounds=40,
	GPU=False, fullset=True, 
	):
	"""
	A function to be optimised using baysian search
	"""
	# XGBoost parameters
	if fullset:
		params = ({

			"verbosity": 0,  # 0 (silent) - 3 (debug)
			"objective": "reg:squarederror",
			"n_estimators": 10000,
			"max_depth": trial.suggest_int("max_depth", 3, 12),
			"num_parallel_tree":trial.suggest_int("num_parallel_tree", 2, 20),
			"learning_rate": trial.suggest_float("learning_rate", 0, 1),
			"colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.),
			"subsample": trial.suggest_float("subsample", 0, 1),
			"alpha": trial.suggest_float("alpha", 0.00, 1),
			"lambda": trial.suggest_float("lambda", 1e-8, 2),
			"gamma": trial.suggest_float("gamma", 0, 2),
			"min_child_weight": trial.suggest_int("min_child_weight", 1, 200),
			})
	else:
		params = ({

			"verbosity": 0,  # 0 (silent) - 3 (debug)
			"objective": "reg:squarederror",
			"n_estimators": 10000,
			"max_depth": trial.suggest_int("max_depth", 3, 12),
			# "num_parallel_tree":trial.suggest_int("num_parallel_tree", 2, 20),
			"learning_rate": trial.suggest_float("learning_rate", 0, 1),
			# "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.),
			"subsample": trial.suggest_float("subsample", 0, 1),
			"alpha": trial.suggest_float("alpha", 0.00, 1),
			# "lambda": trial.suggest_float("lambda", 1e-8, 2),
			# "gamma": trial.suggest_float("gamma", 0, 2),
			"min_child_weight": trial.suggest_int("min_child_weight", 1, 200),
			})
	# ========== Make GPU and CPU mods ==========
	if GPU:
		params['tree_method'] = 'gpu_hist'
		mfunc  = cuml.metrics.mean_squared_error
		y_pred = np.zeros_like(y_train)
	else:
		params['n_jobs']      = -1
		mfunc  = sklMet.mean_squared_error
		y_pred = y_train.copy() * 0

	# ========== Make the Regressor and the Kfold ==========
	regressor        = xgb.XGBRegressor(**params)
	pruning_callback = XGBoostPruningCallback(trial, "validation_0-rmse")
	gkf              = GroupKFold(n_splits=n_splits, )#, n_repeats=n_repeats,) #random_state=random_state
	
	# ========== setup my values and loop through the grouped kfold splits ==========
	for train_index, test_index in gkf.split(X_train, y_train, groups=g_train):
		X_A, X_B = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
		y_A, y_B = y_train.iloc[train_index], y_train.iloc[test_index]
		regressor.fit(
			X_A,
			y_A,
			eval_set=[(X_B, y_B)],
			eval_metric="rmse",
			verbose=0,
			callbacks=[pruning_callback],
			early_stopping_rounds=early_stopping_rounds,
		)
		try:
			if GPU:
				y_pred.iloc[test_index] = regressor.predict(X_B)
			else:
				y_pred.iloc[test_index, ] = np.expand_dims(regressor.predict(X_B), axis=1)
		except Exception as er: 
			warn.warn(str(er))
			breakpoint()
	# y_pred /= n_repeats
	# breakpoint()
	return np.sqrt(mfunc(y_train, y_pred))


def _pairFinder(exper, experiment, setup, pair, version, perf, force, t0, branch):
	# ========== check if the files exist ==========
	psetup   = exper[pair].copy()
	ppath    = f"./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/{pair}/" 
	pfn_br   = ppath + "Exp%03d_%s_vers%02d_BranchItteration.csv" % (pair, psetup["name"], version)
	pfn_res  = ppath + "Exp%03d_%s_vers%02d_Results.csv" % (pair, psetup["name"], version)
	pfn_PI   = ppath + "Exp%03d_%s_vers%02d_%sImportance.csv" % (pair, psetup["name"], version, psetup["ImportanceMet"])
	
	pfn_RFE  =  ppath + "Exp%03d_%s_vers%02d_%sImportance_RFECVfeature.csv" % (pair, psetup["name"], version, psetup["ImportanceMet"])
	pfn_RCV  =  ppath + "Exp%03d_%s_vers%02d_%sImportance_RFECVsteps.csv" % (pair, psetup["name"], version, psetup["ImportanceMet"])# a check to see if i can skip the RFECV
	inhRFECV = False
	pairdf   = None

	if all([os.path.isfile(fn) for fn in [pfn_br, pfn_res, pfn_PI]]) and not force:
		print ("Using paired run to shorten computation time")
		# ========== add metrics to performance ==========
		dfp = pd.read_csv(pfn_br)
		for exp in range(dfp.shape[0]-1):
			bx = dfp.iloc[exp]
			perf["Branch%02d" % exp] = ({"experiment":experiment, "version":version, 
				"RFtime":pd.to_timedelta(bx["RFtime"]), "TimeCumulative":pd.to_timedelta(bx["TimeCumulative"]),  
				"R2":bx["R2"], "NumVar":bx["NumVar"],  "SiteFraction":bx["SiteFraction"]})
			t0     -= pd.to_timedelta(bx["RFtime"])
			branch += 1
		
		# ========== pull the column names ==========
		dfc = pd.read_csv(pfn_PI)
		ColNm = dfc.Variable.values.tolist()
		
		# ========== Check to see if RFECV has already been done ==========
		if psetup['AltMethod'] == "RFECV":
			inhRFECV = True
			pairdf = ({
				"pfn_RFE":pd.read_csv(pfn_RFE, index_col=0),
				"pfn_RCV":pd.read_csv(pfn_RCV, index_col=0),
				})
		return ColNm, True, t0, perf, branch, inhRFECV, pairdf

	else:
		breakpoint()
		return None, False, t0, perf, branch, inhRFECV, pairdf

def ml_regression( 
	X_train, X_test, y_train, y_test, path, col_nms, orig_clnm, 
	experiment, version, branch, setup, corr_linkage, fn_RFE, 
	fn_RCV, dbg, inhRFECV, pairdf, df_site,
	verbose=True, perm=True, final = False):
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
		if setup["AltMethod"] in ["RFECV", "RFECVBHYP"] and final:
			#This should be the same as the 
			warn.warn("Not Implemented yet")
			breakpoint()
		else:
			regressor.fit(X_train, y_train.values.ravel())


		# ========== Testing out of prediction ==========
		print("starting regression prediction at:", pd.Timestamp.now())
		y_pred = regressor.predict(X_test)

	
	elif setup["model"] == "XGBoost":

		# ========== convert the values ==========\
		# ========= Try the GPU version here =========
		
		reg = xgb.XGBRegressor(objective ='reg:squarederror', 
			tree_method='hist', colsample_bytree = 0.3, 
			learning_rate = 0.1, max_depth = setup['max_depth'], 
			n_estimators =setup['nbranch'],  num_parallel_tree=setup["ntree"])

		# if branch > 0 :
		# 	breakpoint()

		if setup["AltMethod"] in ["RFECV", "RFECVBHYP"] and final:
			if not inhRFECV:
				print(f"Start RFECV at: {pd.Timestamp.now()}")
				selector = RFECV(reg, step=setup["Step"], cv=5, verbose=1)#, scoring='neg_mean_absolute_error')#, n_jobs=-1
				selector.fit(X_train.values, y_train.values.ravel())

				# Build a table about the features
				feat = OrderedDict()
				for nm, rank, infinal in zip(col_nms, selector.ranking_, selector.support_):
					feat[nm]={"rank":rank, "InFinal":infinal}
				df_feat = pd.DataFrame(feat).T
				df_feat.to_csv(fn_RFE)

				# build a grid score table
				itp = OrderedDict()
				for nx, (score, cnt) in enumerate(zip(selector.grid_scores_, np.flip(np.arange(selector.n_features_in_, 0, -setup["Step"])))):	itp[nx] = { "score":score, "N_features":cnt}
				df_itt = pd.DataFrame(itp).T
				df_itt["N_features"][df_itt.N_features <1] = 1
				df_itt.to_csv(fn_RCV)

				col_nms = col_nms[selector.support_]
				X_train =  X_train[col_nms]#selector.transform(X_train)
				X_test  =  X_test[col_nms]#selector.transform(X_test)
				
				# ========== Do the debug scoring ==========
				if setup["debug"]:
					dbg["X_test"] = dbg["X_test"][col_nms]
			else:
				# ========== Copy across the files ==========
				pairdf["pfn_RFE"].to_csv(fn_RFE)
				pairdf["pfn_RCV"].to_csv(fn_RCV)
			
			# ==========  pull the regressor out ==========
			if setup["AltMethod"] == "RFECV":
				regressor = selector.estimator_
			else:
				# ========== Create a filename ==========
				fnout = f"{path}models/Exp{experiment:02d}_ver{version:02d}_optuna.pkl"
				
				# ========== Setup the experiment ==========
				t0x      = pd.Timestamp.now()
				sampler  = TPESampler(multivariate=True)
				study    = optuna.create_study(direction="minimize", sampler=sampler)
				n_trials = 100
				# warn.warn("Set to 5 for some debugging")
				
				# ========== Pull out the gtraub ==========
				g_train = df_site.loc[X_train.index, "group"]
				# ========== Add some good basline defulats ==========
				# A study with origial defualts
				study.enqueue_trial({
					'num_parallel_tree' :10,
					'max_depth'         :5,
					"colsample_bytree"  :0.3,
					"learning_rate"     :0.3,
					"subsample"         :1, 
					"alpha"             :0, 
					"lambda"            :1,
					"min_child_weight"  :1,
					"gamma"             :0
					})
				# A study with some best guess defualts
				study.enqueue_trial({ 
					"max_depth": 7,
					"num_parallel_tree":15,
					"alpha":0.15, 
					"lambda":0.001, 
					})
				# ========== perform the optimisation ==========
				study.optimize(
					lambda trial: Objective(trial,
						X_train, y_train, g_train, n_splits=3,
						n_repeats=2, early_stopping_rounds=40,	GPU=False, 
						fullset=True
					),
					n_trials=n_trials,
					n_jobs=1,
				)
				joblib.dump(study, fnout)
				print(f"Hyperpram Optimisation took: {pd.Timestamp.now() - t0x}")
				hp = study.best_params
				XGB_dict = _XGBdict(XGB_dict=hp)

				# ========== create the XGBoost object ========== 
				eval_set  = [(X_test.values, y_test.values.ravel())]
				regressor = xgb.XGBRegressor(**XGB_dict)
				regressor.fit(X_train.values, y_train.values.ravel(), 
					early_stopping_rounds=40, verbose=True, eval_set=eval_set)
				# breakpoint()
		else:
			
			eval_set  = [(X_test.values, y_test.values.ravel())]
			regressor = reg
			regressor.fit(X_train.values, y_train.values.ravel(), 
				early_stopping_rounds=40, verbose=True, eval_set=eval_set)

		# ========== Testing out of prediction ==========
		print("starting regression prediction at:", pd.Timestamp.now())
		y_pred = regressor.predict(X_test.values)
	else:
		warn.warn("Method Not Implemented")
		breakpoint()
		raise ValueError
	# ========== make a list of names ==========
	clnames = X_train.columns.values
	r2 = sklMet.r2_score(y_test, y_pred)

	# ========== Do the debug scoring ==========
	if setup["debug"]:
		DGBy_pred = regressor.predict(dbg["X_test"].values)
		score_debug = OrderedDict()
		score_debug["FWH:R2"]   = sklMet.r2_score(dbg["y_test"], DGBy_pred)
		score_debug["FWH:MAE"]  = sklMet.mean_absolute_error(dbg["y_test"], DGBy_pred)
		score_debug["FWH:RMSE"] = np.sqrt(sklMet.mean_squared_error(dbg["y_test"], DGBy_pred))
		print('DEBUG: r squared score:',        score_debug["FWH:R2"]  )
		print('DEBUG: Mean Absolute Error:',    score_debug["FWH:MAE"] )
		print('DEBUG: Root Mean Squared Error:',score_debug["FWH:RMSE"])
	else:
		score_debug = None
	# ========== print all the infomation if verbose ==========
	if verbose or r2 < 0.6:
		print('r squared score:',         sklMet.r2_score(y_test, y_pred))
		print('Mean Absolute Error:',     sklMet.mean_absolute_error(y_test, y_pred))
		print('Root Mean Squared Error:', np.sqrt(sklMet.mean_squared_error(y_test, y_pred)))

	# ========== Convert Feature importance to a dictionary ==========
	FI = OrderedDict()

	# +++++ use permutation importance +++++
	if setup["ImportanceMet"] == "Permutation":
		try:
			# print(y_test.shape, X_test.shape, y_train.shape, X_train.shape)
			print("starting sklearn permutation importance calculation at:", pd.Timestamp.now())
			result = permutation_importance(regressor, X_test.values, y_test.values.ravel(), n_repeats=5) #n_jobs=cores
			impMet = result.importances_mean
		except Exception as er:
			warn.warn(str(er))
			breakpoint()
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
		# ========== Save out the model ==========
		if experiment >= 400:
			cf.pymkdir(path+"models/")
			try:
				if setup["model"] == "XGBoost":
					fn_mod = f"{path}models/XGBoost_model_exp{experiment}_version{version}"
					regressor.save_model(f"{fn_mod}.json")
					pickle.dump(regressor, open(f"{fn_mod}.dat", "wb"))
					if (not setup["Transformer"] is None) or  (not setup["yTransformer"] is None):
						pickle.dump(setup, open(f"{fn_mod}_setuptransfromers.dat", "wb"))
					df_pack = syspackinfo()
					df_pack.to_csv(f"{fn_mod}_packagelist.csv")
					print(f"Model saved at: {pd.Timestamp.now()}")

				else:
					warn.warn("This has not been set up. Going interactive to stop model loss")
					breakpoint()

			except Exception as er:
				warn.warn(str(er))
				warn.warn("Model save failed. going interactive to stop model loss")
				breakpoint()

		return tDif, r2, FI, col_nms, score_debug

	else:
		ColNm = Variable_selection(corr_linkage, branch, FI, col_nms, orig_clnm)
		# if len(ColNm) <=  setup['SlowPoint']:
		# 	# ========== Implement some fof stopping here ==========
		# 	breakpoint()
		return tDif, sklMet.r2_score(y_test, y_pred), FI, ColNm, score_debug


def _predictedVSobserved(y_test, y_pred, experiment, version, branch, setup):
	"""
	function to save out the predicted vs the observed values
	"""
	# ========== Create the path ==========
	path = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/%d/" % experiment
	cf.pymkdir(path)

	dfy  = pd.DataFrame(y_test).rename({setup["predvar"]:"Observed"}, axis=1)
	dfy["Estimated"] = y_pred
	
	fnameout = path + "Exp%03d_%s_vers%02d_OBSvsPREDICTED.csv" % (experiment, setup["name"], version)
	# print(fnameout)
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

	return ColNm

def _XGBdict(GPU=False, XGB_dict=None):
	if XGB_dict is None:
		XGB_dict = ({
			'objective'         : 'reg:squarederror',
			'num_parallel_tree' :10,
			'max_depth'         :5,
			"n_estimators"      :2000,
			"colsample_bytree"  :0.3,
			})
	else:
		XGB_dict['objective']    = 'reg:squarederror'
		XGB_dict["n_estimators"] = 2000


	if GPU:
		XGB_dict['tree_method'] = 'gpu_hist'
	else:
		XGB_dict['n_jobs']      = -1
	return XGB_dict

def Region_calculation(basestr, experiment, version, setup, path, fn_PI, fn_res,fnamein, sfnamein, res=None):
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
	
	# # ========== load in the data ==========
	#  # bf.datasplit(experiment, version,  0, setup, final=True, cols_keep=ColNm, 
	# if not setup['predictwindow'] is None:
	# 	bsestr = f"TTS_VI_df_{setup['predictwindow']}years"
	# else:
	# 	bsestr = f"TTS_VI_df_AllSampleyears" 

	# 	if not setup["FullTestSize"] is None:
	# 		bsestr += f"_{int(setup['FullTestSize']*100)}FWH"
	# 		if setup["splitvar"] == ["site", "yrend"]:
	# 			bsestr += "_siteyear"

	# breakpoint()
	# breakpoint()
	loadstats = bf.datasplit(setup["predvar"], experiment, version,  0, setup, 
		final=True,  cols_keep=ColNm, RStage=True, sitefix=True, 
		vi_fn=fnamein, region_fn=sfnamein, basestr=basestr)

	
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


def syspackinfo():
	import scipy
	import sklearn
	pack = OrderedDict()
	pack["Python"]  = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
	pack["pandas"]  = pd.__version__
	pack["numpy"]   = np.__version__
	pack["sklearn"] = sklearn.__version__
	pack["XGBoost"] = xgb.__version__
	pack["scipy"]   = scipy.__version__
	return pd.DataFrame({"Packages":pack})


# ==============================================================================

def experiments(ncores = -1):
	""" Function contains all the infomation about what experiments i'm 
	performing """
	expr = OrderedDict()

	expr[300] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :300,
		"predvar"          :"lagged_biomass",
		"dropvar"          :[],
		"name"             :"OneStageXGBOOST_5ypred",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window",
		"window"           :5,
		"predictwindow"    :5,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :10, 
		"DropNAN"          :0.0, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, 
		"Step"             :None,
		"FullTestSize"     :None,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[301] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :301,
		"predvar"          :"lagged_biomass",
		"dropvar"          :[],
		"name"             :"OneStageXGBOOST_5ypred_10yrLS",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window",
		"window"           :10,
		"predictwindow"    :5,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :10, 
		"DropNAN"          :0.0, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, 
		"Step"             :None,
		"FullTestSize"     :None,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[302] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :302,
		"predvar"          :"lagged_biomass",
		"dropvar"          :[],
		"name"             :"OneStageXGBOOST_5ypred_15yrLS",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window",
		"window"           :15,
		"predictwindow"    :5,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :10, 
		"DropNAN"          :0.0, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, 
		"Step"             :None,
		"FullTestSize"     :None,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[303] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :303,
		"predvar"          :"lagged_biomass",
		"dropvar"          :[],
		"name"             :"OneStageXGBOOST_10ypred_5yrLS",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window",
		"window"           :5,
		"predictwindow"    :10,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"debug"            : False,
		"maxitter"         :10, 
		"DropNAN"          :0.0, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, 
		"Step"             :None,
		"FullTestSize"     :None,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[304] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :304,
		"predvar"          :"lagged_biomass",
		"dropvar"          :[],
		"name"             :"OneStageXGBOOST_10ypred_10yrLS",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window",
		"window"           :10,
		"predictwindow"    :10,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :10, 
		"DropNAN"          :0.0, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, 
		"Step"             :None,
		"FullTestSize"     :None,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[305] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :305,
		"predvar"          :"lagged_biomass",
		"dropvar"          :[],
		"name"             :"OneStageXGBOOST_10ypred_15yrLS",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window",
		"window"           :15,
		"predictwindow"    :10,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :10, 
		"DropNAN"          :0.0, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, 
		"Step"             :None,
		"FullTestSize"     :None,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[310] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :310,
		"predvar"          :"lagged_biomass",
		"dropvar"          :[],
		"name"             :"OneStageXGBOOST_AllGap",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window",
		"window"           :5,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :10, 
		"DropNAN"          :0.0, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, 
		"Step"             :None,
		"FullTestSize"     :None,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})

	expr[320] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :320,
		"predvar"          :"lagged_biomass",
		"dropvar"          :[],
		"name"             :"OneStageXGBOOST_5ypred_WithNA",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window and a nan fraction",
		"window"           :5,
		"predictwindow"    :5,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :20, 
		"DropNAN"          :0.25, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, 
		"Step"             :None,
		"FullTestSize"     :None,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[321] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :321,
		"predvar"          :"lagged_biomass",
		"dropvar"          :[],
		"name"             :"OneStageXGBOOST_5ypred_50perNA",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window and a nan fraction",
		"window"           :5,
		"predictwindow"    :5,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :20, 
		"DropNAN"          :0.50, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, 
		"Step"             :None,
		"FullTestSize"     :None,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[322] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :322,
		"predvar"          :"lagged_biomass",
		"dropvar"          :[],
		"name"             :"OneStageXGBOOST_5ypred_75perNA",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window and a nan fraction",
		"window"           :5,
		"predictwindow"    :5,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :20, 
		"DropNAN"          :0.75, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, 
		"Step"             :None,
		"FullTestSize"     :None,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[323] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :323,
		"predvar"          :"lagged_biomass",
		"dropvar"          :[],
		"name"             :"OneStageXGBOOST_5ypred_100perNA",
		"desc"             :"Gradient boosted regression in place of Random Forest with a 5 year prediction window 1and a nan fraction",
		"window"           :5,
		"predictwindow"    :5,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :20, 
		"DropNAN"          :1.0, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, 
		"Step"             :None,
		"FullTestSize"     :None,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[330] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :330,
		"predvar"          :"lagged_biomass",
		"dropvar"          :[],
		"name"             :"OneStageXGBOOST_AllGap_50perNA",
		"desc"             :"Gradient boosted regression in place of Random Forest with variable prediction window and a nan fraction",
		"window"           :5,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :10, 
		"DropNAN"          :0.5, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, 
		"Step"             :None,
		"FullTestSize"     :None,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[331] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :331,
		"predvar"          :"lagged_biomass",
		"dropvar"          :[],
		"name"             :"OneStageXGBOOST_AllGap_100perNA",
		"desc"             :"Gradient boosted regression in place of Random Forest with variable prediction window and a nan fraction",
		"window"           :5,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :10, 
		"DropNAN"          :1.0, 
		"DropDist"         :True,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, 
		"Step"             :None,
		"FullTestSize"     :None,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[332] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :332,
		"predvar"          :"lagged_biomass",
		"dropvar"          :[],
		"name"             :"OneStageXGBOOST_AllGap_50perNA_Disturbance",
		"desc"             :"Gradient boosted regression in place of Random Forest with variable prediction window, a nan fraction and stand age",
		"window"           :5,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :10, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :35,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, 
		"Step"             :None,
		"FullTestSize"     :None,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	expr[333] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :333,
		"predvar"          :"lagged_biomass",
		"dropvar"          :[],
		"name"             :"OneStageXGBOOST_AllGap_50perNA_FeatureImp",
		"desc"             :"Gradient boosted regression with variable prediction window, a nan fraction and Feature Importance",
		"window"           :5,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :10, 
		"DropNAN"          :0.5, 
		"DropDist"         :True,
		"StopPoint"        :5,
		"SlowPoint"        :0, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, 
		"Step"             :None,
		"FullTestSize"     :None,
		"AltMethod"        :None # alternate method to use after slowdown point is reached
		})
	
	expr[334] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :334,
		"predvar"          :"lagged_biomass",
		"dropvar"          :[],
		"name"             :"OneStageXGBOOST_AllGap_50perNA_Permutation_backstep",
		"desc"             :"After the slow down point the model can backtrack if performance degrades too much",
		"window"           :5,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :10, 
		"DropNAN"          :0.5, 
		"DropDist"         :True,
		"StopPoint"        :5,
		"SlowPoint"        :150, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, 
		"Step"             :None,
		"FullTestSize"     :None,
		"AltMethod"        :"BackStep" # alternate method to use after slowdown point is reached
		})
	expr[335] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :335,
		"predvar"          :"lagged_biomass",
		"dropvar"          :[],
		"name"             :"OneStageXGBOOST_AllGap_50perNA_FeatureImp_backstep",
		"desc"             :"After the slow down point the model can backtrack if performance degrades too much",
		"window"           :5,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :10, 
		"DropNAN"          :0.5, 
		"DropDist"         :True,
		"StopPoint"        :5,
		"SlowPoint"        :150, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, 
		"Step"             :None,
		"FullTestSize"     :None,
		"AltMethod"        :"BackStep" # alternate method to use after slowdown point is reached
		})

	expr[336] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :336,
		"predvar"          :"lagged_biomass",
		"dropvar"          :[],
		"name"             :"OneStageXGBOOST_AllGap_50perNA_PermutationImp_RFECV",
		"desc"             :"After the slow down point the model switches to RFECV to do feature selection",
		"window"           :5,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :10, 
		"DropNAN"          :0.5, 
		"DropDist"         :True,
		"StopPoint"        :5,
		"SlowPoint"        :150, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :334, # identical runs except at the last stage
		"Step"             :5,
		"FullTestSize"     :None,
		"AltMethod"        :"RFECV" # alternate method to use after slowdown point is reached
		})
	expr[337] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :337,
		"predvar"          :"lagged_biomass",
		"dropvar"          :[],
		"name"             :"OneStageXGBOOST_AllGap_50perNA_FeatureImp_RFECV",
		"desc"             :"After the slow down point the model switches to RFECV to do feature selection",
		"window"           :5,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :10, 
		"DropNAN"          :0.5, 
		"DropDist"         :True,
		"StopPoint"        :5,
		"SlowPoint"        :150, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :335, # identical runs except at the last stage
		"Step"             :5,
		"FullTestSize"     :None,
		"AltMethod"        :"RFECV" # alternate method to use after slowdown point is reached
		})
	
	expr[400] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :400,
		"predvar"          :"lagged_biomass",
		"dropvar"          :[],
		"name"             :"OneStageXGBOOST_AllGap_50perNA_PermutationImp_RFECV_FINAL",
		"desc"             :"Fist attempt at a paper final model configuration",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, # identical runs except at the last stage
		"Step"             :4,
		"FullTestSize"     :None,
		"AltMethod"        :"RFECV" # alternate method to use after slowdown point is reached
		})
	expr[401] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :401,
		"predvar"          :"Obs_biomass",
		"dropvar"          :["Delta_biomass"],
		"name"             :"OneStageXGBOOST_AllGap_50perNA_PermutationImp_RFECV_FINAL_Obs_biomass",
		"desc"             :"Testing different prediction approaches with paper final model configuration",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, # identical runs except at the last stage
		"Step"             :4,
		"FullTestSize"     :None,
		"AltMethod"        :"RFECV" # alternate method to use after slowdown point is reached
		})
	expr[402] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :402,
		"predvar"          :"Delta_biomass",
		"dropvar"          :["Obs_biomass"],
		"name"             :"OneStageXGBOOST_AllGap_50perNA_PermutationImp_RFECV_FINAL_Delta_biomass",
		"desc"             :"Testing different prediction approaches with paper final model configuration",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, # identical runs except at the last stage
		"Step"             :4,
		"FullTestSize"     :None,
		"AltMethod"        :"RFECV" # alternate method to use after slowdown point is reached
		})
	expr[403] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :403,
		"predvar"          :"Obs_biomass",
		"dropvar"          :["Delta_biomass"],
		"name"             :"OneStageXGBOOST_AllGap_50perNA_PermutationImp_RFECV_FINAL_Obs_biomass",
		"desc"             :"Testing different prediction approaches with paper final model configuration",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :QuantileTransformer(output_distribution='normal', ignore_implicit_zeros=True),
		"yTransformer"     :None, #QuantileTransformer(output_distribution='normal'), 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, # identical runs except at the last stage
		"Step"             :4,
		"FullTestSize"     :None,
		"AltMethod"        :"RFECV" # alternate method to use after slowdown point is reached
		})
	expr[404] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :404,
		"predvar"          :"Obs_biomass",
		"dropvar"          :["Delta_biomass"],
		"name"             :"OneStageXGBOOST_AllGap_50perNA_PermutationImp_RFECV_FINAL_Obs_biomass",
		"desc"             :"Testing different prediction approaches with paper final model configuration",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :QuantileTransformer(output_distribution='normal', ignore_implicit_zeros=True),
		"yTransformer"     :QuantileTransformer(output_distribution='normal'), 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, # identical runs except at the last stage
		"Step"             :4,
		"FullTestSize"     :None,
		"AltMethod"        :"RFECV" # alternate method to use after slowdown point is reached
		})
	expr[405] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :405,
		"predvar"          :"Delta_biomass",
		"dropvar"          :["Obs_biomass"],
		"name"             :"OneStageXGBOOST_AllGap_50perNA_PermutationImp_RFECV_FINAL_Obs_biomass",
		"desc"             :"Testing different prediction approaches with paper final model configuration",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :QuantileTransformer(output_distribution='normal', ignore_implicit_zeros=True),
		"yTransformer"     :None, #QuantileTransformer(output_distribution='normal'), 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, # identical runs except at the last stage
		"Step"             :4,
		"FullTestSize"     :None,
		"AltMethod"        :"RFECV" # alternate method to use after slowdown point is reached
		})
	expr[406] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :406,
		"predvar"          :"Delta_biomass",
		"dropvar"          :["Obs_biomass"],
		"name"             :"OneStageXGBOOST_AllGap_50perNA_PermutationImp_RFECV_FINAL_Obs_biomass",
		"desc"             :"Testing different prediction approaches with paper final model configuration",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :QuantileTransformer(output_distribution='normal', ignore_implicit_zeros=True),
		"yTransformer"     :QuantileTransformer(output_distribution='normal'), 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, # identical runs except at the last stage
		"Step"             :4,
		"FullTestSize"     :None,
		"AltMethod"        :"RFECV" # alternate method to use after slowdown point is reached
		})
	expr[410] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :410,
		"predvar"          :"Delta_biomass",
		"dropvar"          :["Obs_biomass"],
		"name"             :"OneStageXGBOOST_AllGap_50perNA_PermutationImp_FINAL_Delta_biomass_altsplit",
		"desc"             :"Testing different prediction approaches with paper final model configuration with an additional split param",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, # identical runs except at the last stage
		"Step"             :4,
		"FullTestSize"     :0.1,
		"AltMethod"        :"BackStep", # alternate method to use after slowdown point is reached
		"FutDist"          :20, 
		"splitmethod"      :"",
		"splitvar"         :"site",
		"Hyperpram"        :False,
		})

	expr[411] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :411,
		"predvar"          :"Delta_biomass",
		"dropvar"          :["Obs_biomass"],
		"name"             :"OneStageXGBOOST_AllGap_50perNA_PermutationImp_RFECV_FINAL_Delta_biomass_altsplit",
		"desc"             :"Testing different prediction approaches with paper final model configuration with an additional split param",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :410, # identical runs except at the last stage
		"Step"             :4,
		"FullTestSize"     :0.1,
		"AltMethod"        :"RFECV", # alternate method to use after slowdown point is reached
		"FutDist"          :20
		})

	expr[412] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :412,
		"predvar"          :"Delta_biomass",
		"dropvar"          :["Obs_biomass"],
		"name"             :"OneStageXGBOOST_AllGap_50perNA_PermutationImp_FINAL_Delta_biomass_altsplit_NOFULLTEST",
		"desc"             :"Testing different prediction approaches with paper final model configuration with an additional split param",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, # identical runs except at the last stage
		"Step"             :4,
		"FullTestSize"     :0,
		"AltMethod"        :"BackStep", # alternate method to use after slowdown point is reached
		"FutDist"          :20, 
		"splitmethod"      :"",
		"splitvar"         :"site",
		"Hyperpram"        :False,
		})
	expr[413] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :413,
		"predvar"          :"Delta_biomass",
		"dropvar"          :["Obs_biomass"],
		"name"             :"OneStageXGBOOST_AllGap_50perNA_PermutationImp_FINAL_Delta_biomass_altsplit_nofutdis",
		"desc"             :"Testing different prediction approaches with paper final model configuration with an additional split param",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, # identical runs except at the last stage
		"Step"             :4,
		"FullTestSize"     :0.1,
		"AltMethod"        :"BackStep", # alternate method to use after slowdown point is reached
		"splitmethod"      :"",
		"splitvar"         :"site",
		"Hyperpram"        :False,
		})
	expr[414] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :414,
		"predvar"          :"Delta_biomass",
		"dropvar"          :["Obs_biomass"],
		"name"             :"OneStageXGBOOST_AllGap_50perNA_PermutationImp_FINAL_Delta_biomass_altsplit_40futdis",
		"desc"             :"Testing different prediction approaches with paper final model configuration with an additional split param",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            :False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, # identical runs except at the last stage
		"Step"             :4,
		"FullTestSize"     :0.1,
		"AltMethod"        :"BackStep", # alternate method to use after slowdown point is reached
		"FutDist"          :40,
		"splitmethod"      :"",
		"splitvar"         :"site",
		"Hyperpram"        :False,
		})
	expr[415] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :415,
		"predvar"          :"Delta_biomass",
		"dropvar"          :["Obs_biomass"],
		"name"             :"OneStageXGBOOST_AllGap_50perNA_PermutationImp_FINAL_Delta_biomass_altsplit",
		"desc"             :"Testing different prediction approaches with paper final model configuration with an additional split param",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            : False,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, # identical runs except at the last stage
		"Step"             :4,
		"FullTestSize"     :0.1,
		"AltMethod"        :"BackStep", # alternate method to use after slowdown point is reached
		"FutDist"          :20, 
		"splitmethod"      :"",
		"splitvar"         :["site", "yrend"],
		"Hyperpram"        :False,
		})
	expr[416] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :416,
		"predvar"          :"Delta_biomass",
		"dropvar"          :["Obs_biomass"],
		"name"             :"OneStageXGBOOST_AllGap_Debug_Sitesplit",
		"desc"             :"Testing different prediction approaches with paper final model configuration with an additional split param",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            :True,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, # identical runs except at the last stage
		"Step"             :4,
		"FullTestSize"     :0.1,
		"AltMethod"        :"BackStep", # alternate method to use after slowdown point is reached
		"FutDist"          :20, 
		"splitmethod"      :"",
		"splitvar"         :"site",
		"Hyperpram"        :False,
		})
	expr[417] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :417,
		"predvar"          :"Delta_biomass",
		"dropvar"          :["Obs_biomass"],
		"name"             :"OneStageXGBOOST_AllGap_Debug_yrfnsplit",
		"desc"             :"Testing different prediction approaches with paper final model configuration with an additional split param",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            :True,
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
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, # identical runs except at the last stage
		"Step"             :4,
		"FullTestSize"     :0.1,
		"AltMethod"        :"BackStep", # alternate method to use after slowdown point is reached
		"FutDist"          :20, 
		"splitmethod"      :"",
		"splitvar"         :["site", "yrend"],
		"Hyperpram"        :False,
		})
	
	# ==========================================================================
	# ==========================================================================
	# ========== new params after digging into the performance issues ==========
	expr[420] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :420,
		"predvar"          :"Delta_biomass",
		"dropvar"          :["Obs_biomass"],
		"name"             :"XGBAllGap_Debug_yrfnsplit_CV",
		"desc"             :"Taking what i've learn't in my simplidfied experiments and incoperating it back in",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            :True,
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.1, 
		"FullTestSize"     :0.05,
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, # identical runs except at the last stage
		"Step"             :4,
		"AltMethod"        :"BackStep", # alternate method to use after slowdown point is reached
		"FutDist"          :0, 
		"splitmethod"      :"GroupCV",
		"splitvar"         :["site", "yrend"],
		"Hyperpram"        :False,
		})
	expr[421] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :421,
		"predvar"          :"Delta_biomass",
		"dropvar"          :["Obs_biomass"],
		"name"             :"XGBAllGap_Debug_yrfnsplit_Futdis_CV",
		"desc"             :"Taking what i've learn't in my simplidfied experiments and incoperating it back in",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            :True,
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.1, 
		"FullTestSize"     :0.05,
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, # identical runs except at the last stage
		"Step"             :4,
		"AltMethod"        :"BackStep", # alternate method to use after slowdown point is reached
		"FutDist"          :100, 
		"splitmethod"      :"GroupCV",
		"splitvar"         :["site", "yrend"],
		"Hyperpram"        :False,
		})
	expr[422] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :422,
		"predvar"          :"Delta_biomass",
		"dropvar"          :["Obs_biomass"],
		"name"             :"XGBAllGap_Debug_yrfnsplit_Futfire_CV",
		"desc"             :"Taking what i've learn't in my simplidfied experiments and incoperating it back in",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            :True,
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.1, 
		"FullTestSize"     :0.05,
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, # identical runs except at the last stage
		"Step"             :4,
		"AltMethod"        :"BackStep", # alternate method to use after slowdown point is reached
		"FutDist"          :100,
		"FutFire"          :0, 

		"splitmethod"      :"GroupCV",
		"splitvar"         :["site", "yrend"],
		"Hyperpram"        :False,
		})

	expr[423] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :423,
		"predvar"          :"Delta_biomass",
		"dropvar"          :["Obs_biomass"],
		"name"             :"XGBAllGap_Debug_yrfnsplit_CV_RFECV",
		"desc"             :"Taking what i've learn't in my simplidfied experiments and incoperating it back in",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            :True,
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.1, 
		"FullTestSize"     :0.05,
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :420, # identical runs except at the last stage
		"Step"             :4,
		"AltMethod"        :"RFECV", # alternate method to use after slowdown point is reached
		"FutDist"          :0, 
		"splitmethod"      :"GroupCV",
		"splitvar"         :["site", "yrend"],
		"Hyperpram"        :False,
		})

	expr[424] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :424,
		"predvar"          :"Delta_biomass",
		"dropvar"          :["Obs_biomass"],
		"name"             :"XGBAllGap_Debug_yrfnsplit_CV_RFECVBHYP",
		"desc"             :"Taking what i've learn't in my simplidfied experiments and incoperating it back in",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            :True,
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.1, 
		"FullTestSize"     :0.05,
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :423, # identical runs except at the last stage
		"Step"             :4,
		"AltMethod"        :"RFECVBHYP", # alternate method to use after slowdown point is reached
		"FutDist"          :0, 
		"splitmethod"      :"GroupCV",
		"splitvar"         :["site", "yrend"],
		"Hyperpram"        :False,
		})

	# ===============================================================================
	expr[430] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :430,
		"predvar"          :"Delta_biomass",
		"dropvar"          :["Obs_biomass"],
		"name"             :"XGBAllGap_Debug_sitesplit_CV",
		"desc"             :"Taking what i've learn't in my simplidfied experiments and incoperating it back in",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            :True,
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.1, 
		"FullTestSize"     :0.05,
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, # identical runs except at the last stage
		"Step"             :4,
		"AltMethod"        :"BackStep", # alternate method to use after slowdown point is reached
		"FutDist"          :0, 
		"splitmethod"      :"GroupCV",
		"splitvar"         :"site",
		"Hyperpram"        :False,
		})
	expr[431] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :431,
		"predvar"          :"Delta_biomass",
		"dropvar"          :["Obs_biomass"],
		"name"             :"XGBAllGap_Debug_sitesplit_Futdis_CV",
		"desc"             :"Taking what i've learn't in my simplidfied experiments and incoperating it back in",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            :True,
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.1, 
		"FullTestSize"     :0.05,
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, # identical runs except at the last stage
		"Step"             :4,
		"AltMethod"        :"BackStep", # alternate method to use after slowdown point is reached
		"FutDist"          :100, 
		"splitmethod"      :"GroupCV",
		"splitvar"         :"site",
		"Hyperpram"        :False,
		})
	expr[432] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :432,
		"predvar"          :"Delta_biomass",
		"dropvar"          :["Obs_biomass"],
		"name"             :"XGBAllGap_Debug_sitesplit_FutBurn_CV",
		"desc"             :"Taking what i've learn't in my simplidfied experiments and incoperating it back in",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            :True,
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.1, 
		"FullTestSize"     :0.05,
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :None, # identical runs except at the last stage
		"Step"             :4,
		"AltMethod"        :"BackStep", # alternate method to use after slowdown point is reached
		"FutDist"          :100, 
		"FutFire"          :0,
		"splitmethod"      :"GroupCV",
		"splitvar"         :"site",
		"Hyperpram"        :False,
		})
	expr[433] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :433,
		"predvar"          :"Delta_biomass",
		"dropvar"          :["Obs_biomass"],
		"name"             :"XGBAllGap_Debug_sitesplit_CV_RFECV",
		"desc"             :"Taking what i've learn't in my simplidfied experiments and incoperating it back in",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            :True,
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.1, 
		"FullTestSize"     :0.05,
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :430, # identical runs except at the last stage
		"Step"             :4,
		"AltMethod"        :"RFECV", # alternate method to use after slowdown point is reached
		"FutDist"          :0, 
		"splitmethod"      :"GroupCV",
		"splitvar"         :"site",
		"Hyperpram"        :False,
		})
	expr[434] = ({
		# +++++ The experiment name and summary +++++
		"Code"             :434,
		"predvar"          :"Delta_biomass",
		"dropvar"          :["Obs_biomass"],
		"name"             :"XGBAllGap_Debug_sitesplit_CV_RFECVBHYP",
		"desc"             :"Taking what i've learn't in my simplidfied experiments and incoperating it back in",
		"window"           :10,
		"predictwindow"    :None,
		"Nstage"           :1, 
		"model"            :"XGBoost",
		"debug"            :True,
		# +++++ The Model setup params +++++
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		# +++++ The experiment details +++++
		"test_size"        :0.1, 
		"FullTestSize"     :0.05,
		"SelMethod"        :"RecursiveHierarchicalPermutation",
		"ImportanceMet"    :"Permutation",
		"Transformer"      :None,
		"yTransformer"     :None, 
		"ModVar"           :"ntree, max_depth", "dataset"
		"classifer"        :None, 
		"cores"            :ncores,
		"maxitter"         :14, 
		"DropNAN"          :0.5, 
		"DropDist"         :False,
		"StopPoint"        :5,
		"SlowPoint"        :120, # The point i start to slow down feature selection and allow a different method
		"maxR2drop"        :0.025,
		"pariedRun"        :433, # identical runs except at the last stage
		"Step"             :4,
		"AltMethod"        :"RFECVBHYP", # alternate method to use after slowdown point is reached
		"FutDist"          :0, 
		"splitmethod"      :"GroupCV",
		"splitvar"         :"site",
		"Hyperpram"        :False,
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