"""
Script goal, 

Interrogate the drop in performance
this is to see if its a bug or a random chance 
"""

# ==============================================================================

__title__ = "XGBoost data manupliation"
__author__ = "Arden Burrell"
__version__ = "v1.0(p.07.2021)"
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
import ipdb
# import ipdb
import matplotlib as mpl
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
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from tqdm import tqdm
import cudf
import cuml
import cupy
import optuna 
from optuna.samplers import TPESampler
from optuna.integration import XGBoostPruningCallback
import joblib

print("seaborn version : ", sns.__version__)
print("xgb version : ", xgb.__version__)
# breakpoint()


# ==============================================================================
def main():
	# ========= make some paths ==========
	dpath = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/"
	path  = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"
	opath = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/Debugging/"
	cf.pymkdir(opath)
	cf.pymkdir(opath+"hyperp")
	force = False
	fix   = False
	
	# ========== load in the stuff used by every run ==========
	# Datasets
	vi_df   = pd.read_csv(f"{dpath}ModDataset/VI_df_AllSampleyears_ObsBiomass.csv", index_col=0)
	df_site = pd.read_csv(f"{dpath}ModDataset/SiteInfo_AllSampleyears_ObsBiomass.csv", index_col=0)
	df_site.rename({"Plot_ID":"site"}, axis=1, inplace=True)
	predvar = "Delta_biomass"

	# ========== Deal with the rate and site problem ==========
	# df_site["rate"] = vi_df[predvar]/vi_df["ObsGap"]
	# dfrate = df_site[["site", "rate"]].groupby("site").max()


	# Column names, THis was chose an s model with not too many vars
	colnm   = pd.read_csv(
		f"{path}411/Exp411_OneStageXGBOOST_AllGap_50perNA_PermutationImp_RFECV_FINAL_Delta_biomass_altsplit_vers04_PermutationImportance_RFECVfeature.csv", index_col=0)#.index.values
	colnm   = colnm.loc[colnm.InFinal].index.values
	# colnm   = colnm.index.values

	# ========== Subset the data ==========
	setup = batchmaker(opath, dpath)
	fnlist = []

	
	for expnum in setup:
		# if setup[expnum]['hyperp']:
		# 	force=True 
		# else:
		# 	force=False
		scores = OrderedDict()
		# breakpoint()
		fnlist.append(f"{setup[expnum]['fnout']}.csv")
		if os.path.isfile(f"{setup[expnum]['fnout']}.csv") and not force:
			if fix:
				save=False
				df_check = pd.read_csv(f"{setup[expnum]['fnout']}.csv", index_col=0)
				for clm in ["Xtransf", "Ytransf", ]:#["preclean", "hyperp"	"OptunaSt"	
					if not clm in df_check.columns:
						df_check[clm] = ""
						save=True
				# if not setup[expnum]["Xtransf"] == "":
				# 	df_check["Xtransf"] = setup[expnum]["Xtransf"]
				# 	save=True
					# breakpoint()
				if save:
					df_check.to_csv(f"{setup[expnum]['fnout']}.csv")

			continue

		# for useGPU in [True]: #True, FalseTrue, 
		# if setup[expnum]['hyperp']:
		# 	useGPU=True 
		# else:
		# 	useGPU=False 
		useGPU=True
		# ========== Subset the data ==========
		y, X, group = datasubset(
			vi_df, colnm, predvar, df_site, setup[expnum],
			FutDist=setup[expnum]["FutDist"], 
			DropNAN=setup[expnum]["DropNAN"],)



		print(f'EXP:{expnum} {setup[expnum]["name"]} {"Using GPU" if useGPU else ""}')
		if isinstance(setup[expnum]["dfk"], pd.DataFrame):
			# ========== Setup to use existing data indexing ==========
			ptrl, ptsl   = lookuptable(
				y.index.values, 
				setup[expnum]["dfk"],
				in_train=setup[expnum]["in_train"], 
				in_test=setup[expnum]["in_test"])
			
			if useGPU:
				# breakpoint()
				y = cudf.from_pandas(y)
				X = cudf.from_pandas(X)
				X = X.fillna(np.NaN)
			# ========== Iterate over the ecperiments ==========
			for nx, (train, test) in tqdm(enumerate(zip(ptrl, ptsl)), total=setup[expnum]['n_splits']):
				scores[len(scores)] = XGBR(
					nx, X.loc[train], X.loc[test], y.loc[train], 
					y.loc[test], group.loc[train], setup[expnum], opath, 
					GPU=useGPU, expnm=setup[expnum]["name"], resample=False)
				# ========== Added an indent here so that it saves slow runs more often ==========
				if nx >=1  and setup[expnum]['hyperp']:
					try:
						dfs = pd.DataFrame(scores).T
						dfs.to_csv(f"{setup[expnum]['fnout']}.csv")
					except Exception as er:
						warn.warn(str(er))
						breakpoint()
		else:
			# breakpoint()
			itr = TTSspliter(y, X, df_site.loc[y.index.values,], setup[expnum])

			if useGPU:
				# breakpoint()
				y = cudf.from_pandas(y)
				X = cudf.from_pandas(X)
			# ========== Iterate over the ecperiments ==========
			for nx, (train, test) in tqdm(enumerate(itr), total=setup[expnum]['n_splits']):
				scores[len(scores)] = XGBR(
					nx, X.iloc[train], X.iloc[test], 
					y.iloc[train], y.iloc[test], group.iloc[train],
					setup[expnum],  opath, GPU=useGPU, expnm=setup[expnum]["name"], 
					resample=True)
				# breakpoint()
				# ========== Added an indent here so that it saves slow runs more often ==========
				if nx >=1  and setup[expnum]['hyperp']:
					try:
						dfs = pd.DataFrame(scores).T
						dfs.to_csv(f"{setup[expnum]['fnout']}.csv")
					except Exception as er:
						warn.warn(str(er))
						breakpoint()
		dfs = pd.DataFrame(scores).T
		dfs.to_csv(f"{setup[expnum]['fnout']}.csv")
		# breakpoint()
	
	# ========== load the multiple results ==========

	# df = pd.concat([pd.read_csv(fnp, index_col=0) for fnp in fnlist])
	# sns.violinplot(y="R2", x="testname", data=df)
	# plt.show()
	breakpoint()

# ==============================================================================
def preprocessing(test, train, transform, useGPU):
	"""
	Function to do data scaling
	"""
	if useGPU:
		import cuml.preprocessing as pp 
		import sklearn.preprocessing as skpp
		transformdict = ({
			"StandardScaler":pp.StandardScaler(),
			"RobustScaler":pp.RobustScaler(quantile_range=(25, 75)),
			"MinMaxScaler":pp.MinMaxScaler(),
			"MaxAbsScaler":pp.MaxAbsScaler(),
			"Normalizer":pp.Normalizer(),
			"PowerTransformerYJ":skpp.PowerTransformer(method='yeo-johnson'),
			# "PowerTransformerBC":skpp.PowerTransformer(method='box-cox'),
			"QuantileTransformerN":skpp.QuantileTransformer(output_distribution='normal'),
			"QuantileTransformerU":skpp.QuantileTransformer(output_distribution='uniform'),
			})
	else:
		import sklearn.preprocessing as pp
		transformdict = ({
			"StandardScaler":pp.StandardScaler(),
			"RobustScaler":pp.RobustScaler(quantile_range=(25, 75)),
			"MinMaxScaler":pp.MinMaxScaler(),
			"MaxAbsScaler":pp.MaxAbsScaler(),
			"PowerTransformerYJ":pp.PowerTransformer(method='yeo-johnson'),
			# "PowerTransformerBC":pp.PowerTransformer(method='box-cox'),
			"Normalizer":pp.Normalizer(),
			"QuantileTransformerN":pp.QuantileTransformer(output_distribution='normal'),
			"QuantileTransformerU":pp.QuantileTransformer(output_distribution='uniform'),
			})



	# ========== Sanity check my transforms ==========
	assert transform in ([
		"StandardScaler", "RobustScaler", "MinMaxScaler", "MaxAbsScaler",  
		"PowerTransformerYJ",  "Normalizer", #"PowerTransformerBC",
		"QuantileTransformerN", "QuantileTransformerU"])

	sclr   = transformdict[transform]
	# ========== make the transform ==========
	if useGPU: 
		try:
			if transform in (["PowerTransformerYJ", "PowerTransformerBC", 
				"QuantileTransformerN", "QuantileTransformerU"]):
				ttrain = sclr.fit_transform(train.to_pandas())
				ttest  = sclr.transform(test.to_pandas())
				ttrain = cudf.from_pandas(pd.DataFrame(ttrain))
				ttest  = cudf.from_pandas(pd.DataFrame(ttest ), )

			elif transform in ["Normalizer"]:
				ttrain = sclr.fit_transform(train.fillna(0))	
				ttest  = sclr.transform(test.fillna(0))
				ttrain.columns = train.columns
				ttrain.index   = train.index
				ttest.columns  = test.columns
				ttest.index    = test.index
				ttrain[train.isnull()] = np.NaN
				ttest[test.isnull()]   = np.NaN
				# breakpoint()
			else:
				ttrain = sclr.fit_transform(train)
				ttest  = sclr.transform(test)
			ttrain.columns = train.columns
			ttrain.index   = train.index
			ttest.columns  = test.columns
			ttest.index    = test.index
			ttest = ttest.fillna(np.NaN)
			ttrain = ttrain.fillna(np.NaN)
			

			if not all(ttrain.isnull() == train.isnull()):
				print(f"Nan problem with {tsf}")
				breakpoint()


		except Exception as err:
			print(str(err))
			breakpoint()
			# train.to_frame()
	else:
		try:
			ttest  = test.copy()
			ttrain = train.copy()
			ttrain.loc[:] = sclr.fit_transform(train)
			ttest.loc[:]  = sclr.transform(test)
		except Exception as err:
			print(str(err))
			breakpoint()
	
		if not all(ttrain.isnull() == train.isnull()):
			print(f"Nan problem with {tsf}")
			breakpoint()
	
	return ttrain, ttest, sclr


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
			y_pred.iloc[test_index] = regressor.predict(X_B)
		except Exception as er: 
			warn.warn(str(er))
			breakpoint()
	# y_pred /= n_repeats
	# breakpoint()
	return np.sqrt(mfunc(y_train, y_pred))

# ==============================================================================

def XGBR(
	nx, X_train, X_test, y_train, y_test, g_train, stinfo, opath, 
	GPU=False, expnm="", verb=False, esr=40, resample=False, n_trials=5):
	
	# ========== Do data pre transforming ==========
	if not stinfo["Xtransf"] == '':
		X_train, X_test, Xtrans = preprocessing(X_test, X_train, stinfo["Xtransf"], GPU)
	
	if not stinfo["Ytransf"] == '':
		y_train, y_test, Ytrans = preprocessing(y_test, y_train, stinfo["Ytransf"], GPU)
	# ========== Create the hyperprams ==========
	t0 = pd.Timestamp.now()
	if stinfo["hyperp"]:
		# Objective(trial, X_train, y_train, g_train, 
		# 	n_splits=3,	n_repeats=2, early_stopping_rounds=esr,
		# 	GPU=GPU)
		fnout = f"{opath}hyperp/optuna_{stinfo['name']}_v{nx:02d}{'_GPU' if GPU else ''}.pkl"
		if os.path.isfile(fnout):
			print(f"Loading existing Hyperpram Optimisation")
			study = joblib.load(fnout)
			if not len(study.trials) == n_trials:
				print("Additional studies required")
				breakpoint()
				# study.optimize(objective, n_trials=3)
			breakpoint()
		else:
			t0x = pd.Timestamp.now()
			sampler = TPESampler(multivariate=True)
			study   = optuna.create_study(direction="minimize", sampler=sampler)
			# breakpoint()
			if stinfo["OptunaSt"] in ["", "bestpar", "defualt", "both", ]:
				n_trials=10
				fullset = True
				if stinfo["OptunaSt"] =="bestpar":
					fullset = False
			elif stinfo["OptunaSt"] in ["long"]:
				n_trials = 50
				fullset  = True
			else:
				breakpoint()

			# A study with origial defualts
			if stinfo["OptunaSt"] in ["defualt", "both", "long"]:
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
			if stinfo["OptunaSt"] in ["both", "long"]:
				study.enqueue_trial({ 
					"max_depth": 7,
					"num_parallel_tree":15,
					"alpha":0.15, 
					"lambda":0.001, 
					})
			# breakpoint()
			study.optimize(
				lambda trial: Objective(trial,
					X_train, y_train, g_train, n_splits=3,
					n_repeats=2, early_stopping_rounds=esr,	GPU=GPU, 
					fullset=fullset
				),
				n_trials=n_trials,
				n_jobs=1,
			)
			joblib.dump(study, fnout)
			print(f"Hyperpram Optimisation took: {pd.Timestamp.now() - t0x}")
		# breakpoint()
		hp = study.best_params
		XGB_dict = _XGBdict(GPU, XGB_dict=hp)
		# breakpoint()
	else:
		XGB_dict = _XGBdict(GPU)

	# ========== create the XGBoost object ========== 
	regressor = xgb.XGBRegressor(**XGB_dict)

	if GPU:
		# breakpoint()
		X_test    = X_test.fillna(np.NaN)
		X_train   = X_train.fillna(np.NaN)
		eval_set  = [(X_test.values, y_test)]
		# try:
		# except:
		# 	bre
			# eval_set  = [(X_test, y_test)]
		regressor.fit(X_train.values, y_train, 
			early_stopping_rounds=esr, verbose=verb, eval_set=eval_set)
		
		# ========== Use cuml metrics instead here =====
		y_pred = regressor.predict(X_test).astype(np.float64)
		if not stinfo["Ytransf"] == '':
			breakpoint()
		try:

			R2     = cuml.metrics.r2_score( y_test, y_pred)
			MAE    = cuml.metrics.mean_absolute_error(y_test, y_pred)
			RMSE   = np.sqrt(cuml.metrics.mean_squared_error(y_test, y_pred))
		except Exception as er:
			breakpoint()

	else:

		eval_set  = [(X_test.values, y_test.values.ravel())]
		regressor.fit(X_train.values, y_train.values.ravel(), 
			early_stopping_rounds=esr, verbose=verb, eval_set=eval_set)

		y_pred = regressor.predict(X_test)
		R2     = sklMet.r2_score(y_test, y_pred)
		MAE    = sklMet.mean_absolute_error(y_test, y_pred)
		RMSE   = np.sqrt(sklMet.mean_squared_error(y_test, y_pred))

	try:

		score = OrderedDict()
		score["testname"]  = expnm
		score["expn"]      = nx
		score["group"]     = stinfo["group"]
		score["sptname"]   = stinfo["sptname"]
		score["R2"]        = R2
		score["MAE"]       = MAE
		score["RMSE"]      = RMSE
		score["time"]      = pd.Timestamp.now()-t0
		score["GPU"]       = GPU
		score["RAND"]      = resample
		score["test_size"] = stinfo["test_size"]
		score["FutDist"]   = stinfo["FutDist"]  
		score["DropNAN"]   = stinfo["DropNAN"]  
		score["obsnum"]    = y_test.size + y_train.size
		score["preclean"]  = stinfo["preclean"]
		score["hyperp"]    = stinfo["hyperp"]
		score["OptunaSt"]  = stinfo["OptunaSt"]
		score["Xtransf"]   = stinfo["Xtransf"]
		score["Ytransf"]   = stinfo["Ytransf"]

		# breakpoint()
		return score
	except Exception as er:
		breakpoint()
		raise er


# ==============================================================================
def TTSspliter(y, X, site_df, expset, random_state=42):
	"""
	Function to perform new random splits
	"""
	# breakpoint()

	if expset["Sorting"] == "site":
		group = site_df["site"]
	elif expset["Sorting"] == ["site", "yrend"]:
		site_df["yrend"] = X.ObsGap + site_df.year
		site_df["grp"]   = site_df.groupby(expset["Sorting"]).grouper.group_info[0]
		# breakpoint()
		group = site_df["grp"]	
	else:
		warn.warn("Not implemented yet")
		breakpoint()
		raise ValueError

	if expset["group"] == "RandCV":
		gss = GroupKFold(n_splits=expset['n_splits'])

	else:

		gss   = GroupShuffleSplit(
			n_splits     = expset['n_splits'], 
			test_size    = expset['test_size'], 
			random_state = random_state)

	return gss.split(X, y, groups=group) #, group


def lookuptable(ind, dfk, in_train = [0, 1], in_test=[2, 3]):
	"""
	ind:	 	array
		The dataframe index as an array
	dfk:	dataframe
		coded lookup table

	in_test:	list
		values included in train set
	in_train: 	list
		values included in test set
	"""
	
	ptrainl = []
	ptestl  = []

	# ========== loop over the dataframes columns ==========
	for nx in range(dfk.shape[1]):
		ftrain = dfk.loc[ind, str(nx)]. apply(_findcords, test=in_train)
		ftest  = dfk.loc[ind, str(nx)]. apply(_findcords, test=in_test)
		# \\\ Add an assertion so i can catch errors \\\
		# assert np.logical_xor(ftrain.values,ftest.values).all()

		ptrainl.append(ftrain.loc[ftrain.values].index.values)
		ptestl.append(ftest.loc[ftest.values].index.values)
	
	return ptrainl, ptestl 



def datasubset(vi_df, colnm, predvar, df_site, info, FutDist=20, DropNAN=0.5):
	"""
	Function to do the splitting and return the datasets
	"""
	# ========== Implement some form of nan filling here ==========

	# ========== pull out the X values ==========
	X = vi_df.loc[:, colnm].copy() 

	# ========= nan foltering here ==========
	nanccal = X.isnull().mean(axis=1) <= DropNAN
	distcal = df_site.BurnFut + df_site.DisturbanceFut
	distcal.where(distcal<=100., 100, inplace=True)
	# +++++ make a sing dist and nan layer +++++
	dist = (distcal <= FutDist) & nanccal
	X = X.loc[dist]

	y = vi_df.loc[X.index, predvar].copy() 
	# setup[expnum]
	if info['preclean']:
		rate = y/X["ObsGap"]
		# brksites = df_site.loc[y.loc[(rate>20)].index, "site"].unique()
		# for bks in brksites:
		# 	breakpoint()
		y.loc[(rate>20)]= np.NaN
		# breakpoint()
	if bn.anynan(y):
		X = X.loc[~y.isnull()]
		y = y.loc[~y.isnull()]
	
	if info["Sorting"] == "site":
		group = df_site["site"].loc[y.index]
	elif info["Sorting"] == ["site", "yrend"]:
		site_df = df_site.loc[y.index].copy()
		site_df["yrend"] = X.ObsGap + site_df.year
		site_df["grp"]   = site_df.groupby(info["Sorting"]).grouper.group_info[0]
		# breakpoint()
		group = site_df["grp"]	
	# group =
	# breakpoint()
	return y, X, group

# ==============================================================================

def batchmaker(opath, dpath):
	"""
	Place to setup the batch experiments
	
	"""
	setup = OrderedDict()

	# ========== Experiment 0 ==========
	# \\\\\ The benchmarking experiments \\\\\
	dspre   = True
	for modpost, optext in zip([True, False], ["long", ""]):
		for sptvar in [["site", "yrend"], "site"]:

			setup[len(setup)] = expgroup(
				sptvar, "Test", opath, dpath, test_size=0.3, 
				FutDist=0, n_splits=10, dropCFWH=False, DropNAN=0.5, 
				dspre=dspre, modpost=modpost, optext=optext)


			# # ///// Building matched random results \\\\\
			# # +++++ This is the site vs fully withlf validation +++++
			# # experiments with constant testsize +++++
			setup[len(setup)] = expgroup(
				sptvar, "RandCV", opath, dpath, test_size=0.1, 
				FutDist=0, n_splits=10, dropCFWH=False, DropNAN=0.5, 
				dspre=dspre, modpost=modpost, optext=optext)

			if not modpost:
				for transform in ([
					"StandardScaler", "RobustScaler", "MinMaxScaler", "MaxAbsScaler",  
					"PowerTransformerYJ", #"PowerTransformerBC", 
					"Normalizer", "QuantileTransformerN", "QuantileTransformerU"]):

					setup[len(setup)] = expgroup(
						sptvar, "Test", opath, dpath, test_size=0.3, 
						FutDist=0, n_splits=10, dropCFWH=False, DropNAN=0.5, 
						dspre=dspre, Xtrans=transform, modpost=modpost, optext=optext)

					# ===== Y transforms currently dont work ==========
					# setup[len(setup)] = expgroup(
					# 	sptvar, "Test", opath, dpath, test_size=0.3, 
					# 	FutDist=0, n_splits=10, dropCFWH=False, DropNAN=0.5, 
					# 	dspre=dspre, Xtrans=transform, Ytrans=transform, 
					# 	modpost=modpost, optext=optext)


	# ========== Experimant 1 ========== 
	# ///// recreating the existing results \\\\\
	for dspre, modpost in zip([False, True, True], [False, False, True]):

		for sptvar in ["site", ["site", "yrend"]]:
			# +++++ This is the site vs fully withlf validation +++++
			for testgroup in ["Test", "Vali"]:
				# experiments with constant testsize +++++
				if modpost and testgroup == "Vali":
					# Way to skip runs quickly
					continue
				# for optext in ['bestpar', "defualt", "both", "long", ""]:
				# 	setup[len(setup)] = expgroup(
				# 		sptvar, testgroup, opath, dpath, test_size=0.3, 
				# 		FutDist=0, n_splits=10, dropCFWH=False, DropNAN=0.5, 
				# 		dspre=dspre, modpost=modpost, optext=optext)
		

		if modpost:
			print("added a skip to jump slow runs")
			continue
		# ========== Experimant 2 ========== 
		# ///// Building matched random results \\\\\
		for sptvar in ["site", ["site", "yrend"]]:
			# +++++ This is the site vs fully withlf validation +++++
			# experiments with constant testsize +++++
			setup[len(setup)] = expgroup(
				sptvar, "Rand", opath, dpath, test_size=0.3, 
				FutDist=0, n_splits=30, dropCFWH=False, DropNAN=0.5, 
				dspre=dspre, modpost=modpost)

		# if modpost:
		# 	# Way to skip runs quickly
		# 	continue
		
	dspre   = True
	modpost = False
	for sptvar in ["site", ["site", "yrend"]]:
		# ========== Experimant 5 ========== 
		# ///// Varying the nan fraction \\\\\
		# +++++ This is the site vs fully withlf validation +++++
		# experiments with constant testsize +++++
		for DropNAN in np.arange(0, 1.01, 0.1):
			setup[len(setup)] = expgroup(
				sptvar, "Rand", opath, dpath, test_size=0.3, 
				FutDist=0, n_splits=30, dropCFWH=False, dspre=True,
				DropNAN=DropNAN)


		# # ========== Experimant 3 ========== 
		# # ///// Varying th disturbance \\\\\
		# for sptvar in ["site", ["site", "yrend"]]:
		# 	# +++++ This is the site vs fully withlf validation +++++
		# 	# experiments with constant testsize +++++
		for FutDist in np.arange(0, 101, 5):
			setup[len(setup)] = expgroup(
				sptvar, "Rand", opath, dpath, test_size=0.3, 
				FutDist=FutDist, n_splits=30, dropCFWH=False, 
				DropNAN=0.5, dspre=dspre, modpost=modpost)

		# # ========== Experimant 4 ========== 
		# # ///// Varying the test size \\\\\
		# for sptvar in ["site", ["site", "yrend"]]:
		# +++++ This is the site vs fully withlf validation +++++
		# experiments with constant testsize +++++
		for test_size in np.arange(0.05, 1., 0.05):
			setup[len(setup)] = expgroup(
				sptvar, "Rand", opath, dpath, test_size=test_size, 
				FutDist=0, n_splits=30, dropCFWH=False, DropNAN=0.5, 
				dspre=dspre, modpost=modpost)
	
	return setup


def expgroup(sptvar, testgroup, opath, dpath, test_size=0.3, 
	FutDist=20, n_splits=10, dropCFWH=False, DropNAN=0.5, dspre=False, 
	 Xtrans="", Ytrans="", modpost=False, optext=""):
	
	"""
	Container for my experiments to go, returns the requested experiment
	args:
		sptvar   	THe var that the split is made on. must be either site
					or kist of site and year
		testgroup:	str 
			must be either Rand, Test, Vali
	returns:
		expnm
		Filename
		setup
	"""
	# ========== 
	# ========== Setup first experiment ==========\
	# Check to see 
	assert testgroup in ["Rand", "RandCV", "Test", "Vali"]


	# setup    = OrderedDict()
	dfk      = None
	in_train = [None]
	in_test  = [None]

	if sptvar == "site":
		# ========== The container of the predone splits ==========
		spn = "Site"
		if not testgroup in  ["Rand", "RandCV"]:
			dfk = pd.read_csv(f'{dpath}TTS_VI_df_AllSampleyears_10FWH_TTSlookup.csv', index_col=0)
			if testgroup == "Test":
				in_train = [0, 1]
				in_test  = [2, 3]
			else:
				in_train = [0, 2]
				in_test  = [1, 3]
		
	elif sptvar == ["site", "yrend"]:
		spn = "SiteYF"
		if not testgroup in  ["Rand", "RandCV"]:
			dfk = pd.read_csv(f'{dpath}TTS_VI_df_AllSampleyears_10FWH_siteyear_TTSlookup.csv', index_col=0)
			if testgroup == "Test":
				in_train = [0, 1]
				in_test  = [2, 3]
			else:
				in_train = [0, 2]
				in_test  = [1, 3]
			if dropCFWH:
				warn.warn("Not implemented yet")
				breakpoint()
		
	else:
		warn.warn("Not implemented yet")
		breakpoint()
		raise ValueError
	# ========== Remove the FWH set ==========

	if dropCFWH:
		for ls in [in_train, in_test]:
			if 3 in ls:
				ls.remove(3)
	
	expn = f"DBG_{testgroup}_{spn}_{int(test_size*100)}TstSz_{FutDist}FDis_{int(DropNAN*100)}NaN"
	if dspre:
		expn += f"_precleaned{Xtrans}{Ytrans}"

	if modpost:
		expn += f"_hyperp{optext}"

	expsetup = ({
		"name"    :expn, 
		"group"   :testgroup,
		"dfk"     :dfk, 
		"Sorting" :sptvar,
		"sptname" :spn,
		"in_train":in_train, 
		"in_test" :in_test, 
		"FutDist" :FutDist, 
		"n_splits":n_splits, 
		"dropCFWH":dropCFWH, 
		"test_size":test_size,
		"FutDist" :FutDist,
		"DropNAN" :DropNAN, 
		"fnout"   :f"{opath}{expn}",
		"preclean":dspre,
		"hyperp"  :modpost,
		"OptunaSt":optext,
		"Xtransf" :Xtrans, 
		"Ytransf" :Ytrans,
		})

	# breakpoint()
	return expsetup


# ==============================================================================

def _XGBdict(GPU, XGB_dict=None):
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


def _findcords(x, test):
	# Function check if x is in different arrays

	## I might need to use an xor here
	return x in test 
		




# ==============================================================================
if __name__ == '__main__':
	main()