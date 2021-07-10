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
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

print("seaborn version : ", sns.__version__)
print("xgb version : ", xgb.__version__)
# breakpoint()


# ==============================================================================
def main():
	# ========= make some paths ==========
	dpath = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/"
	path  = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"
	
	# ========== load in the stuff used by every run ==========
	# Datasets
	vi_df   = pd.read_csv(f"{dpath}ModDataset/VI_df_AllSampleyears_ObsBiomass.csv", index_col=0)
	df_site = pd.read_csv(f"{dpath}ModDataset/SiteInfo_AllSampleyears_ObsBiomass.csv", index_col=0)
	df_site.rename({"Plot_ID":"site"}, axis=1, inplace=True)
	# Column names, THis was chose an s model with not too many vars

	colnm   = pd.read_csv(
		f"{path}411/Exp411_OneStageXGBOOST_AllGap_50perNA_PermutationImp_RFECV_FINAL_Delta_biomass_altsplit_vers04_PermutationImportance_RFECVfeature.csv", index_col=0)#.index.values
	colnm   = colnm.loc[colnm.InFinal].index.values
	# colnm   = colnm.index.values
	predvar = "Delta_biomass"

	# ========== Subset the data ==========
	y, X = datasubset(vi_df, colnm, predvar, df_site,  FutDist=20, DropNAN=0.5,)


	# ========== Setup first experiment ==========\
	# once i get this working it will be a function
	
	dfksiteyr = pd.read_csv(f'{dpath}TTS_VI_df_AllSampleyears_10FWH_siteyear_TTSlookup.csv', index_col=0)
	dfksite   = pd.read_csv(f'{dpath}TTS_VI_df_AllSampleyears_10FWH_TTSlookup.csv', index_col=0)
	setup = OrderedDict()
	
	setup["30pRand"]       = ({
		"in_train":[0, 1], "in_test":[2, 3],"dfk":{"testsize":0.3, "n_splits":10}, "Sorting":"site", 
		})
	setup["30pTest"]       = ({
		"in_train":[0, 1], "in_test":[2, 3],"dfk":dfksite, "Sorting":"site", 
		})
	setup["30pRand-SYR"]       = ({
		"in_train":[0, 1], "in_test":[2, 3],"dfk":{"testsize":0.3, "n_splits":10}, "Sorting":["site", "yrend"], 
		})
	setup["30pTest-SYR"]       = ({
		"in_train":[0, 1], "in_test":[2, 3],"dfk":dfksiteyr, "Sorting":["site", "yearfn"], 
		})
	# setup["20pValTstDrop"] = ({
	# 	"in_train":[0], "in_test":[1], "dfk":dfksite, "Sorting":"site", 
	# 	"Summary":"Dropping the test set"
	# 	})
	
	setup["20pRand"]       = ({
		"in_train":[0, 1], "in_test":[2, 3],"dfk":{"testsize":0.2, "n_splits":10}, "Sorting":"site", 
		})
	setup["20pTest"]       = ({
		"in_train":[0, 1, 3], "in_test":[2],"dfk":dfksite, "Sorting":"site", 
		})
	setup["20pVal"]        = ({
		"in_train":[0, 2, 3], "in_test":[1], "dfk":dfksite, "Sorting":"site", 
		"Summary":"look at my random validiation splits instead"
		})

	setup["20pRand-SYR"]       = ({
		"in_train":[0, 1], "in_test":[2, 3],"dfk":{"testsize":0.2, "n_splits":10}, "Sorting":["site", "yearfn"], 
		})
	setup["20pTest-SYR"]       = ({
		"in_train":[0, 1, 3], "in_test":[2],"dfk":dfksiteyr, "Sorting":["site", "yearfn"], 
		})
	setup["20pVal-SYR"]        = ({
		"in_train":[0, 2, 3], "in_test":[1], "dfk":dfksiteyr, "Sorting":["site", "yearfn"], 
		"Summary":"look at my random validiation splits instead"
		})
	# setup["20pValTstDrop-SYR"] = ({
	# 	"in_train":[0], "in_test":[1], "dfk":dfksiteyr, "Sorting":["site", "yearfn"], 
	# 	"Summary":"Dropping the test set"
	# 	})
	

	scores = OrderedDict()
	
	for useGPU in [False,]:#True
		for expnm in setup:
			print(f'EXP:{expnm} {"Using GPU" if useGPU else ""}')
			if isinstance(setup[expnm]["dfk"], pd.DataFrame):
				# ========== Setup to use existing data indexing ==========
				ptrl, ptsl   = lookuptable(
					y.index.values, 
					setup[expnm]["dfk"],
					in_train=setup[expnm]["in_train"], 
					in_test=setup[expnm]["in_test"])
				

				# ========== Iterate over the ecperiments ==========
				for nx, (train, test) in tqdm(enumerate(zip(ptrl, ptsl)), total=10):
					scores[len(scores)] = XGBR(
						nx, X.loc[train], X.loc[test], y.loc[train], y.loc[test], 
						GPU=useGPU, expnm=expnm, resample=False)
			else:
				# breakpoint()
				itr = TTSspliter(y, X, df_site.loc[y.index.values,], setup[expnm])
				# ========== Iterate over the ecperiments ==========
				for nx, (train, test) in tqdm(enumerate(itr), total=setup[expnm]['dfk']['n_splits']):
					scores[len(scores)] = XGBR(
						nx, X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test], 
						GPU=useGPU, expnm=expnm, resample=True)

	
	dfs = pd.DataFrame(scores).T
	# print (dfs.groupby(["GPU"])["time"].apply(np.mean))
	# sns.barplot(y="R2", x="expn", hue="testnm", data=dfs)
	# plt.show()

	sns.boxplot(y="R2", x="testnm", data=dfs)
	plt.show()
	breakpoint()

	# fnin  = "TTS_VI_df_AllSampleyears_10FWH_TTSlookup.csv"
	# dfi   = pd.read_csv(fnin, index_col=0) 

# ==============================================================================
def XGBR(
	nx, X_train, X_test, y_train, y_test, 
	GPU=False, expnm="", verb=False, esr=40, resample=False):

	if GPU:
		XGB_dict = ({
			'objective': 'reg:squarederror',
			'num_parallel_tree'     :10,
			# 'n_jobs'           :-1,
			'max_depth'        :5,
			"n_estimators"     :2000,
			'tree_method': 'gpu_hist',
			"colsample_bytree":0.3,
			})
	else:
		XGB_dict = ({
			# +++++ The Model setup params +++++
			'objective': 'reg:squarederror',
			'num_parallel_tree'     :10,
			'n_jobs'           :-1,
			'max_depth'        :5,
			"n_estimators"     :2000,
			"tree_method"      :'hist',
			"colsample_bytree":0.3,
			})

	t0 = pd.Timestamp.now()
	eval_set  = [(X_test.values, y_test.values.ravel())]

	# ========== convert the values ========== 
	regressor = xgb.XGBRegressor(**XGB_dict)
	# regGPU = xgb.XGBRegressor(**gpu_dict)


	regressor.fit(X_train.values, y_train.values.ravel(), early_stopping_rounds=esr, verbose=verb, eval_set=eval_set)

	y_pred = regressor.predict(X_test)
	
	score = OrderedDict()
	score["testnm"] = expnm
	score["expn"]   = nx
	score["R2"]     = sklMet.r2_score(y_test, y_pred)
	score["MAE"]    = sklMet.mean_absolute_error(y_test, y_pred)
	score["RMSE"]   = np.sqrt(sklMet.mean_squared_error(y_test, y_pred))
	score["time"]   = pd.Timestamp.now()-t0
	score["GPU"]    = GPU
	score["RAND"]   = resample
	return score


# ==============================================================================
def TTSspliter(y, X, site_df, expset, random_state=42):
	"""
	Function to perform new random splits
	"""

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


	gss   = GroupShuffleSplit(
		n_splits = expset['dfk']['n_splits'], 
		test_size=expset['dfk']['testsize'], 
		random_state=random_state)

	return gss.split(X, y, groups=group)


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



def datasubset(vi_df, colnm, predvar, df_site, FutDist=20, DropNAN=0.5,):
	"""
	Function to do the splitting and return the datasets
	"""
	# ========== pull out the X values ==========
	X = vi_df.loc[:, colnm].copy() 
	nanccal = X.isnull().sum(axis=1)<= DropNAN
	distcal = df_site.BurnFut + df_site.DisturbanceFut
	distcal.where(distcal<=100., 100, inplace=True)
	# +++++ make a sing dist and nan layer +++++
	dist = (distcal <= FutDist) & nanccal
	X = X.loc[dist]

	y = vi_df.loc[X.index, predvar].copy() 
	# y.shape
	return y, X


# ==============================================================================
def _findcords(x, test):
	# Function check if x is in different arrays

	## I might need to use an xor here
	return x in test 
		







# ==============================================================================
if __name__ == '__main__':
	main()