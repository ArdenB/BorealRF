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
	predvar = "Delta_biomass"

	# ========== Subset the data ==========
	y, X = datasubset(vi_df, colnm, predvar, df_site,  FutDist=20, DropNAN=0.5,)


	# ========== Setup first experiment ==========
	dfk   = pd.read_csv(f'{dpath}TTS_VI_df_AllSampleyears_10FWH_TTSlookup.csv', index_col=0)
	
	ptrl, ptsl   = lookuptable(ind, dfk,)
	breakpoint()

	# fnin  = "TTS_VI_df_AllSampleyears_10FWH_TTSlookup.csv"
	# dfi   = pd.read_csv(fnin, index_col=0) 

# ==============================================================================

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
		assert np.logical_xor(ftrain.values,ftest.values).all()

		ptrainl.append(ftrain.loc[ftrain.values])
		ptestl.append(ftest.loc[ftest.values])
		# breakpoint()
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

	return y, X


# ==============================================================================
def _findcords(x, test):
	# Function check if x is in different arrays

	## I might need to use an xor here
	return x in test 
		

def XGBR():

	cpu_dict = ({
		# +++++ The Model setup params +++++
		'objective': 'reg:squarederror',
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		})

	gpu_dict = ({
		'objective': 'reg:squarederror',
		"ntree"            :10,
		"nbranch"          :2000,
		"max_features"     :'auto',
		"max_depth"        :5,
		"min_samples_split":2,
		"min_samples_leaf" :2,
		"bootstrap"        :True,
		'tree_method': 'gpu_hist'
		})






# ==============================================================================
if __name__ == '__main__':
	main()