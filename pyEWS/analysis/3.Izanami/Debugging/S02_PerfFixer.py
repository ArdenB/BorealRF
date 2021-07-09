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
# print("xgb version : ", xgb.__version__)
# breakpoint()


# ==============================================================================
def main():
	# ========= make some paths ==========
	dpath = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/"
	path  = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"
	
	# ========== load in the stuff used by every run ==========
	# Datasets
	vi_df = pd.read_csv(f"{dpath}ModDataset/VI_df_AllSampleyears_ObsBiomass.csv", index_col=0)
	df_site = pd.read_csv(f"{dpath}ModDataset/SiteInfo_AllSampleyears_ObsBiomass.csv", index_col=0)
	df_site.rename({"Plot_ID":"site"}, axis=1, inplace=True)
	# Column names, THis was chose an s model with not too many vars
	colnm = pd.read_csv(
		f"{411}TTS_VI_df_AllSampleyears_10FWH_vers04_X_train.csv", index_col=0).columns.values

	# ========== Subset the data ==========

	# ---------- Grab the stuff used by some configerations ==========

	fn    = f'{dpath}TTS_VI_df_AllSampleyears_10FWH_TTSlookup.csv'
	dfk   = pd.read_csv(fn, index_col=0)

	fnin  = "TTS_VI_df_AllSampleyears_10FWH_TTSlookup.csv"
	dfi   = pd.read_csv(fnin, index_col=0) 
	predvar = "Delta_biomass"

# ==============================================================================
def datasubset(vi_df, colnm, predvar, FutDist=20, DropNAN=0.5,):
	"""
	Function to do the splitting and return the datasets
	"""
	# splits=None,  intest = [2,3]
	pass 



def XGBR():
	# +++++ The Model setup params +++++
	"ntree"            :10,
	"nbranch"          :2000,
	"max_features"     :'auto',
	"max_depth"        :5,
	"min_samples_split":2,
	"min_samples_leaf" :2,
	"bootstrap"        :True,





# ==============================================================================
if __name__ == '__main__':
	main()