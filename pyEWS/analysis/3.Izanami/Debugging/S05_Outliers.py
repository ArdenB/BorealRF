"""
Script goal, 

to make a draft version of the outlier prediction code that will go into 
the final version of the model
"""

# ==============================================================================

__title__ = "Outlier prediction problem"
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
# import cudf
# import cuml
# import optuna 
# from optuna.samplers import TPESampler
# from optuna.integration import XGBoostPruningCallback
# import joblib

print("seaborn version : ", sns.__version__)
print("xgb version : ", xgb.__version__)
# breakpoint()


# ==============================================================================
def main():
	# ========== Create the matplotlib params ==========
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':12, "axes.labelweight":"bold",})
	font = {'weight' : 'bold', 'size'   : 12}
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	# ========== open the VI dataset ==========
	fnamein  = f"./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/VI_df_AllSampleyears_ObsBiomass.csv"
	sfnamein = f"./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/SiteInfo_AllSampleyears_ObsBiomass.csv"
	vi_df  = pd.read_csv( fnamein, index_col=0)[["site", "year", "biomass", 'Delta_biomass', "ObsGap"]]
	# ["site", "year", "biomass", 'Delta_biomass' "ObsGap"]
	
	# ========== Create the new columns ==========
	# f, (ax1, ax2) = plt.subplots(2, 1,  sharex=True)
	vi_df["AnnualDelta"] = vi_df["Delta_biomass"]/vi_df["ObsGap"]
	sns.kdeplot(x="AnnualDelta", data=vi_df)#, ax=ax1)
	# plt.show()

	vi_df.AnnualDelta.hist(bins=2000)#, ax=ax2)
	plt.show()
	breakpoint()

# ==============================================================================

if __name__ == '__main__':
	main()