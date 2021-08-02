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
import optuna 
from optuna.samplers import TPESampler
from optuna.integration import XGBoostPruningCallback
import joblib

print("seaborn version : ", sns.__version__)
print("xgb version : ", xgb.__version__)
# breakpoint()


# ==============================================================================
def main():
	# Script is to work on my optimisations a little bit
	# ========== work out the important variables ==========
	impl = []
	parl = []
	opath1 = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/Debugging/hyperp/"
	# ========== Load in the data ==========
	fnames = glob.glob(f"./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/Debugging/DBG_Test_Site_30TstSz_20FDis_50NaN_precleaned_hyper*.csv")
	df     = pd.concat([pd.read_csv(fn, index_col=0) for fn in fnames])
	breakpoint()
	paths = OrderedDict()


	paths["Current"]  = opath1
	paths["Tweak1"]   = f"{opath1}old2/"
	paths["Original"] = f"{opath1}old/"

	for sptvar in ["Site", "SiteYF"]:
		for vname in paths:
			imp = []
			par = []
			fnames = glob.glob(f"{paths[vname]}optuna_DBG_Test*_{sptvar}_*.pkl")
			for nx, fn in enumerate(fnames):
				study = joblib.load(fn)
				ip = optuna.importance.get_param_importances(study)
				imp.append(pd.DataFrame({nx:ip}).T)
				# imp[nx] = 
				pms = study.best_params
				pms["best_value"] = study.best_value
				par.append(pd.DataFrame({nx:pms}).T)
				# fig =  optuna.visualization.plot_contour(study)
				# fig.show()


			# breakpoint()
			dfi = pd.concat(imp)
			dfp = pd.concat(par)

			# dfi = pd.DataFrame(imp).T
			dfl = pd.melt(dfi)
			dfl["splitvar"] = sptvar
			dfl["Version"]    = vname

			# dfp = pd.DataFrame(par).T
			# dfp["best_value"] = study.best_value
			dfp["splitvar"]   = sptvar
			dfp["Version"]    = vname
			impl.append(dfl)
			parl.append(dfp)
	dfls = pd.concat(impl)
	dfpv = pd.concat(parl)
	dfpv.reset_index(inplace=True)
	sns.boxplot(y="value", x="variable", hue="splitvar", data=dfls)
	plt.show()
	sns.boxplot(y="best_value", x="splitvar", hue="Version", data=dfpv)
	plt.show()

	sns.barplot(y="best_value", x="index", hue="Version", data=dfpv.loc[dfpv.splitvar == "Site"])
	breakpoint()

# ==============================================================================
if __name__ == '__main__':
	main()