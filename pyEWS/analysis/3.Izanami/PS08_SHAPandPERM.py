"""
Script for dealing with permutation importance and SHAP values
"""

# ==============================================================================

__title__ = "Loss variables "
__author__ = "Arden Burrell"
__version__ = "v1.0(08.06.2021)"
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import OrderedDict, defaultdict
import seaborn as sns
import palettable
# from numba import jit
import matplotlib.colors as mpc
from tqdm import tqdm
import pickle
from itertools import product

# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf
import myfunctions.benchmarkfunctions as bf

# ========== Import packages for parellelisation ==========
# import multiprocessing as mp
import xgboost as xgb
import xarray as xr
import cartopy.crs as ccrs
import dask
from dask.diagnostics import ProgressBar
from tqdm import tqdm
import shap

import cartopy.feature as cpf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ast

# ========== Import ml packages ==========
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn import metrics as sklMet
# from sklearn.utils import shuffle
# from scipy.stats import spearmanr
# from scipy.cluster import hierarchy


# ==============================================================================

def main():
	# ========== Setup the pathways ==========
	formats = None
	path  = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"
	cf.pymkdir(path+"plots/")
	ppath = "./pyEWS/analysis/3.Izanami/Figures/PS08/"
	cf.pymkdir(ppath)

	# ========== Chose the experiment ==========
	exp      = 434
	_ImpOpener(path, exp)


def _ImpOpener(path, exp, var = "PermutationImportance", AddFeature=False, textsize=14):
	"""
	Function to open the feature importance files and return them as a single 
	DataFrame"""
	sns.set_style("whitegrid")
	font = ({'weight' : 'bold', 'size'   : textsize})
	mpl.rc('font', **font)
	plt.rcParams.update({'axes.titleweight':"bold", 
		"axes.labelweight":"bold", 'axes.titlesize':textsize, 'axes.titlelocation': 'left',}) 

	# ========== Loop over the exps ==========
	df_list = []
	SHAPlst = []
	fnames = sorted(glob.glob(f"{path}{exp}/Exp{exp}_*PermutationImportance.csv"))
	for ver, fn in enumerate(fnames):
		print(ver)

		# ========== load the model ==========
		dfin = pd.read_csv( fn, index_col=0)
		fn_mod = f"{path}{exp}/models/XGBoost_model_exp{exp}_version{ver}.dat"
		model  = pickle.load(open(f"{fn_mod}", "rb"))

		ColNm = dfin["Variable"].values
		X_train, X_test, y_train, y_test, col_nms, loadstats, corr, df_site, dbg = _getdata(path, exp, ColNm)
		y_pred = model.predict(X_test)
		# ========== do the increasing permutation and decreasing permutation ==========
		# print("starting sklearn permutation importance calculation at:", pd.Timestamp.now())
		# resultdec = permutation_importance(model, X_test[y_pred<=0].values, y_test[y_pred<=0].values.ravel(), n_repeats=5) #n_jobs=cores
		# impMetdec = resultdec.importances_mean
		# dfin["Loss"] = impMetdec

		# resultinc = permutation_importance(model, X_test[y_pred>0].values, y_test[y_pred>0].values.ravel(), n_repeats=5) #n_jobs=cores
		# impMetinc = resultinc.importances_mean
		# dfin["Gain"] = impMetinc

		dfin["experiment"] = exp
		dfin["version"]    = ver
		df_list.append(dfin)

		# ========== calculate the SHAP values ==========
		explainer   = shap.TreeExplainer(model)
		shap_values = explainer.shap_values(X_test)
		SHAPlst.append(shap_values)
		# breakpoint()
		makeindplots = False
		if makeindplots:
			plt.figure(1)
			shap.summary_plot(shap_values, X_test, max_display=20, show=False)
			shap.dependence_plot("biomass", shap_values, X_test)
			shap.dependence_plot("ObsGap", shap_values, X_test)
			# shap.summary_plot(shap_values, X_test, plot_type="layered_violin")#, color='coolwarm')
			if ver == 2:
				explainer2   = shap.Explainer(model, X_test)
				shap_values2 = explainer2(X_test)
				shap.plots.waterfall(shap_values2[0])
				shap.plots.waterfall(shap_values2[1000])
				breakpoint()
			breakpoint()
			


		# X_output = X_test.copy()
		# X_output.loc[:,'predict'] = y_pred
		# random_picks = np.arange(1,330,50) # Every 50 rows
		# S = X_output.iloc[random_picks]
		# shap_values_Model = explainer.shap_values(S)

		# shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])

	# ========== Group them together ==========
	df = pd.concat(df_list).reset_index(drop=True)
	df["Count"] = df.groupby(["experiment", "Variable"])[var].transform("count")
	breakpoint()


	# df = pd.melt(df.drop(var, axis=1) , 
	# 	id_vars=['Variable','experiment','version', 'Count'], value_vars=['Loss', 'Gain'],
	# 	var_name='ChangeDirection', value_name=var)
	
	# # ========== Calculate the counts ==========
	# # df.sort_values(['experiment', 'Count'], ascending=[True, False], inplace=True)
	# # df.reset_index(drop=True,inplace=True)
	# df["ChangeDirection"] = df["ChangeDirection"].astype("category").cat.set_categories(["Loss", "Gain"])

	# fnout = f"{path}{exp}/Exp{exp}_PermutationImportance_lossgain_PS08.csv"
	# df.to_csv(fnout)

	# g = sns.boxplot(x="Variable", y="PermutationImportance", hue="ChangeDirection", data=df.loc[df.Count >= 5])
	# plt.show()


	breakpoint()

# ==============================================================================

def Smartrenamer(names):
	breakkpoint()
	

def _getdata(path, exp, ColNm,
	dpath = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/", 
	inheritrows=True):
	# ========== load the setup ==========
	setup            = pd.read_csv(f"{path}{exp}/Exp{exp}_setup.csv", index_col=0).T.infer_objects().reset_index()
	for va in ["window", "Nstage", "test_size", "DropNAN", "FutDist", "FullTestSize"]:
		setup[va] = setup[va].astype(float)
	for vaxs in ["dropvar", "splitvar"]:	#
		# breakpoint()
		try:
			setup[vaxs] = [ast.literal_eval(setup[vaxs].values[0])]
		except ValueError:
			continue

	setup = setup.loc[0]
	for tf in ["yTransformer", "Transformer"]:
		if np.isnan(setup[tf]):
			setup[tf] = None
		else:
			breakpoint()
	# breakpoint()
	branch  = 0
	version = 0
	if (setup["predvar"] == "lagged_biomass") or inheritrows:
		basestr = f"TTS_VI_df_AllSampleyears" 
	else:
		basestr = f"TTS_VI_df_AllSampleyears_{setup['predvar']}" 

	if not setup["FullTestSize"] is None:
		basestr += f"_{int(setup['FullTestSize']*100)}FWH"
		if setup["splitvar"] == ["site", "yrend"]:
			basestr += f"_siteyear{setup['splitmethod']}"
		elif setup["splitvar"] == "site":
			basestr += f"_site{setup['splitmethod']}"

	if setup.loc["predvar"] == "lagged_biomass":
		fnamein  = f"{dpath}VI_df_AllSampleyears.csv"
		sfnamein = f"{dpath}SiteInfo_AllSampleyears.csv"
	else:
		fnamein  = f"{dpath}VI_df_AllSampleyears_ObsBiomass.csv"
		sfnamein = f"{dpath}SiteInfo_AllSampleyears_ObsBiomass.csv"
		# bsestr = f"TTS_VI_df_AllSampleyears_{setup.loc[0, 'predvar']}" 

	# ========== load in the data ==========
	X_train, X_test, y_train, y_test, col_nms, loadstats, corr, df_site, dbg = bf.datasplit(
		setup.loc["predvar"], exp, version,  branch, setup,  cols_keep=ColNm, final=True, #force=True,
		vi_fn=fnamein, region_fn=sfnamein, basestr=basestr, dropvar=setup.loc["dropvar"])

	return X_train, X_test, y_train, y_test, col_nms, loadstats, corr, df_site, dbg

# ==============================================================================
if __name__ == '__main__':
	main()