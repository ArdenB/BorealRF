"""
Script for dealing with permutation importance and SHAP values
"""

# ==============================================================================

__title__ = "Database Stat Calculator"
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
	ppath = "./pyEWS/analysis/3.Izanami/Figures/PS09/"
	cf.pymkdir(ppath)
	fpath = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/"
	vi_df   = pd.read_csv(f"{fpath}VI_df_AllSampleyears_ObsBiomass.csv", index_col=0)
	VIfvi   = pd.read_csv(f"{fpath}VI_df_AllSampleyears_FutureBiomass.csv", index_col=0)
	
	site_df = pd.read_csv(f"{fpath}SiteInfo_AllSampleyears_ObsBiomass.csv", index_col=0)
	experiment = [ 434, 424]
	exper = experiments()
	# exp      = 434
	for exp in experiment:
		setup = exper[exp].copy()
		# ========= Load in the observations ==========
		OvP_fnames = glob.glob(f"{path}{exp}/Exp{exp}*_OBSvsPREDICTED.csv")
		df_OvsP    = pd.concat([load_OBS(ofn) for ofn in OvP_fnames], sort=True)
		df_OvsP["Region"] = site_df.loc[df_OvsP.index, "Region"]
		regions    = regionDict()
		df_OvsP.replace(regions, inplace=True)


		df_OvsP["ObsGap"]      = vi_df.loc[df_OvsP.index, "ObsGap"]
		df_OvsP["Residual"]    = df_OvsP["Estimated"].values - df_OvsP["Observed"].values
		df_OvsP["ABSResidual"] = np.abs(df_OvsP["Estimated"].values - df_OvsP["Observed"].values)
		df_OvsP["AnnualResidual"   ] = df_OvsP["Residual"]    /df_OvsP["ObsGap"]
		df_OvsP["AnnualABSResidual"] = df_OvsP["ABSResidual"] /df_OvsP["ObsGap"]
		df_mod     = df_OvsP[~df_OvsP.index.duplicated(keep='first')]
		# breakpoint()
		# ========== Chose the experiment ==========
		sitedtb(path, ppath, exp, fpath, vi_df, site_df, df_mod, VIfvi)

		PredictorInfo(ppath, exp, setup, ColNm = None)

		# ========== predictions ==========
		Futuredfmaker(path, ppath, exp, fpath, vi_df, site_df, df_OvsP)

		ensemblper(path, ppath, exp, fpath, vi_df, site_df, df_OvsP)
		
		predictions(path, ppath, exp, fpath, vi_df, site_df, df_OvsP)
		# breakpoint()

# ==============================================================================
def PredictorInfo(ppath, exp, setup, ColNm = None, inheritrows=True):
	"""Load in the predicto varrs"""

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


	X_train, X_test, y_train, y_test, col_nms, loadstats, corr, df_site, dbg = bf.datasplit(
		setup["predvar"], exp, 0,  0, setup, 
		final=True,  cols_keep=ColNm, vi_fn=fnamein, region_fn=sfnamein, basestr=basestr, 
		dropvar=setup["dropvar"], column_retuner=True)

	Scriptinfo = "Stats about variables Exported using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	keystats = [Scriptinfo, gitinfo]



	# +++++ summary +++++
	# Number of sites
	keystats.append("\n Total number of predictor variables \n")
	keystats.append(col_nms.shape)
	# mean obs per site
	col_df = Grouper(col_nms)
	
	keystats.append("\nVariables by source \n")
	keystats.append(col_df.groupby("VariableGroup").count())

	# breakpoint()

	# ========== Save the info =========
	fname = f'{ppath}PS09_PredictorVarsSummary_exp{exp}.txt'
	f = open(fname,'w')
	for info in keystats:
		f.write("%s\n" %info)
	f.close()



def Futuredfmaker(path, ppath, exp, fpath, vi_df, site_df, df_OvsP):


	Scriptinfo = "Stats Exported using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	keystats = [Scriptinfo, gitinfo]


	years = [2020, 2025, 2030, 2040]
	df    = fpred(path, exp, years)

	dfmean = df.groupby(["Plot_ID", "time"]).mean()
	dfmean["Region"]      = df.groupby(["Plot_ID", "time"]).first()["Region"]
	dfmean.reset_index(inplace=True)
	dfmean["MMMIncrease"] =  dfmean.DeltaBiomass > 0	

	keystats.append("\n Stats About Future Predictions\n")
	keystats.append("\n Fraction of sites that are predicted to increase \n")
	keystats.append(dfmean[["time", "MMMIncrease"]].groupby(["time"]).mean())
	keystats.append(dfmean[["time", "MMMIncrease", "ModelAgreement"]].groupby(["time", "MMMIncrease"]).count())

	keystats.append("\n Number of sites that are predicted to increase by ModelAgreement \n")
	keystats.append(dfmean[["time", "MMMIncrease", "ModelAgreement"]].groupby(["time", "ModelAgreement"]).count())

	# keystats.append(dfmean[["time", "MMMIncrease", "ModelAgreement"]].groupby(["time", "ModelAgreement"]).count())
	
	dfa = dfmean[["time", "Region", "MMMIncrease"]].groupby(["time", "Region"]).mean()
	dfa["MMMDecrease"] = 1- dfa["MMMIncrease"]
	dfa["SiteCount"] = dfmean[["time", "Region", "MMMIncrease"]].groupby(["time", "Region"]).count().values
	dfa["SiteCountInc"] = dfmean[["time", "Region", "MMMIncrease"]].groupby(["time", "Region"]).sum().values
	keystats.append("\n Regional Breakdown predicted to increase by ModelAgreement \n")
	keystats.append(dfa)
	# breakpoint()
	# ========== Save the info =========
	fname = f'{ppath}PS09_FuturePredictionSummary_exp{exp}.txt'
	f = open(fname,'w')
	for info in keystats:
		f.write("%s\n" %info)
	f.close()


# def ensemblper(path, ppath, exp, fpath, vi_df, site_df, df_OvsP):

def ensemblper(path, ppath, exp, fpath, vi_df, site_df, df_OvsP):
	"""
	Look at how the ensemble median does compared to the mean
	"""
	Scriptinfo = "Stats Exported using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	keystats = [Scriptinfo, gitinfo]

	# ========== Summary of prediction scores ==========
	df_O = df_OvsP.reset_index()
	df_O["Ncount"] = df_O.groupby("index")["Estimated"].transform("count")
	dfen = df_O[df_O["Ncount"]==10].sort_values(["index", "version"])
	dfen["CorrectDir"] = (dfen.Estimated > 0) == (dfen.Observed > 0)

	dfen["ObservedGain"] = (dfen.Observed > 0)
	dfen["EstimatedGain"] = (dfen.Estimated > 0)
	dfen["AnnualEst"] = dfen['Estimated'] / dfen["ObsGap"]
	# dfen.groupby("index").mean()
	

	# observed in range of predictions 
	dfens = dfen.groupby("index").agg(
		minest=pd.NamedAgg(column='Estimated', aggfunc='min'),
		maxest=pd.NamedAgg(column='Estimated', aggfunc='max'),
		Observed=pd.NamedAgg(column='Observed', aggfunc='mean'),
		MeanEst=pd.NamedAgg(column='Estimated', aggfunc='mean'),
		MedianEst=pd.NamedAgg(column='Estimated', aggfunc='median'),
		ModelGainFrac=pd.NamedAgg(column='EstimatedGain', aggfunc='mean'),
		# AnnualMeanEst=pd.NamedAgg(column='AnnualEst', aggfunc='mean'),
		ObsGap=pd.NamedAgg(column='ObsGap', aggfunc='mean'),
		)
	dfens["Inrange"  ] = np.logical_and(dfens.maxest > dfens.Observed, dfens.minest < dfens.Observed)
	dfens["MeanRes"  ] = dfens.MeanEst - dfens.Observed
	dfens["AnnMeanRes"]= dfens["MeanRes"]/dfens["ObsGap"]
	dfens["AbsMeanRes"] = dfens["MeanRes"  ].abs()
	dfens["MedianRes"] = dfens.MedianEst - dfens.Observed
	dfens["AbsMedianRes"] = dfens["MedianRes"].abs()
	dfens["CorrectDir"] = (dfens.MeanEst > 0) == (dfens.Observed > 0)
	dfens["ObservedGain"] = (dfens.Observed > 0)
	

	keystats.append("\n The median indivdual model performance \n")
	keystats.append(dfen.groupby("version").median().drop(["CorrectDir", "Ncount", "index"], axis=1))

	keystats.append("\n The median ensemble performance \n")
	keystats.append(dfens[["Inrange", "MeanRes", "AnnMeanRes", "AbsMeanRes", "MedianRes", "AbsMedianRes"]].median())

	keystats.append("\n The mean indivdual model performance \n")
	keystats.append(dfen.groupby("version").mean().drop(["Ncount", "index"], axis=1))

	keystats.append("\n The mean ensemble performance \n")
	keystats.append(dfens[["Inrange", "MeanRes", "AnnMeanRes", "AbsMeanRes", "MedianRes", "AbsMedianRes", "CorrectDir"]].mean())
	
	keystats.append("\n The total perormace accuracy by observed direction \n")
	keystats.append(dfen[["ObservedGain", "CorrectDir"]].groupby("ObservedGain").mean())

	keystats.append("\n The total perormace accuracy by observed direction and model \n")
	keystats.append(dfen[["version", "ObservedGain", "CorrectDir"]].groupby(["version","ObservedGain"]).mean())

	keystats.append("\n The total perormace accuracy by observed direction \n")
	keystats.append(dfens[["ObservedGain", "CorrectDir"]].groupby("ObservedGain").mean())
	
	dfens["ErrorType"]  = ""
	dfens.loc[ np.logical_and(dfens["ObservedGain"], dfens["CorrectDir"]), "ErrorType"] = "CorrectGain"
	dfens.loc[ np.logical_and((~dfens["ObservedGain"]), dfens["CorrectDir"]), "ErrorType"] = "Correctloss"
	
	dfens.loc[ np.logical_and(dfens["ObservedGain"], ~dfens["CorrectDir"]), "ErrorType"] = "FalseLoss"
	dfens.loc[ np.logical_and((~dfens["ObservedGain"]), ~dfens["CorrectDir"]), "ErrorType"] = "FalseGain"
	keystats.append("\n Error types \n")
	keystats.append(dfens.groupby(["ObservedGain", "ErrorType"]).count()["Observed"])
	keystats.append(dfens.groupby(["ModelGainFrac", "ErrorType"]).count()["Observed"])
	# breakpoint()

	# ========== loss frequency ==========
	dfenL  = dfen[dfen.Observed < 0 ]
	dfensL = dfens[dfens.Observed < 0 ]
	keystats.append("\n Performance in areas with loss \n \n")

	keystats.append("\n Mean ensemble member \n")
	keystats.append(dfenL.groupby("version").mean()["CorrectDir"])

	keystats.append("\n Mulitmodel mean \n")
	keystats.append(dfenL["CorrectDir"].mean())

	# keystats.append(f'\n Fraction of measurments where models guess directionalty correctly:{np.sum((y_test > 0) == (y_pred > 0)) / df_OvsP.shape[0]}')

	# ========== Save the info =========
	fname = f'{ppath}PS09_EnsemblePerfSummary_exp{exp}.txt'
	f = open(fname,'w')
	for info in keystats:
		f.write("%s\n" %info)
	f.close()


def predictions(path, ppath, exp, fpath, vi_df, site_df, df_OvsP):
	"""
	open the database and pull out the relevant files
	"""
	Scriptinfo = "Stats Exported using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	keystats = [Scriptinfo, gitinfo]

	# ========== Summary of prediction scores ==========
	y_test = df_OvsP["Observed"].values
	y_pred = df_OvsP["Estimated"].values
	# breakpoint()
	keystats.append('\n Prediction Score Metric for the entire prediction ensemble \n')
	keystats.append(f'\n R squared score: {sklMet.r2_score(y_test, y_pred)}')
	keystats.append(f'\n Mean Absolute Error: {sklMet.mean_absolute_error(y_test, y_pred)}')
	keystats.append(f'\n Median Absolute Error: {sklMet.median_absolute_error(y_test, y_pred)}')
	keystats.append(f'\n Root Mean Squared Error: {np.sqrt(sklMet.mean_squared_error(y_test, y_pred))}')

	keystats.append(f'\n Annual Median Absolute Error: {sklMet.median_absolute_error(y_test/df_OvsP.ObsGap.values, y_pred/df_OvsP.ObsGap.values)}')
	keystats.append(f'\n Annual Mean Absolute Error: {sklMet.mean_absolute_error(y_test/df_OvsP.ObsGap.values, y_pred/df_OvsP.ObsGap.values)}')
	keystats.append(f'\n Annual Root Mean Squared Error: {np.sqrt(sklMet.mean_squared_error(y_test/df_OvsP.ObsGap.values, y_pred/df_OvsP.ObsGap.values))}')
	# df_OvsP['ObsGap'].values
	# breakpoint()
	# ========== overall trends in the model data =======
	keystats.append(f"\n The fraction sites with Observed increases:\n {(df_OvsP['Observed'] > 0).sum() / df_OvsP.shape[0]}")
	keystats.append(f"\n The fraction sites with Estimated increases:\n {(df_OvsP['Estimated'] > 0).sum() / df_OvsP.shape[0]}")
	# +++++ did it get direction correct +++++
	keystats.append(f'\n Fraction of measurments where models guess directionalty correctly:{np.sum((y_test > 0) == (y_pred > 0)) / df_OvsP.shape[0]}')

	# ========== THe number of features per model ==========
	keystats.append('\n Permutation Importantce and feature counts \n')
	df_perm = loadperm(exp, path)
	# regions = regionDict()
	# site_dfM
	# df_perm.replace(regions, inplace=True)
	# breakpoint()
	keystats.append("\n Number of features per model \n")
	keystats.append(df_perm.groupby("Version").count()["index"])

	keystats.append("\n No of models that selected features \n")
	keystats.append(df_perm.groupby("Variable").count()["index"])

	keystats.append("\n No features in 10 models \n")
	df_perm.groupby("Variable").count()["index"].reset_index().groupby("index").count()

	# +++++ Grouped perm +++++
	dfg =  df_perm.groupby(["Version","VariableGroup"]).sum()["PermutationImportance"].reset_index().groupby("VariableGroup").mean()#["PermutationImportance"]
	dfg = dfg.drop("Version", axis=1)
	dfg["STD"] = df_perm.groupby(["Version","VariableGroup"]).sum()["PermutationImportance"].reset_index().groupby("VariableGroup").std()["PermutationImportance"]
	keystats.append("\n Grouped Importance \n")
	keystats.append(dfg)

	keystats.append("\n Mean Importance \n")
	keystats.append(df_perm.groupby("Variable").mean().sort_values("PermutationImportance", ascending=False).drop(["index", "Version"], axis=1).reset_index())


	# ========== Residual Performance by region ==========
	keystats.append('\n Permutation Importantce and feature counts \n')
	# site_dfM

	keystats.append("\n Mean and absmean residuals by region \n")
	keystats.append(df_OvsP.groupby("Region").mean()[["Estimated","Observed", "Residual", "ABSResidual", "ObsGap", "AnnualResidual", "AnnualABSResidual"   ]])
	keystats.append("\n Median and absmedian residuals by region \n")
	keystats.append(df_OvsP.groupby("Region").median()[["Estimated","Observed", "Residual", "ABSResidual", "ObsGap", "AnnualResidual", "AnnualABSResidual"]])

	df_OvsP["ObservedGain"] = y_test > 0
	df_OvsP["CorrectDir"] = (df_OvsP.Estimated > 0) == (df_OvsP.Observed > 0)

	keystats.append("\n The total perormace accuracy by observed direction \n")
	keystats.append(df_OvsP[["ObservedGain", "CorrectDir"]].groupby("ObservedGain").mean())

	df_OvsP["ErrorType"]  = ""
	df_OvsP.loc[ np.logical_and(df_OvsP["ObservedGain"], df_OvsP["CorrectDir"]), "ErrorType"] = "CorrectGain"
	df_OvsP.loc[ np.logical_and((~df_OvsP["ObservedGain"]), df_OvsP["CorrectDir"]), "ErrorType"] = "Correctloss"
	
	df_OvsP.loc[ np.logical_and(df_OvsP["ObservedGain"], ~df_OvsP["CorrectDir"]), "ErrorType"] = "FalseLoss"
	df_OvsP.loc[ np.logical_and((~df_OvsP["ObservedGain"]), ~df_OvsP["CorrectDir"]), "ErrorType"] = "FalseGain"
	keystats.append("\n Error types \n")
	keystats.append(df_OvsP.groupby(["ObservedGain", "ErrorType"]).count()["Observed"])


	keystats.append("\n Mean and absmean residuals by change direction \n")
	keystats.append(df_OvsP.groupby("ObservedGain").mean()[["Estimated","Observed", "Residual", "ABSResidual", "ObsGap", "CorrectDir"]])

	# breakpoint()

	# ========== Save the info =========
	fname = f'{ppath}PS09_ModelPerfSummary_exp{exp}.txt'
	f = open(fname,'w')
	for info in keystats:
		f.write("%s\n" %info)
	f.close()


def sitedtb(path, ppath, exp, fpath, vi_df, site_df, df_mod, VIfvi):
	"""
	open the database and pull out the relevant files
	"""
	Scriptinfo = "Stats Exported using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	keystats = [Scriptinfo, gitinfo]



	# +++++ summary +++++
	# Number of sites
	keystats.append("\n Total number of sites \n")
	keystats.append(vi_df.groupby(["site"]).count()["year"].shape[0])
	# mean obs per site
	keystats.append("\n Observations per sites \n")
	keystats.append((vi_df.groupby(["site", "year"]).first().reset_index().groupby("site").count()["year"]+1).mean())
	# total measurments 
	keystats.append("\n Total number of site measurements in the database \n")
	keystats.append(vi_df.groupby(["site", "year"]).first().reset_index().groupby("site").count()["year"].sum()+VIfvi.shape[0])

	keystats.append("\n Total number of all measurements (all gaps) in the database \n")
	keystats.append(vi_df.shape[0])
	keystats.append("\n Total number of Site that can be used for prediction in the database \n")
	keystats.append(VIfvi.shape[0])

	keystats.append("\n First year in database \n")
	keystats.append(vi_df.year.min())

	keystats.append("\n Last year in database \n")
	keystats.append(VIfvi.year.max())

	site_dfM = site_df.loc[df_mod.index]
	site_dfM["Observed"] = df_mod["Observed"]
	site_dfM["Biomass"]  = vi_df.loc[df_mod.index, "biomass"]
	site_dfM["ObsGap"]  = vi_df.loc[df_mod.index, "ObsGap"]
	regions = regionDict()
	# site_dfM
	site_dfM.replace(regions, inplace=True)
	keystats.append("\n Modelled Total number of sites \n")
	keystats.append(site_dfM.groupby(["Plot_ID"]).count()["year"].shape[0])

	keystats.append("\n Modelled Observations per sites \n")
	keystats.append((site_dfM.groupby(["Plot_ID", "year"]).first().reset_index().groupby("Plot_ID").count()["year"]+1).mean())
	# total measurments 
	keystats.append("\n Modelled Total number of site measurements in the database \n")
	keystats.append(site_dfM.groupby(["Plot_ID", "year"]).first().reset_index().groupby("Plot_ID").count()["year"].sum() + VIfvi[VIfvi.year>1990].shape[0])

	keystats.append("\n Modelled Total number of all measurements (all gaps) in the database \n")
	keystats.append(site_dfM.shape[0])

	keystats.append("\n Total number of Site that can be used for prediction valid window in the database \n")
	keystats.append(VIfvi[VIfvi.year>1990].shape[0])

	keystats.append("\n First year in Modelled database \n")
	keystats.append(site_dfM.year.min())

	keystats.append("\n Last year in Modelled database \n")
	keystats.append(VIfvi.year.max())
	# keystats.append(vi_df.shape[0])
	# +++++ Sites included in the models +++++
	# keystats.append("\n Total number of measurements Modelled \n")



	# number of site per region
	keystats.append("\n Total number of sites Modelled by region \n")
	keystats.append(site_dfM.groupby("Region").count()["Plot_ID"])
	# number of observations per site
	# print(site_dfM.groupby("Region").count()["Plot_ID"])
	# print(site_dfM.groupby(["Plot_ID", "year"]).first().reset_index().groupby(["Plot_ID"]).count()["year"].mean())#
	# dfp = site_dfM.groupby(["Plot_ID", "year"]).first().reset_index().groupby(["Region","Plot_ID"]).count()
	# print(dfp.reset_index().groupby("Region").mean())

	# obsgaps per site 
	dfos = site_dfM.groupby(["Plot_ID", "Region"]).count()[["year"]]
	# number of measurments per site per region
	keystats.append("\n measurments per site Modelled by region \n")
	keystats.append(dfos.reset_index().groupby("Region").mean())

	# ========== overall trends in the model data =======
	keystats.append(f"\n The fraction of Modelled sites with increases:\n {(site_dfM['Observed'] > 0).sum() / site_dfM.shape[0]}")
	# sites increasing by region
	inc = (site_dfM.reset_index().set_index(["index", "Region"])["Observed"] > 0).reset_index().groupby("Region").sum() ["Observed"]
	cnt = (site_dfM.reset_index().set_index(["index", "Region"])["Observed"] > 0).reset_index().groupby("Region").count() ["Observed"]
	keystats.append(f"The fraction of Modelled sites with increases by region:\n {inc/cnt}")


	# ========== Save the info =========
	fname = f'{ppath}PS09_DatasetSummary_exp{exp}.txt'
	f = open(fname,'w')
	for info in keystats:
		f.write("%s\n" %info)
	f.close()




	site_dfM["Region"] = site_dfM.Region.astype('category')#.cat.reorder_categories(ks)
	plotter(ppath, exp, site_dfM, inc, cnt)
	breakpoint()
	warn.warn("Do the future sites as well")

# ==============================================================================
def Grouper(col_nms):
	df = pd.DataFrame({"Variable":col_nms})
	# ========== group the vartypes ==========
	sp_groups  = pd.read_csv("./EWS_package/data/raw_psp/SP_groups.csv", index_col=0)
	soils      = pd.read_csv( "./EWS_package/data/psp/modeling_data/soil_properties_aggregated.csv", index_col=0).columns.values
	permafrost = pd.read_csv("./EWS_package/data/psp/modeling_data/extract_permafrost_probs.csv", index_col=0).columns.values
	def _getgroup(VN, species=[], soils = [], permafrost=[]):
		if VN in ["biomass", "stem_density", "ObsGap", "StandAge"]:
			return "Survey"
		elif VN in ["Disturbance", "DisturbanceGap", "Burn", "BurnGap", "DistPassed"]:
			return "Disturbance"
		elif (VN.startswith("Group")) or (VN in species):
			return "Species"
		elif VN.startswith("LANDSAT"):
			return "RS VI"
		elif VN.endswith("30years"):
			return "Climate"
		elif VN in soils:
			return "Soil"
		elif VN in permafrost:
			return "Permafrost"
		else: 
			print(VN)
			breakpoint()
			return "Unknown"

	
	df["VariableGroup"] = df.Variable.apply(_getgroup, 
		species = sp_groups.scientific.values, soils=soils, permafrost=permafrost).astype("category")
	return df

def plotter(ppath, exp, site_dfM, inc, cnt):

	plt.rcParams.update({
		'axes.titleweight':"bold", 
		'axes.titlesize':12, 
		"axes.labelweight":"bold", 
		'axes.titlelocation': 'left'})

	font = {'family' : 'normal',
	        'weight' : 'bold', #,
	        'size'   : 12}
	mpl.rc('font', **font)
	sns.set_style("whitegrid")

	# ========== Create the figure ==========
	fig  = plt.figure(constrained_layout=True, figsize=(16,19))
	spec = gridspec.GridSpec(ncols=1, nrows=4, figure=fig)

	# +++++ the plot of the number of sites +++++
	ax0  = fig.add_subplot(spec[0, :])
	# breakpoint()
	sns.countplot(x= "Region", data=site_dfM, ax=ax0)# y ="Plot_ID",
	ax0.set_xticklabels(ax0.get_xticklabels(), rotation=15, horizontalalignment='right')
	ax0.set_title("a)")
	ax0.set_ylabel("No. of Modelled Measurements")
	ax0.set_xlabel("")

	# +++++ Mean Biomass at observation +++++
	ax1  = fig.add_subplot(spec[1, :])
	sns.violinplot(y = "Biomass", x="Region", data = site_dfM.groupby(["Plot_ID", "year"]).first(), ax=ax1)
	ax1.set_ylim(0, 1000)
	ax1.set_title("b)")
	unt = r"t $ha^{-1}$"
	ax1.set_ylabel(f"Biomass ({unt})")
	ax1.set_xlabel("")
	ax1.set_xticklabels(ax1.get_xticklabels(), rotation=15, horizontalalignment='right')

	ax2  = fig.add_subplot(spec[2, :])
	data = 1-(inc / cnt)
	# data.plot.bar(ax=ax2)
	sns.barplot(data=pd.DataFrame(data).reset_index(), x="Region", y="Observed", ax=ax2)
	ax2.set_title("c)")
	ax2.set_ylabel("Loss Fraction")
	ax2.set_xlabel("")
	ax2.set_xticklabels(ax2.get_xticklabels(), rotation=15, horizontalalignment='right')
	
	# Mean annual delta biomass
	ax3  = fig.add_subplot(spec[3, :])
	site_dfM["AnnualBiomassChange"] = site_dfM["Observed"] / site_dfM["ObsGap"]
	sns.violinplot(y = "AnnualBiomassChange", x="Region", data = site_dfM, ax=ax3)
	# ax1.set_ylim(0, 1000)
	unit = r"t $ha^{-1} yr^{-1}$"
	ax3.set_ylim(-20, 20)
	ax3.set_title("d)")
	ax3.set_ylabel(f"Biomass Change Rate ({unit})")
	ax3.set_xlabel("")
	ax3.set_xticklabels(ax3.get_xticklabels(), rotation=15, horizontalalignment='right')


	# ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS09_RegionalOverview_exp{exp}" 
	for ext in [".png", ]:#".pdf",
		plt.savefig(fnout+ext)#, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()

	ipdb.set_trace()
	breakpoint()

# ==============================================================================
def fpred(path, exp, years, var = "DeltaBiomass",
	fpath    = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/", 
	maxdelta = 30):
	"""
	function to predict future biomass
	args:
	path:	str to files
	exp:	in of experiment
	years:  list of years to predict 
	"""
	# warn.warn("\nTo DO: Implemnt obsgap filtering")
	# ========== Load the variables ==========
	site_df = pd.read_csv(f"{fpath}SiteInfo_AllSampleyears_FutureBiomass.csv", index_col=0)
	vi_df   = pd.read_csv(f"{fpath}VI_df_AllSampleyears_FutureBiomass.csv", index_col=0)
	setup   = pd.read_csv(f"{path}{exp}/Exp{exp}_setup.csv", index_col=0)
	pvar    = setup.loc["predvar"].values[0]
	if type(pvar) == float:
		# deal with the places i've alread done
		pvar = "lagged_biomass"

	# ========== Loop over the model versions
	est_list = []
	for ver in tqdm(range(10)):
		fn_mod = f"{path}{exp}/models/XGBoost_model_exp{exp}_version{ver}.dat"
		if not os.path.isfile(fn_mod):
			# missing file
			continue
		# ========== Load the run specific params ==========
		model  = pickle.load(open(f"{fn_mod}", "rb"))
		fname  = glob.glob(f"{path}{exp}/Exp{exp}_*_vers{ver:02}_PermutationImportance.csv")
		feat   = pd.read_csv(fname[0], index_col=0)["Variable"].values

		# ========== Make a dataframe ==========
		dfout = site_df.loc[:, ["Plot_ID", "Longitude", "Latitude", "Region", "year"]].copy()
		dfout["Version"]  = ver
		dfout["Original"] = vi_df["biomass"].values

		# ========== Make a filter for bad latitudes ==========
		# dfout.loc[dfout["Longitude"] == 0, ["Longitude", "Latitude"]] = np.NaN
		# dfout.loc[dfout["Longitude"] < -180, ["Longitude", "Latitude"]] = np.NaN
		# dfout.loc[dfout["Longitude"] >  180, ["Longitude", "Latitude"]] = np.NaN
		# dfout.loc[dfout["Latitude"] <= 0, ["Longitude", "Latitude"]] = np.NaN
		# dfout.loc[dfout["Latitude"] >  90, ["Longitude", "Latitude"]] = np.NaN

		for yr in years:
			dfoutC = dfout.copy()
			# ========== Check for missing columns ==========
			fcheck = []
			for ft in feat:	
				fcheck.append(ft not in vi_df.columns)

			if any(fcheck):
				print("Fixing missing columns")
				vi_dfX = pd.read_csv(f"{fpath}VI_df_AllSampleyears.csv", index_col=0)
				for clnm in feat[fcheck]:
					vi_df[clnm] = vi_dfX.loc[:, ["site", clnm]].groupby("site").median().loc[vi_df.site]
				vi_df.to_csv(f"{fpath}VI_df_AllSampleyears_FutureBiomass.csv")
			# ========== pull out the variables and apply transfors ==========
			dfX = vi_df.loc[:, feat].copy()
			# ========== pull out the variables and apply transfors ==========
			# try:
			# 	dfX = vi_df.loc[:, feat].copy()	
			# except Exception as err:
			# 	warn.warn(str(err))
			# 	# vi_dfo = pd.read_csv(f"{fpath}VI_df_AllSampleyears_ObsBiomass.csv", index_col=0)
			# 	breakpoint()
			if not type(setup.loc["Transformer"].values[0]) == float:
				warn.warn("Not implemented yet")
				breakpoint()

			# ========== calculate the obsgap ==========
			if "ObsGap" in feat:
				dfX["ObsGap"] = yr - site_df["year"].values

			# ========== Perform the prediction ==========
			est = model.predict(dfX.values)
			if not type(setup.loc["yTransformer"].values[0]) == float:
				warn.warn("Not implemented yet")
				breakpoint()

			# ========== Convert to common forms ==========
			if pvar == "lagged_biomass":
				breakpoint()
			elif pvar == 'Delta_biomass':
				dfoutC[f"Biomass"]      = vi_df["biomass"].values + est
				dfoutC[f"DeltaBiomass"] = est
				# breakpoint()
			elif pvar == 'Obs_biomass':
				# dfout[f"BIO_{yr}"]   = est
				# dfout[f"DELTA_{yr}"] = est - vi_df["biomass"].values
				dfoutC[f"Biomass"]      = est
				dfoutC[f"DeltaBiomass"] = est - vi_df["biomass"].values
			
			dfoutC["time"] = pd.Timestamp(f"{yr}-12-31")
			dfoutC.loc[(dfoutC.time.dt.year - dfoutC.year) > maxdelta, ['Biomass', 'DeltaBiomass']] = np.NaN
			est_list.append(dfoutC)

	df = pd.concat(est_list)
	# df["longitude"] = pd.cut(df["Longitude"], lons, labels=bn.move_mean(lons, 2)[1:])
	# df["latitude"]  = pd.cut(df["Latitude" ], lats, labels=bn.move_mean(lats, 2)[1:])
	df["Region"] = site_df.loc[df.index, "Region"]
	regions = regionDict()
	df.replace(regions, inplace=True)


	df["ObsGap"] = df.time.dt.year - df.year
	df = df[df.ObsGap <= 30]

	# ========== do the direction stuff ==========
	dft = df[[var, "Plot_ID", "year", "ObsGap"]].copy()
	dft[var] = dft[var]>=0
	df["ModelAgreement"] = (dft.groupby(["Plot_ID", "year", "ObsGap"]).transform("sum") - 5) / 5
	# breakpoint()
	return df


def loadperm(exp, path):

	perm = []
	for ver in range(10):
		fname = glob.glob(f"{path}{exp}/Exp{exp}_*{ver}_PermutationImportance.csv")[0]
		dfin = pd.read_csv(fname, index_col=0)
		dfin["Version"] = ver
		perm.append(dfin)
	# ========== group the vartypes ==========
	sp_groups  = pd.read_csv("./EWS_package/data/raw_psp/SP_groups.csv", index_col=0)
	soils      = pd.read_csv( "./EWS_package/data/psp/modeling_data/soil_properties_aggregated.csv", index_col=0).columns.values
	permafrost = pd.read_csv("./EWS_package/data/psp/modeling_data/extract_permafrost_probs.csv", index_col=0).columns.values
	def _getgroup(VN, species=[], soils = [], permafrost=[]):
		if VN in ["biomass", "stem_density", "ObsGap", "StandAge"]:
			return "Survey"
		elif VN in ["Disturbance", "DisturbanceGap", "Burn", "BurnGap", "DistPassed"]:
			return "Disturbance"
		elif (VN.startswith("Group")) or (VN in species):
			return "Species"
		elif VN.startswith("LANDSAT"):
			return "RS VI"
		elif VN.endswith("30years"):
			return "Climate"
		elif VN in soils:
			return "Soil"
		elif VN in permafrost:
			return "Permafrost"
		else: 
			print(VN)
			breakpoint()
			return "Unknown"

	df = pd.concat(perm).reset_index()
	
	df["VariableGroup"] = df.Variable.apply(_getgroup, 
		species = sp_groups.scientific.values, soils=soils, permafrost=permafrost).astype("category")
	return df

def load_OBS(ofn):
	df_in = pd.read_csv(ofn, index_col=0)
	df_in["experiment"] = int(ofn.split("/")[-2])
	df_in["experiment"] = df_in["experiment"].astype("category")
	df_in["version"]    = float(ofn.split("_vers")[-1][:2])
	return df_in

def regionDict():
	regions = ({
		'BC': "British Columbia", 
		'AB': "Alberta", 
		'SK': "Saskatchewan", 
		'MB': "Manitoba", 
		'ON': "Ontario", 
		'QC': "Quebec", 
		'NL': "Newfoundland and Labrador", 
		'NB': "New Brunswick", 
		'NS': "Nova Scotia", 
		'YT': "Yukon", 
		'NWT':"Northwest Territories", 
		'CAFI':"Alaska"
		})
	return regions


def experiments(ncores = -1):
	""" Function contains all the infomation about what experiments i'm 
	performing """
	expr = OrderedDict()

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
# ==============================================================================
if __name__ == '__main__':
	main()