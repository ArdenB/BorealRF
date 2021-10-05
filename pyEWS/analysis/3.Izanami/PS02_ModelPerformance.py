"""
Boreal EWS PSP data anlysis 
 
Script to  make individaul psps plots  
"""

# ==============================================================================

__title__ = "Examining the biomass predictions"
__author__ = "Arden Burrell"
__version__ = "v1.0(15.04.2021)"
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
import xarray as xr
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
from itertools import product
import pickle


# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf
import myfunctions.benchmarkfunctions as bf

import xgboost as xgb
import cartopy.crs as ccrs

import cartopy.feature as cpf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import shap


# ========== Import ml packages ==========
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.inspection import permutation_importance
from sklearn import metrics as sklMet
# from sklearn.utils import shuffle
# from scipy.stats import spearmanr
# from scipy.cluster import hierarchy

# ==============================================================================
def main():
	# ========== Get the file names and open the files ==========
	formats = None
	path  = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"
	cf.pymkdir(path+"plots/")
	ppath = "./pyEWS/analysis/3.Izanami/Figures/PS02/"
	cf.pymkdir(ppath)
	# +++++ the model Infomation +++++
	setup_fnames = glob.glob(path + "*/Exp*_setup.csv")
	df_setup     = pd.concat([pd.read_csv(sfn, index_col=0).T for sfn in setup_fnames], sort=True)
	
	# +++++ the final model results +++++
	mres_fnames = glob.glob(path + "*/Exp*_Results.csv")
	df_mres = pd.concat([fix_results(mrfn) for mrfn in mres_fnames], sort=True)
	df_mres["TotalTime"]  = df_mres.TotalTime / pd.to_timedelta(1, unit='m')
	df_mres, keys = Experiment_name(df_mres, df_setup, var = "experiment")


	# ========= Load in the observations ==========
	OvP_fnames = glob.glob(path + "*/Exp*_OBSvsPREDICTED.csv")
	df_OvsP    = pd.concat([load_OBS(ofn) for ofn in OvP_fnames], sort=True)
	
	gclass     = glob.glob(path + "*/Exp*_OBSvsPREDICTEDClas_y_test.csv")
	df_clest   = pd.concat([load_OBS(mrfn) for mrfn in gclass], sort=True)

	branch     = glob.glob(path + "*/Exp*_BranchItteration.csv")
	df_branch  = pd.concat([load_OBS(mrfn) for mrfn in branch], sort=True)

	# experiments = [400]
	# exp = 402
	exp = 434
	FigureModelPerfomance(df_setup, df_mres, keys, df_OvsP, df_clest, df_branch, path, exp, ppath)
	FigureModelPerfomanceV2(df_setup, df_mres, keys, df_OvsP, df_clest, df_branch, path, exp, ppath)
	breakpoint()
	FigureModelPerfomancePresentation(df_setup, df_mres, keys, df_OvsP, df_clest, df_branch, path, exp, ppath)
	breakpoint()
	# ========== old plots that might end up in supplementary material ==========
	oldplots(df_setup, df_mres, keys, df_OvsP, df_clest, df_branch, path, ppath)

# ================================================================================
def FigureModelPerfomanceV2(
	df_setup, df_mres, keys, df_OvsP, 
	df_clest, df_branch, path, exp, ppath, years=[2020], huex = "VariableGroup",	
	lons = np.arange(-170, -50.1,  0.5),
	lats = np.arange(  42,  70.1,  0.5), textsize=12):
	"""
	Build model performace figure
	"""
	# ========== Create the figure ==========
	sns.set_style("whitegrid")
	font = ({'weight' : 'bold', 'size'   : textsize})
	mpl.rc('font', **font)
	plt.rcParams.update({'axes.titleweight':"bold", 
		"axes.labelweight":"bold", 'axes.titlesize':textsize, 'axes.titlelocation': 'left',})
	# plt.rcParams.update({'axes.titleweight':"bold", })

	# ========== Create the map projection ==========
	map_proj = ccrs.LambertConformal(central_longitude=lons.mean(), central_latitude=lats.mean())
	df = Translator(df_setup, df_mres, keys, df_OvsP, df_clest, df_branch, [exp], path)

	# ========== Convert to a dataarray ==========
	# ds = gridder(path, exp, years, fpred(path, exp, years), lats, lons)
	
	pred = OrderedDict()
	pred['Delta_biomass'] = ({
		"obsvar":"ObsDelta",
		"estvar":"EstDelta", 
		"limits":(-300, 300),
		"Resname":"Residual", 
		"gap":10
		})
	pred["Biomass"] = ({
		"obsvar":"Observed",
		"estvar":"Estimated", 
		"limits":(0, 1000), 
		"Resname":"Residual"
		})

	# ========== Create the figure ==========
	fig  = plt.figure(constrained_layout=True, figsize=(18,18))
	spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)#, width_ratios=[6,1,6,1], )#height_ratios=[5, 10, 5]


	ax1 = fig.add_subplot(spec[0, 0])
	_confusion_plots(df, keys, exp, fig, ax1, pred, df_setup, norm=False, title="a)")

	ax2 = fig.add_subplot(spec[0, 1])
	_confusion_plots(df, keys, exp, fig, ax2, pred, df_setup, title="b)")
	# Temporal_predictability(ppath, [exp], df_setup, df, keys,  fig, ax1, "Residual", hue="ChangeDirection",
	# 	va = "Residual", CI = "QuantileInterval", single=False)
	
	# ========== Create the figure ==========
	df, ver, hueord = _ImpOpener(path, [exp], AddFeature=True)
	ax3  = fig.add_subplot(spec[1, :])
	# g = sns.catplot(x="Variable", y="PermutationImportance", hue=huex, 	dodge=False, data=df, palette= hueord["cmap"], col="experiment",  ax=ax3)
	g = sns.boxplot(
		x="Variable", y="PermutationImportance", hue=huex, 
		dodge=False, data=df.loc[df.Count >= 7], palette= hueord["cmap"], ax=ax3)
	ax3.set_yscale('log')
	g.set_xticklabels(g.get_xticklabels(), rotation=15, horizontalalignment='right')
	ax3.set_title(f"c)")
	# breakpoint()
	# ax2 = fig.add_subplot(spec[1, 0])
	# _confusion_plots(df, keys, exp, fig, ax2, pred, df_setup)

	# ax3 = fig.add_subplot(spec[1, 2:])
	# _regplot(df, ax3, fig)

	
	# ax4 = fig.add_subplot(spec[2, 2:], projection= map_proj)
	# _simplemapper(ds, "MedianDeltaBiomass", fig, ax4, map_proj, 0, "Delta Biomass", lats, lons,  dim="Version")

	# # ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS02_PaperFig03_ModelPerformanceV2" 
	for ext in [".png", ".pdf"]:#".pdf",
		plt.savefig(fnout+ext)#, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()

	breakpoint()


def FigureModelPerfomancePresentation(
	df_setup, df_mres, keys, df_OvsP, 
	df_clest, df_branch, path, exp, ppath, years=[2020], 	
	lons = np.arange(-170, -50.1,  0.5),
	lats = np.arange(  42,  70.1,  0.5), textsize=24):
	"""
	Build model performace figure. This makes each subplot as its own figure so i can 
	use them in presentations
	"""
	# ========== Create the figure ==========
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':textsize})
	font = {'weight' : 'bold', #,
	        'size'   : textsize}
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	plt.rcParams.update({'axes.titleweight':"bold", "axes.labelweight":"bold"})

	# ========== Create the map projection ==========
	map_proj = ccrs.LambertConformal(central_longitude=lons.mean(), central_latitude=lats.mean())
	df = Translator(df_setup, df_mres, keys, df_OvsP, df_clest, df_branch, [exp], path)

	# ========== Convert to a dataarray ==========
	ds = gridder(path, exp, years, fpred(path, exp, years), lats, lons)
	
	pred = OrderedDict()
	pred['Delta_biomass'] = ({
		"obsvar":"ObsDelta",
		"estvar":"EstDelta", 
		"limits":(-300, 300),
		"Resname":"Residual", 
		"gap":10
		})
	pred["Biomass"] = ({
		"obsvar":"Observed",
		"estvar":"Estimated", 
		"limits":(0, 1000), 
		"Resname":"Residual"
		})

	# ========== Make the plots ==========
	fig, ax = plt.subplots(constrained_layout=True, figsize=(13,7))
	Temporal_predictability(ppath, [exp], df_setup, df, keys,  fig, 
		ax, "Residual", hue="ChangeDirection",	va = "Residual", 
		CI = "QuantileInterval", single=False)
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS02_PaperFig03_ModelPerformance_tempgapperf_CD" 
	for ext in [".png", ".pdf"]:#".pdf",
		plt.savefig(fnout+ext)#, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()
	
	fig, ax = plt.subplots(constrained_layout=True, figsize=(13,7))
	Temporal_predictability(ppath, [exp], df_setup, df, keys,  fig, 
		ax, "Residual", hue="experiment",	va = "Residual", 
		CI = "QuantileInterval", single=False)
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS02_PaperFig03_ModelPerformance_tempgapperf" 
	for ext in [".png", ".pdf"]:#".pdf",
		plt.savefig(fnout+ext)#, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()

	# ========== Make the plots ==========
	fig, ax = plt.subplots(constrained_layout=True, figsize=(13,11))
	_confusion_plots(df, keys, exp, fig, ax, pred, df_setup)
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS02_PaperFig03_ModelPerformance_NormConf" 
	for ext in [".png", ".pdf"]:#".pdf",
		plt.savefig(fnout+ext)#, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()

	# ========== Make the plots ==========
	fig, ax = plt.subplots(constrained_layout=True, figsize=(13,11))
	_confusion_plots(df, keys, exp, fig, ax, pred, df_setup, norm=False)
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS02_PaperFig03_ModelPerformance_RawConf" 
	for ext in [".png", ".pdf"]:#".pdf",
		plt.savefig(fnout+ext)#, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()


	# ax3 = fig.add_subplot(spec[1, 2:])
	fig, ax = plt.subplots(constrained_layout=True, figsize=(13,7))
	_regplot(df, ax, fig)
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS02_PaperFig03_ModelPerformance_RegionProDistFunc" 
	for ext in [".png", ".pdf"]:#".pdf",
		plt.savefig(fnout+ext)#, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()


	# fig, ax = plt.subplots(#constrained_layout=True, 
	# 	subplot_kw={'projection':map_proj}, figsize=(12,7))
	# _simplemapper(ds, "MedianDeltaBiomass", fig, ax, map_proj, 0, 
	# 	"Delta Biomass", lats, lons,  dim="Version")

	# # # ========== Save tthe plot ==========
	# print("starting save at:", pd.Timestamp.now())
	# fnout = f"{ppath}PS02_PaperFig03_ModelPerformance_Map" 
	# for ext in [".png", ".pdf"]:#".pdf",
	# 	plt.savefig(fnout+ext)#, dpi=130)
	
	# plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
	# 	__title__, __version__,  __author__, pd.Timestamp.now())
	# gitinfo = cf.gitmetadata()
	# cf.writemetadata(fnout, [plotinfo, gitinfo])
	# plt.show()

	breakpoint()



def FigureModelPerfomance(
	df_setup, df_mres, keys, df_OvsP, 
	df_clest, df_branch, path, exp, ppath, years=[2020], 	
	lons = np.arange(-170, -50.1,  0.5),
	lats = np.arange(  42,  70.1,  0.5), textsize=14):
	"""
	Build model performace figure
	"""
	# ========== Create the figure ==========
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':textsize})
	font = ({'weight' : 'bold', 'size'   : textsize})
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	plt.rcParams.update({'axes.titleweight':"bold", "axes.labelweight":"bold"})

	# ========== Create the map projection ==========
	map_proj = ccrs.LambertConformal(central_longitude=lons.mean(), central_latitude=lats.mean())
	df = Translator(df_setup, df_mres, keys, df_OvsP, df_clest, df_branch, [exp], path)

	# ========== Convert to a dataarray ==========
	# ds = gridder(path, exp, years, fpred(path, exp, years), lats, lons)
	
	pred = OrderedDict()
	pred['Delta_biomass'] = ({
		"obsvar":"ObsDelta",
		"estvar":"EstDelta", 
		"limits":(-300, 300),
		"Resname":"Residual", 
		"gap":10
		})
	pred["Biomass"] = ({
		"obsvar":"Observed",
		"estvar":"Estimated", 
		"limits":(0, 1000), 
		"Resname":"Residual"
		})

	# ========== Create the figure ==========
	fig  = plt.figure(constrained_layout=True, figsize=(16,20))
	spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)#, width_ratios=[5,1,5,5], height_ratios=[5, 10, 5])


	ax1 = fig.add_subplot(spec[0, :])
	Temporal_predictability(ppath, [exp], df_setup, df, keys,  fig, ax1, "Residual", hue="ChangeDirectionAll",
		va = "Residual", CI = "QuantileInterval", single=False, title="a)")

	ax2 = fig.add_subplot(spec[1, :])
	Temporal_predictability(ppath, [exp], df_setup, df, keys,  fig, ax2, "Residual", hue="ChangeDirectionNorm",
		va = "Residual", CI = "QuantileInterval", single=False, title="b)")
	# ax2 = fig.add_subplot(spec[1, 0])
	# _confusion_plots(df, keys, exp, fig, ax2, pred, df_setup)

	ax3 = fig.add_subplot(spec[2, :])
	_regplot(df, ax3, fig, title="c)")

	
	# ax4 = fig.add_subplot(spec[2, 2:], projection= map_proj)
	# _simplemapper(ds, "MedianDeltaBiomass", fig, ax4, map_proj, 0, "Delta Biomass", lats, lons,  dim="Version")

	# # ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS02_PaperFig03_ModelPerformanceLimits" 
	for ext in [".png", ".pdf"]:#".pdf",
		plt.savefig(fnout+ext)#, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()

	breakpoint()

def _simplemapper(ds, vas, fig, ax, map_proj, indtime, title, lats, lons,  dim="Version"):
	f = ds[vas].mean(dim=dim).isel(time=indtime).plot(
		x="longitude", y="latitude", #col="time", col_wrap=2, 
		transform=ccrs.PlateCarree(), 
		cbar_kwargs={"pad": 0.015, "shrink":0.95},#, "extend":extend}
		# subplot_kws={'projection': map_proj}, 
		# size=6,	aspect=ds.dims['longitude'] / ds.dims['latitude'],  
		ax=ax)
	# breakpoint()
	# for ax in p.axes.flat:
	ax.set_extent([lons.min()+10, lons.max()-5, lats.min()-13, lats.max()])
	ax.gridlines()
	coast = cpf.GSHHSFeature(scale="intermediate")
	ax.add_feature(cpf.LAND, facecolor='dimgrey', alpha=1, zorder=0)
	ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
	ax.add_feature(coast, zorder=101, alpha=0.5)
	ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
	ax.add_feature(cpf.RIVERS, zorder=104)
	ax.add_feature(cpf.BORDERS, linestyle='--', zorder=102)


def _regplot(df, ax, fig, title=""):
	regions   = regionDict()
	dfx = df.copy()
	dfx["Region"].replace(regions, inplace=True)
	g = sns.violinplot( y="Residual", x="Region", hue="ChangeDirection", data=dfx, ax=ax, split=True, cut=0)
	# ax.set_ylim((-500, 500))
	g.set_xticklabels(g.get_xticklabels(), rotation=15, horizontalalignment='right')
	g.set(ylim=(-400, 400))
	ax.set_title(f"{title}", loc= 'left')


def oldplots(df_setup, df_mres, keys, df_OvsP, df_clest, df_branch, path, ppath, textsize=24):
	# ========== Create the figure ==========
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':textsize})
	font = ({
		'weight' : 'bold', #,
		'size'   : textsize})
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	plt.rcParams.update({'axes.titleweight':"bold", "axes.labelweight":"bold"})

	experiments = [400, 401, 402]
	df = Translator(df_setup, df_mres, keys, df_OvsP, df_clest, df_branch, experiments, path)

	pred = OrderedDict()
	pred["DeltaBiomass"] = ({
		"obsvar":"ObsDelta",
		"estvar":"EstDelta", 
		"limits":(-300, 300),
		"Resname":"Residual"
		})
	pred["Biomass"] = ({
		"obsvar":"Observed",
		"estvar":"Estimated", 
		"limits":(0, 1000), 
		"Resname":"Residual"
		})
	# breakpoint()

	for exp, var in product(experiments, pred):
		print(exp, var)
		if var == "Biomass":
			fig, ax = plt.subplots(1, 1, figsize=(14,6))
			Temporal_predictability(ppath, [exp], df_setup, df, keys,  fig, ax, var, va=pred[var]['Resname'])	

		fig, ax = plt.subplots(1, 1, figsize=(14,6))
		pdfplot(ppath, df, exp, keys, fig, ax, pred[var]['obsvar'], pred[var]['estvar'], var, pred[var]['limits'])

	# vi_df, fcount = VIload()

	warn.warn("TThe following plots have not been adjusted for different variable types")
	breakpoint()
	
	
	for var, ylab, ylim in zip(["R2", "TotalTime", "colcount"], [r"$R^{2}$", r"$\Delta$t (min)", "# Predictor Vars."], [(0., 1.), None, None]):
		fig, ax = plt.subplots(1, 1, figsize=(15,13))
		branchplots(exp, df_mres, keys, var, ylab, ylim,  fig, ax)


	# breakpoint()
	splts = np.arange(-1, 1.05, 0.10)
	splts[ 0] = -1.00001
	splts[-1] = 1.00001

	fig, ax = plt.subplots(1, 1, figsize=(15,13))
	confusion_plots(path, df_mres, df_setup, df_OvsP, keys,  exp, fig, ax, 
		inc_class=False, split=splts, sumtxt="", annot=False, zline=True)

	splts = np.arange(-1, 1.05, 1.0)
	splts[ 0] = -1.00001
	splts[-1] = 1.00001
	fig, ax = plt.subplots(1, 1, figsize=(15,13))
	confusion_plots(path, df_mres, df_setup, df_OvsP, keys,  exp, fig, ax, 
		inc_class=False, split=splts, sumtxt="", annot=True, zline=True)
	# breakpoint()
	
	breakpoint()

# ==============================================================================
def _confusion_plots(
	df, keys, exp, fig, ax, pred, df_setup,  #split,
	inc_class=False, sumtxt="", annot=False, zline=True, 
	num=0, cbar=True, mask=True, norm=True, title=""):

	# ========== Create the figure ==========
	"""Function to create and plot the confusion matrix"""
	setup   = df_setup.loc[f"Exp{exp}"]
	
	obsvar  = pred[setup["predvar"]]["obsvar"] 
	estvar  = pred[setup["predvar"]]["estvar"] 
	gap     = pred[setup["predvar"]]["gap"]
	# pred[setup["predvar"]]
	split   = np.arange(pred[setup["predvar"]]['limits'][0] - gap, pred[setup["predvar"]]['limits'][1]+gap+1, gap)
	expsize = len(split) -1 # df_class.experiment.unique().size

	labels = [f"{lb}" for lb in bn.move_mean(split, 2)[1:]]
	labels[ 0] = f"<{pred[setup['predvar']]['limits'][0]}"
	labels[-1] = f">{pred[setup['predvar']]['limits'][1]}"
	colconv = {cl:sp for cl, sp in zip([int(i) for i in np.arange(expsize)], labels)}
	
	# ========== Get the observed and estimated values ==========
	df_c         = df.loc[:, [obsvar,estvar]].copy()
	# +++++ tweak splits to match the min and max bounds of the data
	if split[0] > df_c.values.min() - 1:
		split[0]  = df_c.values.min() - 1
	
	# +++++ solve some issues +++++
	if split[-1] < df_c.values.max() + 1:
		split[-1] = df_c.values.max() + 1
	# breakpoint()
	df_c[obsvar] = pd.cut(df_c[obsvar], split, labels=np.arange(expsize))
	df_c[estvar] = pd.cut(df_c[estvar], split, labels=np.arange(expsize))
	# df_set       = dfS[dfS.experiment == keys[exp]]

	if any(df_c[estvar].isnull()):
		breakpoint()
		df_c = df_c[~df_c[estvar].isnull()]

	if any(df_c[obsvar].isnull()):
		breakpoint()
		df_c = df_c[~df_c[obsvar].isnull()]

	try:
		sptsze = split.size
	except:
		sptsze = len(split)
	
	df_c.sort_values(obsvar, axis=0, ascending=True, inplace=True)
	print(exp, sklMet.accuracy_score(df_c[obsvar], df_c[estvar]))
	
	# ========== set the colorbar ==========
	if norm:
		normalize='true'
		cmap = mpc.ListedColormap(palettable.matplotlib.Inferno_20_r.mpl_colors)
	else:
		normalize=None
		cmap = mpc.ListedColormap(palettable.matplotlib.Inferno_17_r.mpl_colors)
	# ========== Calculate the confusion matrix ==========
	# \\\ confustion matrix  observed (rows), predicted (columns), then transpose and sort
	df_cm  = pd.DataFrame(
		sklMet.confusion_matrix(df_c[obsvar], df_c[estvar],  
			labels=df_c[obsvar].cat.categories,  normalize=normalize),  
		index = [int(i) for i in np.arange(expsize)], 
		columns = [int(i) for i in np.arange(expsize)]).T.sort_index(ascending=False)

	df_cm.rename(colconv, axis='columns', inplace=True)
	df_cm.rename(colconv, inplace=True)
	# breakpoint()

	cmap = mpc.ListedColormap(palettable.matplotlib.Inferno_20_r.mpl_colors)
	if mask:
		#+++++ remove 0 values +++++
		df_cm.replace(0, np.NaN, inplace=True)
		# breakpoint()

	if annot:
		ann = df_cm.round(3)
	else:
		ann = False

	if norm:
		g = sns.heatmap(
			df_cm, annot=ann, vmin=0, vmax=1, ax = ax, 
			cbar=cbar, square=True, cmap=cmap, cbar_kws={"shrink":0.65},
			)
	else:
		print(f"Checking nanmax counts {bn.nanmax(df_cm)}")
		g = sns.heatmap(df_cm, annot=ann,  
			ax = ax, cbar=cbar, square=True, 
			cmap=cmap, norm=LogNorm(vmin=1, vmax=10000,),
			cbar_kws={"extend":"max", "shrink":0.65})
	ax.plot(np.flip(np.arange(expsize+1)), np.arange(expsize+1), "darkgrey", alpha=0.75)


	# 	# ========== Calculate the zero lines ==========
	if zline:
		# ========== Calculate the zero line location ==========
		((x0),) = np.where(np.array(split) < 0)
		x0a = x0[-1]
		((x1),) = np.where(np.array(split) >=0)
		x1a = x1[0]
		zeroP =  x0a + (0.-split[x0a])/(split[x1a]-split[x0a])
		# breakpoint()
		# ========== Add the cross hairs ==========
		ax.axvline(x=zeroP, alpha =0.5, linestyle="--", c="darkgrey", zorder=101)
		ax.axhline(y=zeroP, alpha =0.5, linestyle="--", c="darkgrey", zorder=102)


	# ax.set_title(f"{exp}-{keys[exp]} $R^{2}$ {df_set.R2.mean()}", loc= 'left')
	delt = r"$\Delta$"
	ax.set_xlabel(f'Observed {delt}Biomass')
	ax.set_ylabel(f'Predicted {delt}Biomass')
	ax.set_title(f"{title}")

	
def pdfplot(ppath, df, exp, keys, fig, ax, obsvar, estvar, var, clip, single=True):
	""" Plot the probability distribution function """
	dfin = df[df.experiment == exp]
	dfin = pd.melt(dfin[[obsvar, estvar]])
	# breakpoint()
	g = sns.kdeplot(data=dfin, x="value", hue="variable", fill=True, ax=ax, clip=clip)#clip=(-1., 1.)
	ax.set_xlabel(var)
	if single:
		plt.title(f'{exp} - {keys[exp]}', fontweight='bold')
		plt.tight_layout()

		# ========== Save tthe plot ==========
		print("starting save at:", pd.Timestamp.now())
		fnout = f"{ppath}PS02_{exp}_{var}_ProbDistFunc" 
		for ext in [".png"]:#".pdf",
			plt.savefig(fnout+ext, dpi=130)
		
		plotinfo = "PLOT INFO: PDF plots made using %s:v.%s by %s, %s" % (
			__title__, __version__,  __author__, pd.Timestamp.now())
		gitinfo = cf.gitmetadata()
		cf.writemetadata(fnout, [plotinfo, gitinfo])
		plt.show()


def branchplots(exp, df_mres, keys, var, ylab, ylim,  fig, ax):

	# +++++ R2 plot +++++
	df_set = df_mres[df_mres.experiment == keys[exp]]
	# breakpoint()
	sns.barplot(y=var, x="version", data=df_set,  ax=ax, ci=None)
	ax.set_xlabel("")
	ax.set_ylabel(ylab)
	ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
	if not ylim is None:
		ax.set(ylim=ylim)
	
	plt.show()


def confusion_plots(path, df_mres, df_setup, df_OvsP, keys, exp, fig, ax,
	inc_class=False, split=None, sumtxt="", annot=False, zline=True, num=0, cbar=True, 
	mask=True):
	"""Function to create and plot the confusion matrix"""

	# ========== Check if i needed to pull out the splits ==========
	if split is None:
		splitstr = df_setup.splits.loc["Exp200"]
		split = []
		for sp in splitstr.split(", "):
			if not sp == "":
				split.append(float(sp))

	# ========== Get the observed values ==========
	df_class = df_OvsP.copy()
	expsize  = len(split) -1 # df_class.experiment.unique().size
	df_class["Observed"]  =  pd.cut(df_class["Observed"], split, labels=np.arange(expsize))
	df_class["Estimated"] =  pd.cut(df_class["Estimated"], split, labels=np.arange(expsize))
	df_set = df_mres[df_mres.experiment == keys[exp]]
	# ========== Pull out the classification only accuracy ==========
	expr   = OrderedDict()
	cl_on  = OrderedDict() #dict to hold the classification only 
	# for num, expn in enumerate(experiments):
	if (exp // 100 in [1, 3, 4]):# or exp // 100 == 3:
		expr[exp] = keys[exp]
	elif exp // 100 == 2:
		expr[exp] = keys[exp]
		if inc_class:

			# +++++ load in the classification only +++++
			OvP_fn = glob.glob(path + "%d/Exp%d*_OBSvsPREDICTEDClas_y_test.csv"% (exp,  exp))
			df_OvP  = pd.concat([load_OBS(ofn) for ofn in OvP_fn])

			df_F =  df_class[df_class.experiment == exp].copy()
			for vr in df_F.version.unique():
				dfe = (df_OvP[df_OvP.version == vr]).reindex(df_F[df_F.version==vr].index)
				df_F.loc[df_F.version==vr, "Estimated"] = dfe.class_est
			# +++++ Check and see i haven't added any null vals +++++
			if (df_F.isnull()).any().any():
				breakpoint()
			# +++++ Store the values +++++
			cl_on[exp+0.1] = df_F
			expr[ exp+0.1] = keys[exp]+"_Class"
	else:
		breakpoint()

	try:
		sptsze = split.size
	except:
		sptsze = len(split)

	# ========== Pull out the data for each experiment ==========
	if (exp % 1) == 0.:
		df_c = df_class[df_class.experiment == exp]
	else: 
		df_c = cl_on[exp]
	
	if any(df_c["Estimated"].isnull()):
		# warn.warn(str(df_c["Estimated"].isnull().sum())+ " of the estimated Values were NaN")
		df_c = df_c[~df_c.Estimated.isnull()]
	
	df_c.sort_values("Observed", axis=0, ascending=True, inplace=True)
	print(exp, sklMet.accuracy_score(df_c["Observed"], df_c["Estimated"]))
	
	# ========== Calculate the confusion matrix ==========
	# \\\ confustion matrix  observed (rows), predicted (columns), then transpose and sort
	df_cm  = pd.DataFrame(
		sklMet.confusion_matrix(df_c["Observed"], df_c["Estimated"],  labels=df_c["Observed"].cat.categories,  normalize='true'),  
		index = [int(i) for i in np.arange(expsize)], columns = [int(i) for i in np.arange(expsize)]).T.sort_index(ascending=False)

	cmap = mpc.ListedColormap(palettable.matplotlib.Inferno_20_r.mpl_colors)
	if mask:
		#+++++ remove 0 values +++++
		df_cm.replace(0, np.NaN, inplace=True)
		# breakpoint()

	if annot:
		ann = df_cm.round(3)
	else:
		ann = False
	sns.heatmap(df_cm, annot=ann, vmin=0, vmax=1, ax = ax, cbar=cbar, square=True, cmap=cmap)
	ax.plot(np.flip(np.arange(expsize+1)), np.arange(expsize+1), "w", alpha=0.5)
	# plt.title(expr[exp])
	# ========== fix the labels +++++
	if (sptsze > 10):
		# +++++ The location of the ticks +++++
		interval = int(np.floor(sptsze/10))
		location = np.arange(0, sptsze, interval)
		# +++++ The new values +++++
		values = np.round(np.linspace(-1., 1, location.size), 2)
		ax.set_xticks(location)
		ax.set_xticklabels(values)
		ax.set_yticks(location)
		ax.set_yticklabels(np.flip(values))
		# ========== Add the cross hairs ==========
		if zline:

			ax.axvline(location[values == 0][0], alpha =0.25, linestyle="--", c="grey")
			ax.axhline(location[values == 0][0], alpha =0.25, linestyle="--", c="grey")
	else:
		# warn.warn("Yet to fix the ticks here")
		# breakpoint()
		# ========== Calculate the zero lines ==========
		if zline:
			# ========== Calculate the zero line location ==========
			((x0),) = np.where(np.array(split) < 0)
			x0a = x0[-1]
			((x1),) = np.where(np.array(split) >=0)
			x1a = x1[0]
			zeroP =  x0a + (0.-split[x0a])/(split[x1a]-split[x0a])
			# ========== Add the cross hairs ==========
			ax.axvline(x=zeroP, ymin=0.1, alpha =0.25, linestyle="--", c="w")
			ax.axhline(y=zeroP, xmax=0.9, alpha =0.25, linestyle="--", c="w")

		# +++++ fix the values +++++
		location = np.arange(0., sptsze)#+1
		location[ 0] += 0.00001
		location[-1] -= 0.00001
		ax.set_xticks(location)
		ax.set_xticklabels(np.round(split, 2), rotation=90)
		ax.set_yticks(location)
		ax.set_yticklabels(np.flip(np.round(split, 2)), rotation=0)


	ax.set_title(f"a) {exp}-{keys[exp]} $R^{2}$ {df_set.R2.mean()}", loc= 'left')
	ax.set_xlabel("Observed")
	ax.set_ylabel("Predicted")


	plt.tight_layout()

	# ========== Save tthe plot ==========
	# print("starting save at:", pd.Timestamp.now())
	# fnout = path+ "plots/BM03_Normalised_Confusion_Matrix_" + sumtxt
	# for ext in [".pdf", ".png"]:
	# 	plt.savefig(fnout+ext, dpi=130)
	
	# plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
	# 	__title__, __version__,  __author__, pd.Timestamp.now())
	# gitinfo = cf.gitmetadata()
	# cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()

def Temporal_predictability(
	ppath, experiments, df_setup, df, keys,  fig, ax, var, hue="experiment",
	va = "Residual", CI = "QuantileInterval", single=True, title=""):

	"""
	Function to make a figure that explores the temporal predictability. This 
	figure will only use the runs with virable windows
	"""
	cats = experiments
	lab = [keys[expn] for expn in experiments]
	huex = hue
	if len(experiments) > 1:
		# pick only the subset that are matched 
		dfX = df.dropna()
		huecount = len(experiments)
	elif hue =="ChangeDirection":
		dfX      = df.copy()
		# dfX[hue] = "All" #dfX[hue].cat.categories[0]
		# dfX      = pd.concat([dfX,df]).reset_index(drop=True)
		dfX[hue] = pd.Categorical(dfX[hue].values, categories=["Gain", "Loss"], ordered=True)
		huecount = len(dfX[hue].cat.categories)
		cats     = dfX[hue].cat.categories
		lab      = [ct for ct in cats]
	elif hue =="ChangeDirectionAll":
		huex = "ChangeDirection"
		dfX      = df.copy()
		dfX[huex] = "All" #dfX[hue].cat.categories[0]
		dfX      = pd.concat([dfX,df]).reset_index(drop=True)
		dfX[huex] = pd.Categorical(dfX[huex].values, categories=["All", "Gain", "Loss"], ordered=True)
		huecount = len(dfX[huex].cat.categories)
		cats     = dfX[huex].cat.categories
		lab      = [ct for ct in cats]
		# breakpoint()
	elif hue == "ChangeDirectionNorm":
		dfX      = df.copy()
		huex = "ChangeDirection"
		try:
			mean = dfX.loc[:,["ObsDelta", "ObsGap"]].abs().groupby("ObsGap").transform("mean").values.flatten()
			dfX[va] /= mean 
		except Exception as er:
			warn.warn(str(er))
			breakpoint()
		dfX[huex] = pd.Categorical(dfX[huex].values, categories=["Gain", "Loss"], ordered=True)
		huecount = len(dfX[huex].cat.categories)
		cats     = dfX[huex].cat.categories
		lab      = [ct for ct in cats]

	elif hue is None:
		dfX = df.dropna()
		huecount = 1
	elif not hue=="experiment":
		warn.warn("Not implemented yet")
		breakpoint()
	else:
		dfX = df.loc[df["experiment"] == experiments[0]]
		huecount = len(experiments)



	# ========== make the plot ==========
	# for va in ["AbsResidual", "residual"]:
	# 	for CI in ["SD", "QuantileInterval"]:
	print(f"{va} {CI} {pd.Timestamp.now()}")
	# Create the labels

	# ========== set up the colours and build the figure ==========
	colours = palettable.cartocolors.qualitative.Vivid_10.hex_colors
	# ========== Build the first part of the figure ==========
	if CI == "SD":
		sns.lineplot(y=va, x="ObsGap", data=dfX, 
			hue=huex, ci="sd", ax=ax, 
			palette=colours[:huecount], legend=False)
	else:
		# Use 
		sns.lineplot(y=va, x="ObsGap", data=dfX, 
			hue=huex, ci=None, ax=ax, 
			palette=colours[:huecount], legend=False)
		# if hue=="experiment":
		for cat, colr in zip(cats, colours[:huecount]) :
			df_ci = dfX[dfX[huex] == cat].groupby("ObsGap")[va].quantile([0.05, 0.95]).reset_index()
			# breakpoint()
			ax.fill_between(
				df_ci[df_ci.level_1 == 0.05]["ObsGap"].values, 
				df_ci[df_ci.level_1 == 0.95][va].values, 
				df_ci[df_ci.level_1 == 0.05][va].values, alpha=0.20, color=colr)
		# else:
	# ========== fix the labels ==========
	ax.set_xlabel('Years Between Observation', fontsize=12, fontweight='bold')
	# ========== Create hhe legend ==========
	ax.legend(title=huex, loc='upper right', labels=lab)
	if hue == "ChangeDirectionNorm":
		# breakpoint()
		ax.set_ylim(-5, 5)
		ax.set_ylabel(r'Normalised Mean Residual ($\pm$ %s)' % CI, fontsize=12, fontweight='bold')
		# ax.set_title(f"{var} {va} {CI}", loc= 'left')
	else:
		ax.set_ylabel(r'Mean Residual ($\pm$ %s)' % CI, fontsize=12, fontweight='bold')
		pass

	ax.set_title(f"{title}", loc= 'left')

	# ========== The second subplot ==========
	# breakpoint()
	# sns.histplot(data=df_OvsP, x="ObsGap", hue="Region",  
	# 	multiple="dodge",  ax=ax2) #palette=colours[:len(experiments)]
	# # ========== fix the labels ==========
	# ax2.set_xlabel('Years Between Observation', fontsize=8, fontweight='bold')
	# ax2.set_ylabel(f'# of Obs.', fontsize=8, fontweight='bold')
	# ========== Create hhe legend ==========
	# ax2.legend(title='Experiment', loc='upper right', labels=lab)
	# ax2.set_title(f"b) ", loc= 'left')

	if single:
		plt.tight_layout()

		# ========== Save tthe plot ==========
		print("starting save at:", pd.Timestamp.now())
		if len (experiments) == 0:
			fnout = f"{ppath}PS02_{var}_{va}_{CI}_{experiments[0]}_TemporalPred" 
		else:
			fnout = f"{ppath}PS02_{var}_{va}_{CI}_TemporalPred" 
		for ext in [".png"]:#".pdf",
			plt.savefig(fnout+ext)#, dpi=130)
		plotinfo = "PLOT INFO: PDF plots made using %s:v.%s by %s, %s" % (
			__title__, __version__,  __author__, pd.Timestamp.now())
		gitinfo = cf.gitmetadata()
		cf.writemetadata(fnout, [plotinfo, gitinfo])
		plt.show()
	# plt.show()

# ==============================================================================
def Translator(df_setup, df_mres, keys, df_OvsP, df_clest, df_branch, experiments, path):
	""" Function to transfrom different methods of calculating biomass 
	into a comperable number """
	bioMls = []
	dpath = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/"
	vi_fn  = f"{dpath}VI_df_AllSampleyears_ObsBiomass.csv"
	vi_df  = pd.read_csv( vi_fn, index_col=0).loc[:, ['biomass', 'Obs_biomass', 'Delta_biomass','ObsGap']]

	
	# ========== Fill in the missing sites ==========
	regions   = regionDict()
	region_fn =f"{dpath}SiteInfo_AllSampleyears_ObsBiomass.csv"
	site_df   = pd.read_csv(region_fn, index_col=0)
	assert site_df.shape[0] == vi_df.shape[0]

	# site_df.replace(regions, inplace=True)
	# ========== Loop over each experiment ==========
	for exp in tqdm(experiments):
		pvar =  df_setup.loc[f"Exp{exp}", "predvar"]
		if type(pvar) == float:
			# deal with the places i've alread done
			pvar = "lagged_biomass"

		# +++++ pull out the observed and predicted +++++
		df_OP  = df_OvsP.loc[df_OvsP.experiment == exp]
		df_act = vi_df.loc[df_OP.index]
		df_s   = site_df.loc[df_OP.index]
		dfC    = df_OP.copy()
		
		if pvar == "lagged_biomass":
			vfunc            = np.vectorize(_delag)
			dfC["Estimated"] = vfunc(df_act["biomass"].values, df_OP["Estimated"].values)
			
		elif pvar == 'Obs_biomass':
			if type(df_setup.loc[f"Exp{exp}", "yTransformer"]) == float:
				pass
			else:
				for ver in dfC.version.unique().astype(int):
					ppath = f"./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/{exp}/" 
					fn_mod = f"{ppath}models/XGBoost_model_exp{exp}_version{ver}"

					setup = pickle.load(open(f"{fn_mod}_setuptransfromers.dat", "rb"))
					dfC.loc[dfC.version == ver, "Estimated"] = setup['yTransformer'].inverse_transform(dfC.loc[dfC.version == ver, "Estimated"].values.reshape(-1, 1))

		elif pvar == 'Delta_biomass':
			dfC["Estimated"] += df_act["biomass"]
		else:
			breakpoint()
		dfC["Observed"]      = df_act["Obs_biomass"].values
		dfC["Residual"]      = dfC["Estimated"] - dfC["Observed"]
		dfC["Original"]      = df_act["biomass"].values
		dfC["ObsDelta"]      = df_act["Delta_biomass"].values
		dfC["EstDelta"]      = dfC["Estimated"] - dfC["Original"]
		# dfC["DeltaResidual"] = dfC["EstDelta"] - dfC["ObsDelta"]
		dfC["ObsGap"]        = df_act["ObsGap"].values
		dfC["Region"]        = df_s["Region"].values
		bioMls.append(dfC)
		# breakpoint()

	# ========== Convert to a dataframe ==========
	df = pd.concat(bioMls).reset_index().sort_values(["version", "index"]).reset_index(drop=True)

	# ========== Perform grouped opperations ==========
	df["Rank"]     = df.drop("Region", axis=1).abs().groupby(["version", "index"])["Residual"].rank(na_option="bottom").apply(np.floor)
	df["RunCount"] = df.drop("Region", axis=1).abs().groupby(["version", "index"])["Residual"].transform("count")
	df.loc[df["RunCount"] < len(experiments), "Rank"] = np.NaN

	def _inc(val):
		if val <=0:
			return "Loss"
		else:
			return "Gain"

	df["ChangeDirection"] = pd.Categorical([_inc(val) for val in df['ObsDelta'].values], categories=["Gain", "Loss"], ordered=True)
	# df.replace(regions, inplace=True)
	return df


def gridder(path, exp, years, df, lats, lons, var = "DeltaBiomass", textsize=24):

	# ========== Setup params ==========
	# plt.rcParams.update({'axes.titleweight':"bold","axes.labelweight":"bold", 'axes.titlesize':textsize})
	# font = {'family' : 'normal',
	#         'weight' : 'bold', #,
	#         'size'   : textsize}
	# mpl.rc('font', **font)
	# sns.set_style("whitegrid")
	""" Function to convert the points into a grid """
	# ========== Copy the df so i can export multiple grids ==========
	dfC = df.copy()#.dropna()
	# breakpoint()

	dfC["longitude"] = pd.cut(dfC["Longitude"], lons, labels=bn.move_mean(lons, 2)[1:])
	dfC["latitude"]  = pd.cut(dfC["Latitude" ], lats, labels=bn.move_mean(lats, 2)[1:])
	dfC["ObsGap"]    = dfC.time.dt.year - dfC.year
	if var == 'DeltaBiomass':
		dfC["AnnualBiomass"] = dfC[var] / dfC["ObsGap"]
	else:
		breakpoint()


	# ========== Convert the different measures into xarray formats ==========
	dscount  = dfC.groupby(["time","latitude", "longitude", "Version"])[var].count().to_xarray().sortby("latitude", ascending=False)
	dscount  = dscount.where(dscount>0)
	dsp      = dfC.loc[dfC["DeltaBiomass"]> 0].groupby(["time","latitude", "longitude", "Version"])[var].count().to_xarray().sortby("latitude", ascending=False)
	dsn      = dfC.loc[dfC["DeltaBiomass"]<=0].groupby(["time","latitude", "longitude", "Version"])[var].count().to_xarray().sortby("latitude", ascending=False)
	dspos    = (dsp-dsn)/dscount
	
	dsmean   = dfC.groupby(["time","latitude", "longitude", "Version"])[var].mean().to_xarray().sortby("latitude", ascending=False)
	dsmedian = dfC.groupby(["time","latitude", "longitude", "Version"])[var].median().to_xarray().sortby("latitude", ascending=False)
	dsannual = dfC.groupby(["time","latitude", "longitude", "Version"])["AnnualBiomass"].mean().to_xarray().sortby("latitude", ascending=False)
	# ========== Convert the different measures into xarray formats ==========
	ds = xr.Dataset({
		"sites":dscount, 
		"sitesInc":dspos, 
		f"Mean{var}":dsmean, 
		f"Median{var}":dsmedian, 
		f"AnnualMeanBiomass":dsannual})
	return ds
	


def fpred(path, exp, years, 
	fpath    = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/", 
	maxdelta = 30):
	"""
	function to predict future biomass
	args:
	path:	str to files
	exp:	in of experiment
	years:  list of years to predict 
	"""
	warn.warn("\nTo DO: Implemnt obsgap filtering")
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
		# breakpoint()
		# dfout.loc[dfout["Longitude"] == 0, ["Longitude", "Latitude"]] = np.NaN
		# dfout.loc[dfout["Longitude"] < -180, ["Longitude", "Latitude"]] = np.NaN
		# dfout.loc[dfout["Longitude"] >  180, ["Longitude", "Latitude"]] = np.NaN
		# dfout.loc[dfout["Latitude"] <= 0, ["Longitude", "Latitude"]] = np.NaN
		# dfout.loc[dfout["Latitude"] >  90, ["Longitude", "Latitude"]] = np.NaN
		# exp = shap.Explainer(model)
		# breakpoint()
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
			# ========== pull out the variables and apply transfors ==========
			dfX = vi_df.loc[:, feat].copy()
			
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



	return pd.concat(est_list)

def _ImpOpener(path, exps, var = "PermutationImportance", AddFeature=False):
	"""
	Function to open the feature importance files and return them as a single 
	DataFrame"""

	# ========== Loop over the exps ==========
	df_list = []
	for exp in exps:
		fnames = sorted(glob.glob(f"{path}{exp}/Exp{exp}_*PermutationImportance.csv"))
		for ver, fn in enumerate(fnames):
			dfin = pd.read_csv( fn, index_col=0)
			if AddFeature:
				ver    = int(fn[-28:-26])
				fn_mod = f"{path}{exp}/models/XGBoost_model_exp{exp}_version{ver}.dat"
				model  = pickle.load(open(f"{fn_mod}", "rb"))
				try:
					dfin["FeatureImportance"] = model.feature_importances_
				except :
					warn.warn("This usually fails when there is an XGBoost version mismatch")
					breakpoint()
					raise ValueError
				
				# if not os.path.isfile()
				if var == "Importance":
					dfin = pd.melt(dfin, id_vars="Variable", value_vars=["PermutationImportance", "FeatureImportance"], var_name="Metric", value_name=var)
					dfin.replace({"PermutationImportance":"PI", "FeatureImportance":"FI"}, inplace=True)
				# breakpoint()
			dfin["experiment"] = exp
			dfin["version"]    = ver
			df_list.append(dfin)
	
	# ========== Group them together ==========
	df = pd.concat(df_list).reset_index(drop=True)
	
	# ========== Calculate the counts ==========
	df["Count"] = df.groupby(["experiment", "Variable"])[var].transform("count")
	df.sort_values(['experiment', 'Count'], ascending=[True, False], inplace=True)
	df.reset_index(drop=True,inplace=True)

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
	hueord =  cmapper(df["VariableGroup"], AddFeature, var)
	if AddFeature and var == "Importance":
		df["VariableGroup"] = pd.Series([f"{MI}-{VG}" for MI, VG in zip(df["Metric"], df["VariableGroup"])]).astype("category")
	
	# ========== set the cat order ==========
	df["VariableGroup"].cat.set_categories(hueord["HueOrder"], inplace=True)
	return df, ver, hueord

# ==============================================================================
def cmapper(varlist, AddFeature, var):
	# pick the cmap
	# "Disturbance", 
	if "Disturbance" in varlist:
		warn.warn("THis is not implemented, currently no provision for disturbance")
		breakpoint()
		sys.exit()

	vaorder = ["Climate", "RS VI", "Survey", "Species","Permafrost", "Soil",]
	cmapHex = palettable.colorbrewer.qualitative.Paired_12.hex_colors
	if AddFeature and var == "Importance":
		vaorder = [f"{MI}-{VG}" for VG, MI in product(vaorder, ["FI", "PI"])]
		# breakpoint()
	else:
		cmapHex = cmapHex[1::2]
	# ========== Decide the order of stuff =========
	return ({"cmap":cmapHex, "HueOrder":vaorder})


def _delag(b0, bl):
	if bl == 1.:
		return np.nan
	else:
		return ((bl*b0)+b0)/(1-bl)

def Experiment_name(df, df_setup, var = "experiment"):
	keys = {}
	for cat in df["experiment"].unique():
		# =========== Setup the names ============
		try:
			if cat == 100:
				nm = "10yr LS"
			elif cat == 101:
				nm = "No LS"
			elif cat == 102:
				nm = "5yr LS"
			elif cat == 103:
				nm = "15yr LS"
			elif cat == 104:
				nm = "20yr LS"
			elif cat == 120:
				nm = "10yr XGBOOST"
			elif cat == 200:
				nm = "RF2 7 Quantile splits"
			elif cat == 201:
				nm = "RF2 3 Quantile splits"
			elif cat == 202:
				nm = "RF2 2 Interval splits"
			elif cat == 204:
				nm = "RF2 4 Interval splits"
			# elif cat == 300:
			# 	breakpoint()
			elif cat // 100 == 3.:
				pred  = df_setup[df_setup.Code.astype(int) == cat]["predictwindow"][0]
				if pred is None:
					breakpoint()
				lswin = df_setup[df_setup.Code.astype(int) == cat]["window"][0]
				if cat < 320:
					nm = f"DataMOD_{pred}yrPred_{lswin}yrLS"
				else:
					NAfrac = int(float(df_setup[df_setup.Code.astype(int) == cat]["DropNAN"][0]) *100)
					nm = f"DataMOD_{pred if not np.isnan(float(pred)) else 'AllSample' }yrPred_{lswin}yrLS_{NAfrac}percNA"
				if cat == 332:
					nm += "_disturbance"
				elif cat == 333:
					nm += "_FeatureSel"
			elif cat >= 400:
				mdva = df_setup[df_setup.Code.astype(int) == cat]["predvar"][0]
				if type(mdva) == float:
					mdva = "lagged_biomass"
				
				def _TFnamer(stn, tfname):
					# Function to quickly rename transfomrs 
					if type(tfname) == float:
						return ""
					elif tfname in ["QuantileTransformer(output_distribution='normal')", "QuantileTransformer(ignore_implicit_zeros=True, output_distribution='normal')"]:
						return f" {stn}_QTn"
					else:
						breakpoint()
						return f" {stn}_UNKNOWN"


				nm = f'{mdva}{_TFnamer("Ytf", df_setup.loc[f"Exp{int(cat)}", "yTransformer"])}{_TFnamer("Xtf", df_setup.loc[f"Exp{int(cat)}", "Transformer"])}'
			else:
				nm = "%d.%s" % (cat, df_setup[df_setup.Code.astype(int) == int(cat)].name.values[0])
		except Exception as er:
			print(str(er))
			breakpoint()
		keys[cat] = nm
		df[var].replace({cat:nm}, inplace=True)
	return df, keys

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

def load_OBS(ofn):
	df_in = pd.read_csv(ofn, index_col=0)
	df_in["experiment"] = int(ofn.split("/")[-2])
	df_in["experiment"] = df_in["experiment"].astype("category")
	df_in["version"]    = float(ofn.split("_vers")[-1][:2])
	return df_in

def fix_results(fn):
	# ========== Fill in the missing sites ==========
	region_fn ="./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/TTS_sites_and_regions.csv"
	site_df = pd.read_csv(region_fn, index_col=0)

	Rcounts = site_df.groupby(["Region"]).count()["site"]
	# ========== convert the types
	df_in   = pd.read_csv(fn, index_col=0).T#, parse_dates=["TotalTime"])
	for col in df_in.columns:
		if col == "TotalTime":
			df_in[col] = pd.to_timedelta(df_in[col])
		elif col in ["Computer"]:
			df_in[col] = df_in[col].astype('category')
		else:
			try:
				df_in[col] = df_in[col].astype(float)
			except:
				breakpoint()
	# ========== Loop over the regions ==========

	for region in site_df.Region.unique():
		try:
			if df_in["%s_sitefrac" % region].values[0] >1:
				warn.warn("value issue here")
				nreakpoint()
		except KeyError:
			# Places with missing regions
			df_in["%s_siteinc"  % region] = 0.
			df_in["%s_sitefrac" % region] = 0.
		# 	# df_in["%s_siteinc" % region] = df_in["%s_siteinc" % region].astype(float)
		# 	df_in["%s_sitefrac" % region] = (df_in["%s_siteinc" % region] / Rcounts[region])
	return df_in

def VIload():
	print(f"Loading the VI_df, this can be a bit slow: {pd.Timestamp.now()}")
	vi_df = pd.read_csv("./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/VI_df_AllSampleyears.csv", index_col=0)#[['lagged_biomass','ObsGap']]
	vi_df["NanFrac"] = vi_df.isnull().mean(axis=1)

	# ========== Fill in the missing sites ==========
	region_fn ="./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/SiteInfo_AllSampleyears.csv"
	site_df = pd.read_csv(region_fn, index_col=0)
	# breakpoint()

	vi_df = vi_df[['year', 'biomass', 'lagged_biomass','ObsGap', "NanFrac"]]
	for nanp in [0, 0.25, 0.50, 0.75, 1.0]:	
		isin = (vi_df["NanFrac"] <=nanp).astype(float)
		isin[isin == 0] = np.NaN
		vi_df[f"{int(nanp*100)}NAN"]  = isin

	fcount = pd.melt(vi_df.drop(["lagged_biomass","NanFrac"], axis=1).groupby("ObsGap").count(), ignore_index=False).reset_index()
	fcount["variable"] = fcount["variable"].astype("category")
	vi_df["Region"]    = site_df["Region"]
	# breakpoint()

	return vi_df, fcount
# ==============================================================================

if __name__ == '__main__':
	main()