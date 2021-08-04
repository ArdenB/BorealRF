"""
This script aims to allow me to quickly assess the diferences in peromance 
i get with different runs.  
"""


# ==============================================================================

__title__ = "Complete run intercomparison"
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
from sklearn import metrics as sklMet

# ==============================================================================
def main():
	# ========== Setup the matplotlib params ==========
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':12, "axes.labelweight":"bold"})
	font = ({'family' : 'normal','weight' : 'bold', 'size'   : 12})
	mpl.rc('font', **font)
	sns.set_style("whitegrid")

	# ========== Get the file names and open the files ==========
	formats = None
	path  = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"
	cf.pymkdir(path+"plots/")
	ppath = "./pyEWS/analysis/3.Izanami/Debugging/plots/"
	cf.pymkdir(ppath)
	# +++++ the model Infomation +++++
	setup_fnames = glob.glob(path + "*/Exp4[2-9][0-9]*_setup.csv")
	df_setup     = pd.concat([pd.read_csv(sfn, index_col=0).T for sfn in setup_fnames], sort=True)
	df_setup.loc[df_setup.splitvar == "['site', 'yrend']", "splitvar"] = "SiteYF"
	# +++++ the final model results +++++
	mres_fnames = glob.glob(path + "*/Exp4[2-9][0-9]*_Results.csv")
	df_mres = pd.concat([fix_results(mrfn) for mrfn in mres_fnames], sort=True)
	df_mres["TotalTime"]  = df_mres.TotalTime / pd.to_timedelta(1, unit='m')
	df_mres, keys = Experiment_name(df_mres, df_setup, var = "experiment")


	df_mres = df_mres[['experiment', 'R2', 'Computer', "TotalTime", 'FBranch', 'colcount', 'fractrows', 'itterrows', 'totalrows', 'version']]

	# ========= Load in the observations ==========
	OvP_fnames = glob.glob(path + "*/Exp4[2-9][0-9]*_OBSvsPREDICTED.csv")
	df_OvsP    = pd.concat([load_OBS(ofn) for ofn in OvP_fnames], sort=True)
	df_OvsP["Residual"] = df_OvsP["Estimated"] - df_OvsP["Observed"]

	branch     = glob.glob(path + "*/Exp4[2-9][0-9]*_BranchItteration.csv")
	df_branch  = pd.concat([load_OBS(mrfn) for mrfn in branch], sort=True)

	# ========== basic comparison ==========
	basiccomparison(path, ppath, df_setup, df_mres, keys, df_OvsP)

	# ========== Heatmaps ==========
	confusion_plotter(keys, ppath, df_setup, df_OvsP, df_mres)
	confusion_plotter(keys, ppath, df_setup, _matchedfinder(df_OvsP, keys, ret_matched=True), df_mres)
	breakpoint()


# ==============================================================================
def confusion_plotter(keys, ppath, df_setup, df_OvsP, df_mres):
	maxval =  350
	minval = -350
	gap    = 10 
	# split  = np.hstack([np.min([-maxval ,-1000.]),np.arange(-400., 401, 10), np.max([1000., maxval])])
	split  = np.arange(minval, maxval+1, gap)
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

	for exp, ax in zip(df_OvsP.experiment.unique(), [ax1, ax2, ax3, ax4]):
		_confusion_plots(df_OvsP, keys, exp, fig, ax, split)
	
	plt.show()
	# breakpoint()



def basiccomparison(path, ppath, df_setup, df_mres, keys, df_OvsP):
	# ========== Create the exis ==========
	# \\\\\ R2 /////
	f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
	sns.barplot(y="R2", x="experiment", data = df_mres, ax=ax1)
	# plt.show()

	# R2 matched
	df_rank, df_score = _matchedfinder(df_OvsP, keys)

	sns.barplot(y="R2", x="experiment", data = df_score, ax=ax2)
	ax2.set_title("matched R2")
	# plt.show()

	sns.barplot(y="MAE", x="experiment", data = df_score, ax=ax3)
	ax3.set_title("matched MAE")
	# plt.show()

	# ranking performance
	sns.barplot(y="count", x="rank", hue="experiment", data = df_rank, ax=ax4)
	ax4.set_title("Min ABs Residual Rank (1st, 2nd, 3rd)")
	plt.show()
	# breakpoint()


# ==============================================================================
# ==============================================================================
def _confusion_plots(
	df_OvsP, keys, exp, fig, ax, split, 
	obsvar="Observed", estvar="Estimated", inc_class=False, 
	sumtxt="", annot=False, zline=True, num=0, cbar=True, mask=True):
	"""Function to create and plot the confusion matrix"""

	# ========== Get the observed and estimated values ==========
	# breakpoint()
	df_c         = df_OvsP.loc[df_OvsP.experiment == exp, [obsvar,estvar]].copy()
	expsize      = len(split) -1 # df_class.experiment.unique().size
	df_c[obsvar] = pd.cut(df_c[obsvar], split, labels=np.arange(expsize))
	df_c[estvar] = pd.cut(df_c[estvar], split, labels=np.arange(expsize))
	
	# ========== See if there are any insane values ==========
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
	r2 = sklMet.r2_score(df_c[obsvar], df_c[estvar]).round(decimals=4)
	print(exp, r2)
	
	# ========== Calculate the confusion matrix ==========
	# \\\ confustion matrix  observed (rows), predicted (columns), then transpose and sort
	df_cm  = pd.DataFrame(
		sklMet.confusion_matrix(df_c[obsvar], df_c[estvar],  
			labels=df_c[obsvar].cat.categories,  normalize='true'),  
		index = [int(i) for i in np.arange(expsize)], 
		columns = [int(i) for i in np.arange(expsize)]).T.sort_index(ascending=False)

	cmap = mpc.ListedColormap(palettable.matplotlib.Inferno_20_r.mpl_colors)
	if mask:
		#+++++ remove 0 values +++++
		df_cm.replace(0, np.NaN, inplace=True)
		# breakpoint()

	if annot:
		ann = df_cm.round(3)
	else:
		ann = False


	g = sns.heatmap(df_cm, annot=ann, vmin=0, vmax=1, ax = ax, cbar=cbar, square=True, cmap=cmap)
	ax.plot(np.flip(np.arange(expsize+1)), np.arange(expsize+1), "k", alpha=0.1)
	# plt.title(expr[exp])
	# ========== fix the labels +++++
	if (sptsze > 10):
		# +++++ The location of the ticks +++++
		interval = int(np.floor(sptsze/10))
		location = np.arange(0, sptsze, interval)
		# +++++ The new values +++++
		values = np.round(np.linspace(split[0], split[-1], location.size))
		ax.set_xticks(location)
		ax.set_xticklabels(values, rotation=90)
		ax.set_yticks(location)
		ax.set_yticklabels(np.flip(values), rotation=0)
		# ========== Add the cross hairs ==========
		if zline:
			try:
				ax.axvline(location[values == 0][0], alpha =0.25, linestyle="--", c="grey")
				ax.axhline(location[values == 0][0], alpha =0.25, linestyle="--", c="grey")
			except Exception as er:
				print(er)
				breakpoint()
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
		# az.set_ylabel("")


	ax.set_title(f"{exp}-{keys[exp]} $R^{2}$ {r2}", loc= 'left')
	# ax.set_xlabel(obsvar)
	ax.set_xlabel("")
	ax.set_ylabel(estvar)
	# plt.show()()

# ============================================================================================

def _matchedfinder(df_OvsP, keys, ret_matched=False):
	df_obsm = df_OvsP.drop(["version"], axis=1).reset_index().groupby(["experiment","index"]).mean().reset_index()
	dfpm    = df_obsm.pivot(index='index', columns='experiment', values='Estimated').dropna()
	dfob    = df_obsm.pivot(index='index', columns='experiment', values='Observed').dropna()
	
	score = OrderedDict()
	for rn in dfpm.columns:
		score[rn] = ({
			"R2"  :sklMet.r2_score(dfob[rn], dfpm[rn]),
			"MAE" :sklMet.mean_absolute_error(dfob[rn], dfpm[rn]),
			"RMSE":np.sqrt(sklMet.mean_squared_error(dfob[rn], dfpm[rn])),
			})
	df_score = pd.DataFrame(score).T.reset_index().rename({"index":"experiment"},axis=1)
	df_score["experiment"].replace(keys, inplace=True)
	# ranking performance
	dfrs    = df_obsm.pivot(index='index', columns='experiment', values='Residual').dropna()
	df_rank = dfrs.abs().rank(axis=1).melt().groupby(["experiment", "value"]).size().reset_index()
	df_rank.rename({0:"count", "value":"rank"},axis=1, inplace=True)
	df_rank["experiment"].replace(keys, inplace=True)
	# breakpoint()
	if ret_matched:
		return df_OvsP.loc[dfpm.index,]
	else:
		return df_rank, df_score


def Experiment_name(df, df_setup, var = "experiment"):
	keys = {}
	for cat in df["experiment"].unique():
		# =========== Setup the names ============
		info = df_setup.loc[f"Exp{int(cat)}"]
		nm = f"{info['splitvar']} {info['FutDist']}%FutDist"
		# breakpoint()
		
		keys[int(cat)] = nm
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