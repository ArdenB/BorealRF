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
	site_df = pd.read_csv(f"{fpath}SiteInfo_AllSampleyears_ObsBiomass.csv", index_col=0)
	exp      = 434
	# ========= Load in the observations ==========
	OvP_fnames = glob.glob(f"{path}{exp}/Exp{exp}*_OBSvsPREDICTED.csv")
	df_OvsP    = pd.concat([load_OBS(ofn) for ofn in OvP_fnames], sort=True)
	df_mod     = df_OvsP[~df_OvsP.index.duplicated(keep='first')]

	predictions(path, ppath, exp, fpath, vi_df, site_df, df_OvsP)
	breakpoint()
	# ========== Chose the experiment ==========
	sitedtb(path, ppath, exp, fpath, vi_df, site_df, df_mod)

# ==============================================================================
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
	keystats.append('\n Prediction Score Metric for the entire prediction ensemble \n')
	keystats.append(f'\n R squared score: {sklMet.r2_score(y_test, y_pred)}')
	keystats.append(f'\n Mean Absolute Error: {sklMet.mean_absolute_error(y_test, y_pred)}')
	keystats.append(f'\n Root Mean Squared Error: {np.sqrt(sklMet.mean_squared_error(y_test, y_pred))}')
	# +++++ did it get direction correct +++++

	keystats.append(f'\n Fraction of measurments where models guess directionalty correctly:{np.sum((y_test > 0) == (y_pred > 0)) / df_OvsP.shape[0]}')

	# ========== THe number of features per model ==========
	df_perm = loadperm(exp, path)
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
	keystats.append(df_perm.groupby("Variable").mean().sort_values("PermutationImportance"))


	breakpoint()
	# ========== Save the info =========
	fname = f'{ppath}DatasetSummary.txt'
	f = open(fname,'w')
	for info in keystats:
		f.write("%s\n" %info)
	f.close()


def sitedtb(path, ppath, exp, fpath, vi_df, site_df, df_mod):
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
	keystats.append(vi_df.groupby(["site", "year"]).first().reset_index().groupby("site").count()["year"].mean())
	# total measurments 
	keystats.append("\n Total number of measurements in the database \n")
	keystats.append(vi_df.shape[0])

	site_dfM = site_df.loc[df_mod.index]
	site_dfM["Observed"] = df_mod["Observed"]
	site_dfM["Biomass"]  = vi_df.loc[df_mod.index, "biomass"]
	site_dfM["ObsGap"]  = vi_df.loc[df_mod.index, "ObsGap"]
	regions = regionDict()
	# site_dfM
	site_dfM.replace(regions, inplace=True)
	# +++++ Sites included in the models +++++
	keystats.append("\n Total number of measurements Modelled \n")
	keystats.append(site_dfM.shape[0])

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
	fname = f'{ppath}DatasetSummary.txt'
	f = open(fname,'w')
	for info in keystats:
		f.write("%s\n" %info)
	f.close()




	site_dfM["Region"] = site_dfM.Region.astype('category')#.cat.reorder_categories(ks)
	plotter(ppath, site_dfM, inc, cnt)
	breakpoint()
	warn.warn("Do the future sites as well")

# ==============================================================================


def plotter(ppath, site_dfM, inc, cnt):

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
	fig  = plt.figure(constrained_layout=True, figsize=(16,13))
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
	ax1.set_ylabel("Biomass (t/ha)")
	ax1.set_xlabel("")
	ax1.set_xticklabels(ax1.get_xticklabels(), rotation=15, horizontalalignment='right')

	ax2  = fig.add_subplot(spec[2, :])
	data = 1-(inc / cnt)
	data.plot.bar(ax=ax2)
	ax2.set_title("c)")
	ax2.set_ylabel("Loss Frac.")
	ax2.set_xlabel("")
	ax2.set_xticklabels(ax2.get_xticklabels(), rotation=15, horizontalalignment='right')
	
	# Mean annual delta biomass
	ax3  = fig.add_subplot(spec[3, :])
	site_dfM["AnnualBiomassChange"] = site_dfM["Observed"] / site_dfM["ObsGap"]
	sns.violinplot(y = "AnnualBiomassChange", x="Region", data = site_dfM, ax=ax3)
	# ax1.set_ylim(0, 1000)
	ax3.set_title("d)")
	ax3.set_ylabel("Biomass Change Rate (t/ha/yr)")
	ax3.set_xlabel("")
	ax3.set_xticklabels(ax3.get_xticklabels(), rotation=15, horizontalalignment='right')


	# ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS09_RegionalOverview" 
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
# ==============================================================================
if __name__ == '__main__':
	main()