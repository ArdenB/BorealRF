"""
Boreal EWS PSP data anlysis 
 
Script to  make individaul psps plots  
"""

# ==============================================================================

__title__ = "Examining the PSP database"
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
import xarray as xr
import cartopy.crs as ccrs

import cartopy.feature as cpf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf
import myfunctions.benchmarkfunctions as bf

# ========== Import packages for parellelisation ==========
# import multiprocessing as mp

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
	# +++++ create the paths +++++
	path  = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"
	ppath = "./pyEWS/analysis/3.Izanami/Figures/PS01/"
	cf.pymkdir(ppath)
	exp   = 402
	lons = np.arange(-170, -50.1,  0.5)
	lats = np.arange(  42,  70.1,  0.5)

	# +++++ the model Infomation +++++
	setup_fnames  = glob.glob(path + "*/Exp*_setup.csv")
	df_setup      = pd.concat([pd.read_csv(sfn, index_col=0).T for sfn in setup_fnames], sort=True)
	regions       = regionDict()
	vi_df, fcount = VIload(regions, path, exp = exp)
	
	PSPfigure(ppath, vi_df, fcount, exp, lons, lats)

	# ========== Old figures that might end up as supplemeentary material ==========
	yearcount(ppath, vi_df, fcount)
	breakpoint()


# ==============================================================================

def PSPfigure(ppath, vi_df, fcount, exp, lons, lats):
	"""
	Build the first figure in the paper

	"""
	# ========== Setup the matplotlib params ==========
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':12, "axes.labelweight":"bold"})
	font = ({'family' : 'normal','weight' : 'bold', 'size'   : 12})
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	map_proj = ccrs.LambertConformal(central_longitude=lons.mean(), central_latitude=lats.mean())

	# ========== Create the figure ==========
	fig  = plt.figure(constrained_layout=True, figsize=(10,15))
	spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)

	# +++++ the plot of the number of sites +++++
	ax1  = fig.add_subplot(spec[0, :])
	_annualcount(vi_df, fig, ax1)

	# +++++ Map of the number of used sites +++++
	ax2  = fig.add_subplot(spec[1, :], projection= map_proj)
	_mapgridder(exp, vi_df, fig, ax2, map_proj, lons, lats, modelled=True,)

	# +++++ KDE of the gabs beteen observations +++++
	ax3 = fig.add_subplot(spec[2, 0])
	_obsgap(vi_df, fig, ax3)


	# +++++ the plot of the number of sites +++++


	# ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS01_PaperFig01_PSPdatabase" 
	for ext in [".png", ".pdf",]:
		plt.savefig(fnout+ext)#, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()
	breakpoint()

# ==============================================================================

def _obsgap(vi_df, fig, ax):
	sns.kdeplot(data=vi_df, x="ObsGap", hue="NanFrac", fill=True, alpha=0.50, ax=ax)

def _mapgridder(exp, vi_df, fig, ax, map_proj, lons, lats, modelled=True, vmin=0, vmax=1000):
	# ========== Simple lons and lats ========== 
	# ========== Setup params ==========
	""" Function to convert the points into a grid """
	
	# ========== Copy the df so i can export multiple grids ==========
	# 	dfC = vi_df.loc[vi_df["NanFrac"] == 1].copy()#.dropna()	
	# else:
	dfC = vi_df.copy()

	dfC["longitude"] = pd.cut(dfC["Longitude"], lons, labels=bn.move_mean(lons, 2)[1:])
	dfC["latitude"]  = pd.cut(dfC["Latitude" ], lats, labels=bn.move_mean(lats, 2)[1:])
	dfC["time"]      = pd.Timestamp(f"2020-12-31")
	# setup            = pd.read_csv(f"{path}{exp}/Exp{exp}_setup.csv", index_col=0)



	# ========== Convert the different measures into xarray formats ==========
	dscount  = dfC.groupby(["time","latitude", "longitude", 'NanFrac'])["biomass"].count().to_xarray().sortby("latitude", ascending=False)
	dscount  = dscount.where(dscount>0)

	# ========== Convert the different measures into xarray formats ==========
	ds = xr.Dataset({
		"TotalSites":dscount.sum(dim="NanFrac").where(dscount.sum(dim="NanFrac")>0),
		"ModelledSites":dscount.isel(NanFrac=1).drop("NanFrac")
		})

	# ========== Make some example plots ==========
	# map_proj = ccrs.LambertConformal(central_longitude=lons.mean(), central_latitude=lats.mean())

	if modelled:
		vas   = "ModelledSites"
		title = "No. of Modelled Sites"
	else:
		vas   = "TotalSites"
		title = "No. of Sites"


	f = ds[vas].isel(time=0).plot(
		x="longitude", y="latitude", transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax,
		cbar_kwargs={"pad": 0.015, "shrink":0.85, "extend":"max"},	ax=ax)
	ax.set_extent([lons.min()+10, lons.max()-5, lats.min()-13, lats.max()])
	ax.gridlines()
	coast = cpf.GSHHSFeature(scale="intermediate")
	ax.add_feature(cpf.LAND, facecolor='dimgrey', alpha=1, zorder=0)
	ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
	ax.add_feature(coast, zorder=101, alpha=0.5)
	ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
	ax.add_feature(cpf.RIVERS, zorder=104)
	ax.add_feature(cpf.BORDERS, linestyle='--', zorder=102)

	ax.set_title("")
	ax.set_title(f"{title}", loc= 'left')
	# plt.show()
	# breakpoint()

def _annualcount(vi_df, fig, ax):
	"""Line graph of the amount of sites included"""
	# ========== Duplicate the datafrema ==========
	sub = vi_df[vi_df["NanFrac"] == 1].copy()
	tot = vi_df.copy()
	sub["Count"] = "Modelled"
	sub["Count"] = pd.Categorical(["Modelled" for i in range(sub.shape[0])], categories=["Total", "Modelled"], ordered=True)
	tot["Count"] = pd.Categorical(["Total" for i in range(tot.shape[0])], categories=["Total", "Modelled"], ordered=True)

	# ========== stack the results ==========
	df = pd.concat([tot, sub]).reset_index(drop=True)
	vi_yc = df.groupby(["Count", "year"])['biomass'].count().reset_index().rename(
		{"biomass":"Observations"}, axis=1).replace(0, np.NaN)
	# ========== Make the plot ==========
	sns.lineplot(y="Observations",x="year", hue="Count",dashes=[True, False], data=vi_yc, ci=None, legend=True, ax = ax)

# ==============================================================================

def yearcount(ppath, vi_df, fcount):

	# ========== Create the figure ==========
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':12, "axes.labelweight":"bold"})
	font = ({'family' : 'normal','weight' : 'bold', 'size'   : 12})
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	# plt.rcParams.update({'axes.titleweight':"bold", })

	vi_yc = vi_df.groupby(["year"])['biomass'].count().reset_index().rename({"biomass":"Observations"}, axis=1).replace(0, np.NaN)
	fig, ax = plt.subplots(1, 1, figsize=(20,10))
	sns.lineplot(y="Observations",x="year", data=vi_yc, ci=None, legend=True, ax = ax)
	fig.tight_layout()
	# ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS01_ObsYear_single" 
	for ext in [".png"]:#".pdf",
		plt.savefig(fnout+ext)
	
	plt.show()

	vi_yc = vi_df.groupby(["year", "Region"])['biomass'].count().reset_index().rename({"biomass":"Observations"}, axis=1).replace(0, np.NaN)
	fig, ax = plt.subplots(1, 1, figsize=(20,10))
	sns.lineplot(y="Observations",x="year", data=vi_yc, hue="Region", ci=None, legend=True, ax = ax)
	fig.tight_layout()
	# ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS01_ObsYear_Region_single" 
	for ext in [".png"]:#".pdf",
		plt.savefig(fnout+ext)
	
	plt.show()
	

	g = sns.FacetGrid(vi_yc, col="Region", col_wrap=4, hue="Region", height=4)
	# sns.displot(y="Observations",x="year", data=vi_yc, col="Region", col_wrap=4, kind="kde")
	g.map(sns.lineplot, "year", "Observations")
	g.fig.tight_layout()
	# ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS01_ObsYear_Region_Multi" 
	for ext in [".png"]:#".pdf",
		plt.savefig(fnout+ext)

	plotinfo = "PLOT INFO: PDF plots made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	
	plt.show()

# ==============================================================================

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

# ==============================================================================

def load_OBS(ofn):
	df_in = pd.read_csv(ofn, index_col=0)
	df_in["experiment"] = int(ofn.split("/")[-2])
	df_in["experiment"] = df_in["experiment"].astype("category")
	df_in["version"]    = float(ofn.split("_vers")[-1][:2])
	return df_in

def VIload(regions, path, exp = None):
	print(f"Loading the VI_df, this can be a bit slow: {pd.Timestamp.now()}")
	vi_df = pd.read_csv("./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/VI_df_AllSampleyears.csv", index_col=0)#[['lagged_biomass','ObsGap']]

	# ========== Fill in the missing sites ==========
	region_fn ="./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/SiteInfo_AllSampleyears.csv"
	site_df = pd.read_csv(region_fn, index_col=0)
	site_df.replace(regions, inplace=True)
	if exp is None:
		vi_df["NanFrac"] = vi_df.isnull().mean(axis=1)
	else:
		# Load in the different colkeys for each version 
		setup   = pd.read_csv(f"{path}{exp}/Exp{exp}_setup.csv", index_col=0)
		fnames  = sorted(glob.glob(f"{path}{exp}/Exp{exp}_*PermutationImportance.csv"))
		rowpass = np.zeros((vi_df.shape[0], len(fnames)))
		for nu, fn in enumerate(fnames):
			# ========== get the list of cols ==========
			dfin = pd.read_csv( fn, index_col=0)
			cols = dfin["Variable"].values
			rowpass[:, nu] = (vi_df[cols].isnull().mean(axis=1) <=  setup.loc["DropNAN"].astype(float).values[0]).astype(float).values

		vi_df["NanFrac"] = rowpass.max(axis=1)

	vi_df = vi_df[['year', 'biomass', 'lagged_biomass','ObsGap', "NanFrac", 'site']]
	# for nanp in [0, 0.25, 0.50, 0.75, 1.0]:	
	# 	isin = (vi_df["NanFrac"] <=nanp).astype(float)
	# 	isin[isin == 0] = np.NaN
	# 	vi_df[f"{int(nanp*100)}NAN"]  = isin

	fcount = pd.melt(vi_df.drop(["lagged_biomass","NanFrac"], axis=1).groupby("ObsGap").count(), ignore_index=False).reset_index()
	fcount["variable"] = fcount["variable"].astype("category")
	
	vi_df["Region"]    = site_df["Region"].astype("category")
	vi_df["Longitude"] = site_df["Longitude"]
	vi_df["Latitude"]  = site_df["Latitude"]
	# breakpoint()

	return vi_df, fcount

# ==============================================================================

if __name__ == '__main__':
	main()