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
# import matplotlib.ticker as tck
import matplotlib.colors as mpc
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from collections import OrderedDict, defaultdict
import seaborn as sns
import palettable
# from numba import jit
import xarray as xr
import cartopy.crs as ccrs

import cartopy.feature as cpf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf
import myfunctions.benchmarkfunctions as bf
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates

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

	warn.warn("\n\n I have not excluded sites that fail on future disturbance thresholds yet \n\n")

	# exp   = 402
	exp = 434
	lons = np.arange(-170, -50.1,  0.5)
	lats = np.arange(  42,  70.1,  0.5)

	# +++++ the model Infomation +++++
	setup_fnames  = glob.glob(path + "*/Exp*_setup.csv")
	df_setup      = pd.concat([pd.read_csv(sfn, index_col=0).T for sfn in setup_fnames], sort=True)
	regions       = regionDict()

	for inclfin in [True]:#False, 
	
		vi_df, fcount = VIload(regions, path, exp = exp, inclfin=inclfin)
		# PSPfigurePres(ppath, vi_df, fcount, exp, lons, lats, inclfin=inclfin)
		
		PSPfigure(ppath, vi_df, fcount, exp, lons, lats, inclfin=inclfin)

		# # ========== Old figures that might end up as supplemeentary material ==========
	breakpoint()
	yearcount(ppath, vi_df, fcount)


# ==============================================================================
def PSPfigurePres(ppath, vi_df, fcount, exp, lons, lats, inclfin=True, textsize=24):
	"""
	Build presentation versions versions of the figures

	"""
	# ========== Setup the matplotlib params ==========
	plt.rcParams.update({
		'axes.titleweight':"bold", 'axes.titlesize':textsize, "axes.labelweight":"bold"})
	font = ({'family' : 'normal','weight' : 'bold', 'size': textsize})
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	map_proj = ccrs.LambertConformal(central_longitude=lons.mean(), central_latitude=lats.mean())

	fig, ax = plt.subplots(constrained_layout=True, figsize=(13,7))
	# +++++ the plot of the number of sites +++++
	_annualcount(vi_df, fig, ax, inclfin=inclfin)
	# ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS01_PaperFig01_PSPdatabase_sitecount" 
	if inclfin:
		fnout += "_WithLastObs"
	for ext in [".png", ".pdf",]:
		plt.savefig(fnout+ext)#, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()


	# ========== Create the figure ==========


	# +++++ Map of the number of used sites +++++
	fig, ax = plt.subplots(constrained_layout=True, 
		subplot_kw={'projection':map_proj}, figsize=(14,7))
	# ax2  = fig.add_subplot(spec[1, :], projection= map_proj)
	_mapgridder(exp, vi_df, fig, ax, map_proj, lons, lats, modelled=True,)
	# ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS01_PaperFig01_PSPdatabase_mapsites" 
	if inclfin:
		fnout += "_WithLastObs"
	for ext in [".png", ".pdf",]:
		plt.savefig(fnout+ext)#, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()

	# +++++ KDE of the gabs beteen observations +++++
	fig, ax = plt.subplots(constrained_layout=True, figsize=(13,7))
	_obsgap(vi_df, fig, ax, inclfin=inclfin)
	# ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS01_PaperFig01_PSPdatabase_kdegaps" 
	if inclfin:
		fnout += "_WithLastObs"
	for ext in [".png", ".pdf",]:
		plt.savefig(fnout+ext)#, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()
	breakpoint()



def PSPfigure(ppath, vi_df, fcount, exp, lons, lats, inclfin=True, textsize=12):
	"""
	Build the first figure in the paper

	"""
	# print("includefin not yet implemneted here")
	# ========== Setup the matplotlib params ==========
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':textsize, "axes.labelweight":"bold"})
	font = ({'family' : 'normal','weight' : 'bold', 'size'   : textsize})
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	map_proj = ccrs.LambertConformal(central_longitude=lons.mean(), central_latitude=lats.mean())

	# ========== Create the figure ==========
	fig  = plt.figure(constrained_layout=True, figsize=(12,13))
	spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)

	# +++++ the plot of the number of sites +++++
	ax1  = fig.add_subplot(spec[0, :])
	_annualcount(vi_df, fig, ax1, title="a)")

	# +++++ Map of the number of used sites +++++
	ax2  = fig.add_subplot(spec[1, :], projection= map_proj)
	_mapgridder(exp, vi_df, fig, ax2, map_proj, lons, lats, title="b)", modelled=True,)

	# +++++ KDE of the gabs beteen observations +++++
	ax3 = fig.add_subplot(spec[2, 0])
	_obsgap(vi_df, fig, ax3, title="c)")


	# +++++ the plot of the number of sites +++++
	ax4 = fig.add_subplot(spec[2, 1])
	_biomasschange(vi_df, fig, ax4, title="d)")
	# breakpoint()


	# ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS01_PaperFig01_PSPdatabase" 
	for ext in [".png", ".pdf"]:#".pdf",
		plt.savefig(fnout+ext)#, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()
	breakpoint()

# ==============================================================================
def _biomasschange(vi_df, fig, ax,title="",):
	test = vi_df.dropna().loc[vi_df.dropna().groupby(['site', 'year'])['ObsGap'].idxmin()]
	test = test.loc[test.year >=1982]
	test["Obs.Year"] = (test.year + test.ObsGap).astype(int)# pd.to_datetime(test.year + test.ObsGap, format='%Y')
	test.sort_values("Obs.Year", inplace=True)
	test["Biomass Increase"] = test.Delta_biomass > 0
	sns.countplot(x="Obs.Year", hue="Biomass Increase", data=test, ax=ax)
	# ax.set_xlabel("Year of Observations")
	ax.set_xlabel("")

	# minorticks = []
	# for num in ax.get_xticks():
	# 	if not num in np.arange(0, 33, 5):
	# 		minorticks.append(num)
	# ax.set_xticks(minorticks, minor=True)
	# ax.minorticks_on()

	# ++++++ create ticks ++++++
	ax.tick_params(bottom=True)
	ax.set_xticks(np.arange(0, 33, 5))#, labels=np.arange(1985, 2020, 5))#, horizontalalignment='right')
	
	# ++++++ create minor tick ++++++
	ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
	ax.tick_params(which='minor', bottom=True)

	
	ax.set_title("")
	ax.set_title(f"{title}", loc= 'left')
	ax.set_xlabel("")
	ax.set_ylabel("Observations")
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles=handles, labels=["AGB loss", "AGB gain"])


def _obsgap(vi_df, fig, ax, title="", currentyr=pd.Timestamp.now().year, inclfin=True):
	if inclfin:
		cats = ["Unmodelled", "Modelled", "Final"]
	else:
		cats = ["Unmodelled", "Modelled"]

	# ========== create a column to hold the obstype ==========

	vi_df["Obs. Type"] = pd.Categorical(["Unmodelled" for i in range(vi_df.shape[0])], 
		categories=cats, ordered=True)
	if inclfin:
		vi_df.loc[vi_df["Future"]==1, "Obs. Type"] = "Final"
		# ========== add the gap from the current year ==========
		vi_df.loc[vi_df["Future"]==1, "ObsGap"] = currentyr - vi_df.loc[vi_df["Future"]==1, "year"]
	
	vi_df.loc[np.logical_and(vi_df["Future"]==0, vi_df["NanFrac"]==1), "Obs. Type"] = "Modelled"

	sns.kdeplot(data=vi_df, x="ObsGap", hue="Obs. Type", fill=True, 
		alpha=0.50, ax=ax, common_norm=False, legend=False,)
	# breakpoint()
	# handles, labels = ax.get_legend_handles_labels()
	# ax.legend()
	ax.set_xlabel("Years between Observations")
	ax.set_ylabel("Probability Density")
	ax.set_title("")
	ax.set_title(f"{title}", loc= 'left')

def _mapgridder(exp, vi_df, fig, ax, map_proj, lons, lats, title="", modelled=True, 
	future=False, vmin=0, vmax=1000):
	# ========== Simple lons and lats ========== 
	# ========== Setup params ==========
	""" Function to convert the points into a grid """
	
	# ========== Copy the df so i can export multiple grids ==========
	if future:
		# included so that i can pull out the future data if needed
		dfC = vi_df.loc[vi_df["Future"]==1].copy()
	else:
		dfC = vi_df.loc[vi_df["Future"]==0].copy()

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
		if title == "":
			title = "No. of Modelled Sites"
	else:
		vas   = "TotalSites"
		if title == "":
			title = "No. of Sites"
	levels = [0, 1,  5, 10, 50, 100, 500, 1000]

	f = ds[vas].isel(time=0).plot(
		x="longitude", y="latitude", transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, levels=levels,
		cbar_kwargs={"pad": 0.015, "shrink":0.80, "extend":"max", "label": "Observations"},	ax=ax)

	ax.set_extent([lons.min()+15, lons.max()-3, lats.min()-5, lats.max()-10])
	# print([lons.min()+15, lons.max()-5, lats.min()-5, lats.max()-10])
	# ax.set_extent([lons.min()+10, lons.max()-5, lats.min()-13, lats.max()])
	ax.gridlines()
	os.environ["CARTOPY_USER_BACKGROUNDS"] = "./data/Background/"
	# breakpoint()
	ax.background_img(name='BM', resolution='low')

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

def _annualcount(vi_df, fig, ax, inclfin=True, title="",):
	if inclfin:
		cats = ["Total", "Modelled", "Final"]
	else:
		cats = ["Total", "Modelled"]
	"""Line graph of the amount of sites included"""
	# ========== Duplicate the datafrema ==========
	sub = vi_df[np.logical_and((vi_df["NanFrac"] == 1), (vi_df["Future"] == 0))].copy()
	fut = vi_df[(vi_df["Future"] == 1)].copy()
	tot = vi_df.copy()
	sub["Count"] = "Modelled"
	sub["Count"] = pd.Categorical(["Modelled" for i in range(sub.shape[0])], categories=cats, ordered=True)
	if inclfin:
		fut["Count"] = pd.Categorical(["Final" for i in range(fut.shape[0])], categories=cats, ordered=True)

	# tot["Count"] = "Total"
	tot["Count"] = pd.Categorical(["Total" for i in range(tot.shape[0])], categories=cats, ordered=True)

	# ========== stack the results ==========
	if inclfin:
		df = pd.concat([tot, sub, fut]).reset_index(drop=True)
	else:
		df = pd.concat([tot, sub]).reset_index(drop=True)
	
	vi_yc = df.groupby(["Count", "year"])['biomass'].count().reset_index().rename(
		{"biomass":"Observations"}, axis=1).replace(0, np.NaN)
	# ========== Make the plot ==========
	sns.lineplot(y="Observations",x="year", hue="Count",dashes=[True, False, False], 
		data=vi_yc, ci=None, legend=True, ax = ax)
	# +++++ remove the title from the legend +++++
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles=handles, labels=labels)
	ax.set_xlabel("")
	ax.set_title("")
	ax.set_title(f"{title}", loc= 'left')

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

def VIload(regions, path, exp = None, 
	fpath  = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/", inclfin=True):
	

	print(f"Loading the VI_df, this can be a bit slow: {pd.Timestamp.now()}")
	
	vi_df   = pd.read_csv(f"{fpath}VI_df_AllSampleyears_ObsBiomass.csv", index_col=0)#[['lagged_biomass','ObsGap']]
	site_df = pd.read_csv(f"{fpath}SiteInfo_AllSampleyears_ObsBiomass.csv", index_col=0)
	vi_df["Future"] = 0
	# ========== open up the future sites ========= 
	if inclfin:
		vi_dfu   = pd.read_csv(f"{fpath}VI_df_AllSampleyears_FutureBiomass.csv", index_col=0)
		site_dfu = pd.read_csv(f"{fpath}SiteInfo_AllSampleyears_FutureBiomass.csv", index_col=0)
		vi_dfu["Future"] = 1

		# ========== Merge the dataframes ==========
		vi_df   = pd.concat([vi_df, vi_dfu])
		site_df = pd.concat([site_df, site_dfu])

	# ========== Fill in the missing sites ==========
	site_df.replace(regions, inplace=True)
	# region_fn =
	if exp is None:
		vi_df["NanFrac"] = vi_df.isnull().mean(axis=1)
		var = 'Delta_biomass'
	else:
		# Load in the different colkeys for each version 
		setup   = pd.read_csv(f"{path}{exp}/Exp{exp}_setup.csv", index_col=0)
		fnames  = sorted(glob.glob(f"{path}{exp}/Exp{exp}_*PermutationImportance.csv"))
		rowpass = np.zeros((vi_df.shape[0], len(fnames)))
		var     = setup.loc["predvar"].values[0]
		# breakpoint()
		for nu, fn in enumerate(fnames):
			# ========== get the list of cols ==========
			dfin = pd.read_csv( fn, index_col=0)
			cols = dfin["Variable"].values
			rowpass[:, nu] = (vi_df[cols].isnull().mean(axis=1) <=  setup.loc["DropNAN"].astype(float).values[0]).astype(float).values

		vi_df["NanFrac"] = rowpass.max(axis=1)

	vi_df = vi_df[['year', 'biomass', var,'ObsGap', "NanFrac", 'site', 'Future']]
	# for nanp in [0, 0.25, 0.50, 0.75, 1.0]:	
	# 	isin = (vi_df["NanFrac"] <=nanp).astype(float)
	# 	isin[isin == 0] = np.NaN
	# 	vi_df[f"{int(nanp*100)}NAN"]  = isin

	fcount = pd.melt(vi_df.drop([var,"NanFrac"], axis=1).groupby("ObsGap").count(), ignore_index=False).reset_index()
	fcount["variable"] = fcount["variable"].astype("category")
	
	vi_df["Region"]    = site_df["Region"].astype("category")
	vi_df["Longitude"] = site_df["Longitude"]
	vi_df["Latitude"]  = site_df["Latitude"]
	# breakpoint()

	return vi_df, fcount

# ==============================================================================

if __name__ == '__main__':
	main()