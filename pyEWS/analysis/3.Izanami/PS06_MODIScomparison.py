"""
Modis Comparison
"""

# ==============================================================================

__title__ = "Script to compare MODIS and BIOMASS changes"
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

import cartopy.feature as cpf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ========== Import ml packages ==========
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.inspection import permutation_importance
from sklearn import metrics as sklMet
# from sklearn.utils import shuffle
from scipy.stats import spearmanr
# from scipy.cluster import hierarchy


# ==============================================================================

def main():
	# ========== Setup the pathways ==========
	formats = None
	path  = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"
	cf.pymkdir(path+"plots/")
	ppath = "./pyEWS/analysis/3.Izanami/Figures/PS06/"
	cf.pymkdir(ppath)
	fn="./data/NDVI/5.MODIS/NorthAmerica/MOD13Q1.006_250m_aid0001.nc"
	# dpath = 
	# fn="/mnt/f/Data51/NDVI/5.MODIS/NorthAmerica/MOD13Q1.006_250m_aid0001.nc"
	
	yr       = 2020
	exp      = 434
	maxdelta = 20

	# ========== Load the data ==========
	df = fulldata(path, exp, yr, maxdelta=maxdelta)
	# regions       = regionDict()
	# vi_df, fcount = VIload(regions, path, exp = exp)
	# df_ls = pd.read_csv("./data/NDVI/6.Landsat/lsat_nam_psps_ndvi_timeseries_20210628.csv")
	# def cusfun(dss):
	# 	lr = sp.stats.linregress(x=dss.year, y=dss["ndvi.max"])
	# 	# breakpoint()
	# 	return lr[0:3]
	# test = df_ls.groupby("site")[["year","ndvi.max"]].apply(cusfun)
	# out = pd.DataFrame(test.tolist(), columns=['slope','Intercept', "R2"], index=test.index)
	
	# ========== Simple lons and lats ========== 
	lons = np.arange(-170, -50.1,  0.5)
	lats = np.arange(  42,  70.1,  0.5)


	
	# ========== Convert to a dataarray ==========
	ds = gridder(path, exp, df, lats, lons)
	print()
	print(spearmanr(df.dropna().lsVItrend.values, df.dropna().DeltaBiomass.values))
	# print(spearmanr(df.dropna().lsVItrend.values, df.dropna().DeltaBiomass.values))
	# breakpoint()

	# ========== Setup and build the maps ========== 
	VIMapper(df, ds, ppath, lats, lons, var = "AGBgain", vivar = "lsVItrendgain")
	VIMapper(df, ds, ppath, lats, lons)
	breakpoint()


# ==============================================================================

def fulldata(path, exp, yr, maxdelta=20, fn="./data/NDVI/5.MODIS/NorthAmerica/MOD13Q1.006_250m_aid0001.nc"):

	# ds = xr.open_dataset(fn).chunk()
	# da = ds["_250m_16_days_VI_Quality"]
	# with ProgressBar(): 
	# 	dap = da.polyfit(dim="time", deg=1)
	# breakpoint()

	
	fnout = f"./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/{exp}/Exp{exp}_MODISandLANDSATndvi_delta.csv"
	if os.path.isfile(fnout):
		dfin = pd.read_csv(fnout, index_col=0)
		dfin["time"] = pd.to_datetime(dfin.time)
		return dfin
	else:
		# ========== Load the Modeled data ==========
		df = fpred(path, exp, yr, maxdelta=maxdelta)

		# ========== Load the MODIS data ========== 
		df = modis(df, yr, exp, fn=fn)

	# ========== Bring in the landsat data ========== 
	df_ls = pd.read_csv("./data/NDVI/6.Landsat/lsat_nam_psps_ndvi_timeseries_20210628.csv")
	Mean  = df_ls.groupby(["site"])["ndvi.max"].transform('mean')    
	Std   = df_ls.groupby(['site'])["ndvi.max"].transform('std')
	df_ls["Zscore"] = (df_ls["ndvi.max"] - Mean)/Std
	
	dfsimp = df.groupby(["Plot_ID"]).first().reset_index()

	pd.options.mode.chained_assignment = None
	df["lsVItrend"] = np.NaN
	df["lsVIdelta"] = np.NaN
	df["lsVIzscore"] = np.NaN
	for site, yrls in tqdm(zip(dfsimp.Plot_ID.values, dfsimp.year.values), total=dfsimp.Plot_ID.values.size):
		lssub = df_ls.loc[df_ls.site == site]
		try:
			df.loc[df.Plot_ID == site, "lsVIdelta"] = float(lssub.loc[lssub.year==int(2020), "ndvi.max"].values - lssub.loc[lssub.year==int(yrls), "ndvi.max"].values)
			df.loc[df.Plot_ID == site, "lsVIzscore"] = float(lssub.loc[lssub.year==int(2020), "Zscore"].values - lssub.loc[lssub.year==int(yrls), "Zscore"].values)
			df.loc[df.Plot_ID == site, "lsVItrend"] = sp.stats.linregress(x=lssub.year, y=lssub["ndvi.max"])[0]
			# breakpoint()
		except TypeError:
			# df.loc[df.Plot_ID == site, "lsVIdelta"] = np.NaN
			pass
		# raise e
	# LSVI.append(lssub.loc[lssub.year==int(yrls), "ndvi.max"].values - lssub.loc[lssub.year==int(2020), "ndvi.max"].values)
	df["AGBincrease"]  = df.DeltaBiomass > 0
	df["VIincrease"]   = df.VIdelta > 0
	df["lsVIincrease"] = df.lsVIdelta > 0
	df["lsVITincrease"] = df.lsVItrend > 0
	df.loc[np.isnan(df.lsVIdelta), "lsVIincrease"] = np.NaN
	df.to_csv(fnout)
	return df



def VIMapper(df, ds, ppath, lats, lons, var = "MeanDeltaBiomass", vivar = "MeanVIdelta"):

	# ========== Create the mapp projection ==========
	map_proj = ccrs.LambertConformal(central_longitude=lons.mean(), central_latitude=lats.mean())

	# ========== Create the figure ==========
	fig  = plt.figure(constrained_layout=True, figsize=(16,6))
	spec = gridspec.GridSpec(ncols=4, nrows=1, figure=fig, width_ratios=[11,1,11,1])

	for pos in range(ds.time.size):
		ax1 = fig.add_subplot(spec[pos, 0], projection= map_proj)
		_simplemapper(ds, var, fig, ax1, map_proj, pos, "Delta Biomass", lats, lons,  dim="Version")

		ax2 = fig.add_subplot(spec[pos, 2], projection= map_proj)
		_simplemapper(ds, vivar, fig, ax2, map_proj, pos, "Delta NDVI", lats, lons,  dim="Version")
	# vas   = 
	# title = 
	# # ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS06_PaperFig06_PredvsVI" 
	for ext in [".png", ".pdf"]:#".pdf",
		plt.savefig(fnout+ext)#, dpi=130)
	
	plotinfo = "PLOT INFO: Paper figure made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()
	breakpoint()
# ==============================================================================
def gridder(path, exp, df, lats, lons, var = "DeltaBiomass", vivar = "VIdelta", lsvivar = "lsVIdelta"):

	# ========== Setup params ==========
	# plt.rcParams.update({'axes.titleweight':"bold","axes.labelweight":"bold", 'axes.titlesize':10})
	# font = {'family' : 'normal',
	#         'weight' : 'bold', #,
	#         'size'   : 10}
	# mpl.rc('font', **font)
	# sns.set_style("whitegrid")
	""" Function to convert the points into a grid """
	# ========== Copy the df so i can export multiple grids ==========
	dfC = df.copy()#.dropna()
	# breakpoint()
	dfC["longitude"] = pd.cut(dfC["Longitude"], lons, labels=bn.move_mean(lons, 2)[1:])
	dfC["latitude"]  = pd.cut(dfC["Latitude" ], lats, labels=bn.move_mean(lats, 2)[1:])
	# breakpoint()
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
	dsVImean   = dfC.groupby(["time","latitude", "longitude", "Version"])[vivar].mean().to_xarray().sortby("latitude", ascending=False)
	dsVImedian = dfC.groupby(["time","latitude", "longitude", "Version"])[vivar].median().to_xarray().sortby("latitude", ascending=False)
	dfAGBgain = dfC.groupby(["time","latitude", "longitude", "Version"])["AGBincrease"].mean().to_xarray().sortby("latitude", ascending=False)*2-1
	dfVIgain  = dfC.groupby(["time","latitude", "longitude", "Version"])["VIincrease"].mean().to_xarray().sortby("latitude", ascending=False)*2-1
	dslsVImean   = dfC.groupby(["time","latitude", "longitude", "Version"])[lsvivar].mean().to_xarray().sortby("latitude", ascending=False)
	dflsVIgain  = dfC.groupby(["time","latitude", "longitude", "Version"])["lsVIincrease"].mean().to_xarray().sortby("latitude", ascending=False)*2-1
	dslsVItrend   = dfC.groupby(["time","latitude", "longitude", "Version"])["lsVItrend"].mean().to_xarray().sortby("latitude", ascending=False)
	dslsVItrendgain   = dfC.groupby(["time","latitude", "longitude", "Version"])["lsVITincrease"].mean().to_xarray().sortby("latitude", ascending=False)*2-1
	# breakpoint()
	# ========== Convert the different measures into xarray formats ==========
	ds = xr.Dataset({
		"sites":dscount, 
		"sitesInc":dspos, 
		f"Mean{var}":dsmean, 
		f"Median{var}":dsmedian, 
		f"AnnualMeanBiomass":dsannual,
		"MeanVIdelta":dsVImean, 
		"MedianVIdelta":dsVImedian, 
		"lsMeanVIdelta":dslsVImean,
		"AGBgain":dfAGBgain,
		"VIgain":dfVIgain,
		"lsVIgain":dflsVIgain,
		"lsVItrend":dslsVItrend,
		"lsVItrendgain":dslsVItrendgain
		})
	return ds

def modis(df, yr, exp, fn="/mnt/f/Data51/NDVI/5.MODIS/NorthAmerica/MOD13Q1.006_250m_aid0001.nc"):
	"""
	Func to open modis data and convert it to a format that matchs the existing 
	modeled dataframe
	
	args: 
		df:		pd dataframe
			the predicted values
		fn:		File name of netcdf
	"""
	if not os.path.isfile(fn):
		warn.warn("file is missing")
		fn="/mnt/d/Data51/NDVI/5.MODIS/NorthAmerica/MOD13Q1.006_250m_aid0001.nc"
		# breakpoint()
	
	fnout = f"./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/{exp}/Exp{exp}_MODISndvi_delta.csv"
	if os.path.isfile(fnout):
		dfin = pd.read_csv(fnout, index_col=0)

		if (dfin.shape[0] == df.shape[0]) and (df["Plot_ID"].equals(dfin["Plot_ID"])):
			df["VIdelta"] = dfin["VIdelta"]
			return df
		else:
			breakpoint()
	
	# ========== load the file =========
	ds = xr.open_dataset(fn).rename({"_250m_16_days_NDVI":"NDVI", "lon":"longitude", "lat":"latitude"}).chunk({"latitude":500})
	df["VIdelta"] = np.NaN
	# ========== index the dataset ==========
	# with dask.config.set(**{'array.slicing.split_large_chunks': True}):
	gb   = df.groupby("Plot_ID").mean().loc[:, ["Longitude", "Latitude"]].sort_values("Latitude", ascending=False).reset_index()
	
	# with dask.config.set(**{'array.slicing.split_large_chunks': True}):
	# 	with ProgressBar():	
	# 		da = ds["NDVI"].sel({"latitude":gb[:500].Latitude.values, "longitude":gb[:500].Longitude.values}, method="nearest").compute()
	# breakpoint()



	vals = OrderedDict()
	pd.options.mode.chained_assignment = None
	for ind in tqdm(np.arange(gb.shape[0])):
		dt = ds["NDVI"].sel({"latitude":gb.iloc[ind].Latitude, "longitude":gb.iloc[ind].Longitude}, method="nearest").compute()
		vals[gb.iloc[ind].Plot_ID] = dt

		dt = dt.groupby("time.year").max("time")#.compute(
		# df.loc[df.Plot_ID == gb.iloc[ind].Plot_ID]

		dfsel = df.loc[df.Plot_ID == gb.iloc[ind].Plot_ID]
		for yrst in dfsel.year.astype(int).unique():
			val = dt.sel(year=yr).values - dt.sel(year=yrst).values
			df["VIdelta"].loc[np.logical_and(df.Plot_ID == gb.iloc[ind].Plot_ID, df.year==yrst)] = val
		# breakpoint()

	df.loc[:, ["Plot_ID", "Longitude", "Latitude", "VIdelta"]].to_csv(fnout)
	# da = ds["NDVI"].sel({"latitude":gb[:50].Latitude.values, "longitude":gb[:50].Longitude.values}, method="nearest")
	# breakpoint()
	return df


	# breakpoint()
	# da = ds["NDVI"].groupby("time.year").max("time").compute()

def fpred(path, exp, yr, 
	fpath    = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/", 
	maxdelta = 30, drop=True):
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
		# breakpoint()
		# dfout.loc[dfout["Longitude"] == 0, ["Longitude", "Latitude"]] = np.NaN
		# dfout.loc[dfout["Longitude"] < -180, ["Longitude", "Latitude"]] = np.NaN
		# dfout.loc[dfout["Longitude"] >  180, ["Longitude", "Latitude"]] = np.NaN
		# dfout.loc[dfout["Latitude"] <= 0, ["Longitude", "Latitude"]] = np.NaN
		# dfout.loc[dfout["Latitude"] >  90, ["Longitude", "Latitude"]] = np.NaN

		# breakpoint()
		dfoutC = dfout.copy()

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
		if drop:
			dfoutC = dfoutC.loc[dfoutC.year > (yr - maxdelta)].dropna()
		est_list.append(dfoutC)



	return pd.concat(est_list)

def VIload(regions, path, exp = None):
	print(f"Loading the VI_df, this can be a bit slow: {pd.Timestamp.now()}")
	vi_df = pd.read_csv("./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/VI_df_AllSampleyears_ObsBiomass.csv.csv", index_col=0)#[['lagged_biomass','ObsGap']]

	# ========== Fill in the missing sites ==========
	region_fn ="./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/SiteInfo_AllSampleyears_ObsBiomass.csv.csv"
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

def _simplemapper(ds, vas, fig, ax, map_proj, indtime, title, lats, lons,  dim="Version"):
	f = ds[vas].mean(dim=dim).isel(time=indtime).plot(
		x="longitude", y="latitude", #col="time", col_wrap=2, 
		transform=ccrs.PlateCarree(), 
		cbar_kwargs={"pad": 0.015, "shrink":0.65},#, "extend":extend}
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
# ==============================================================================
if __name__ == '__main__':
	main()
