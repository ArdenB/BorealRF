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
# from scipy.stats import spearmanr
# from scipy.cluster import hierarchy


# ==============================================================================

def main():
	# ========== Setup the pathways ==========
	formats = None
	path  = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"
	cf.pymkdir(path+"plots/")
	# ppath = "./pyEWS/analysis/3.Izanami/Figures/PS06/"
	# cf.pymkdir(ppath)
	# dpath = 
	# fn="/mnt/f/Data51/NDVI/5.MODIS/NorthAmerica/MOD13Q1.006_250m_aid0001.nc"
	fn="./data/NDVI/5.MODIS/NorthAmerica/MOD13Q1.006_250m_aid0001.nc"
	yr       = 2020
	exp      = 402
	maxdelta = 20

	# ========== Load the data ==========
	# regions       = regionDict()
	# vi_df, fcount = VIload(regions, path, exp = exp)
	
	# ========== Simple lons and lats ========== 
	lons = np.arange(-170, -50.1,  0.5)
	lats = np.arange(  42,  70.1,  0.5)

	# ========== Load the Modeled data ==========
	df = fpred(path, exp, yr, maxdelta=maxdelta)


	# ========== Load the MODIS data ========== 
	df = modis(df, yr, fn=fn)
	
	# ========== Convert to a dataarray ==========
	ds = gridder(path, exp, years, df, lats, lons)

	# ========== Setup and build the maps ========== 


# ==============================================================================

# ==============================================================================
def modis(df, yr, fn="/mnt/f/Data51/NDVI/5.MODIS/NorthAmerica/MOD13Q1.006_250m_aid0001.nc"):
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
	
	# ========== load the file =========
	ds = xr.open_dataset(fn).rename({"_250m_16_days_NDVI":"NDVI", "lon":"longitude", "lat":"latitude"}).chunk({"latitude":500})
	df["VIdelta"] = np.NaN
	# ========== index the dataset ==========
	# with dask.config.set(**{'array.slicing.split_large_chunks': True}):
	gb   = df.groupby("Plot_ID").mean().loc[:, ["Longitude", "Latitude"]].sort_values("Latitude", ascending=False).reset_index()
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
	# with ProgressBar():	
	# 	da = ds["NDVI"].sel({"latitude":gb[:500].Latitude.values, "longitude":gb[:500].Longitude.values}, method="nearest").compute()
	breakpoint()
	# da = ds["NDVI"].sel({"latitude":gb[:50].Latitude.values, "longitude":gb[:50].Longitude.values}, method="nearest")
	breakpoint()


	breakpoint()
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
