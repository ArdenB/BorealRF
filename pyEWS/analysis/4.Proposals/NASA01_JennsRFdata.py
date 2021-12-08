"""
NASAproposal
"""

# ==============================================================================

__title__ = "Recuitment data analysis"
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
import geopandas as gpd
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
import geopandas as gpd

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
		# ========== Setup the matplotlib params ==========
	plt.rcParams.update({
		'axes.titleweight':"bold", 'axes.titlesize':14, "axes.labelweight":"bold"})
	font = ({'family' : 'normal','weight' : 'bold', 'size': 14})
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	# map_proj = ccrs.LambertConformal(central_longitude=lons.mean(), central_latitude=lats.mean())

	# fig, ax = plt.subplots(constrained_layout=True, figsize=(13,7))



	df = pd.read_csv("./data/RFdata/BorealNA_Postfire_Regeneration_1989_2014.csv")
	df = df[~(df["total_density"] == 0)]
	df["RecRatio"] =  df["total_sdlng_sucker_dens"] / df["total_density"]
	# df2 = pd.read_csv("../fireflies/data/field/ProcessedFD.csv", index_col=0)

	# df.hist(column="RecRatio", range=(0, 1), bins=100)
	# plt.figure(2)
	# sns.histplot(data=df, x="RecRatio")
	df["Pathway"] = 0
	# df["total_sdlng_sucker_dens"]
	df.loc[df["RecRatio"] >= 1, "Pathway"] = 2
	df.loc[df["RecRatio"] < 1, "Pathway"] = 1
	df.loc[df["total_sdlng_sucker_dens"] == 0, "Pathway"] = 0
	df2 = Field_data()
	df3 = Field_data(year=2017)

	# breakpoint()

	# dfg = df.groupby(["Region","Plot_ID", "time"]).median().reset_index()
	gdf = gpd.GeoDataFrame(pd.concat([df[['longitude', 'latitude', 'Pathway']], df2, df3]))
	gdf.set_geometry(
	    gpd.points_from_xy(gdf['longitude'], gdf['latitude']),
	    inplace=True, crs='EPSG:4326')
	gdf.drop(['latitude', 'longitude'], axis=1, inplace=True)
	# gdf[["Pathway", "geometry"]].to_file('./data/RFdata/BorealNA_Postfire_Regeneration_1989_2014.shp')


	da = xr.open_rasterio("./data/RFdata/EuroasiaLidar.tif").rename({"y":"latitude", "x":"longitude"})
	da = da.where(da>=0)
	# dfx = df2.copy()
	dfx = pd.concat([df2, df3]).reset_index()#.groupby("sn").mean()
	dfx.replace({0:"Failure", 1:"Poor", 2:"Successfull"}, inplace=True)
	dfx["Pathway"] = dfx["Pathway"].astype("category")

	cph = np.array([da.sel({"band":1, "longitude":lon, "latitude":lat}, method='nearest').values	for lon , lat  in zip(dfx.longitude.values.tolist(), dfx.latitude.values.tolist())])
	cph[cph > 8] = np.NaN
	# breakpoint()
	dfx["CanopyHeight (m)"] = cph
	# sns.violinplot(y="CanopyHeight", x="Pathway",data=dfx)
	fig, ax = plt.subplots()
	sns.violinplot(y="CanopyHeight (m)", x="Pathway",data=dfx, ax=ax)
	sns.swarmplot(y="CanopyHeight (m)", x="Pathway",data=dfx, ax=ax, color='w')

	plt.savefig("./data/RFdata/testplot2.png")
	plt.show()
	# da.sel(dict(longitude=df2.longitude.values.tolist(), latitude=df2.latitude.values.tolist()), method='nearest').values
	breakpoint()

	
# ==============================================================================

def Field_data(year = 2018):
	"""
	# Aim of this function is to look at the field data a bit
	To start it just opens the file and returns the lats and longs 
	i can then use these to look up netcdf fils
	"""
	# ========== Load in the relevant data ==========
	if year == 2018:
		fd18 = pd.read_csv("../fireflies/data/field/2018data/siteDescriptions18.csv")
	else:
		fd18 = pd.read_csv("../fireflies/data/field/2018data/siteDescriptions17.csv")

	fd18.sort_values(by=["site number"],inplace=True) 
	# ========== Create and Ordered Dict for important info ==========
	info = OrderedDict()
	info["sn"]  = fd18["site number"]
	try:
		info["latitude"] = fd18.lat
		info["longitude"] = fd18.lon
		info["Pathway"]  = fd18.rcrtmnt
	except AttributeError:
		info["latitude"] = fd18.strtY
		info["longitude"] = fd18.strtX
		info["Pathway"]  = fd18.recruitment
	
	# ========== function to return nan when a value is missing ==========
	def _missingvalfix(val):
		try:
			return float(val)
		except Exception as e:
			return np.NAN

	def _fireyear(val):
		try:
			year = float(val)
			if (year <= 2018):
				return year
			else:
				return np.NAN
		except ValueError: #not a simple values
			try:
				year = float(str(val[0]).split(" and ")[0])
				if year < 1980:
					warn.warn("wrong year is being returned")
					year = float(str(val).split(" ")[0])
					# ipdb.set_trace()

				return year
			except Exception as e:
				# ipdb.set_trace()
				# print(e)
				print(val)
				return np.NAN
		
	# ========== Convert to dataframe and replace codes ==========
	# info["fireyear"] = [_fireyear(fyv) for fyv in fd18["estimated fire year"].values]
	RFinfo = pd.DataFrame(info).set_index("sn")

	# ipdb.set_trace()
	RFinfo.Pathway[    RFinfo["Pathway"].str.contains("poor")] = "RF"  #"no regeneration"
	RFinfo.Pathway[    RFinfo["Pathway"].str.contains("no regeneration")] = "RF" 
	RFinfo.Pathway[RFinfo["Pathway"].str.contains("singular")] = "IR"  
	for repstring in ["abundunt", "sufficient", "abundant", "sufficent", "sifficient"]:
		RFinfo.Pathway[RFinfo["Pathway"].str.contains(repstring)] = "AR"  
	

	RFinfo.replace({"RF":0, "IR":1, "AR":2}, inplace=True)
	return RFinfo

if __name__ == '__main__':
	main()