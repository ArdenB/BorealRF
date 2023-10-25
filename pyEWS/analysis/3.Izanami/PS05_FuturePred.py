"""
Boreal EWS PSP data anlysis 
 
Script to  make individaul psps plots  
"""

# ==============================================================================

__title__ = "Future Biomass Prediction"
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
from matplotlib.colors import LogNorm
import string
# from sklearn.utils import shuffle
# from scipy.stats import spearmanr
# from scipy.cluster import hierarchy


# ==============================================================================

def main():
	# ========== Get the file names and open the files ==========
	formats = None
	path  = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"
	cf.pymkdir(path+"plots/")
	ppath = "./pyEWS/analysis/3.Izanami/Figures/PS05/"
	cf.pymkdir(ppath)
	os.environ["CARTOPY_USER_BACKGROUNDS"] = "./data/Background/"
	
	# expr = OrderedDict()
	# expr['DeltaBiomass']  = [402, 405]
	# expr['Delta_biomass'] = [402, 405, 406] 
	# expr["Predictors"]    = [400, 401, 402] 
	# expr['Obs_biomass']   = [401, 403, 404] 
	# expr["Complete"]      = [400, 401, 402, 403, 404, 405, 406] 
	# var  = "PermutationImportance"
	# var  = "Importance"


	experiments = [434, 424]
	years1       = [2030]
	years2       = [2020, 2025, 2030, 2040]
	# ========== Simple lons and lats ========== 
	lons = np.arange(-170, -50.1,  0.5)
	lats = np.arange(  42,  70.1,  0.5)

	for exp in experiments:
		for years in [years1, years2]:
			# +++++ the final model results +++++



			df  = fpred(path, exp, years, lats, lons)

			# ========== Convert to a dataarray ==========
			ds = gridder(path, exp, years, df, lats, lons)

			# breakpoint()
			
			for fvar in ["EnsembleDirection","MeanDeltaBiomass", "sites",]:
				splotmap(exp, df, ds, ppath, lats, lons, fvar, years,)
			breakpoint()


			# FutureMapper(df, ds, ppath, lats, lons, var = "DeltaBiomass")
			
			# breakpoint()
			# dfg = df.groupby(["Region","Plot_ID", "time"]).median().reset_index()
			# gdf = gpd.GeoDataFrame(dfg)
			# gdf.set_geometry(
			#     geopandas.points_from_xy(gdf['Longitude'], gdf['Latitude']),
			#     inplace=True, crs='EPSG:4326')
			# gdf.drop(['Latitude', 'Longitude'], axis=1, inplace=True)
			# gdf[["DeltaBiomass", "geometry"]].to_file(f'{ppath}test.shp')

			# fnout = f"{ppath}Examplenetcdf.nc"
			# ds.to_netcdf(fnout, 
			# 	format         = 'NETCDF4', 
			# 	# encoding       = encoding,
			# 	unlimited_dims = ["time"])
			# breakpoint()


	# for epnm in expr:
	# 	for exp in expr[epnm]:



# ==============================================================================
def splotmap(exp, df, ds, ppath, lats, lons, var, years, 
	textsize=14, col_wrap=1, dim="Version", norm=None, robust=False):
	"""
	Plot function for a single var
	"""
	if var == "EnsembleDirection":
		cmap = mpc.ListedColormap(palettable.colorbrewer.diverging.PiYG_11.mpl_colors)
		cbkw = {"pad": 0.015, "shrink":1./len(years), "label": r"$\Delta$AGB Direction"}
	elif var == "sites":
		norm=LogNorm(vmin=1, vmax=1000,)
		cmap = mpc.ListedColormap(palettable.cmocean.sequential.Matter_20.mpl_colors)
		cbkw = {"pad": 0.015, "shrink":1./len(years), "extend":"max", "label":"No. Sites"}
	elif var in ["MedianDeltaBiomass", "MeanDeltaBiomass"]:
		cmap = mpc.ListedColormap(palettable.colorbrewer.diverging.BrBG_11.mpl_colors)
		cbkw = {"pad": 0.015, "shrink":1./len(years), "extend":"both", "label": r"$\Delta$AGB (t/ha/yr)"}
		robust=True
	else:
		breakpoint()
		cmap = mpc.ListedColormap(palettable.cmocean.sequential.Ice_20_r.mpl_colors)
	# ========== Create the figure ==========
	# plt.rcParams.update({'axes.titleweight':"bold", })
	font = ({'weight' : 'bold', 'size'   : textsize})
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	plt.rcParams.update({
		'axes.titleweight':"bold", 
		"axes.labelweight":"bold", 
		'axes.titlelocation': 'left',
		'axes.titlesize':textsize})

	# ========== Create the mapp projection ==========
	map_proj = ccrs.LambertConformal(central_longitude=lons.mean(), central_latitude=lats.mean())
	# fig  = plt.figure(constrained_layout=True, figsize=(18,ds.time.size*7))
	# breakpoint()
	if len(years) >1:
		f = ds[var].mean(dim).plot(
			x="longitude", y="latitude", col="time", 
			col_wrap=col_wrap, 
			transform=ccrs.PlateCarree(), 
			cbar_kwargs=cbkw,
			subplot_kws={'projection': map_proj}, 
			cmap=cmap, #size =8,
			figsize=(12, ds.time.size*3.75),
			norm=norm, 
			robust=robust,
			)
		fax = f.axes.flat
	else:
		# Defining the figure
		fig = plt.figure(figsize=(12, ds.time.size*4.5))

		# Axes with Cartopy projection
		ax1 = plt.axes(projection=map_proj)

		# x="longitude", y="latitude"
		f = ds[var].mean(dim).isel(time=0).plot(
			x="longitude", y="latitude", 
			transform=ccrs.PlateCarree(), 
			cbar_kwargs=cbkw,
			cmap=cmap, #size =8,
			norm=norm, 
			robust=robust, ax=ax1
			)
		fax = [ax1]
		plt.tight_layout()
		# norm=LogNorm(vmin=1, vmax=1000,)
		# size=6,	aspect=ds.dims['longitude'] / ds.dims['latitude'],  
	# for ax in :

	for ax, tit, year in zip(fax, string.ascii_lowercase, years):
		# ax.set_extent([lons.min()+15, lons.max()-3, lats.min()-5, lats.max()-10])
		# ax.background_img(name='BM', resolution='low')
		# ax.gridlines()
		ax.set_extent([lons.min()+15, lons.max()-3, lats.min()-3, lats.max()-6])
		ax.gridlines(alpha=0.5)
		ax.stock_img()

		coast = cpf.GSHHSFeature(scale="intermediate")
		ax.add_feature(cpf.LAND, facecolor='dimgrey', alpha=1, zorder=0)
		ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
		ax.add_feature(coast, zorder=101, alpha=0.5)
		ax.add_feature(cpf.BORDERS, linestyle='--', zorder=104)
		# ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
		# ax.add_feature(cpf.RIVERS, zorder=104)

		provinc_bodr = cpf.NaturalEarthFeature(category='cultural', 
			name='admin_1_states_provinces_lines', scale='50m', facecolor='none', edgecolor='k')

		ax.add_feature(provinc_bodr, linestyle='--', linewidth=0.6, edgecolor="k", zorder=105)

		if len(years) >1:
			ax.set_title(f"{tit}) {year}", loc= 'left')
		else:
			ax.set_title(f"{year}", loc= 'left')

	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS05_PaperFig04_FuturePredSvar_{ds.time.size}_{var}_exp{exp}" 
	for ext in [".png", ".pdf"]:#".pdf",
		plt.savefig(fnout+ext, dpi=300)
	
	plotinfo = "PLOT INFO: Paper figure made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()
	# breakpoint()

def FutureMapper(df, ds, ppath, lats, lons, var = "DeltaBiomass", textsize=24):



	# for vas, title in zip(["sites", "sitesInc",  f"Mean{var}", f"Median{var}", "AnnualMeanBiomass"],
	# 	["No. of Sites", "Direction of change",  f"Mean {var}", f"Median {var}", "Annual Mean Delta Biomass"]):
	# 	# break
	# 	# ========== Create the figure ==========
	# 	fig, ax = plt.subplots(
	# 		1, 1, sharex=True, subplot_kw={'projection': map_proj}, 
	# 		figsize=(20,9)
	# 		)
	# ========== Create the figure ==========
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':textsize})
	font = ({'weight' : 'bold', 'size'   : textsize})
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	plt.rcParams.update({'axes.titleweight':"bold", "axes.labelweight":"bold"})

	# ========== Create the mapp projection ==========
	map_proj = ccrs.LambertConformal(central_longitude=lons.mean(), central_latitude=lats.mean())

	# ========== Create the figure ==========
	fig  = plt.figure(constrained_layout=True, figsize=(10, 13))
	spec = gridspec.GridSpec(ncols=4, nrows=ds.time.size, figure=fig, width_ratios=[11,1,11,1])

	for pos in range(ds.time.size):
		ax1 = fig.add_subplot(spec[pos, 0], projection= map_proj)
		_simplemapper(ds, "sitesInc", fig, ax1, map_proj, pos, "Direction of change", lats, lons,  dim="Version")

		ax2 = fig.add_subplot(spec[pos, 2], projection= map_proj)
		_simplemapper(ds, "sites", fig, ax2, map_proj, pos, 
			"Number of Sites", lats, lons,  dim="Version", norm=True)
	# vas   = 
	# title = 
	# plt.subplots_adjust(left=0.04, right=1, top=0.95,)# wspace=0, hspace=0,  bottom=0, )
	# # ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS05_PaperFig04_FuturePred_{ds.time.size}" 
	for ext in [".png", ".pdf"]:#".pdf",
		plt.savefig(fnout+ext, dpi=300)
	
	plotinfo = "PLOT INFO: Paper figure made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()
	breakpoint()
	
	

def _simplemapper(ds, vas, fig, ax, map_proj, indtime, title, lats, lons,  
	dim="Version", norm=False, extend=None):
	if norm:
		dsa = ds[vas].mean(dim=dim).isel(time=indtime)
		dsa.attrs = ds[vas].attrs
		f = dsa.plot(
			x="longitude", y="latitude", #col="time", col_wrap=2, 
			transform=ccrs.PlateCarree(), 
			cbar_kwargs={"pad": 0.015, "shrink":0.65, "extend":"max"},
			# subplot_kws={'projection': map_proj}, 
			# size=6,	aspect=ds.dims['longitude'] / ds.dims['latitude'],  
			norm=LogNorm(vmin=1, vmax=1000,),
			ax=ax)
	else:
		cmap = palettable.colorbrewer.diverging.PiYG_11.mpl_colormap
		dsa = ds[vas].mean(dim=dim).isel(time=indtime)
		dsa.attrs = ds[vas].attrs
		f = dsa.plot(
			x="longitude", y="latitude", #col="time", col_wrap=2, 
			transform=ccrs.PlateCarree(), 
			cbar_kwargs={"pad": 0.015, "shrink":0.65,},
			cmap=cmap,
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

	# ax.set_title("")
	# ax.set_title(f"{title}", loc= 'left')
	# plt.tight_layout()
	# plt.show()

def gridder(path, exp, years, df, lats, lons, var = "DeltaBiomass"):

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
	dspos.attrs["units"] = "Fraction of sites Increasing"
	
	dsmean   = dfC.groupby(["time","latitude", "longitude", "Version"])[var].mean().to_xarray().sortby("latitude", ascending=False)
	dsmedian = dfC.groupby(["time","latitude", "longitude", "Version"])[var].median().to_xarray().sortby("latitude", ascending=False)
	dsannual = dfC.groupby(["time","latitude", "longitude", "Version"])["AnnualBiomass"].mean().to_xarray().sortby("latitude", ascending=False)
	dschange = dfC.groupby(["time","latitude", "longitude", "Version"])["ModelAgreement"].mean().to_xarray().sortby("latitude", ascending=False)
	# ========== Convert the different measures into xarray formats ==========
	ds = xr.Dataset({
		"sites":dscount, 
		"sitesInc":dspos, 
		f"Mean{var}":dsmean, 
		f"Median{var}":dsmedian, 
		f"AnnualMeanBiomass":dsannual,
		"EnsembleDirection":dschange})
	# breakpoint()
	return ds
	


def fpred(path, exp, years, lats, lons, var = "DeltaBiomass",
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
		# breakpoint()
		# dfout.loc[dfout["Longitude"] == 0, ["Longitude", "Latitude"]] = np.NaN
		# dfout.loc[dfout["Longitude"] < -180, ["Longitude", "Latitude"]] = np.NaN
		# dfout.loc[dfout["Longitude"] >  180, ["Longitude", "Latitude"]] = np.NaN
		# dfout.loc[dfout["Latitude"] <= 0, ["Longitude", "Latitude"]] = np.NaN
		# dfout.loc[dfout["Latitude"] >  90, ["Longitude", "Latitude"]] = np.NaN

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
				print(f"The Mean Observation gap for {yr} is {dfX.ObsGap.mean()}")

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
			dfoutC[f"DeltaBiomass"] /=(dfoutC.time.dt.year - dfoutC.year)
			est_list.append(dfoutC)

	df = pd.concat(est_list)
	df["longitude"] = pd.cut(df["Longitude"], lons, labels=bn.move_mean(lons, 2)[1:])
	df["latitude"]  = pd.cut(df["Latitude" ], lats, labels=bn.move_mean(lats, 2)[1:])
	df["ObsGap"]    = df.time.dt.year - df.year
	df = df[df.ObsGap <= 30]

	# ========== do the direction stuff ==========
	dft = df[[var, "Plot_ID", "year", "ObsGap"]].copy()
	dft[var] = dft[var]>=0
	df["ModelAgreement"] = (dft.groupby(["Plot_ID", "year", "ObsGap"]).transform("sum") - 5) / 5
	# breakpoint()
	return df




# ==============================================================================
if __name__ == '__main__':
	main()