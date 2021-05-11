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

# ========== Import ml packages ==========
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.inspection import permutation_importance
from sklearn import metrics as sklMet
# from sklearn.utils import shuffle
# from scipy.stats import spearmanr
# from scipy.cluster import hierarchy

import xgboost as xgb
import xarray as xr
import cartopy.crs as ccrs

# ==============================================================================

def main():
	# ========== Get the file names and open the files ==========
	formats = None
	path  = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"
	cf.pymkdir(path+"plots/")
	ppath = "./pyEWS/analysis/3.Izanami/Figures/PS04/"
	cf.pymkdir(ppath)
	
	# expr = OrderedDict()
	# expr['DeltaBiomass']  = [402, 405]
	# expr['Delta_biomass'] = [402, 405, 406] 
	# expr["Predictors"]    = [400, 401, 402] 
	# expr['Obs_biomass']   = [401, 403, 404] 
	# expr["Complete"]      = [400, 401, 402, 403, 404, 405, 406] 
	# var  = "PermutationImportance"
	# var  = "Importance"
	experiments = [402, 401]
	years       = [2020, 2030, 2040]
	# ========== Simple lons and lats ========== 
	lons = np.arange(-170, -50.1,  0.50)
	lats = np.arange(  42,  70.1,  0.50)

	for exp in experiments:
		df = fpred(path, exp, years)
		# ========== Convert to a dataarray ==========
		gridder(path, exp, years, df, lats, lons)
		breakpoint()


	# for epnm in expr:
	# 	for exp in expr[epnm]:



# ==============================================================================
def gridder(path, exp, years, df, lats, lons, var = "DeltaBiomass"):
	""" Function to convert the points into a grid """
	# ========== Copy the df so i can export multiple grids ==========
	dfC = df.copy().dropna()

	dfC["longitude"] = pd.cut(dfC["Longitude"], lons, labels=bn.move_mean(lons, 2)[1:])
	dfC["latitude"]  = pd.cut(dfC["Latitude" ], lats, labels=bn.move_mean(lats, 2)[1:])


	# ========== Convert the different measures into xarray formats ==========
	dscount  = dfC.groupby(["time","latitude", "longitude", "Version"])[var].count().to_xarray().sortby("latitude", ascending=False)
	dscount  = dscount.where(dscount>0)
	dspos    = dfC.loc[dfC["DeltaBiomass"]>0].groupby(["time","latitude", "longitude", "Version"])[var].count().to_xarray().sortby("latitude", ascending=False)/dscount
	dsmean   = dfC.groupby(["time","latitude", "longitude", "Version"])[var].mean().to_xarray().sortby("latitude", ascending=False)
	dsmedian = dfC.groupby(["time","latitude", "longitude", "Version"])[var].median().to_xarray().sortby("latitude", ascending=False)

	# ========== Convert the different measures into xarray formats ==========
	ds = xr.Dataset({"sites":dscount, "sitesInc":dspos, f"Mean{var}":dsmean, f"Median{var}":dsmedian})

	# ========== Make some example plots ==========
	map_proj = ccrs.LambertConformal(central_longitude=lons.mean(), central_latitude=lats.mean())
	
	for vas in ["sites", "sitesInc",  f"Mean{var}", f"Median{var}"]:
		p = ds[vas].mean(dim="Version").plot(
			transform=ccrs.PlateCarree(), 
			x="longitude", y="latitude", col="time", 
			col_wrap=2, 
			# size=6,	aspect=ds.dims['longitude'] / ds.dims['latitude'],  
			subplot_kws={'projection': map_proj})

		for ax in p.axes.flat:
		    ax.coastlines()
		    ax.set_extent([lons.min()+10, lons.max(), lats.min()-13, lats.max()])
		plt.show()
	breakpoint()



def fpred(path, exp, years, fpath  = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/"):
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
		dfout.loc[dfout["Longitude"] == 0, ["Longitude", "Latitude"]] = np.NaN
		dfout.loc[dfout["Longitude"] < -180, ["Longitude", "Latitude"]] = np.NaN
		dfout.loc[dfout["Longitude"] >  180, ["Longitude", "Latitude"]] = np.NaN
		dfout.loc[dfout["Latitude"] <= 0, ["Longitude", "Latitude"]] = np.NaN
		dfout.loc[dfout["Latitude"] >  90, ["Longitude", "Latitude"]] = np.NaN

		# breakpoint()
		for yr in years:
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
			est_list.append(dfoutC)
			# breakpoint()



	return pd.concat(est_list)




# ==============================================================================
if __name__ == '__main__':
	main()