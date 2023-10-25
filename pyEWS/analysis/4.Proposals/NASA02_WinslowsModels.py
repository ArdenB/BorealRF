"""
NASAproposal
"""

# ==============================================================================

__title__ = "Model"
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
<<<<<<< HEAD
import rioxarray 
=======
>>>>>>> 12bf2f27a583055a62b58c7de5523e335e4f4330
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
<<<<<<< HEAD
from pyproj import CRS
# ==============================================================================

def main():


	# crs = CRS("ESRI:102003") 
	# cf_grid_mapping = crs.to_cf()
	crs = CRS("EPSG:4326") 
	cf_grid_mapping = crs.to_cf()

	# ========== Load the files ==========
	dsrf = xr.open_dataset("./data/RFdata/Model/regen.failure.nc")#.rename({"easting":"latitude", "northing":"longitute"})
	dsa = dsrf["regen.failure"]
	dsrf["regen.failure"].attrs = {}
	dsr = dsrf["regen.failure"].rio.set_spatial_dims(x_dim= "easting", y_dim = "northing")
	ds  = dsr.rio.write_crs("ESRI:102039")
	# ds.rio.to_raster("./data/RFdata/Model/regen_failure_fixedv20.tif")

	dsc = dsr.coarsen({"easting":3, "northing":3}, boundary="trim").max()
	dsc = dsc.fillna(-1)
	dsc = dsc.where(dsc != dsc.rio.nodata)
	dsc = dsc.rio.write_nodata(dsc.rio.nodata, encoded=True)
	dsc = dsc.rio.set_spatial_dims(x_dim= "easting", y_dim = "northing")


	dscr  = dsc.rio.write_crs("ESRI:102039")
	dscr.rio.to_raster("./data/RFdata/Model/regen_failure_coarsenv3.tif")
	breakpoint()
	

	dssf = xr.open_dataset("./data/RFdata/Model/stable.forest.nc")#.rename({"easting":"latitude", "northing":"longitute"})
	# dssa = dsrf["stable.forest"]
	dssf["stable.forest"].attrs = {}
	dssr = dssf["stable.forest"].rio.set_spatial_dims(x_dim= "easting", y_dim = "northing")
	dss  = dssr.rio.write_crs("ESRI:102039")
	# dss.rio.to_raster("./data/RFdata/Model/stable_forest_fixed.tif")
	
	


	# "EPSG",8823



	# dsr2 = dsrf["regen.failure"].rio.set_spatial_dims(x_dim= "northing", y_dim = "easting" )
	# ds2  = dsr2.rio.write_crs("ESRI:102003")

	# ds2.rio.to_raster("./data/RFdata/Model/regen_failure_fixedv4.tif")

	# breakpoint()
	# rds = rds.rename(lon=longitute, lat=latitude)

	# xds_lonlat = ds.rio.reproject("EPSG:4326")
	# xds = xds_lonlat.rio.write_crs("EPSG:4326")
	# xds.rio.to_raster("./data/RFdata/Model/regen_failure_fixedv13.tif")

	# breakpoint()

	# xds.y -= 18.4
	# xds.crs = cf_grid_mapping
	# xds_lonlat.to_netcdf("./data/RFdata/Model/regen_failure_fixed.nc")

	# ds2  = dsr.rio.write_crs("SR-ORG:7480")

	# xds_lonlat = ds.rio.reproject("EPSG:4326")
	# xds = xds_lonlat.rio.write_crs("EPSG:4326")
	# xds.rio.to_raster("./data/RFdata/Model/regen_failure_fixedv13.tif")

	breakpoint()



if __name__ == '__main__':
	main()
=======
>>>>>>> 12bf2f27a583055a62b58c7de5523e335e4f4330
