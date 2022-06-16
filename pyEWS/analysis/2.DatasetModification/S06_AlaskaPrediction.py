"""
Script goal, 

Rerbuild a new version of the vegetation datasets. This script will be modular in a way that allows for new 
and imporved datasets to be built on the fly 
	- Find the survey dates and biomass estimates
	- Add a normalisation option to see if that imporves things 
	- Add remotly sensed estimates of stand age 

"""

# ==============================================================================

__title__ = "DatasetSite check"
__author__ = "Arden Burrell"
__version__ = "v1.0(18.06.2020)"
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
# import ipdb
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
import seaborn as sns
from tqdm import tqdm
# from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf
import myfunctions.benchmarkfunctions as bf
import utm

# ========== Import packages for parellelisation ==========
# import multiprocessing as mp

# ========== Import ml packages ==========
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.inspection import permutation_importance
# from sklearn import metrics as sklMet
# from sklearn.utils import shuffle
# from scipy.stats import spearmanr
# from scipy.cluster import hierarchy
# import xgboost as xgb


# ==============================================================================

def main():

	df = pd.read_csv("./data/GPS_Reading_USAGAK_FIA.csv")
	cords = np.array([ utmconvert(Est, Nort, Zone) for Est, Nort, Zone in zip(df.Easting, df.Northing, df.Zone)])
	# dfout = pd.DataFrame(df["Loc"])
	# breakpoint()
	dfout = df[["Loc", "Pt"]].rename({"Loc":"ID1", "Pt":"ID2"}, axis=1)
	dfout["lat"]  = cords[:, 0]
	dfout["long"] = cords[:, 1]
	# Pace holder
	dfout["el"]   = 300
	# dfout[["Latitude", "Longitude"]] = cords

	breakpoint()

	dfout.to_csv("./data/Serdp_locations.csv")
	dfot = dfout.groupby(["ID1"]).mean().reset_index()
	dfot.to_csv("./data/Serdp_locations_abridged.csv")

def utmconvert(Est, Nort, Zone):
	if any(np.isnan([Est, Nort, Zone])):
		return (np.NaN, np.NaN)
	if Est < 100000:
		print(Est)
		Est *= 10
	

	try:
		return utm.to_latlon(Est, Nort, Zone, northern=True)
	except:
		breakpoint()

if __name__ == '__main__':
	main()