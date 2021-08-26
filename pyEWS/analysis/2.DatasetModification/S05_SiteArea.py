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


	# ========== select the relevant data files ==========
	fpath = "./EWS_package/data/psp/"
	dpath = "./EWS_package/data/raw_psp/"
	# rpath = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"

	folder   = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/"
	fnamein  = f"./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/VI_df_AllSampleyears_ObsBiomass.csv"
	sfnamein = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/SiteInfo_AllSampleyears_ObsBiomass.csv"
	regdf    = pd.read_csv("./EWS_package/data/raw_psp/All_sites_Regions.csv", index_col=0)
	site_df  = pd.read_csv(sfnamein, index_col=0)
	# vi_df    = pd.read_csv(fnamein, index_col=0)
	# vi_df["Plot_ID"] = site_df["Plot_ID"]
	sitelist = regdf["Plot_ID"].values.tolist()
	regkey  = regions()
	areaOD = OrderedDict()
	# area    = []
	# group_kfold = GroupKFold(n_splits=10)
	for region in regkey:
		print(region)
		fnames = glob.glob(f"{dpath}{region}/checks/*.csv")
		for fn in tqdm(fnames, total=len(fnames)):
			# ===== Pull out the site name =====
			sn   = fn.split("/")[-1].replace("_check.csv", '')
			site = f"{regkey[region]}{sn}" 
			try:
				dfin = pd.read_csv(fn, index_col=0)
				miss = False
				if not site in sitelist:
					miss=True

				if not dfin.plot_size.unique().size==1:
					breakpoint()

				areaOD[site] = {"plot_size": dfin.plot_size.unique()[0], "region":region, "missingGPS":miss}
			except Exception as err:
				areaOD[site] = {"plot_size": np.NaN, "region":region, "missingGPS":miss}
				# print(err)
				# breakpoint()
	dfout = pd.DataFrame(areaOD)
	dfout.to_csv("./EWS_package/data/raw_psp/All_sites_area.csv", index_col=0)
	breakpoint()

def regions():
	regkey = ({# values from survey_years.R
		"BC"   :"1_",
		"AB"   :"2_",
		"SK"   :"3",
		"MB"   :"4_",
		"ON"   :"5", # Possibly 5_
		"QC"   :"6_",
		"NL"   :"7_",
		"NB"   :"8_",
		"NS"   :"9_", 
		"YT"   :"11_",
		"NWT"  :"12_",
		"CAFI" :"13_",
		"CIPHA":"14", 
		})
	return regkey


if __name__ == '__main__':
	main()	