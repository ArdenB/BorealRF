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
	rpath = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"

	folder   = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/"
	fnamein  = f"./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/VI_df_AllSampleyears_ObsBiomass.csv"
	sfnamein = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/SiteInfo_AllSampleyears_ObsBiomass.csv"
	site_df  = pd.read_csv(sfnamein, index_col=0)
	vi_df    = pd.read_csv(fnamein, index_col=0)
	vi_df["Plot_ID"] = site_df["Plot_ID"]
	# group_kfold = GroupKFold(n_splits=10)

	dfl  = pd.read_csv(f"{folder}TTS_VI_df_AllSampleyears_10FWH_TTSlookup.csv", index_col=0)
	expo = 402
	expn = 411
	dfp  = pd.read_csv(f"{rpath}{expn}/Exp411_OneStageXGBOOST_AllGap_50perNA_PermutationImp_RFECV_FINAL_Delta_biomass_altsplit_vers00_OBSvsPREDICTED.csv", index_col=0)
	metl = []
	
	# ========== Take a look for fully witheld sites ==========
	for n in range(10):
		X_test  = pd.read_csv(f"{folder}TTS_VI_df_AllSampleyears_vers0{n}_X_test.csv", index_col=0)
		X_train = pd.read_csv(f"{folder}TTS_VI_df_AllSampleyears_vers0{n}_X_train.csv", index_col=0)
		# ========== read in the results ==========
		dfpo = pd.read_csv(f"{rpath}{expo}/Exp{expo}_OneStageXGBOOST_AllGap_50perNA_PermutationImp_RFECV_FINAL_Delta_biomass_vers0{n}_OBSvsPREDICTED.csv", index_col=0)
		# dfpo["Plot_ID"] = site_df.loc[dfpo.index,"Plot_ID"]
		test_sites      = site_df.loc[dfpo.index,"Plot_ID"]#.unique()
		
		train_sites = site_df.loc[X_train.index,"Plot_ID"].unique()
		all_site    = np.unique(np.hstack([site_df.loc[X_test.index,"Plot_ID"], site_df.loc[X_train.index,"Plot_ID"]]))

		isuni     = [ts not in train_sites for ts in test_sites]

		# dfl.loc[X_test[isuni].index.values, "0"]
		
		inboth = [tsi in dfp.index.values for tsi in dfpo[isuni].index.values]
		# dfpo[isuni]

		dfm = dfpo[isuni].loc[inboth].copy()
		dfm["ResO"] = dfm.Estimated - dfm.Observed  
		dfm["EstP"] = dfp.loc[dfm.index, "Estimated"]
		dfm["ResP"] = dfm.EstP - dfm.Observed  
		metl.append(dfm)
		print(sum(isuni)/float(dfpo.shape[0]) * 100.0)

	dftm = pd.concat(metl, ignore_index=True)
	dftm.ResP.abs() < dftm.ResO.abs() 

	breakpoint()
	# ========== group tts ==========
	gss = GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
	y = vi_df['Delta_biomass']


	info = OrderedDict()
	regions = OrderedDict()
	for num, (train, test) in enumerate(gss.split(site_df, y, groups=site_df["Plot_ID"])):
		X_test  = site_df.loc[test,]
		X_train = site_df.loc[train,]
		regions[num] = X_test.groupby("Region").size() / site_df.groupby("Region").size()
		info[num] = ({"Total":test.size / site_df.shape[0]})
	
	lsa = [(((site_df.DisturbanceFut + site_df.BurnFut) <=x).sum() / site_df.shape[0]) for x in np.arange(0, 101)]
	breakpoint()




	breakpoint()
	# group_kfold.get_n_splits(site_df, y, groups)





	breakpoint()

if __name__ == '__main__':
	main()	