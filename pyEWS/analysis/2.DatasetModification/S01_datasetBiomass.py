"""
Script goal, 

Rerbuild a new version of the vegetation datasets. 
	- This script is designed to ask basic questions about the input datasets and buld some simple interrogiation 
"""

# ==============================================================================

__title__ = "Dataset interoogation"
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
	# The biomass estimates
	biomass  = pd.read_csv(fpath+"PSP_total_changesV2.csv", index_col=0)
	# GPS locations
	regions  = pd.read_csv("./EWS_package/data/raw_psp/All_sites_Regions.csv", index_col=0)#.set_index("Plot_ID")
	# sample dates
	df_SD   = pd.read_csv(fpath+"survey_datesV2.csv", index_col=0)
	# Stand age 
	StandAge = pd.read_csv(fpath+"stand_origin_years_v1.csv", index_col=0).rename({"x":"StandAge"}, axis=1)
	# Remotly sensed stand age 

	# =========== Apply a value conversion function ===========
	# tfv = np.vectorize(_timefix)
	df_sa = StandAge.reset_index().rename({"index":"Plot_ID"}, axis=1) #.apply(tfv)
	# df_SD = survey.apply(tfv)#.reset_index().rename({"index":"Plot_ID"}, axis=1) 
	
	breakpoint()
	# =========== create a combined region dataset summary ==========
	df_reg = regions.merge(df_sa, how="left", on="Plot_ID")

	# =========== Loop over the datasets ===========
	warn.warn("I currently drop sites with one one measurement and sites where the stand age is older than pandas datetime supported")
	surveyint(biomass, df_reg, df_SD, df_sa)
	datacombined(biomass, df_reg, df_SD, df_sa)

	breakpoint()

# ==============================================================================

def surveyint(biomass, df_reg, df_SD, df_sa):
	# ========== pull out the survey interval ==========
	# si = df_SD.diff(axis=1).values
	si = np.hstack([df_SD.diff(periods=pe, axis=1).values for pe in np.arange(1, df_SD.shape[1])])
	
	# si[si <0] = np.NaN
	si = np.where(si >0, si, np.NaN)
	si1d = si[~np.isnan(si)]

	# ========== check site fraction ==========
	sfOD = OrderedDict()
	for gap in range(1, int(bn.nanmax(si)+1)):
		count = np.sum(np.any(si == gap, axis=1))
		sfOD[gap] = {"Total":count, "percentage":float(count)/float(si.shape[0])}
	
	sgaps = pd.DataFrame(sfOD).T
	sgaps.plot(y="percentage")
	
	plt.figure(2)
	sns.distplot(si1d, bins=np.arange(-0.5, 30.5, 1), kde=False)
	plt.show()
	breakpoint()


def datacombined(biomass, df_reg, df_SD, df_sa):
	# ========== Make a longform biomass 
	filter_col = [col for col in biomass.columns if col.startswith('live_mass')]
	OD = OrderedDict()
	counts = []
	# bmlist = []
	# =========== Itterate rowns ============
	for index, row in df_reg.iterrows():
		# ========== Check and see if standage data exists ==========
		if pd.isnull(row.StandAge):
			continue

		# ========== Pull out the biomass and the dates ==========
		try:
			dates = df_SD.loc[row.Plot_ID].dropna()
			counts.append(dates.size)
			if dates.size >1:
					bmass = biomass.loc[row.Plot_ID, filter_col].dropna()
			else:
				# Skip the places with oonly one because they dont have biomass
				continue
		except:
			continue

		if not bmass.size == dates.size:
			warn.warn("Size missmatcher here")
			breakpoint()
		else:
			for yr, bm in zip(dates, bmass):
				# ===== Workout the time delta =====
				td = yr - row.StandAge
				if td >= 0:
					OD[len(OD)] = ({
						"Plot_ID":row.Plot_ID,
						"year":yr,
						"biomass":bm,
						"standage":td,
						"region":row.Region})


	df = pd.DataFrame(OD).T

	sns.scatterplot(x="standage", y="biomass", hue="region", data=df)
	plt.show()
	breakpoint()


def _timefix(va):
	# =========== Datasets 
	if np.isnan(va):
		return pd.Timestamp(np.NaN)
	else:
		try:
			return pd.Timestamp("%d-12-31" % va)
		except:
			print(va)
			return pd.Timestamp(np.NaN)
# ==============================================================================
if __name__ == '__main__':
	main()