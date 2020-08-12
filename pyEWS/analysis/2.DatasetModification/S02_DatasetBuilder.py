"""
Script goal, 

Rerbuild a new version of the vegetation datasets. This script will be modular in a way that allows for new 
and imporved datasets to be built on the fly 
	- Find the survey dates and biomass estimates
	- Add a normalisation option to see if that imporves things 
	- Add remotly sensed estimates of stand age 

"""

# ==============================================================================

__title__ = "Datasetbuilder"
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

	# ========== Setup the different experiments ==========
	dsmod = setup_experiements()

	# ========== Load any datafiles that need to be used multiple times ==========
	# The biomass calculations
	biomass  = pd.read_csv(fpath+"PSP_total_changesV2.csv", index_col=0)
	# GPS locations
	regions  = pd.read_csv("./EWS_package/data/raw_psp/All_sites_Regions.csv", index_col=0)#.set_index("Plot_ID")
	# sample dates
	df_SD   = pd.read_csv(fpath+"survey_datesV2.csv", index_col=0)

	for ds_set in dsmod:

		# ========== Things to produce ==========
		# - Training data 
		# 	- 
		# - Site info data
		# 	- Site name, region, gps, year

		biomass_extractor(biomass, regions, df_SD, dsmod[ds_set])
		breakpoint()

	# Stand age 
	# StandAge = pd.read_csv(fpath+"stand_origin_years_v1.csv", index_col=0).rename({"x":"StandAge"}, axis=1)
	# Remotly sensed stand age 

	# =========== Apply a value conversion function ===========
	# tfv = np.vectorize(_timefix)
	# df_sa = StandAge.reset_index().rename({"index":"Plot_ID"}, axis=1) #.apply(tfv)
	
	# =========== create a combined region dataset summary ==========
	# df_reg = regions.merge(df_sa, how="left", on="Plot_ID")

	# =========== Loop over the datasets ===========
	# surveyint(biomass, df_reg, df_SD, df_sa)
	# datacombined(biomass, df_reg, df_SD, df_sa)

	breakpoint()

# ==============================================================================

def biomass_extractor(biomass, regions, df_SD, info):
	"""
	Function to pull out the biomass and site level infomation 
	"""
	# ========== Pull out the relevant columns ==========
	filter_col = [col for col in biomass.columns if col.startswith('live_mass') or col.startswith("live_N_t")]
	
	# ========== Setup containeers for new infomation ==========
	bmchange = OrderedDict()
	Sinfo    = []

	# ========== iterate through the rows ==========
	for index, row in biomass[filter_col].iterrows():
		# ========== pull out the relevant data ==========
		re   = (regions[regions.Plot_ID == index]).copy()#.rest_index(drop=True)
		surv = df_SD.loc[index].values

		if info["infillingMethod"] is None:
			# This time i'm only looking for value that match the define interval
			# ===== Find the values =====
			dif = np.subtract.outer(surv,surv)
			dif = np.where(dif>0, dif, np.NaN)

			# =========== Find the indexs that match the survey interval ==========
			loc = np.argwhere(dif == info['PredictInt']) 
			# +++++ skip if empty +++++
			if loc.size==0:
				continue
			# +++++ the locations +++++
			for I2, I1 in loc:
				bio_orig = row.iloc[I1]
				bio_delt = row.iloc[I2]
				year = surv[I1]
				bmchange[len(bmchange)] = ({
					"site":index, 
					"biomass":bio_orig,
					"lagged_biomass": ((bio_delt - bio_orig)/(bio_delt+bio_orig)),
					"year":year
					})
				# ========== This is where i bring in all the other params ==========
				# Soils
				# Climate
				# Site level Stuff
				re["year"] = year
				re["index"] = [len(Sinfo)]
				re.set_index("index", inplace=True, drop=True)

				Sinfo.append(re)
		else:
			warn.warn("Not implemented yet")
			breakpoint()

	# ========== Build the dataframes ==========
	sites  = pd.concat(Sinfo)
	df_exp = pd.DataFrame(bmchange).T
	breakpoint()
	# ========== pull out the survey interval ==========
	# # si = df_SD.diff(axis=1).values
	# si = np.hstack([df_SD.diff(periods=pe, axis=1).values for pe in np.arange(1, df_SD.shape[1])])
	
	# # si[si <0] = np.NaN
	# si = np.where(si >0, si, np.NaN)
	# si1d = si[~np.isnan(si)]

	# # ========== check site fraction ==========
	# sfOD = OrderedDict()
	# for gap in range(1, int(bn.nanmax(si)+1)):
	# 	count = np.sum(np.any(si == gap, axis=1))
	# 	sfOD[gap] = {"Total":count, "percentage":float(count)/float(si.shape[0])}
	
	# sgaps = pd.DataFrame(sfOD).T
	# sgaps.plot(y="percentage")
	
	# plt.figure(2)
	# sns.distplot(si1d, bins=np.arange(-0.5, 30.5, 1), kde=False)
	# plt.show()

# ==============================================================================
def SolSiteParms():
	"""
	The goal of this function it to load the site level stem density infomation 
	that sol used to put it into the database
	"""
	raw_stem_df = pd.read_csv("./EWS_package/data/psp/stem_dens_interpolated_w_over_10yearsV2.csv", index_col=0)
	breakpoint()
# ==============================================================================
def setup_experiements():
	"""
	This function creates a setup table and an ordered dctionary of all the functions to be
	included in a given version of the datasset 
	"""

	dsmod = OrderedDict()
	dsmod[0] = ({
		"desc":"This version is to see if infilling might be impacting results",
		"PredictInt": 5, # the interval of time to use to predict biomass into the future
		"RSveg":"landsat_ews", # the source of the data to be used to calculate vegetation metrics 
		"Clim": "climateNA", # the climate dataset to use
		"StandAge": None, # add any stand age data
		"CO2": None, # add any data
		"infillingMethod":None, # function to use for infilling
		"Norm": True,  # Also produce a normalised version
		})

	# Future experiments
	# Raw  5, 10, 15 and 20 years
	# SOls 5, 10, 15 and 20 years
	# Different inflling methods 5, 10, 15 and 20 years
	# Different climate datasets
	# addinf CO2
	return dsmod


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