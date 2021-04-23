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
from tqdm import tqdm

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

	df_dam  = pd.read_csv("./EWS_package/data/psp/modeling_data/damage_flags.csv", index_col=0).fillna(0.)
	df_burn = pd.read_csv("./EWS_package/data/fire/LANDSAT_fire.csv", index_col=0)
	df_burn = _fix_burn_index(df_burn)
	#Add in soil chracteristics
	soils = pd.read_csv(
		"./EWS_package/data/psp/modeling_data/soil_properties_aggregated.csv", index_col=0).rename(
		{'prop_vals.rownames.samp_loc.':"Plot_ID"}, axis=1)
	soils = _fix_burn_index(soils)
	# add permafrost
	permafrost = pd.read_csv(
		"./EWS_package/data/psp/modeling_data/extract_permafrost_probs.csv", index_col=0).rename(
		{'rownames.samp_loc.':"Plot_ID"}, axis=1)
	permafrost = _fix_burn_index(permafrost)
	permafrost.fillna(0, inplace=True)

	# Stand age 
	df_StandAge = pd.read_csv(fpath+"stand_origin_years_v1.csv").rename({"Unnamed: 0":"Plot_ID","x":"StandAge"}, axis=1)
	df_StandAge = _fix_burn_index(df_StandAge)
	# Remotly sensed stand age 


	for ds_set in dsmod:
		biomass_extractor(biomass, regions, df_SD, dsmod[ds_set], df_dam, df_burn, soils, permafrost, df_StandAge)
		breakpoint()

		# ============ Add data normalisation

	# ========== Things to produce ==========
	# - Training data 
	# 	- 
	# - Site info data
	# 	- Site name, region, gps, year


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

def biomass_extractor(biomass, regions, df_SD, info, df_dam,  df_burn, soils, permafrost, df_StandAge,
	damage_win = 50, t0=pd.Timestamp.now()):
	"""
	Function to pull out the biomass and site level infomation 
	"""
	# ========== Pull out the relevant columns ==========
	Bfilter_col = [col for col in biomass.columns if col.startswith('live_mass')]
	Tfilter_col = [col for col in biomass.columns if col.startswith("live_N_t")]
	
	# ========== Setup containeers for new infomation ==========
	bmchange = OrderedDict()
	Sinfo    = []

	# ========== iterate through the rows to find ones with the correct gap ==========
	for num, (index, row) in enumerate(biomass.iterrows()):
		cf.lineflick(num, biomass.shape[0], t0)
		# ========== pull out the relevant data ==========
		re   = (regions[regions.Plot_ID == index]).copy(deep=True)#.rest_index(drop=True)
		if re.empty:
			# print(f"site {index} is missing")
			re = pd.Series({"Plot_ID":index, "Longitude":np.NaN, "Latitude":np.NaN, "Region":_regionfix(index)}).to_frame().T
		bio  = row[Bfilter_col]
		stem = row[Tfilter_col]
		surv = df_SD.loc[index].values

		if info["infillingMethod"] is None:
			# This time i'm only looking for value that match the define interval
			# ===== Find the values =====
			dif = np.subtract.outer(surv,surv)
			with np.errstate(invalid='ignore'):
				dif = np.where(dif>0, dif, np.NaN)

			# =========== Find the indexs that match the survey interval ==========
			if not info['PredictInt'] is None:
				loc = np.argwhere(dif == info['PredictInt']) 
			else:
				# print("This has yet to be implemented yet")
				loc = np.argwhere(~np.isnan(dif))
			# +++++ skip if empty +++++
			if loc.size==0:
				continue
			# +++++ the locations +++++
			for I2, I1 in loc:
				bio_orig = bio.iloc[I1]
				bio_delt = bio.iloc[I2]
				year     = surv[I1]
				if info["Norm"]:
					if (bio_delt+bio_orig) == 0:
						lagged_biomass = 0
					else:
						lagged_biomass = ((bio_delt - bio_orig)/(bio_delt+bio_orig))
					
					if np.isnan(lagged_biomass):
						# Pull out places with bad values
						# print 
						breakpoint()
						continue

					outdict = OrderedDict({
						"site":index, 
						"year":year,
						"biomass":bio_orig,
						"lagged_biomass": lagged_biomass,
						"stem_density":stem.iloc[I1],
						})
				else:
					outdict = OrderedDict({
						"site":index, 
						"year":year,
						"biomass":bio_orig,
						"Obs_biomass": bio_delt,
						"Delta_biomass":(bio_delt - bio_orig),
						"stem_density":stem.iloc[I1],
						})
				# breakpoint()
				# add the prediction lag when i'm assessing multiple indexes
				if info['PredictInt'] is None:
					outdict["ObsGap"] = dif[I2, I1]
				bmchange[len(bmchange)] = outdict

				# ========== This is where i bring in all the other params ==========
				# +++++ Disturbance +++++
				yrsD  = np.max([1926, year-damage_win]) # THis has been included to deal with issues abbout indexing befo32 1932
				if index in df_dam.index:
					dist = df_dam.loc[index, np.arange(yrsD, year).astype(int).astype(str)].sum()*100.
					# Find the disturbance gap
					if (df_dam.loc[index, np.arange(1926, year).astype(int).astype(str)] > 0).any():
						dist_year = (df_dam.loc[index, np.arange(1926, year).astype(int).astype(str)] > 0).reset_index().rename({"index":"year"}, axis=1)
						dgap = year - np.max((dist_year.loc[dist_year[index], "year"]).astype(int).values)
					else:
						dgap = np.NaN
				else:
					dist = np.NaN
					dgap = np.NaN

				# +++++ fires +++++
				yrsF  = np.max([1917, year-damage_win])
				if index in df_burn.index:
					burn = df_burn.loc[index, np.arange(yrsF, year).astype(int).astype(str)].sum()
					# check for a burn gap
					if (df_burn.loc[index, np.arange(1917, year).astype(int).astype(str)]>0).any():
						burn_years = (df_burn.loc[index, np.arange(1917, year).astype(int).astype(str)]>0).reset_index().rename({"index":"year"}, axis=1)
						bgap = year - np.max((burn_years.loc[burn_years[index], "year"]).astype(int).values)
					else:
						bgap = np.NaN
				elif index.startswith("11_"):
					# Error in indexing
					breakpoint()
				else:
					burn = np.NaN
					bgap = np.NaN


				# ++++++++++ StandAge ++++++++++
				if index in df_StandAge.index:
					standage = year - df_StandAge.loc[index, "StandAge"]
				else:
					standage = np.NaN

				# ========== Add the site infomation ==========
				# breakpoint()
				re2 = re.copy(deep=True)
				re2["year"]           = year
				re2["index"]          = [len(Sinfo)]
				re2["Disturbance"]    = dist
				re2["DisturbanceGap"] = dgap
				re2["Burn"]           = burn
				re2["BurnGap"]        = bgap
				re2["StandAge"]	      = standage
				re2["DistPassed"]     = float((dist+burn) <= 0.1)
				re2.set_index("index", inplace=True, drop=True)
				Sinfo.append(re2)
				# if loc.shape[0]>1:
				# 	print(year)
				# 	breakpoint()
		else:
			warn.warn("Not implemented yet")
			breakpoint()

	# ========== Build the dataframes ==========
	print("\n Building dataframes")
	sites  = pd.concat(Sinfo)
	df_exp = pd.DataFrame(bmchange).T

	# ========== Add the VI's ==========
	df_exp = info['RSveg'](sites, df_exp, info)

	# ========== Add the species data ==========
	df_exp = info['SiteData'](sites, df_exp, info)
	# # ========== add the climate ==========
	df_exp = sol_climate_extractor(sites, df_exp, info, regions)

	# # ========== Add in Soil data ==========
	soils  = soils.reindex(df_exp["site"].values) 
	soilc  = ["30" in cl for cl in soils.columns]
	soils  = soils.loc[:,soilc]
	# breakpoint()
	for col in soils.columns:	
		df_exp[col] = soils[col].values

	# ========== Add the permafrost data ==========
	permafrost  = permafrost.reindex(df_exp["site"].values) 
	for colp in permafrost.columns:	
		df_exp[colp] = permafrost[colp].values

	# df_exp = df_exp.merge(permafrost, how="left", left_on="site", right_index=True)
	sites.set_index(df_exp.index.values, inplace=True)

	# ========== Write the file out ==========
	fpath  = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/"
	if not info['PredictInt'] is None:
		fnameS = fpath + f"SiteInfo_{info['PredictInt']}years.csv"
		fnameV = fpath + f"VI_df_{info['PredictInt']}years.csv"
	else:
		if info["Norm"]:
			fnameS = fpath + f"SiteInfo_AllSampleyears.csv"
			fnameV = fpath + f"VI_df_AllSampleyears.csv"
		else:
			fnameS = fpath + f"SiteInfo_AllSampleyears_ObsBiomass.csv"
			fnameV = fpath + f"VI_df_AllSampleyears_ObsBiomass.csv"
	# breakpoint()
	
	sites.to_csv(fnameS)
	df_exp.to_csv(fnameV)
	# breakpoint()

# ==============================================================================
# ==============================================================================
def sol_climate_extractor(sites, df_exp, info, regions):

	print(f"AddingClimate to the predictor dataset started at: {pd.Timestamp.now()}")
	# ========== Create the site names in the VI format ==========
	zShape =  df_exp.shape[0]
	# sset  = sites[sites["year"]>=1982] # no VI data before that
	# slist = sset["Plot_ID"].values#("X"+sset["year"].astype(int).astype(str)+"_"+sset["Plot_ID"]).values

	#there were some statisitics that I decided afterwards that didn't make sense, like relative trends for a number of variables,
	#so I created absolute trends and am removing the relative trends here. 
	remove_clim = (['MAR_mean_30years','MAR_trend_30years','MAR_abs_trend_30years','MAT_trend_30years','MWMT_trend_30years',
	                'MCMT_trend_30years','TD_trend_30years','FFP_trend_30years','EXT_trend_30years','EMT_trend_30years',
	                'eFFP_trend_30years','DD5_trend_30years','DD18_trend_30years','DD_18_trend_30years','DD_0_trend_30years',
	                'bFFP_trend_30years','RH_trend_30years','NFFD_trend_30years', 'climate_df_30years'])
	def _yconvert(syear):
		cindex=np.arange(1981., 2019)
		if syear < 1981: 
			return -1
		else:
			return np.where(cindex == syear)[0][0]

	vycon = np.vectorize(_yconvert)
	# all_climate = pd.read_csv("./EWS_package/data/psp/modeling_data/climate/1951-2018/climate_df_30years.csv", index_col=0)
	for clfn in glob.glob("./EWS_package/data/psp/modeling_data/climate/1951-2018/*30years.csv"):
		# +++++ make the variable name
		var = clfn.split("/")[-1][:-4]
		if var in remove_clim:
			#Skip places Sol excluded
			continue
		else:
			print(f"Loading {var} at {pd.Timestamp.now()}")
		# ========== load the data ==========
		df_cl = pd.read_csv(clfn, index_col=0)
		if df_cl.shape[0] == regions.shape[0]:
			df_cl.set_index(regions["Plot_ID"].values, inplace=True)
			# add an out of value column
			df_cl["Unknown"]=np.NaN
		else:
			print("Indexing error here")
			breakpoint()
			continue

		# ========== Fix the indexing issues ==========

		dfcl = df_cl.reindex(df_exp["site"].values)

		# ========== Add the variable to the dataframe ==========
		df_exp[var] =  dfcl.values[np.arange(dfcl.shape[0]), vycon(sites["year"])]

	return df_exp


def sol_VI_extractor(sites, df_exp, info):
	"""
	Function takes the sites and bimass data and adds all of sols VI Metrics 
	args:	
		sites:	pd df
		df_exp: pd df
	returns:
		df_exp
	"""
	print(f"Adding VI's to the predictor dataset started at: {pd.Timestamp.now()}")
	zShape =  df_exp.shape[0]
	# ========== Create the site names in the VI format ==========
	sset  = sites[sites["year"]>=1982] # no VI data before that
	slist = ("X"+sset["year"].astype(int).astype(str)+"_"+sset["Plot_ID"]).values
	# ========== Loop over the vi indexes ==========
	for vi in info["VIs"]:
		try:
			print(f"Loading {vi} at {pd.Timestamp.now()}")
			df_vi = pd.read_csv(
				f"./EWS_package/data/VIs/metrics/metric_dataframe_{vi}_noshift.csv", 
				index_col=0)
			# breakpoint()

			df_vi.set_index("obs", inplace=True)
			# +++++ pull out the set ove VI data that matches +++++
			subs = df_vi.loc[slist]
			subs["index"] = sset.index
			subs.set_index("index", inplace=True)
			# ========== Add that to the dataframe ==========
			df_exp = df_exp.merge(subs, how="left", left_index=True, right_index=True)
			# +++++ there is a duplication bug, this adresses this problem but its a bodge +++++
			df_exp.drop_duplicates(inplace=True)

			# +++++ Check to see if the size has gone wonky +++++
			if not df_exp.shape[0] == zShape:
				breakpoint()

		except Exception as er:
			print(str(er))
			breakpoint()
	# ========== Return the results ==========
	return df_exp

def SolSiteParms(sites, df_exp, info):
	"""
	The goal of this function it to load the site level stem density infomation 
	that sol used to put it into the database
	"""
	# Read in species compositions
	# Table used to read species groups and time, different group types
	print(f"Adding Species infomation to the predictor dataset started at: {pd.Timestamp.now()}")
	LUT       = pd.read_csv("./EWS_package/data/raw_psp/SP_LUT.csv", index_col=0)
	sp_groups = pd.read_csv("./EWS_package/data/raw_psp/SP_groups.csv", index_col=0) 
	# fix the groups 
	g1 = sp_groups["Group_1"].str.replace("/", ".")
	sp_groups["Group_1"] = g1.str.replace(" ", "_")

	# sp_out_df = data.frame('site' = rep(sites,37))
	# rows = vector()
	# ========== Loop over the species in LUT ==========
	fails = [] #species that are missing key infomation
	for spec in LUT.index:
		try:
			print(spec, LUT.loc[spec,"scientific"], pd.Timestamp.now())
			fname = f"./EWS_package/data/psp/modeling_data/species/comp_interp_{spec}.csv"
			if not os.path.isfile(fname):
				fails.append(LUT.loc[spec,"scientific"])
				sp_groups.replace(LUT.loc[spec,"scientific"], np.NaN, inplace=True)
				continue
			raw_sp_df = pd.read_csv(fname, index_col=0).reindex(df_exp["site"].values)
			# as nan is the same as no trees, infill gapes
			raw_sp_df.fillna(0, inplace=True)
			
			# +++++ make a function that can be vectorised +++++
			# This is done to make use of numpys super fast indexing in the next line
			def _yconvert(syear):
				cindex=np.arange(1926., 2018.)
				return np.where(cindex == syear)[0][0]
			vycon = np.vectorize(_yconvert)

			# ========== apply the vectorised function to the data ==========
			df_exp[LUT.loc[spec,"scientific"]] = raw_sp_df.values[np.arange(raw_sp_df.shape[0]), vycon(sites["year"].values)]
	
		except Exception as er:
			print(str(er))
			breakpoint()
		# raw_sp_df.reindex(df_exp["site"].values).reindex(df_exp["site"].values
	# breakpoint()
	# ========== Add in Sols Groups ==========
	for gr in ["Group_1", "Group_2", "Group_3",  "Group_4"]:
		for cla in sp_groups[gr].unique():
			# ========== Make a list of the relevant columns ==========
			cols = sp_groups[sp_groups[gr] == cla]["scientific"].dropna().values
			# +++++ check if there are enough observations +++++
			try:
				rowsums = df_exp[cols].values.sum(axis=1)
			except Exception as e:
				print(str(e))
				breakpoint()
				continue
			if (rowsums > 0).sum() < 1500:
				continue
			else:
				df_exp[f"{gr}_{cla}"] = rowsums


	return df_exp

# ==============================================================================
# ==============================================================================
def _regionfix(site):
	"""
	Function to work out which region each site is and return it 
	"""
	Sitekey = ({# values from survey_years.R
		"1":"BC",
		"2":"AB",
		"3":"SK",
		"4":"MB",
		"5":"ON",
		"6":"QC",
		"7":"NL",
		"8":"NB",
		"9":"NS",
		"11":"YT",
		"12":"NWT",
		"13":"CAFI",
		"14":"CIPHA",
		})
	return Sitekey[site.split("_")[0]]

def _fix_burn_index(df_burn):
	# ===== Fix indexing problem =====
	indexs = df_burn.Plot_ID
	newindex = []
	for index in indexs:
		if index.startswith("11_"):
			zone, site = index.split("_")
			newindex.append(zone+"_%03d" % int(site))
		else:
			newindex.append(index)

	df_burn["Plot_ID"] = newindex
	df_burn.set_index("Plot_ID", inplace=True)
	return df_burn
# ==============================================================================
def setup_experiements():
	"""
	This function creates a setup table and an ordered dctionary of all the functions to be
	included in a given version of the datasset 
	"""

	dsmod = OrderedDict()
	# dsmod[5] = ({
	# 	"desc":"This version has a new prediction interval but with raw biomass",
	# 	"PredictInt": None, # the interval of time to use to predict biomass into the future
	# 	"RSveg":sol_VI_extractor, # the source of the data to be used to calculate vegetation metrics 
	# 	"VIs":['ndvi','psri','ndii','ndvsi','msi','nirv','ndwi','nbr','satvi','tvfc'],
	# 	"Clim": "climateNA", # the climate dataset to use
	# 	"SiteData":SolSiteParms,
	# 	"StandAge": None, # add any stand age data
	# 	"CO2": None, # add any data
	# 	"infillingMethod":None, # function to use for infilling
	# 	"Norm": False,  # Also produce a normalised version
	# 	"meth":"Delta"
	# 	})
	dsmod[4] = ({
		"desc":"This version has a new prediction interval but with raw biomass",
		"PredictInt": None, # the interval of time to use to predict biomass into the future
		"RSveg":sol_VI_extractor, # the source of the data to be used to calculate vegetation metrics 
		"VIs":['ndvi','psri','ndii','ndvsi','msi','nirv','ndwi','nbr','satvi','tvfc'],
		"Clim": "climateNA", # the climate dataset to use
		"SiteData":SolSiteParms,
		"StandAge": None, # add any stand age data
		"CO2": None, # add any data
		"infillingMethod":None, # function to use for infilling
		"Norm": False,  # Also produce a normalised version
		})
	dsmod[3] = ({
		"desc":"This version has a new prediction interval",
		"PredictInt": None, # the interval of time to use to predict biomass into the future
		"RSveg":sol_VI_extractor, # the source of the data to be used to calculate vegetation metrics 
		"VIs":['ndvi','psri','ndii','ndvsi','msi','nirv','ndwi','nbr','satvi','tvfc'],
		"Clim": "climateNA", # the climate dataset to use
		"SiteData":SolSiteParms,
		"StandAge": True, # add any stand age data
		"CO2": None, # add any data
		"infillingMethod":None, # function to use for infilling
		"Norm": True,  # Also produce a normalised version
		})
	dsmod[0] = ({
		"desc":"This version is to see if infilling might be impacting results",
		"PredictInt": 5, # the interval of time to use to predict biomass into the future
		"RSveg":sol_VI_extractor, # the source of the data to be used to calculate vegetation metrics 
		"VIs":['ndvi','psri','ndii','ndvsi','msi','nirv','ndwi','nbr','satvi','tvfc'],
		"Clim": "climateNA", # the climate dataset to use
		"SiteData":SolSiteParms,
		"StandAge": None, # add any stand age data
		"CO2": None, # add any data
		"infillingMethod":None, # function to use for infilling
		"Norm": True,  # Also produce a normalised version
		})
	dsmod[1] = ({
		"desc":"This version is to see if infilling might be impacting results",
		"PredictInt": 10, # the interval of time to use to predict biomass into the future
		"RSveg":sol_VI_extractor, # the source of the data to be used to calculate vegetation metrics 
		"VIs":['ndvi','psri','ndii','ndvsi','msi','nirv','ndwi','nbr','satvi','tvfc'],
		"Clim": "climateNA", # the climate dataset to use
		"SiteData":SolSiteParms,
		"StandAge": None, # add any stand age data
		"CO2": None, # add any data
		"infillingMethod":None, # function to use for infilling
		"Norm": True,  # Also produce a normalised version
		})

	dsmod[2] = ({
		"desc":"This version is to see if infilling might be impacting results",
		"PredictInt": 15, # the interval of time to use to predict biomass into the future
		"RSveg":sol_VI_extractor, # the source of the data to be used to calculate vegetation metrics 
		"VIs":['ndvi','psri','ndii','ndvsi','msi','nirv','ndwi','nbr','satvi','tvfc'],
		"Clim": "climateNA", # the climate dataset to use
		"SiteData":SolSiteParms,
		"StandAge": None, # add any stand age data
		"CO2": None, # add any data
		"infillingMethod":None, # function to use for infilling
		"Norm": True,  # Also produce a normalised version
		"meth":None
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