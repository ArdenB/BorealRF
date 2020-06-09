"""
Script goal, 
Currently at an impass for the future direction of this paper. the goal 
of this script is answer some key questions. 
	1. How well is my modeling approach doing 
		- predicted vs observed out of sample 
	2. What is causing the missing values in the input data
	3. Look for regional bias in the dropped points 

"""

# ==============================================================================

__title__ = "Deciding future directions"
__author__ = "Arden Burrell"
__version__ = "v1.0(28.05.2020)"
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
import xarray as xr
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

# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf

# ========== Import packages for parellelisation ==========
# import multiprocessing as mp

# ========== Import ml packages ==========
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn import metrics as sklMet
from sklearn.utils import shuffle
from scipy.stats import spearmanr
from scipy.cluster import hierarchy

# ========== Import Cuda Package packages ==========
# try:
# 	import cupy
# 	import cudf
# 	import cuml

# 	from cuml.dask.common import utils as dask_utils
# 	from dask.distributed import Client, wait
# 	from dask_cuda import LocalCUDACluster
# 	import dask_cudf

# 	from cuml import RandomForestRegressor as cuRFR
# 	# from cuml.dask.ensemble import RandomForestRegressor as cuRFR
# 	# from cuml.preprocessing.model_selection import train_test_split as  CUDAtrain_test_split
# 	# model_selection.train_test_split
# 	cuda = True
# except:
# 	print("Unable to build a cuda environment on this OS and/or conda env")
# 	cuda = False
# breakpoint()

# ==============================================================================

def main():
	
	# ========== Load in the dataframe ==========
	df, site_df = df_proccessing()

	# ========== check for spatial gaps ==========
	Spatial_gaps(df, site_df)

	# ========== Check for temporal gaps ==========
	Temporal_gaps(df, site_df)
	
	# ========== Test model performance ==========
	Perf_testing()



	
	breakpoint()

# ==============================================================================

def Spatial_gaps(df, site_df):
	"""
	function to test the performance of my models testing and training 
	"""
	colnames = (['biomass', #'LANDSAT_ndvi_trend_trend_5',
		# 'LANDSAT_ndvi_trend_trend_6', 'LANDSAT_ndvi_trend_trend_7',
       # 'LANDSAT_ndvi_trend_trend_8', 'LANDSAT_ndvi_trend_trend_9',
       'LANDSAT_ndvi_trend_trend_10'])
	renames = (["sites",
		# 'LANDSAT_5yr_trends',
		# 'LANDSAT_6yr_trends', 
		# 'LANDSAT_7yr_trends',
		# 'LANDSAT_8yr_trends', 
		# 'LANDSAT_9yr_trends',
		'LANDSAT_10yr_trends'
		])


	dfx = df[["date", "Region"]+colnames].groupby(["date", "Region"],as_index=False).count()
	for new, old in zip(renames, colnames):
		dfx.rename({old:new}, axis=1, inplace=True)
	
	# dfx.reset_index(inplace=True)
	df_ls = []
	for col, source in zip(renames, ["Observed", "Included"]):
		dfc = dfx[["date", "Region", col]].rename({col:"sites"}, axis=1)
		dfc["Source"] = source
		df_ls.append(dfc)
	dfs = pd.concat(df_ls)
	dfs["Region"] = dfs["Region"].astype('category')
	dfs["Source"] = dfs["Source"].astype('category')
	dfs["sites"]  = dfs["sites"].astype('float32')
	# dfs["date"]   = dfs["date"].astype('<M8[D]')

	# ========== Build the figures ==========
	# ========== Set the plot params ==========
	font = {'family' : 'normal',
			'weight' : 'bold', #,
		    'size'   : 8}

	sns.set_style("whitegrid")
	mpl.rc('font', **font)
	sns.barplot(x="Region", y="sites", hue="Source", data=dfs, estimator=np.sum, ci=None)

	sns.relplot(x="date", y="sites", hue = "Source", col="Region",
		col_wrap=6, data=dfs, estimator=np.sum, ci=None, kind="line")

	plt.show()

	breakpoint()

def Perf_testing():
	"""
	function to test the performance of my models testing and training 
	"""
	tmpath      = "./pyEWS/experiments/1.FeatureSelection/"
	res = []
	met = []
	for test in range(10):
		fnamein = fnameout = tmpath + "S02_outofsampleEST_test%02d_10.csv" % (test)
		if os.path.isfile(fnamein):
			# ========== Load in the data ==========
			dfx = pd.read_csv(fnamein, index_col=0)
			dfx["Residual"]   = dfx["Estimated"] -  dfx["Observed"]  
			dfx["Experiment"] = test
			res.append(dfx)
			met.append({
				'R2'  : sklMet.r2_score(dfx["Observed"], dfx["Estimated"]),
				'MAE' : sklMet.mean_absolute_error(dfx["Observed"], dfx["Estimated"]),
				'RMSE': np.sqrt(sklMet.mean_squared_error(dfx["Observed"], dfx["Estimated"])),
				})

			# print('Mean Absolute Error:',     sklMet.mean_absolute_error(y_test, y_pred))
			# print('Root Mean Squared Error:', np.sqrt(sklMet.mean_squared_error(y_test, y_pred)))
			# print('Mean Squared Error:',      sklMet.mean_squared_error(y_test, y_pred))

	if len(res)>1:
		df=pd.concat(res)
	else:
		df = res[0]

	# =========== Create xome catogorical bins ==========
	# ========== Bin the data based on the observed change ==========
	df["ObsCat"] = [vls.left for vls in pd.cut(df.Observed, 20)]
	df["ObsCat"].replace(df["ObsCat"].min(), -1.0, inplace=True)

	# ========== Set the plot params ==========
	font = {'family' : 'normal',
			'weight' : 'bold', #,
		    'size'   : 8}

	sns.set_style("whitegrid")
	mpl.rc('font', **font)

	# ========== Make an scatter plot ==========
	gx = sns.relplot(y="Estimated", x="Observed", data=df, 
		col="Experiment", col_wrap=5, kind="scatter")
	for num, ax in enumerate(gx.axes.flat):
		# +++++ Measure the performance +++++
		textstr = '\n'.join((
		    r'R2  =%.4f' % (met[num]["R2"]),
		    r'MAE =%.4f' % (met[num]["MAE"]),
		    r'RMSE=%.4f' % (met[num]["RMSE"])))
		
		# place a text box in upper left in axes coords
		ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
		        verticalalignment='top')

		# +++++ Add a 1 to 1 line +++++
		ln = np.arange(-1, 1, 0.05)
		ax.plot(ln, ln, color="k", alpha=0.25)
		
	plt.show()

	# ========== Make the second figure ==========
	sns.boxplot(y="Residual", x="ObsCat", hue="Experiment", data=df)
	plt.show()
	# def GB_r2(dfin):
	# 	return sklMet.r2_score(dfin["Observed"], dfin["Estimated"])

	# # facet_kws={"gridspec_kws":{"wspace":0.020,  "hspace":.10}}, legend=False,
	# for cat in np.arange(-1., 1, 0.1):
	# 	print(cat)
	# 	dfs = df[df.ObsCat == cat]
	# 	sklMet.r2_score(dfs["Observed"], dfs["Estimated"])
	breakpoint()

def Temporal_gaps(df, site_df):
	"""
	function to test the Nan flags in my data 
	"""

	colnames = (['biomass', 'LANDSAT_ndvi_trend_trend_5',
		'LANDSAT_ndvi_trend_trend_6', 'LANDSAT_ndvi_trend_trend_7',
       'LANDSAT_ndvi_trend_trend_8', 'LANDSAT_ndvi_trend_trend_9',
       'LANDSAT_ndvi_trend_trend_10'])
	renames = (["sites",
		'LANDSAT_5yr_trends',
		'LANDSAT_6yr_trends', 
		'LANDSAT_7yr_trends',
		'LANDSAT_8yr_trends', 
		'LANDSAT_9yr_trends',
		'LANDSAT_10yr_trends'
		])

	dfx = df.groupby("date").count()[colnames]
	for new, old in zip(renames, colnames):
		dfx.rename({old:new}, axis=1, inplace=True)
	
	# ========== Build the figures ==========
	plt.figure(1)
	dfx.plot.bar()
	plt.figure(2)
	dfx.cumsum().plot.line()
	plt.figure(3)
	

	dfs = dfx.copy()
	dfs["TrendPossible"] = dfs["sites"]
	dfs["TrendPossible"][dfs.LANDSAT_5yr_trends== 0 ] = 0
	(dfs.cumsum().iloc[:,1:].div(dfs.cumsum().iloc[:,0], axis=0)).plot.bar()
	plt.show()
	breakpoint()

	pass


# ==============================================================================

def df_proccessing(window = 10, cols_keep=None):
	"""
	This function opens and performs all preprocessing on the dataframes.
	These datasets were originally made using:
		./code/data_prep/create_VI_df.R
		./code/data_prep/vi_calcs/functions/modified......
		./code/data_prep/vi_calcs/full_df_calcs_loop......
	args:
		window:		int
			the max size of the values considered 
		tmpath:		str
			Location to save temp files
		dftype:		str
			How to open th dataset
		cols_keep:		list		
			A list of columns to subest the datasett
		branch: 	int
			the branch number
		experiment: 	int
			the experiment number
		dftype:		str
			dataset loaded using pandas, dask or cudf
		recur:
			dataframe to be used in place of recursive dataset

	"""
	# ============ Set a timeer ============
	t0 = pd.Timestamp.now()

	# ============ Setup the file names ============
	# The level of correlation between datasets
	# cor_fn  = "./EWS_package/data/models/input_data/correlations_2019-10-09.csv"
	vi_fn   = "./EWS_package/data/models/input_data/vi_df_all_V2.csv"

	# ============ Open the variables and correlations ============
	# This may need to change when using cuda 
	print("loading the files at:", t0)
	vi_df   = pd.read_csv(  vi_fn, index_col=0)
	# cor_df  = pd.read_csv( cor_fn, index_col=0)
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

	def _regionmaker(Sitekey):
		site_out = "./EWS_package/data/raw_psp/All_sites_Regions.csv"
		if os.path.isfile(site_out):
			return pd.read_csv(site_out, index_col=0)
		
		site_fn = "./EWS_package/data/raw_psp/All_sites_101218.csv"
		site_df = pd.read_csv(site_fn, index_col=0)
		""" Function to add region infomation """
		raw_checks = glob.glob("./EWS_package/data/raw_psp/*/checks/*_check.csv")
		sitenames   = []
		siteregions = []
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

		# ============ make a region lookup ==========
		for fncheck in raw_checks:
			region = fncheck.split("/")[4]
			siteregions.append(region)
			if region == "YT":
				sitenames.append("%s%d" %(regkey[region], int(fncheck.split("/")[-1].split("_check.csv")[0])))
			else:
				sitenames.append(regkey[region]+fncheck.split("/")[-1].split("_check.csv")[0])

		# ============ Loop over the site_df ==========
		def site_locator(sn, sitenames, siteregions, Sitekey):
			
			if sn in sitenames:
				return siteregions[sitenames.index(sn)]
			else:
				if sn.split("_")[0] in Sitekey.keys():
					return Sitekey[sn.split("_")[0]]
				else:
					breakpoint()
					return "Unknown"
		site_df["Region"] = [site_locator(sn, sitenames, siteregions) for sn in site_df.Plot_ID]
		site_df.to_csv(site_out)
		return site_df

	site_df =_regionmaker(Sitekey)

	print("loading the files took: ", pd.Timestamp.now()-t0)
	
	# ============ Filter the rows ============
	if cols_keep is None:
		# +++++ A container to hold the kept columns  +++++ 
		cols_keep = []
		clnm =  vi_df.columns.values
		for cn in clnm:
			# Test to see if the column is one of the ones i want to keep
			if cn.startswith("LANDSAT"):
				# The VI datasets, check the length of window considered
				if int(cn.split("_")[-1]) <= window:
					cols_keep.append(cn)
			# elif cn in ['site']:
			# 	pass
			else:
				cols_keep.append(cn)
	# else:
	# 	ipdb.set_trace()

	# ========== Fill in any NA rows ==========
	warn.warn("This is the simplified file, might be way faster to load in")
	data      = vi_df[cols_keep]

	# ========== Find the dates ==========
	yr =  [pd.Timestamp("%d-06-30" %(int(nm[1:5]))) for nm in  data.index]
	data["date"] = yr
	data["Region"] = [Sitekey[sn.split("_")[0]] for sn in data.site]
	return data, site_df
# ==============================================================================
if __name__ == '__main__':
	main()