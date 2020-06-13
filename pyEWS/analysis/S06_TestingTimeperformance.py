"""
Script goal, 

To learn about the existing dataset through the use of Random Forest
	- Open the datasets
	- Perform a random forest regression and variable selection using both scipy and CUDA
	- Compare the results

Relevent existing R code
	./code/analysis/modeling/build_model/rf_class......
"""

# ==============================================================================

__title__ = "Random Forest Implementation with some performance metrics"
__author__ = "Arden Burrell"
__version__ = "v1.0(11.06.2020)"
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

# ==============================================================================
def main():
	# ========== These are all from Sol ===========
	region      = 'all' 
	# num_quants  = 7 
	# window      = 10
	ntree       = 500
	test_size   = 0.2
	tmpath      = "./pyEWS/experiments/1.FeatureSelection/tmp/"
	SelMethod   = "hierarchicalPermutation"
	VarImp      = "recursiveDrop"
	force       = False
	writenames  = []

	# ========== Make a folder to store the results  ==========
	folder = "./pyEWS/experiments/1.FeatureSelection/"
	cf.pymkdir(folder)

	# Setup the experiment version
	versions = OrderedDict()
	versions["v3"] = ({
		"desc":"Testing variable trend leng through time",
		"teststr":"_V3", #string to add to written files
		# "windows":np.arange(4, 20),
		"windows":[4, 5, 10, 15, 20],
		"testvariable" : "windows"
		# "maxiter":11, # The maximum number to tests to run. determined observationally in v1
	})
	print("\nCalculating Max branch depth starrting at:", pd.Timestamp.now())
	versions["v3"]["maxdepth"] = [_branchcal(window) for window in versions["v3"]["windows"]] #
	print(versions["v3"]["maxdepth"])
	# versions["v3"]["maxdepth"] = [2, 3, 4, 5, 7, 8, 10, 10, 12, 13, 14, 17, 18, 20, 22, 23]
	versions["v3"]["maxiter"]  = np.max(versions["v3"]["maxdepth"])

	for experiment  in [0, 1, 2, 3, 4, 5]:
		for vers in versions:
			# ========== Set the max itteration ==========
			maxiter = versions[vers]["maxiter"]

			for window, maxDP in zip(versions[vers][versions[vers]["testvariable"]], versions[vers]["maxdepth"]):
				print("\n Version:", vers, " Trend length (yr):", window)

				# ========== Create a max number of branches ==========
				test_branchs = np.arange(maxiter)
				ColNm        = None #will be replaced as i keep adding new columns
				corr_linkage = None # will be replaced after the 0 itteration
				orig_clnm    = None

				# ========== Create a dictionary so i can store performance metrics ==========
				perf  = OrderedDict()

				# ========== Make a file name =============
				fnout =  folder + "S06_RF_TestingLANDSAT_%02dyrs_Exp%02d_%s.csv" % (window, experiment, versions[vers]["teststr"])

				# ========== Check i the file already exists ===========
				if os.path.isfile(fnout) and not force:
					# The file already exists and i'm not focing the reload
					writenames.append(fnout)
					continue
				
				for branch in test_branchs:
					# ========== Check and see if i need to consider the branch at all ==========
					if branch > maxDP:
						# ========== Add the results of the different itterations to OD ==========
						perf["Branch%02d" % branch] = ({
							"LSWindow":window, "branch":branch, "dfloadtime":pd.Timedelta(np.NaN), 
							"RFtime":pd.Timedelta(np.NaN), "TimeCumulative":pd.Timedelta(np.NaN),  
							"R2":np.NaN, "NumVar":np.NaN,  "SiteFraction":np.NaN,
							})
						continue

					# ========== Load the data and get the basic stats ==========
					X_train, X_test, y_train, y_test, col_nms, loadstats, corr = df_proccessing(
						window, tmpath, test_size, branch, experiment, cols_keep=ColNm, recur=None, final = branch == maxDP )


					# ========== Calculate the random forest ==========
					time,  r2, feature_imp  = skl_rf_regression(
						X_train, X_test, y_train, y_test, col_nms, ntree, test_size, 
						branch, window, tmpath, maxDP, experiment, verbose=False, perm=True)
					
					# ========== perform some zeo branch data storage ==========
					if branch == 0:
						# +++++ Calculate the ward hierarchy +++++
						corr_linkage = hierarchy.ward(corr)
						# +++++ Create a zero time deltat +++++
						cum_time     = pd.Timedelta(0)
						orig_clnm    = col_nms.copy()
					else:
						# ========== Pull out the existing total time ==========
						cum_time = perf["Branch%02d" % (branch-1)]["TimeCumulative"]

					# ========== Add the results of the different itterations to OD ==========
					perf["Branch%02d" % branch] = ({
						"LSWindow":window, "branch":branch, "dfloadtime":loadstats["loadtime"], 
						"RFtime":time, "TimeCumulative":cum_time + (loadstats["loadtime"]+time),  
						"R2":r2, "NumVar":loadstats["colcount"],  "SiteFraction":loadstats["fractrows"],
						})
					
					# ========== Perform Variable selection and get new column names ==========
					ColNm = Variable_selection(corr_linkage, branch, feature_imp, X_test, col_nms, orig_clnm)
					print("the number of columns to be cosidered in the next itteration: ", len(ColNm), " R2 of this run: ", r2)

					
				# ========== Pull out the features that i'm going to use on the next loop ==========
				perf_df = pd.DataFrame(perf).T
				for var in ["TimeCumulative", "RFtime", "dfloadtime"]:	
					perf_df[var] = perf_df[var].astype('timedelta64[s]') / 60.
				perf_df["experiment"] = experiment

				# # ========== Save the df with the timing and performance ===========
				perf_df.to_csv(fnout)
				writenames.append(fnout)

	Summaryplots(writenames)
	ipdb.set_trace()

def Summaryplots(writenames):
	"""
	Function takes a list of filenames, concats into a single 
	"""

	# +++++ make a single df that contains the spped and accuracy of the methods +++++
	exp_df  = pd.concat([pd.read_csv(fn).rename( columns={'Unnamed: 0':'BranchName'}) for fn in writenames])
	exp_df["LSWindow"] = exp_df["LSWindow"].astype('category')
	exp_df["R2perVar"] = exp_df["R2"] / exp_df["NumVar"]

	for va in ["TimeCumulative", "R2", "NumVar",'SiteFraction', "R2perVar"]:

		# +++++ make some plots +++++
		plt.figure(va)
		ax = sns.lineplot(x="BranchName", y=va, data=exp_df,  err_style='bars', hue="LSWindow")
		# +++++ Make a log scale to help visulation in some variables +++++
		if va in ["correlated"]: #"NumVar", 
			ax.set_yscale('log')
		for label in ax.get_xticklabels():label.set_rotation(90)
		
		plt.show()
	
	# breakpoint()
	ax = sns.lineplot(x="NumVar", y="R2", data=exp_df, err_style='band', hue="LSWindow")
	for label in ax.get_xticklabels():label.set_rotation(90)
	plt.show()
	# ax = sns.lineplot(x="correlated", y="R2", data=exp_df,  hue="LSWindow") #err_style='band',
	# ax.set_xscale('log')
	# # for label in ax.get_xticklabels():label.set_rotation(90)
	# plt.show()
	breakpoint()

def Variable_selection(corr_linkage, branch, feature_imp, X_train, col_nms, orig_clnm):
	"""
	Function uses the coorrelation linchage and the branch to select the best variables
	args:
		corr_linkage	
			output of hierarchy.ward(corr)
		branch:			int
			the branch of the correlation to select from
		feature_imp:	OrderedDict
			variable name and feature importance
		X_train:		df
			df to get the column names currently being used
	returns:
		ColNm:	list of column names to test on the next round of the itteration

	"""
	# ========== Performing Clustering based on the branch level ==========
	cluster_ids   = hierarchy.fcluster(corr_linkage, branch, criterion='distance')
	clusID_featID = defaultdict(list)

	# ========== Find what variable belong in each cluster ==========
	for idx, cluster_id in enumerate(cluster_ids): clusID_featID[cluster_id].append(idx)

	# ========== Find the most important variable in each cluster ==========
	sel_feat = [] #Container for the featture names

	for clus in  clusID_featID:
		try:
			# +++++ Get the IDs, the feature importance and the name +++++
			IDs = clusID_featID[clus]
			NMs = orig_clnm[IDs]

			# +++++ drop the names that are irrelevant +++++
			# This approach will look at the most recent feature performance
			FIs = []

			for fn in NMs:
				if (fn in col_nms) and (feature_imp[fn]>0):
					FIs.append(feature_imp[fn])
				else:
					FIs.append(np.NAN)
			
			try:
				sel_feat.append(NMs[bn.nanargmax(FIs)])
			except ValueError:
				pass
		except Exception as er:
			warn.warn("something went wrong here " )
			print(str(er))
			ipdb.set_trace()

	# ========== Pull out the features that i'm going to use on the next loop ==========
	ColNm = sel_feat

	# ========== readd lagged biomass ==========
	ColNm.append("lagged_biomass")

	return ColNm

# ========== Calculate the number of loops i need for each variable ==========
def _branchcal(window, tmpath="./pyEWS/experiments/1.FeatureSelection/tmp/", test_size=0.2, brch=0, experiment=0):
	print("Getting the maxt depth for %d yr window" % window)
	# ========== Load the data and get the basic stats ==========
	X_train, X_test, y_train, y_test, col_nms, loadstats, corr = df_proccessing(
		window, tmpath, test_size, brch, experiment, verbose=False)
	
	corr_linkage = hierarchy.ward(corr)

	# ========== loop over the windows ==========
	for branch in np.arange(50):
		# ========== Performing Clustering based on the branch level ==========
		cluster_ids   = hierarchy.fcluster(corr_linkage, branch, criterion='distance')
		if np.unique(cluster_ids).size <40:
			return branch#-1
	breakpoint()
# ==============================================================================
# ============================== Forest modeling ===============================
# ==============================================================================

def skl_rf_regression( X_train, X_test, y_train, y_test, col_nms, ntree, 
	test_size, branch, test, tmpath, maxiter, experiment, cores=-1, verbose=True, perm=False):
	"""
	This function is to test out the  speed of the random forest regressions using
	sklearn 
	args:
		Xin: 			ndarray or pd dataframe
			data to be used for prediction
		yin: 			ndarray or pd dataframe
			data to be predicted
		col_nms:	array
			name of the columns
		cores:		int
			the number of CPU cores to use, Defualt=-1 (the total number of threads)
		verbose: 	bool
			How much infomation to print, Defualt=True
		perm:		bool
			Use the permutation importance rather than feature importance
	"""


	# ========== Setup some skl params ==========
	skl_rf_params = ({
		'n_estimators': ntree,
		'n_jobs': cores })
		# 'max_depth': 13,

	# ========== Start timing  ==========
	t0 = pd.Timestamp.now()
	print("starting sklearn random forest regression at:", t0)

	# ========== Do the RF regression training ==========
	regressor = RandomForestRegressor(**skl_rf_params)
	regressor.fit(X_train, y_train)

	# ========== Testing out of prediction ==========
	print("starting sklearn random forest prediction at:", pd.Timestamp.now())
	y_pred = regressor.predict(X_test)
	
	# ========== make a list of names ==========
	clnames = X_train.columns.values
	
	# ========== print all the infomation if verbose ==========
	if verbose:
		print('r squared score:',         sklMet.r2_score(y_test, y_pred))
		print('Mean Absolute Error:',     sklMet.mean_absolute_error(y_test, y_pred))
		print('Mean Squared Error:',      sklMet.mean_squared_error(y_test, y_pred))
		print('Root Mean Squared Error:', np.sqrt(sklMet.mean_squared_error(y_test, y_pred)))
		
		# ========== print Variable importance ==========
		for var, imp in  zip(clnames, regressor.feature_importances_):
			print("Variable: %s Importance: %06f" % (var, imp))

	# ========== Convert Feature importance to a dictionary ==========
	FI = OrderedDict()
	# +++++ use permutation importance +++++
	print("starting sklearn permutation importance calculation at:", pd.Timestamp.now())
	result = permutation_importance(regressor, X_test, y_test, n_repeats=5) #n_jobs=cores
	
	for fname, f_imp in zip(clnames, result.importances_mean): 
		FI[fname] = f_imp

	# ========== Print the time taken ==========
	tDif = pd.Timestamp.now()-t0
	print("The time taken to perform the random forest regression:", tDif)

	# =========== Save out the results if the branch is approaching the end ==========
	if branch == (maxiter - 1):

		# =========== save the predictions of the last branch ==========
		_predictedVSobserved(y_test, y_pred, branch, test, tmpath, experiment)

	return tDif, sklMet.r2_score(y_test, y_pred), FI


def _predictedVSobserved(y_test, y_pred, branch, test, tmpath, experiment):
	"""
	function to save out the predicted vs the observed values
	"""
	path = "./pyEWS/experiments/1.FeatureSelection/"
	dfy  = pd.DataFrame(y_test).rename({"lagged_biomass":"Observed"}, axis=1)
	dfy["Estimated"] = y_pred
	
	fnameout = path + "S06_OOSest_LS_test%02dyrs_%02d_exp%02d.csv" % (test, branch, experiment)
	print(fnameout)
	dfy.to_csv(fnameout)
# ==============================================================================
# ============================== Data processing ===============================
# ==============================================================================

def df_proccessing(window, tmpath, test_size, branch, experiment,  dftype="pandas", 
	cols_keep=None, recur=None, verbose = True, region=False, final=False):
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
	# cor_fn = "./EWS_package/data/models/input_data/correlations_2019-10-09.csv"
	vi_fn  = "./EWS_package/data/models/input_data/vi_df_all_V2.csv"

	# ============ Open the variables and correlations ============
	# This may need to change when using cuda 
	if verbose:
		print("loading the files using %s at:" % dftype, t0)
	vi_df  = pd.read_csv( vi_fn, index_col=0)
	if region:
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


	if verbose:
		print("loading the files using %s took: " % dftype, pd.Timestamp.now()-t0)
	
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
			elif cn in ['site']:
				pass
			else:
				cols_keep.append(cn)
	# else:
	# 	ipdb.set_trace()
	# ========== Fill in any NA rows ==========
	data      = vi_df[cols_keep].dropna()

	# =========== drop rows with 0 variance ===========
	data.drop(data.columns[data.std() == 0], axis=1, inplace=True)

	# =========== Pull out the data used for prediction ===========
	X        = data.drop(["lagged_biomass"], axis = 1).astype("float32")
	cols_out = X.columns.values

	# =========== Pull out the data the is to be predicted ===========
	y = (data["lagged_biomass"]).astype("float32")

	# ========== Make some simple stats ===========
	def _simplestatus(vi_df, X, corr, threshold=0.5):
		statsOD = OrderedDict()
		statsOD["totalrows"] = vi_df.shape[0]
		statsOD["itterrows"] = X.shape[0]
		statsOD["fractrows"] = float(X.shape[0])/float(vi_df.shape[0])
		statsOD["colcount" ] = X.shape[1]

		# =========== work out how many things correlate ===========
		corr                 = spearmanr(X).correlation
		corr[corr == 1.]     = np.NaN
		statsOD["covariate"] = np.sum(abs(corr)>threshold)
		statsOD["meancorr"]  = bn.nanmean(abs(corr))
		
		statsOD["loadtime"]  = 0
		return statsOD

	# ========== Setup the inital clustering ==========
	# +++++ Build a spearman table +++++
	try:
		corr         = spearmanr(X).correlation
	except Exception as er:
		print(str(er))
		ipdb.set_trace()



	statsOD = _simplestatus(vi_df, X, corr, threshold=0.5)

	# ========== Split the data  ==========
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
	
	# =========== Fnction to quicky save dataframes ==========
	def _quicksave(dftype, tmpath, df, dfname, branch, experiment, window):
		# ========== Quickly save the CSV files of the components
		# +++++ make a file name ++++++ 
		fnout = tmpath + "S06tmp_final_%s_%03d_win%02d_exp%02s.csv" % (dfname, branch,  window, experiment)
		# +++++ write the result ++++++ 
		df.to_csv(fnout)
		# +++++ return the fname ++++++
		return fnout

	# =========== save the split dataframe out so i can reload with dask as needed ============
	if final:
		fnames = {dfname:_quicksave(dftype, tmpath, df, dfname, branch, experiment, window) for df, dfname in zip(
			[X_train, X_test, y_train, y_test], ["X_train", "X_test", "y_train", "y_test"])}

	
	# return the split data and the time it takes
	statsOD["loadtime"] = pd.Timestamp.now() - t0
	# ============ Return the filtered data  ============
	return X_train, X_test, y_train, y_test, cols_out, statsOD, corr

# ==============================================================================
if __name__ == '__main__':
	main()