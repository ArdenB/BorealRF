"""
Function to open data in the benchmarking suite.
If the datasets have not been made yet this script will generate them
"""
# ==============================================================================

__title__ = "DataSplit"
__author__ = "Arden Burrell"
__version__ = "v1.0(11.06.2020)"
__email__ = "arden.burrell@gmail.com"

# ==============================================================================
# ==============================================================================

# ========== Import packages ==========
import numpy as np
import pandas as pd
import os
import sys
import glob
from collections import OrderedDict, defaultdict
import warnings as warn
from sklearn.model_selection import train_test_split
import myfunctions.corefunctions as cf
from scipy.stats import spearmanr
import bottleneck as bn

def _testtrainbuild(version, VI_fnsplit,  vi_df, test_size):
	"""Function to make batch producing new ttsplit datasets eays"""
	# +++++ make new files +++++

	# ========== Partition the dataset into different window lengths ==========
	vi_df["landsatgroup"] = 1
	# ============ landsat group is the last ls windown that a dataset has no nans ============
	LandsatWindows = [1, 5, 10, 15, 20]
	for lswin in LandsatWindows:
		# +++++ A container to hold the kept columns  +++++ 
		c_keep = []
		clnm =  vi_df.columns.values
		for cn in clnm:
			# Test to see if the column is one of the ones i want to keep
			if cn.startswith("LANDSAT"):
				# The VI datasets, check the length of window considered
				if int(cn.split("_")[-1]) <= lswin:
					c_keep.append(cn)
			elif cn in ['site']:
				pass
			else:
				c_keep.append(cn)
		# ========== set the group ==========
		vi_df["landsatgroup"][vi_df[c_keep].dropna().index] = lswin

	# ========== make containers for the test train split results
	ls_X_train = []  
	ls_X_test  = [] 
	ls_y_train = [] 
	ls_y_test  = []
	# ========== Loop over the windos again ==========
	for lswin in LandsatWindows:
		data = vi_df[vi_df["landsatgroup"]== lswin]
		# =========== Pull out the data used for prediction ===========
		X        = data.drop(["lagged_biomass", "landsatgroup", 'site'], axis = 1).astype("float32")
		cols_out = X.columns.values

		# =========== Pull out the data the is to be predicted ===========
		y = (data["lagged_biomass"]).astype("float32")

		# ========== Split the data  ==========
		X_tr, X_tt, y_tr, y_tt = train_test_split(X, y, test_size=test_size)
		for ls, dt in zip([ls_X_train, ls_X_test,ls_y_train,ls_y_test, ], [X_tr, X_tt, y_tr, y_tt]):
			ls.append(dt)

	# ========== build a single df for each  ==========
	X_train = pd.concat(ls_X_train)
	X_test  = pd.concat(ls_X_test )
	y_train = pd.DataFrame(pd.concat(ls_y_train))
	y_test  = pd.DataFrame(pd.concat(ls_y_test ))
	# ========== Save them out  ==========
	for df, fn in zip([X_train, X_test, y_train, y_test ] , VI_fnsplit):
		df.to_csv(fn)

	return X_train, X_test, y_train, y_test

def _regionbuilder(region_fn, vi_df):
	"""
	Function to work out which region each site is in and save it out to a file
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

	# ========== fix the missing problem ==========
	site_df = site_df.rename({"Plot_ID":"site"}, axis=1)#.set_index("site")
	site_vi = vi_df[["site", "lagged_biomass"]].copy()#.set_index("site")
	site_df = site_df.merge(
		site_vi, on="site", how="outer").drop("lagged_biomass", axis=1)
	site_df.drop_duplicates(subset="site", keep="first", inplace = True, ignore_index=True)
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
	
	site_df["Region"] = [site_locator(sn, sitenames, siteregions, Sitekey) for sn in site_df.site]

	# ========== Make metadata infomation ========== 
	maininfo = "All data in this folder is written from %s (%s):%s by %s, %s" % (__title__, __file__, 
		__version__, __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	
	site_df.to_csv(region_fn)
	cf.writemetadata(region_fn, [maininfo, gitinfo])

	return site_df

# ========== Make some simple stats ===========
def _simplestatus(vi_df, X, df_site):
	statsOD = OrderedDict()
	statsOD["totalrows"] = vi_df.shape[0]
	statsOD["itterrows"] = X.shape[0]
	statsOD["fractrows"] = float(X.shape[0])/float(vi_df.shape[0])
	statsOD["colcount" ] = X.shape[1]

	# ========== create the full list of sites and regions ==========
	df_full = vi_df.reset_index().merge(
		df_site, on="site", how="left")[["index","site", "Region"]].drop_duplicates(keep='first')
	if (df_full.Region.isnull()).any():
		warn.warn("\n\n There are areas with null values for reason unknown, going interactive. \n\n")
		breakpoint()
	# df_full.loc[df_full.Region.isnull(), "Region"] = "YT"
	df_sub  = X.reset_index().merge(df_full, on="index", how="left")
	
	# ========== Count the number of sites in each region ==========
	for region in df_full.Region.unique():
		try:
			inc_reg = float(df_sub.Region.value_counts()[region])
			tot_ref = float(df_full.Region.value_counts()[region])
			if inc_reg > tot_ref:
				print("got an insane value here")
				breakpoint()
			statsOD["%s_siteinc" % region]  = inc_reg
			statsOD["%s_sitepos" % region]  = tot_ref
			statsOD["%s_sitefrac" % region] = inc_reg/tot_ref
		except:
			statsOD["%s_siteinc" % region]  = 0 
			statsOD["%s_sitepos" % region]  = df_full.Region.value_counts()[region]
			statsOD["%s_sitefrac" % region] = 0.0
	return statsOD
# ==============================================================================
# The main part of the function is here
# ==============================================================================
def datasplit(experiment, version,  branch, setup,  group=None, test_size=0.2, dftype="pandas", 
	cols_keep=None, verbose = True, region=False,  final=False, RStage=True,
	vi_fn = "./EWS_package/data/models/input_data/vi_df_all_V2.csv", 
	folder = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/", 
	force = False, y_names=None, sitefix = False
	):
	"""
	This function opens and performs all preprocessing on the dataframes.
	These datasets were originally made using:
		./code/data_prep/create_VI_df.R
		./code/data_prep/vi_calcs/functions/modified......
		./code/data_prep/vi_calcs/full_df_calcs_loop......
	this function has two parts, the first is to generate and save the dataset if the
	file down't already exist. the second is to do any processing on the dataframe 
	that the higher level scripts require.  
	args:
		experiment:		int
			the experiment number
		Version:		int
			the experiment number
		branch: 	int
			the branch number, will be used to seed random humber generator
		setup:			dict
			container will all the setop infomation
		test_size:		float
			defualt 0.2, the size of the rest train split
		dftype:		str
			dataset loaded using pandas, dask or cudf
		cols_keep:		list		
			A list of columns to subest the datasett
		verbose:		bool
			print additional info
		region:			bool
			return the region specific infomation 
		vi_fn:			str
			the filname of the vi dataset to split
		folder:			str
			path to location where split datasets should be saved 


	"""
	# ====================================================================
	# ====================== Version data creation =======================
	# ====================================================================
	# ============ Set a timer and make sure the path folder  ============
	t0 = pd.Timestamp.now()
	cf.pymkdir(folder)

	# ============ Open the datasets ============
	if verbose:
		print("loading the files using %s at:" % dftype, t0)

	# ============ Setup the file names ============
	VI_fnsplit = [folder + "TTS_vers%02d_%s.csv" % (version, sptyp) for sptyp in ["X_train", "X_test", "y_train", "y_test"]]
	vi_df  = pd.read_csv( vi_fn, index_col=0)

	# ============Check if the files already exist ============
	if all([os.path.isfile(fn) for fn in VI_fnsplit]) and not force:
		# +++++ open existing files +++++
		X_train, X_test, y_train, y_test = [pd.read_csv( fn, index_col=0) for fn in VI_fnsplit]
	else:
		print("Building Test/train dataset for version: ", version, pd.Timestamp.now())
		X_train, X_test, y_train, y_test = _testtrainbuild(version, VI_fnsplit,  vi_df.copy(), test_size)

	# ========== Look for a region file ==========
	region_fn = folder + "TTS_sites_and_regions.csv"
	if os.path.isfile(region_fn) and not force:
		df_site = pd.read_csv(region_fn, index_col=0)
	else:
		df_site = _regionbuilder(region_fn, vi_df)
	
	# ========== print time taken to get all the files ==========
	if verbose:
		print("loading the files using %s took: " % dftype, pd.Timestamp.now()-t0)


	# ========== Perform extra splits ==========
	if setup['Nstage']!=1:
		if not RStage:
			# ///// the classification stage \\\\\\\\\\
			# ========== Make the split filenames =========
			VI_fnsplit2 = [folder + "TTS_vers%02d_%s_twostage.csv" % (version, sptyp) for sptyp in ["X_train", "X_test", "y_train", "y_test"]]

			# ========== Check if i need to calculate the splits ==========
			y_testCal = y_test.copy()
			X_testCal = X_test.copy()
			# y_train, y_test
			# ============Check if the files already exist ============
			if all([os.path.isfile(fn) for fn in VI_fnsplit2]) and not force:
				# +++++ open existing files +++++
				X_train, X_test, y_train, y_test = [pd.read_csv( fn, index_col=0) for fn in VI_fnsplit2]
			else:
				print("Building a two stage Test/train dataset for version: ", version, pd.Timestamp.now())
				# X_train, X_test, y_train, y_test = _testtrainbuild(version, VI_fnsplit,  vi_df.copy(), test_size)
				X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size)
				y_train = pd.DataFrame(y_train)
				y_test  = pd.DataFrame(y_test)
				for df, fn in zip([X_train, X_test, y_train, y_test ] , VI_fnsplit2):
					df.to_csv(fn)
			# ========== Perform any classification and extra splits ==========
			y_testCal, y_train, y_test = setup["classMethod"]([y_testCal, y_train, y_test], setup)
			
	# ====================================================================
	# ======================  Branch data creation =======================
	# ====================================================================
	
	# ============ Filter the rows ============
	if cols_keep is None:
		# +++++ A container to hold the kept columns  +++++ 
		cols_keep = []
		clnm =  vi_df.columns.values
		for cn in clnm:
			# Test to see if the column is one of the ones i want to keep
			if cn.startswith("LANDSAT"):
				# The VI datasets, check the length of window considered
				if int(cn.split("_")[-1]) <= setup["window"]:
					cols_keep.append(cn)
			elif cn in ['site', 'lagged_biomass', 'landsatgroup']:
				pass
			else:
				cols_keep.append(cn)

	# ========== Remove any nans ==========
	# try:
	X_train = X_train[cols_keep].dropna()
	X_train.drop(X_train.columns[X_train.std() == 0], axis=1, inplace=True)
	X_test  =  X_test[X_train.columns].dropna()
	y_train = pd.DataFrame(y_train["lagged_biomass"][X_train.index])
	y_test  = pd.DataFrame(y_test["lagged_biomass"][X_test.index])
	
	# ========== Do the samee for the test datasets that a held over to the last stage ==========
	if sitefix == True:
		# =========== Pull out the data used for prediction ===========
		X        = pd.concat([X_test, X_train])
		cols_out = X.columns.values
		statsOD = _simplestatus(vi_df, X, df_site)
		return statsOD

	elif setup['Nstage']!=1:
		if not RStage:
			X_testCal = X_testCal[X_train.columns].dropna()
			y_testCal = pd.DataFrame(y_testCal["lagged_biomass"][X_testCal.index])
			classified = {"X_test":X_testCal, "y_test":y_testCal}
		else:
			# ///// the regression stage \\\\\\\\\\
			# load in the guessed R2 values and do some form of class filtering
			# ========== Load the data in ==========
			y_testcat  = pd.read_csv(y_names[0], index_col=0)
			y_traincat = pd.read_csv(y_names[1], index_col=0)
			X_test["group"]  = y_testcat.reindex(X_test.index)
			X_train["group"] = y_traincat.reindex(X_train.index)
			
			# ========== Subset the data ==========
			X_train = (X_train[X_train["group"] == group]).drop("group", axis=1)
			X_train.drop(X_train.columns[X_train.std() == 0], axis=1, inplace=True)
			
			# ///// Added to remove some weird zero variance issues \\\\\
			X_test  = ( X_test[X_test["group"] == group]).drop("group", axis=1)
			X_test  =  X_test[X_train.columns]

			y_train = pd.DataFrame(y_train["lagged_biomass"][X_train.index])
			y_test  = pd.DataFrame(y_test["lagged_biomass"][X_test.index])

	# except Exception as er:
	# 	print(str(er))
	# 	breakpoint()
	# =========== Pull out the data used for prediction ===========
	X        = pd.concat([X_test, X_train])
	cols_out = X.columns.values
	
	# ========== Build the spearmans rank correlation clusters ==========
	# +++++ Build a spearman table +++++
	try:
		corr = spearmanr(X).correlation
		if bn.anynan(corr):
			warn.warn("Nan values slipped through the correlation, going interactive")
			breakpoint()
	except Exception as er:
		print(str(er))
		breakpoint()




	statsOD = _simplestatus(vi_df, X, df_site)

	# ========== Split the data  ==========
	if final:
		print("Returing the true test train set. Loading and processing the files took:",  pd.Timestamp.now()-t0)
		if setup['Nstage']!=1 and not RStage:
			return X_train, X_test, y_train, y_test, cols_out, statsOD, corr, df_site, classified
		else:
			return X_train, X_test, y_train, y_test, cols_out, statsOD, corr, df_site
	else:
		# ========== Non final Need to split the training set ==========
		# /// This is done so that my test set is pristine when looking at final models \\\
		X_trainRec, X_testRec, y_trainRec, y_testRec = train_test_split(X_train, y_train, test_size=test_size, random_state=branch)

		# ============ Return the filtered data  ============
		print("Returing the twice split test train set. Loading and processing the files took:",  pd.Timestamp.now()-t0)
		if setup['Nstage']!=1 and not RStage:
			return X_trainRec, X_testRec, y_trainRec, y_testRec, cols_out, statsOD, corr, df_site, classified
		else:
			return X_trainRec, X_testRec, y_trainRec, y_testRec, cols_out, statsOD, corr, df_site



