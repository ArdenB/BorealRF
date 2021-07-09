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
from sklearn.model_selection import GroupShuffleSplit


def altsplit(setup, df_site, vi_df, test_size, predvar, dropvar, version, 
	basestr, folder, batchsplit=True,  n_splits=10, random_state=0, vers = None, columns=None):
	"""group=None, y=None, x=None
	An Alternative splitting function to the test train split mode
	This will work 
	"""
	# ========== Inital split, if i'm not doing stuff then this will  ==========
	if n_splits == 1:
		ts = test_size
	else:
		ts  = setup['FullTestSize']
		ts2 = np.round(test_size/(1-setup['FullTestSize']), decimals=3)

	gss   = GroupShuffleSplit(n_splits=1, test_size=ts, random_state=random_state)
	fnl   = f"{folder}{basestr}_TTSlookup.csv"
	try:
		if setup["splitvar"] == "site":
			group = vi_df["site"]
		elif setup["splitvar"] == ["site", "yrend"]:
			df_site["yrend"] = vi_df.ObsGap + df_site.year
			df_site["grp"]   = df_site.groupby(setup["splitvar"]).grouper.group_info[0]
			group = df_site["grp"]
			
		else:
			warn.warn("Not implemented yet")
			breakpoint()
		y     = vi_df[predvar].astype("float32")
		X     = vi_df.drop([predvar, 'site'] + dropvar, axis = 1).astype("float32")
		if not columns is None:
			X = X.loc[:, columns]
	except:
		breakpoint()
	# ========== make the prelim test train group ==========
	if setup['FullTestSize'] > 0:
		for ptrain, ptest in gss.split(X, y, groups=group):
			# ========== Check and see if the values have already been randomised ==========
			if np.all(np.diff(X.index.values) > 0):
				rng = np.random.default_rng(random_state)
				rng.shuffle(ptest)
			# else:
			# 	# ========== go to lookup table ==========
			# 	dfl = pd.read_csv(fnl, index_col=0)
			# 	
			if n_splits == 1:
				# ========== sace for second splits =====
				if not _sizecheck(X.iloc[ptrain], X.iloc[ptest], y.iloc[ptrain], y.iloc[ptest]):
					warn.warn("Size Check failed here")
					breakpoint()
				return X.iloc[ptrain], X.iloc[ptest], y.iloc[ptrain], y.iloc[ptest]
		# ========== loop over the the second gss ==========
		Xp  = X.iloc[ptrain]
		yp  = y.iloc[ptrain]
		gp  = group.iloc[ptrain]
	else:

		Xp  = X.copy()
		yp  = y.copy()
		gp  = group.copy()
		ptest = []


	# ========== Now move on to the alternat split approach ==========	
	gssM = GroupShuffleSplit(n_splits=n_splits, test_size=ts2, random_state=random_state)
	dfc  = pd.DataFrame(np.zeros((vi_df.shape[0], n_splits))*np.NaN, index=vi_df.index, columns=np.arange(n_splits))
	vfc  = np.vectorize(_findcords, excluded=["ptest", "itest","vtest", "vtrain"])

	for num, (train, test) in enumerate(gssM.split(Xp, yp, groups=gp)):
		print("Building Test/train dataset for version: ", num, pd.Timestamp.now())
		VI_fnsplit = [folder + "%s_vers%02d_%s.csv" % (basestr, num, sptyp) for sptyp in ["X_train", "X_test", "y_train", "y_test"]]
		# ========== Save them out  ==========
		rng = np.random.default_rng(num)
		rng.shuffle(test)
		rng.shuffle(train)
		Xt_train =  Xp.iloc[train]
		yt_train =  yp.iloc[train]
		gt_train =  gp.iloc[train]
		
		if setup['FullTestSize'] > 0:
			Xt_test  = Xp.iloc[test].append(X.iloc[ptest])
			yt_test  = yp.iloc[test].append(y.iloc[ptest])
		else:
			Xt_test  = Xp.iloc[test]
			yt_test  = yp.iloc[test]
		# breakpoint()
		# ========== CHECK THE SIZE ==========
		if not _sizecheck(Xt_train, Xt_test, yt_train, yt_test):
			warn.warn("Size Check failed here")
			breakpoint()
		# ========== create the validation set ==========
		gssV = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=num)
		for vtrain, vtest in gssV.split(Xt_train, yt_train, groups=gt_train):
			# ========== Create a lookup table ==========
			dfc.loc[:, num] = vfc(
				dfc.index.values, 
				ptest  = ptest, 
				itest  = yp.index.values[test], 
				vtest  = yt_train.index.values[vtest], 
				vtrain = yt_train.index.values[vtrain])

			# breakpoint()
		
			
		# ========== Write the file out ==========
		for df, fn in zip([Xt_train, Xt_test, yt_train, yt_test ] , VI_fnsplit):
			df.to_csv(fn)
		
		# ========== Check and see if its the requested version ==========
		if num == version:
			X_train, X_test, y_train, y_test = [Xt_train, Xt_test, yt_train, yt_test]
		

	if not _sizecheck(X_train, X_test, y_train, y_test):
		warn.warn("Size Check failed inside altsplit function")
		breakpoint()
	
	# ========== save the lookup table ==========
	dfc.to_csv(fnl)

	return X_train, X_test, y_train, y_test

def _findcords(x, ptest, itest, vtest, vtrain):
	# Function check if x is in different arrays

	## I might need to use an xor here
	if x in ptest: #multi run fully witheld 
		return 3
	elif x in itest: #full witheld fract
		return 2
	elif x in vtest: #validation test
		return 1
	elif x in vtrain: #training set
		return 0
	else:
		raise ValueError("x not in any index")

def _testtrainbuild(version, VI_fnsplit,  vi_df, test_size, predvar, dropvar):
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
		X        = data.drop([predvar, "landsatgroup", 'site']+ dropvar, axis = 1).astype("float32")
		cols_out = X.columns.values

		# =========== Pull out the data the is to be predicted ===========
		y = (data[predvar]).astype("float32")

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

def _regionbuilder(region_fn, vi_df, predvar):
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
	site_vi = vi_df[["site", predvar]].copy()#.set_index("site")
	site_df = site_df.merge(
		site_vi, on="site", how="outer").drop(predvar, axis=1)
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

def _sizecheck(X_train, X_test, y_train, y_test):
	checks=([
		X_train.shape[1]== X_test.shape[1], 
		X_train.shape[0]== y_train.shape[0], 
		X_test.shape[0]== y_test.shape[0], 
		not any(X_train.index.duplicated()),
		not any(X_test.index.duplicated()),  
		y_test.index.equals(X_test.index), 
		y_train.index.equals(X_train.index)])
	return all(checks)
# ========== Make some simple stats ===========
def _simplestatus(vi_df, X, df_site):
	statsOD = OrderedDict()
	statsOD["totalrows"] = vi_df.shape[0]
	statsOD["itterrows"] = X.shape[0]
	statsOD["fractrows"] = float(X.shape[0])/float(vi_df.shape[0])
	statsOD["colcount" ] = X.shape[1]

	# ========== create the full list of sites and regions ==========
	if "Region" in df_site.columns:
		# +++++ This is the new column values +++++
		df_full = df_site.reset_index()
	else:
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

def datasplit(predvar, experiment, version,  branch, setup, trans=None,  group=None, test_size=0.2, dftype="pandas", 
	cols_keep=None, verbose = True, region=False,  final=False, RStage=True,
	vi_fn = "./EWS_package/data/models/input_data/vi_df_all_V2.csv", 
	region_fn=None,
	folder = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/", 
	force = False, y_names=None, sitefix = False, basestr="TTS", dropvar=[]
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
	VI_fnsplit = [folder + "%s_vers%02d_%s.csv" % (basestr,version, sptyp) for sptyp in ["X_train", "X_test", "y_train", "y_test"]]
	vi_df  = pd.read_csv( vi_fn, index_col=0)

	# ========== Look for a region file ==========
	if region_fn is None:
		region_fn = folder + "TTS_sites_and_regions.csv"
		if os.path.isfile(region_fn) and not force:
			df_site = pd.read_csv(region_fn, index_col=0)
		else:
			df_site = _regionbuilder(region_fn, vi_df, predvar)
	else:
		df_site = pd.read_csv(region_fn, index_col=0)
		df_site.rename({"Plot_ID":"site"}, axis=1, inplace=True)

	# breakpoint()
	# ============Check if the files already exist ============
	if all([os.path.isfile(fn) for fn in VI_fnsplit]) and not force:
		# +++++ open existing files +++++
		X_train, X_test, y_train, y_test = [pd.read_csv( fn, index_col=0) for fn in VI_fnsplit]
		# ========== Checking the preedictor variable ==========
		if not y_test.columns[0] == predvar:
			# +++ This existis so i can mach the analysis between differnet predictors +++
			print(f"Swapping predictor variables from {y_test.columns[0]} to {predvar} at: {pd.Timestamp.now()}")
			#swap out the predicto variable 
			y_test  = pd.DataFrame(vi_df.loc[y_test.index, predvar])

			y_train = pd.DataFrame(vi_df.loc[y_train.index, predvar])
		if y_test[predvar].isnull().any():
			X_test = X_test[~y_test[predvar].isnull()]
			y_test = y_test[~y_test[predvar].isnull()]

		if y_train[predvar].isnull().any():
			X_train = X_train[~y_train[predvar].isnull()]
			y_train = y_train[~y_train[predvar].isnull()]

	else:
		# here i have some callout method 
		# Is should save all 10 versions in one go as well as a full withheld cross fraction
		if not setup["FullTestSize"] is None:
			X_train, X_test, y_train, y_test = altsplit(
				setup, df_site,  vi_df, test_size, predvar, dropvar, version, basestr, folder, 
				batchsplit=True, n_splits=10, random_state=0, vers = None)
			# breakpoint()
		else:
			print("Building Test/train dataset for version: ", version, pd.Timestamp.now())
			X_train, X_test, y_train, y_test = _testtrainbuild(version, VI_fnsplit,  vi_df.copy(), test_size, predvar, dropvar)

	# ========== print time taken to get all the files ==========
	if not _sizecheck(X_train, X_test, y_train, y_test):
		warn.warn("Size Check failed here 1")
		breakpoint()

	if verbose:
		print("loading the files using %s took: " % dftype, pd.Timestamp.now()-t0)

	# ========== Perform extra splits ==========
	if setup['Nstage']!=1:
		if not RStage:
			# ///// the classification stage \\\\\\\\\\
			# ========== Make the split filenames =========
			VI_fnsplit2 = [folder + "%s_vers%02d_%s_twostage.csv" % (basestr,version, sptyp) for sptyp in ["X_train", "X_test", "y_train", "y_test"]]

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
	
	# ========== Droplist for columns ========== 
	droplist = ['site', "year",  predvar, 'landsatgroup']+dropvar
	

	# ========== Drop disturbed sites ==========
	if 'DropDist' in setup.keys():
		if setup['DropDist']:
			X_train = X_train[df_site.loc[X_train.index.values]["DistPassed"] == 1]
			X_test  = X_test [df_site.loc[X_test.index.values]["DistPassed"] == 1]
			y_train = y_train[df_site.loc[y_train.index.values]["DistPassed"] == 1]
			y_test  = y_test [df_site.loc[y_test.index.values]["DistPassed"] == 1]

		else:
			print(f"Adding Disturbance columns at: {pd.Timestamp.now()}")
			distcols = ["Disturbance",  "DisturbanceGap",   "Burn",  "BurnGap",  "StandAge"]
			# ========== Add additional columns here ==========
			X_train =  pd.concat([X_train, df_site.loc[X_train.index, distcols]], axis=1)
			X_test  =  pd.concat([X_test,  df_site.loc[X_test.index, distcols]], axis=1)
			y_train =  pd.concat([y_train, df_site.loc[y_train.index, distcols]], axis=1)
			y_test  =  pd.concat([y_test,  df_site.loc[y_test.index, distcols]], axis=1)
			# breakpoint()
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
			elif cn in droplist:
				pass
			else:
				cols_keep.append(cn)

	# ========== Drop disturbed sites ==========		
	if "FutDist" in setup.keys():
		distcal = df_site.BurnFut + df_site.DisturbanceFut
		distcal.where(distcal<=100., 100, inplace=True)
		
		dst = distcal <= setup["FutDist"]
		X_train = X_train.loc[dst.loc[X_train.index.values]]
		X_test  = X_test .loc[dst.loc[X_test.index.values] ]
		y_train = y_train.loc[dst.loc[y_train.index.values]]
		y_test  = y_test .loc[dst.loc[y_test.index.values] ]

		if not _sizecheck(X_train, X_test, y_train, y_test):
			warn.warn("Size Check failed here 2")
			breakpoint()
		# breakpoint()
	# ========== Check sites with NaNs ==========

	if 'DropNAN' in setup.keys() and not setup["DropNAN"] == 0:
		# =========== Get rid of columnns with les that 10% of value ===========
		# /// This is to solve a nan correllation problem. 
		# ========== add a sneaky column killer here ==========
		# Changed from 0.01 to speed up first run
		X_train = X_train[cols_keep]
		X_train.drop(X_train.columns[(
			np.sum(~np.logical_or(X_train.values == 0, 
				np.isnan(X_train.values)), axis=0) /  X_train.values.shape[0]) < 0.01],axis=1,inplace=True)
		# ========== Leave the NaNs in up to a specific value ==========
		X_train = X_train.loc[X_train.isnull().mean(axis=1) <= setup["DropNAN"],]
		X_train.drop(X_train.columns[X_train.std() == 0], axis=1, inplace=True)
		


		X_test  =  X_test.loc[X_test[X_train.columns].isnull().mean(axis=1) <= setup["DropNAN"], X_train.columns]
		
		y_train = pd.DataFrame(y_train[predvar][X_train.index])
		y_test  = pd.DataFrame(y_test[predvar][X_test.index])
		
		if not _sizecheck(X_train, X_test, y_train, y_test):
			warn.warn("Size Check failed here 3")
			breakpoint()
	else:
		# ========== Remove any nans ==========

		X_train = X_train[cols_keep].dropna()
		X_train.drop(X_train.columns[X_train.std() == 0], axis=1, inplace=True)
		X_test  =  X_test[X_train.columns].dropna()

		y_train = pd.DataFrame(y_train[predvar][X_train.index])
		y_test  = pd.DataFrame(y_test[predvar][X_test.index])
	
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
			y_testCal = pd.DataFrame(y_testCal[predvar][X_testCal.index])
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

			y_train = pd.DataFrame(y_train[predvar][X_train.index])
			y_test  = pd.DataFrame(y_test[predvar][X_test.index])



	# =========== Pull out the data used for prediction ===========
	X        = pd.concat([X_test, X_train])
	cols_out = X.columns.values
	
	# ========== Build the spearmans rank correlation clusters ==========
	# +++++ Build a spearman table +++++
	try:
		if 'DropNAN' in setup.keys() and not setup["DropNAN"] == 0:
			Xa = X.copy().dropna()
			if Xa.shape[0]/X.shape[0] < 0.6:
				warn.warn("too small fraction")
				breakpoint()
			corr = spearmanr(Xa).correlation	
		else:
			corr = spearmanr(X).correlation
		
		if bn.anynan(corr):
			warn.warn("Nan values slipped through the correlation, going interactive")
			breakpoint()
		# breakpoint()
	except Exception as er:
		print(str(er))
		breakpoint()
	statsOD = _simplestatus(vi_df, X, df_site)

	# ========== See if the dataset needs to be transfored ==========
	if not setup["Transformer"] is None:
		print(f"Applying X Transfomer at {pd.Timestamp.now()}")
		# ========== covert to sparse ==========
		X_train = X_train.astype(pd.SparseDtype(float, fill_value=np.NaN))
		X_test  = X_test.astype(pd.SparseDtype(float, fill_value=np.NaN))
		X_train = pd.DataFrame.sparse.from_spmatrix(setup["Transformer"].fit_transform(X_train), index=X_train.index, columns=X_train.columns).astype(pd.SparseDtype(float, fill_value=np.NaN)).sparse.to_dense()
		X_test  = pd.DataFrame.sparse.from_spmatrix(setup["Transformer"].transform(X_test), index=X_test.index, columns=X_test.columns).astype(pd.SparseDtype(float, fill_value=np.NaN)).sparse.to_dense()
		
	if not setup["yTransformer"] is None:
		print(f"Applying Y Transfomer at {pd.Timestamp.now()}")
		y_train = pd.DataFrame(setup["yTransformer"].fit_transform(y_train), index=y_train.index, columns=y_train.columns )
		y_test  = pd.DataFrame(setup["yTransformer"].transform(y_test), index=y_test.index, columns=y_test.columns )
		# breakpoint()



	if not _sizecheck(X_train, X_test, y_train, y_test):
		warn.warn("Size Check failed at the end")
		breakpoint()
	# ========== Split the data and return ==========
	if setup["debug"]:
		# Container for the debugging stuff
		dbg = {"X_test":X_test, "y_test":y_test}
	else:
		dbg = None
	
	if final:

		print("Returing the true test train set. Loading and processing the files took:",  pd.Timestamp.now()-t0)
		if setup['Nstage']!=1 and not RStage:
			return X_train, X_test, y_train, y_test, cols_out, statsOD, corr, df_site, dbg, classified
		else:
			return X_train, X_test, y_train, y_test, cols_out, statsOD, corr, df_site, dbg
	else:
		# ========== Non final Need to split the training set ==========
		# /// This is done so that my test set is pristine when looking at final models \\\
		if not setup["FullTestSize"] is None:
			fnl   = f"{folder}{basestr}_TTSlookup.csv"
			dfl   = pd.read_csv(fnl, index_col=0)
			ind   = y_train.index.values
			# ========== Check to make sure i'm pulling out the right values ==========
			assert dfl.iloc[ind, version].max() == 1
			X_trainRec = X_train[dfl.iloc[ind, version] == 0]
			X_testRec  = X_train[dfl.iloc[ind, version] == 1]
			y_trainRec = y_train[dfl.iloc[ind, version] == 0]
			y_testRec  = y_train[dfl.iloc[ind, version] == 1]
			
			# dfl.loc[ind, version]
			# breakpoint()
			# this will also need to call into the random state
			# X_trainRec, X_testRec, y_trainRec, y_testRec = altsplit(
			# 	setup, df_site.loc[X_train.index.values],   vi_df.loc[X_train.index.values], test_size, predvar, dropvar, version, basestr, folder, 
			# 	batchsplit=False, n_splits=1, random_state=version, vers = None, columns=cols_out)
			# breakpoint()
		else:
			# In versions pre 410, there was a mistake here where random state was branch not versions
			X_trainRec, X_testRec, y_trainRec, y_testRec = train_test_split(
				X_train, y_train, test_size=test_size, random_state=version)

		if not _sizecheck(X_trainRec, X_testRec, y_trainRec, y_testRec):
			warn.warn("Size Check failed at the end 2")
			breakpoint()
		# ============ Return the filtered data  ============
		print("Returing the twice split test train set. Loading and processing the files took:",  pd.Timestamp.now()-t0)
		if setup['Nstage']!=1 and not RStage:
			return X_trainRec, X_testRec, y_trainRec, y_testRec, cols_out, statsOD, corr, df_site, dbg, classified
		else:
			return X_trainRec, X_testRec, y_trainRec, y_testRec, cols_out, statsOD, corr, df_site, dbg



