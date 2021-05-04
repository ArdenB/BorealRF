"""
Boreal EWS PSP data anlysis 
 
Script to  make individaul psps plots  
"""

# ==============================================================================

__title__ = "Compare Biomass transform"
__author__ = "Arden Burrell"
__version__ = "v1.0(15.04.2021)"
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
import ipdb
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
import seaborn as sns
import palettable
# from numba import jit
import matplotlib.colors as mpc
from tqdm import tqdm
import pickle


# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf
import myfunctions.benchmarkfunctions as bf

# ========== Import packages for parellelisation ==========
# import multiprocessing as mp

# ========== Import ml packages ==========
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.inspection import permutation_importance
from sklearn import metrics as sklMet
# from sklearn.utils import shuffle
# from scipy.stats import spearmanr
# from scipy.cluster import hierarchy

# ==============================================================================
def main():
	# ========== Get the file names and open the files ==========
	formats = None
	path  = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"
	cf.pymkdir(path+"plots/")
	ppath = "./pyEWS/analysis/3.Izanami/Figures/PS03/"
	cf.pymkdir(ppath)
	# +++++ the model Infomation +++++
	setup_fnames = glob.glob(path + "*/Exp*_setup.csv")
	df_setup     = pd.concat([pd.read_csv(sfn, index_col=0).T for sfn in setup_fnames], sort=True)
	
	# +++++ the final model results +++++
	mres_fnames = glob.glob(path + "*/Exp*_Results.csv")
	df_mres = pd.concat([fix_results(mrfn) for mrfn in mres_fnames], sort=False)
	df_mres["TotalTime"]  = df_mres.TotalTime / pd.to_timedelta(1, unit='m')
	df_mres, keys = Experiment_name(df_mres, df_setup, var = "exp")


	# ========= Load in the observations ==========
	OvP_fnames = glob.glob(path + "*/Exp*_OBSvsPREDICTED.csv")
	df_OvsP    = pd.concat([load_OBS(ofn) for ofn in OvP_fnames], sort=True)
	
	gclass     = glob.glob(path + "*/Exp*_OBSvsPREDICTEDClas_y_test.csv")
	df_clest   = pd.concat([load_OBS(mrfn) for mrfn in gclass], sort=True)

	branch     = glob.glob(path + "*/Exp*_BranchItteration.csv")
	df_branch  = pd.concat([load_OBS(mrfn) for mrfn in branch], sort=True)

	experiments = [400, 401, 402, 403, 404] 
	# ========== get the scores ==========
	df = Translator(df_setup, df_mres, keys, df_OvsP, df_clest, df_branch, experiments, path)
	transmet(df, experiments, df_mres)
	breakpoint()

# ==============================================================================

def transmet(df, experiments, df_mres):
	"""Function to look at ther overall performance of the different approaches"""

		# ========== Create the figure ==========
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':8})
	font = {'family' : 'normal',
	        'weight' : 'bold', #,
	        'size'   : 8}
	mpl.rc('font', **font)
	sns.set_style("whitegrid")

	# ========== pull out matched runs so i can compare across runs ==========
	dfM = df.dropna()
	metrics = OrderedDict()
	for exp in experiments:
		print( f"{exp} max version: {dfM.version.max()}")
		for ver in dfM.version.unique():
			dfOG = df_mres.loc[np.logical_and(df_mres.experiment == exp, df_mres.version == ver)]
			dfMe = dfM.loc[np.logical_and(dfM.experiment == exp, dfM.version == ver)].copy()
			bad  = (dfMe.Estimated < 0 ).sum() + (dfMe.Estimated > (df.Observed.max())).sum()
			Insane  = (dfMe.Estimated < -100 ).sum() + (dfMe.Estimated > (500+df.Observed.max())).sum()

			dfMe.loc[(dfMe.Estimated < 0), "Estimated"] = np.NaN
			dfMe.loc[(dfMe.Estimated > 2*df.Observed.max()), "Estimated"] = np.NaN
			# dfMe.loc[(dfMe.Observed < 0), "Observed"]   = 0
			dfMe.dropna(inplace=True)

			mets = OrderedDict({
				"experiment":exp,
				"Version":ver,
				"BadValueCount":bad, 
				"InsaneValueCount":Insane, 
				"R2":sklMet.r2_score(dfMe.Observed.values, dfMe.Estimated.values), 
				"ExplainedVarianceScore":sklMet.explained_variance_score(dfMe.Observed.values, dfMe.Estimated.values),
				"MAE":sklMet.mean_absolute_error(dfMe.Observed.values, dfMe.Estimated.values),
				"MedianAE":sklMet.median_absolute_error(dfMe.Observed.values, dfMe.Estimated.values),
				"rawR2":dfOG["R2"].values[0],
				"TotalTime":dfOG["TotalTime"].values[0],#pd.to_timedelta(, unit='s'), 
				"Computer":dfOG["Computer"].values[0],
				})
				# "MAPE":sklMet.mean_absolute_percentage_error(dfMe.Observed.values, dfMe.Estimated.values),
				# "MeanGammaDeviance":sklMet.mean_gamma_deviance(dfMe.Observed.values+0.0000001, dfMe.Estimated.values+0.0000001),

			metrics[len(metrics)] = mets
			print(mets)


	dfS = pd.DataFrame(metrics).T.infer_objects()
	dfS["experiment"] = dfS["experiment"].astype('category')
	sns.barplot(y="TotalTime", x="experiment", hue="Computer", data=dfS)
	sns.barplot(y="R2", x="experiment",  data=dfS)#hue="Computer",
	plt.show()
	sns.barplot(y="R2", x="experiment",  data=dfS)
	plt.show()
	# breakpoint()
	# dfS["TotalTime"] = pd.to_timedelta(dfS["TotalTime"], unit='s')
	dfScores = dfM.groupby(["experiment","Rank"]).nunique()["index"].reset_index()
	# breakpoint()

	fig = sns.relplot(data=dfM, x="Observed", y="Estimated", hue="experiment", col="experiment", col_wrap=3)
	fig.set(ylim=(0, 2000))
	fig.set(xlim=(0, 2000))
	plt.show()

	fig = sns.relplot(data=dfM, x="ObsDelta", y="EstDelta", hue="experiment", col="experiment", col_wrap=3)
	fig.set(ylim=(-500, 500))
	fig.set(xlim=(-500, 500))
	plt.show()

	sns.barplot(y="index", x="Rank", hue="experiment", data=dfScores)
	plt.show()

	ipdb.set_trace()
	breakpoint()
	# dfS.groupby(["experiment","Computer"]).mean()


def Translator(df_setup, df_mres, keys, df_OvsP, df_clest, df_branch, experiments, path):
	""" Function to transfrom different methods of calculating biomass 
	into a comperable number """
	bioMls = []
	vi_fn = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/VI_df_AllSampleyears_ObsBiomass.csv"
	vi_df  = pd.read_csv( vi_fn, index_col=0).loc[:, ['biomass', 'Obs_biomass', 'Delta_biomass','ObsGap']]

	# ========== Loop over each experiment ==========
	for exp in tqdm(experiments):
		pvar =  df_setup.loc[f"Exp{exp}", "predvar"]
		if type(pvar) == float:
			# deal with the places i've alread done
			pvar = "lagged_biomass"

		# +++++ pull out the observed and predicted +++++
		df_OP  = df_OvsP.loc[df_OvsP.experiment == exp]
		df_act = vi_df.iloc[df_OP.index]
		dfC    = df_OP.copy()
		
		if pvar == "lagged_biomass":
			vfunc            = np.vectorize(_delag)
			dfC["Estimated"] = vfunc(df_act["biomass"].values, df_OP["Estimated"].values)
			
		elif pvar == 'Obs_biomass':
			if type(df_setup.loc[f"Exp{exp}", "yTransformer"]) == float:
				pass
			else:
				for ver in dfC.version.unique().astype(int):
					ppath = f"./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/{exp}/" 
					fn_mod = f"{ppath}models/XGBoost_model_exp{exp}_version{ver}"

					setup = pickle.load(open(f"{fn_mod}_setuptransfromers.dat", "rb"))
					dfC.loc[dfC.version == ver, "Estimated"] = setup['yTransformer'].inverse_transform(dfC.loc[dfC.version == ver, "Estimated"].values.reshape(-1, 1))

				# breakpoint()
		elif pvar == 'Delta_biomass':
			dfC["Estimated"] += df_act["biomass"]
		else:
			breakpoint()
		dfC["Observed"] = df_act["Obs_biomass"].values
		dfC["Residual"] = dfC["Estimated"] - dfC["Observed"]
		dfC["Original"] = df_act["biomass"].values
		dfC["ObsDelta"] = df_act["Delta_biomass"].values
		dfC["EstDelta"] = dfC["Estimated"] - dfC["Original"]
		dfC["ObsGap"]   = df_act["ObsGap"].values
		bioMls.append(dfC)

	# ========== Convert to a dataframe ==========
	df = pd.concat(bioMls).reset_index().sort_values(["version", "index"]).reset_index(drop=True)

	# ========== Perform grouped opperations ==========
	df["Rank"]     = df.abs().groupby(["version", "index"])["Residual"].rank(na_option="bottom").apply(np.floor)
	df["RunCount"] = df.abs().groupby(["version", "index"])["Residual"].transform("count")
	df.loc[df["RunCount"] < len(experiments), "Rank"] = np.NaN
	
	return df


# ==============================================================================

def _delag(b0, bl):
	if bl == 1.:
		return np.nan
	else:
		return ((bl*b0)+b0)/(1-bl)


def Experiment_name(df, df_setup, var = "experiment"):
	keys = {}
	for cat in df["experiment"].unique():
		# =========== Setup the names ============
		try:
			if cat == 100:
				nm = "10yr LS"
			elif cat == 101:
				nm = "No LS"
			elif cat == 102:
				nm = "5yr LS"
			elif cat == 103:
				nm = "15yr LS"
			elif cat == 104:
				nm = "20yr LS"
			elif cat == 120:
				nm = "10yr XGBOOST"
			elif cat == 200:
				nm = "RF2 7 Quantile splits"
			elif cat == 201:
				nm = "RF2 3 Quantile splits"
			elif cat == 202:
				nm = "RF2 2 Interval splits"
			elif cat == 204:
				nm = "RF2 4 Interval splits"
			# elif cat == 300:
			# 	breakpoint()
			elif cat // 100 == 3.:
				pred  = df_setup[df_setup.Code.astype(int) == cat]["predictwindow"][0]
				if pred is None:
					breakpoint()
				lswin = df_setup[df_setup.Code.astype(int) == cat]["window"][0]
				if cat < 320:
					nm = f"DataMOD_{pred}yrPred_{lswin}yrLS"
				else:
					NAfrac = int(float(df_setup[df_setup.Code.astype(int) == cat]["DropNAN"][0]) *100)
					nm = f"DataMOD_{pred if not np.isnan(float(pred)) else 'AllSample' }yrPred_{lswin}yrLS_{NAfrac}percNA"
					if cat == 332:
						nm += "_disturbance"
					elif cat == 333:
						nm += "_FeatureSel"
			elif cat == 400:
				nm = "Final XGBoost Model"
			elif cat == 401:
				nm = "Final XGBoost Model Observed Biomass"
			elif cat == 402:
				nm = "Final XGBoost Model Delta Biomass"
			elif cat == 404:
				nm = "Final XGBoost Model Obs Biomass Quantile Transform"
			else:
				nm = "%d.%s" % (cat, df_setup[df_setup.Code.astype(int) == int(cat)].name.values[0])
		except Exception as er:
			print(str(er))
			breakpoint()
		keys[cat] = nm
		if not  var == "experiment":
			df[var] = df["experiment"].copy()
		df[var].replace({cat:nm}, inplace=True)
	return df, keys

def load_OBS(ofn):
	df_in = pd.read_csv(ofn, index_col=0)
	df_in["experiment"] = int(ofn.split("/")[-2])
	df_in["experiment"] = df_in["experiment"].astype("category")
	df_in["version"]    = float(ofn.split("_vers")[-1][:2])
	return df_in

def fix_results(fn):
	# ========== Fill in the missing sites ==========
	region_fn ="./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/TTS_sites_and_regions.csv"
	site_df = pd.read_csv(region_fn, index_col=0)

	Rcounts = site_df.groupby(["Region"]).count()["site"]
	# ========== convert the types
	df_in   = pd.read_csv(fn, index_col=0).T#, parse_dates=["TotalTime"])
	for col in df_in.columns:
		if col == "TotalTime":
			df_in[col] = pd.to_timedelta(df_in[col])
		elif col in ["Computer"]:
			df_in[col] = df_in[col].astype('category')
		else:
			try:
				df_in[col] = df_in[col].astype(float)
			except:
				breakpoint()
	# ========== Loop over the regions ==========

	for region in site_df.Region.unique():
		try:
			if df_in["%s_sitefrac" % region].values[0] >1:
				warn.warn("value issue here")
				breakpoint()
		except KeyError:
			# Places with missing regions
			df_in["%s_siteinc"  % region] = 0.
			df_in["%s_sitefrac" % region] = 0.
		# 	# df_in["%s_siteinc" % region] = df_in["%s_siteinc" % region].astype(float)
		# 	df_in["%s_sitefrac" % region] = (df_in["%s_siteinc" % region] / Rcounts[region])
	return df_in

def VIload():
	print(f"Loading the VI_df, this can be a bit slow: {pd.Timestamp.now()}")
	vi_df = pd.read_csv("./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/VI_df_AllSampleyears.csv", index_col=0)#[['lagged_biomass','ObsGap']]
	vi_df["NanFrac"] = vi_df.isnull().mean(axis=1)

	# ========== Fill in the missing sites ==========
	region_fn ="./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/SiteInfo_AllSampleyears.csv"
	site_df = pd.read_csv(region_fn, index_col=0)
	# breakpoint()

	vi_df = vi_df[['year', 'biomass', 'lagged_biomass','ObsGap', "NanFrac"]]
	for nanp in [0, 0.25, 0.50, 0.75, 1.0]:	
		isin = (vi_df["NanFrac"] <=nanp).astype(float)
		isin[isin == 0] = np.NaN
		vi_df[f"{int(nanp*100)}NAN"]  = isin

	fcount = pd.melt(vi_df.drop(["lagged_biomass","NanFrac"], axis=1).groupby("ObsGap").count(), ignore_index=False).reset_index()
	fcount["variable"] = fcount["variable"].astype("category")
	vi_df["Region"]    = site_df["Region"]
	# breakpoint()

	return vi_df, fcount
# ==============================================================================

if __name__ == '__main__':
	main()