"""
Script goal, 

Look at how the performance changes as a function of prediction interval


elevent existing R code
	./code/analysis/modeling/build_model/rf_class......
"""

# ==============================================================================

__title__ = "Future Predictability"
__author__ = "Arden Burrell"
__version__ = "v1.0(07.12.2020)"
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
from numba import jit


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
	"""Goal of the script is to open the results files from different models 
	and produce some figures """
	# ========== Get the file names and open the files ==========
	path  = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"
	cf.pymkdir(path+"plots/")

	# +++++ the model Infomation +++++
	setup_fnames = glob.glob(path + "*/Exp*_setup.csv")
	df_setup     = pd.concat([pd.read_csv(sfn, index_col=0).T for sfn in setup_fnames])
	
	# +++++ the final model results +++++
	mres_fnames = glob.glob(path + "*/Exp*_Results.csv")
	df_mres = pd.concat([fix_results(mrfn) for mrfn in mres_fnames])
	df_mres["TotalTime"]  = df_mres.TotalTime / pd.to_timedelta(1, unit='m')
	df_mres, keys = Experiment_name(df_mres, df_setup, var = "experiment")
	df_mres["experiment"] = df_mres["experiment"].astype('category')


	# ========== Setup the experiment ==========
	expn = 310 #temporal varibility exp
	# gclass     = glob.glob(f"{path}{expn}/Exp{expn}_*_OBSvsPREDICTEDClas_y_test.csv")
	# df_clest   = pd.concat([load_OBS(mrfn) for mrfn in gclass])



	# ========= Load in the observations
	OvP_fnames = glob.glob(f"{path}{expn}/Exp{expn}_*_OBSvsPREDICTED.csv")
	df_OvsP    = pd.concat([load_OBS(ofn) for ofn in OvP_fnames])

	vi_df = pd.read_csv("./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/VI_df_AllSampleyears.csv", index_col=0)[['lagged_biomass','ObsGap']]
	
	# ========== pull out the obs gap ==========
	df_OvsP["ObsGap"]   = vi_df.loc[df_OvsP.index]["ObsGap"]
	df_OvsP["ObsGap"]   = df_OvsP["ObsGap"].astype("category")
	df_OvsP["residual"] = df_OvsP["Estimated"] = df_OvsP["Observed"]
	df_OvsP["AbsResidual"] = df_OvsP["residual"].abs()

	# ========== make the plot ==========
	for va in ["AbsResidual", "residual"]:
		plt.figure(va)
		df_ci = df_OvsP.groupby("ObsGap")[va].quantile([0.05, 0.95]).reset_index()
		ax = sns.lineplot(y=va, x="ObsGap", data=df_OvsP, ci=None)
		ax.fill_between(
			df_ci[df_ci.level_1 == 0.05]["ObsGap"].values, 
			df_ci[df_ci.level_1 == 0.95][va].values, 
			df_ci[df_ci.level_1 == 0.05][va].values, alpha=0.10)
	plt.show()
	breakpoint()
# ==============================================================================


# ==============================================================================
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
				lswin = df_setup[df_setup.Code.astype(int) == cat]["window"][0]
				if cat < 320:
					if np.isnan(float(pred)):
						nm = f"DataMOD_AllSampleyrPred_{lswin}yrLS"
					else:
						nm = f"DataMOD_{pred}yrPred_{lswin}yrLS"
				else:
					NAfrac = int(float(df_setup[df_setup.Code.astype(int) == cat]["DropNAN"][0]) *100)
					nm = f"DataMOD_{pred}yrPred_{lswin}yrLS_{NAfrac}percNA"

			else:
				nm = "%d.%s" % (cat, df_setup[df_setup.Code.astype(int) == int(cat)].name.values[0])
		except Exception as er:
			print(str(er))
			breakpoint()
		keys[cat] = nm
		df[var].replace({cat:nm}, inplace=True)
	return df, keys

def load_OBS(ofn):
	df_in = pd.read_csv(ofn, index_col=0)
	breakpoint()
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
		# elif col == "experiment":
		# 	df_in[col] = df_in[col].astype('category')
		else:
			df_in[col] = df_in[col].astype(float)
	# ========== Loop over the regions ==========

	for region in site_df.Region.unique():
		try:
			if df_in["%s_sitefrac" % region].values[0] >1:
				warn.warn("value issue here")
				nreakpoint()
		except KeyError:
			# Places with missing regions
			df_in["%s_siteinc"  % region] = 0.
			df_in["%s_sitefrac" % region] = 0.
		# 	# df_in["%s_siteinc" % region] = df_in["%s_siteinc" % region].astype(float)
		# 	df_in["%s_sitefrac" % region] = (df_in["%s_siteinc" % region] / Rcounts[region])
	return df_in


if __name__ == '__main__':
	main()

