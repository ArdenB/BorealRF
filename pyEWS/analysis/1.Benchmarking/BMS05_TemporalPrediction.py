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
import dask.dataframe as dd
from dask.diagnostics import ProgressBar


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
	# for expn in df_mres.experiment.unique():
	# 	def _tcheck(ser):
	# 		if ser.max() > 1.5 * ser.median():
	# 			breakpoint()
	# 		else:
	# 			return ser
	# 	df_mres.loc[df_mres.experiment == expn, 'TotalTime'] = _tcheck(df_mres.loc[df_mres.experiment == expn, 'TotalTime'])
	# breakpoint()
	df_mres["TotalTime"]  = df_mres.TotalTime / pd.to_timedelta(1, unit='m')
	df_mres, keys = Experiment_name(df_mres, df_setup, var = "experiment")
	df_mres["experiment"] = df_mres["experiment"].astype('category')
	formats = [".png"]
	vi_df, fcount = VIload()
	

	# ========== Nan Tollerance =========== 
	# NanExp = [300, 320, 321, 322, 323]
	# NanTol_performance(path, NanExp, df_setup, df_mres.copy(), formats, vi_df, keys)
	
	# ========== Setup the experiment for temporal ==========
	experiments = [310, 330, 331]
	MLmethod_performance(path, experiments, df_setup, df_mres, formats, vi_df, keys, ncol = 4)
	Temporal_predictability(path, experiments, df_setup, df_mres, formats, vi_df)
	
	# ========== Experiments for ML methods ==========
	experiments = [100, 110, 200, 120, 304]
	MLmethod_performance(path, experiments, df_setup, df_mres, formats, vi_df, keys, ncol = 4)
	# for expn in experiments:
	# 	 #temporal varibility exp
	# 	# gclass     = glob.glob(f"{path}{expn}/Exp{expn}_*_OBSvsPREDICTEDClas_y_test.csv")
	# 	# df_clest   = pd.concat([load_OBS(mrfn) for mrfn in gclass])

	# 	# ========= Load in the observations
	# 	OvP_fnames = glob.glob(f"{path}{expn}/Exp{expn}_*_OBSvsPREDICTED.csv")

	# 	breakpoint()
# ==============================================================================
def NanTol_performance(path, experiments, df_setup, df_mres, formats, vi_df, keys,  ncol = 4):
	"""
	function to make map of overall ML method performance 
	"""
	# ========== make a container for the data and get the file names ==========
	OvP_fnames = []
	for expn in experiments:	
		OvP_fnames += glob.glob(f"{path}{expn}/Exp{expn}_*_OBSvsPREDICTED.csv")
	
	# ========== load the results and the VI data ==========
	df_OvsP    = pd.concat([load_OBS(ofn) for ofn in OvP_fnames])

	#                   ========== pull out the Nan Fraction ==========
	df_OvsP["NanFrac"]  = vi_df.loc[df_OvsP.index]["NanFrac"]
	df_OvsP["residual"] = df_OvsP["Estimated"] = df_OvsP["Observed"]
	df_OvsP["NanPer"]   = pd.cut(df_OvsP['NanFrac'], np.arange(0.00, 1.01, 0.05), right=False).cat.codes.astype(float) * 5

	# ========== Create the figure ==========
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':8})
	font = {'family' : 'normal',
	        'weight' : 'bold', #,
	        'size'   : 8}
	mpl.rc('font', **font)
	sns.set_style("whitegrid")

	# ========== Subset the data if i need an experiment name ==========
	exp_names = [f"{nanp}% Missing Data " for nanp in [0, 25, 50, 75, 100]]
	old_names = [keys[expn] for expn in experiments]
	df_mres.replace(old_names, exp_names, inplace=True)


	df_set    = df_mres[df_mres.experiment.isin(exp_names)].copy()
	# df_set["experiment"].cat.remove_unused_categories(inplace=True)
	colours = palettable.cartocolors.qualitative.Vivid_10.hex_colors
	# breakpoint()

	# ========== Plot Broad summary  ==========
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
		2, 2, figsize=(15,10),num=("General Summary"), dpi=130)
	# +++++ R2 plot +++++
	sns.barplot(y="R2", x="experiment", data=df_set, ci="sd", ax=ax1, order=exp_names, palette=colours[:len(experiments)])
	ax1.set_xlabel("")
	ax1.set_ylabel(r"$R^{2}$")
	ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
	ax1.set(ylim=(0., 1.))
	# +++++ time taken plot +++++
	sns.barplot(y="TotalTime", x="experiment", data=df_set, ci="sd", ax=ax2, order=exp_names, palette=colours[:len(experiments)])
	ax2.set_xlabel("")
	ax2.set_ylabel(r"$\Delta$t (min)")
	ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')
	# +++++ site fraction plot +++++
	sns.barplot(y="fractrows", x="experiment", data=df_set, ci="sd", ax=ax3, order=exp_names, palette=colours[:len(experiments)])
	ax3.set_xlabel("")
	ax3.set_ylabel("% of sites")
	ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, horizontalalignment='right')
	ax3.set(ylim=(0., 1.))
	
	# +++++ site fraction plot +++++
	sns.histplot(data=vi_df, x="NanFrac", bins=10, binrange=[0,1], ax=ax4, palette=colours[0], cumulative=True)
	# sns.barplot(y="itterrows", x="experiment", data=df_set, ci="sd", ax=ax4, order=exp_names, palette=colours[:len(experiments)])
	ax4.set_xlabel(f"% of missing data")
	ax4.set_ylabel("No. of sites")
	# ax4.set_xticklabels(ax3.get_xticklabels(), rotation=30, horizontalalignment='right')
	# ax4.set(ylim=(0., np.ceil(df_set.itterrows.max()/1000)*1000))
	
	plt.tight_layout()
	# ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{path}plots/BM05_Nan_performance"
	for ext in formats:
		plt.savefig(fnout+ext)
	
	plotinfo = "PLOT INFO: General Multimodel Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()
	# # ========== Nan Fraction plot ========
	# ax = sns.lineplot(y="residual", x="NanPer", data=df_OvsP[df_OvsP.experiment == 323], 
	# 	ci=None, palette=colours[:len(experiments)], legend=False)
	# df_ci = df_OvsP[df_OvsP.experiment == 323].groupby("NanPer")["residual"].quantile([0.05, 0.95]).reset_index()
	# ax.fill_between(
	# 	df_ci[df_ci.level_1 == 0.05]["NanPer"].values, 
	# 	df_ci[df_ci.level_1 == 0.95]["residual"].values, 
	# 	df_ci[df_ci.level_1 == 0.05]["residual"].values, alpha=0.10)#, color=hue)
		# ax=ax, 
		# hue="experiment", 
	breakpoint()

def MLmethod_performance(path, experiments, df_setup, df_mres, formats, vi_df, keys,  ncol = 4):
	"""
	function to make map of overall ML method performance 
	"""
	# def Main_plots(path, df_mres, df_setup, df_OvsP, keys, experiments=None,  , sumtxt=""):

	font = ({
		'family' : 'normal',
		'weight' : 'bold', 
		'size'   : 8})
	axes = ({
		'titlesize':8,
		'labelweight':'bold'
		})


	sns.set_style("whitegrid")
	mpl.rc('font', **font)
	mpl.rc('axes', **axes)
	colours = palettable.cartocolors.qualitative.Vivid_10.hex_colors

	# ========== Subset the data if i need an experiment name ==========
	exp_names = [keys[expn] for expn in experiments]
	df_set    = df_mres[df_mres.experiment.isin(exp_names)].copy()
	df_set["experiment"].cat.remove_unused_categories(inplace=True)
	# breakpoint()

	# ========== Plot Broad summary  ==========
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
		2, 2, figsize=(15,10),num=("General Summary"), dpi=130)
	# +++++ R2 plot +++++
	sns.barplot(y="R2", x="experiment", data=df_set, ci="sd", ax=ax1, order=exp_names, palette=colours[:len(experiments)])
	ax1.set_xlabel("")
	ax1.set_ylabel(r"$R^{2}$")
	ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
	ax1.set(ylim=(0., 1.))
	# +++++ time taken plot +++++
	sns.barplot(y="TotalTime", x="experiment", data=df_set, ci="sd", ax=ax2, order=exp_names, palette=colours[:len(experiments)])
	ax2.set_xlabel("")
	ax2.set_ylabel(r"$\Delta$t (min)")
	ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')
	# +++++ site fraction plot +++++
	sns.barplot(y="fractrows", x="experiment", data=df_set, ci="sd", ax=ax3, order=exp_names, palette=colours[:len(experiments)])
	ax3.set_xlabel("")
	ax3.set_ylabel("% of sites")
	ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, horizontalalignment='right')
	ax3.set(ylim=(0., 1.))
	
	# +++++ site fraction plot +++++
	sns.barplot(y="itterrows", x="experiment", data=df_set, ci="sd", ax=ax4, order=exp_names, palette=colours[:len(experiments)])
	ax4.set_xlabel("")
	ax4.set_ylabel("No. of sites")
	ax4.set_xticklabels(ax3.get_xticklabels(), rotation=30, horizontalalignment='right')
	ax4.set(ylim=(0., np.ceil(df_set.itterrows.max()/1000)*1000))
	plt.tight_layout()
	# ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{path}plots/BM05_MLmethod_performance"
	for ext in formats:
		plt.savefig(fnout+ext)
	
	plotinfo = "PLOT INFO: General Multimodel Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()

	# .tick_params(axis='x', labelrotation= )
	breakpoint()


def Temporal_predictability(path, experiments, df_setup, df_mres, formats, vi_df):
	"""
	Function to make a figure that explores the temporal predictability. This 
	figure will only use the runs with virable windows
	"""
	# ========== make a container for the data and get the file names ==========
	OvP_fnames = []
	for expn in experiments:	
		OvP_fnames += glob.glob(f"{path}{expn}/Exp{expn}_*_OBSvsPREDICTED.csv")
	
	# ========== load the results and the VI data ==========
	df_OvsP    = pd.concat([load_OBS(ofn) for ofn in OvP_fnames])


	# ========== pull out the obs gap ==========
	df_OvsP["ObsGap"]   = vi_df.loc[df_OvsP.index]["ObsGap"]
	df_OvsP["ObsGap"]   = df_OvsP["ObsGap"].astype("category")
	df_OvsP["residual"] = df_OvsP["Estimated"] = df_OvsP["Observed"]
	df_OvsP["AbsResidual"] = df_OvsP["residual"].abs()

	# ========== Create the figure ==========
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':8})
	font = {'family' : 'normal',
	        'weight' : 'bold', #,
	        'size'   : 8}
	mpl.rc('font', **font)
	sns.set_style("whitegrid")

	# ========== make the plot ==========
	for va in ["AbsResidual", "residual"]:
		for CI in ["SD", "QuantileInterval"]:
			print(f"{va} {CI} {pd.Timestamp.now()}")
			# Create the labels
			lab = [df_setup.loc[df_setup.Code.astype(int) == expn, "name"].values[0] for expn in experiments]
			# ========== set up the colours and build the figure ==========
			colours = palettable.cartocolors.qualitative.Vivid_10.hex_colors
			fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14,13))
			# ========== Build the first part of the figure ==========
			if CI == "SD":
				sns.lineplot(y=va, x="ObsGap", data=df_OvsP, 
					hue="experiment", ci="sd", ax=ax, 
					palette=colours[:len(experiments)], legend=False)
			else:
				# Use 
				sns.lineplot(y=va, x="ObsGap", data=df_OvsP, 
					hue="experiment", ci=None, ax=ax, 
					palette=colours[:len(experiments)], legend=False)
				for expn, hue in zip(experiments, colours[:len(experiments)]) :
					df_ci = df_OvsP[df_OvsP.experiment == expn].groupby("ObsGap")[va].quantile([0.05, 0.95]).reset_index()
					ax.fill_between(
						df_ci[df_ci.level_1 == 0.05]["ObsGap"].values, 
						df_ci[df_ci.level_1 == 0.95][va].values, 
						df_ci[df_ci.level_1 == 0.05][va].values, alpha=0.10, color=hue)
			# ========== fix the labels ==========
			ax.set_xlabel('Years Between Observation', fontsize=8, fontweight='bold')
			ax.set_ylabel(r'Mean Residual ($\pm$ %s)' % CI, fontsize=8, fontweight='bold')
			# ========== Create hhe legend ==========
			ax.legend(title='Experiment', loc='upper right', labels=lab)
			ax.set_title(f"a) ", loc= 'left')


			# ========== The second subplot ==========
			sns.histplot(data=df_OvsP.astype(float), x="ObsGap", hue="experiment",  
				multiple="dodge", palette=colours[:len(experiments)], ax=ax2)
			# ========== fix the labels ==========
			ax2.set_xlabel('Years Between Observation', fontsize=8, fontweight='bold')
			ax2.set_ylabel(f'# of Obs.', fontsize=8, fontweight='bold')
			# ========== Create hhe legend ==========
			ax2.legend(title='Experiment', loc='upper right', labels=lab)
			ax2.set_title(f"b) ", loc= 'left')
			plt.tight_layout()
			# +++++ Save the plot out +++++
			if not formats is None:
				for fmt in formats:
					fn = f"{path}plots/BMS05_TemporalPrediction_{va}_{CI}{fmt}"
					plt.savefig(fn)
			# 
			plt.show()

		plotinfo = "PLOT INFO: Temporal Predicability plots made using %s:v.%s by %s, %s" % (
			__title__, __version__,  __author__, pd.Timestamp.now())
		gitinfo = cf.gitmetadata()
		cf.writemetadata(f"{path}plots/BMS05_TemporalPrediction_{va}", [plotinfo, gitinfo])
	ipdb.set_trace()
	breakpoint()

# ==============================================================================



def VIload():
	print(f"Loading the VI_df, this can be a bit slow: {pd.Timestamp.now()}")
	vi_df = pd.read_csv("./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/VI_df_AllSampleyears.csv", index_col=0)#[['lagged_biomass','ObsGap']]
	vi_df["NanFrac"] = vi_df.isnull().mean(axis=1)
	vi_df = vi_df[['lagged_biomass','ObsGap', "NanFrac"]]
	for nanp in [0, 0.25, 0.50, 0.75, 1.0]:	
		isin = (vi_df["NanFrac"] <=nanp).astype(float)
		isin[isin == 0] = np.NaN
		vi_df[f"{int(nanp*100)}NAN"]  = isin

	fcount = pd.melt(vi_df.drop(["lagged_biomass","NanFrac"], axis=1).groupby("ObsGap").count(), ignore_index=False).reset_index()
	fcount["variable"] = fcount["variable"].astype("category")

	return vi_df, fcount

def Experiment_name(df, df_setup, var = "experiment"):
	keys = {}
	for cat in df["experiment"].unique():
		# =========== Setup the names ============
		[100, 110, 200, 120, 304]
		try:
			if cat == 100:
				nm = "RandomForest. . ."
			elif cat == 101:
				nm = "No LS"
			elif cat == 102:
				nm = "5yr LS"
			elif cat == 103:
				nm = "15yr LS"
			elif cat == 104:
				nm = "20yr LS"
			elif cat == 110:
				nm = "RandomForest. Hypervisor Parameterisation. ."
			elif cat == 120:
				nm = "XGBoost. . ."
			elif cat == 200:
				nm = "RandomForest. Two Stage. ."
			elif cat == 201:
				nm = "RF2 3 Quantile splits"
			elif cat == 202:
				nm = "RF2 2 Interval splits"
			elif cat == 204:
				nm = "RF2 4 Interval splits"
			elif cat == 304:
				nm = "XGBoost. . No Dependant Interpolation."
			elif cat ==310:
				nm = "XGBoost VPW and 0% Missing data" #, 330, 331]
			elif cat ==330:
				nm = "XGBoost VPW and 50% Missing data"
			elif cat ==331:
				nm = "XGBoost VPW and 100% Missing data"
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
					nm = f"DataMOD_{pred if not np.isnan(float(pred)) else 'AllSample' }yrPred_{lswin}yrLS_{NAfrac}percNA"

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

