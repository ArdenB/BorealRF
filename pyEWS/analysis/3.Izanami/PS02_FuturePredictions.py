"""
Boreal EWS PSP data anlysis 
 
Script to  make individaul psps plots  
"""

# ==============================================================================

__title__ = "Examining the biomass predictions"
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

		# ========== Get the file names and open the files ==========
	path  = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"
	cf.pymkdir(path+"plots/")
	ppath = "./pyEWS/analysis/3.Izanami/Figures/PS02/"
	cf.pymkdir(ppath)
	# +++++ the model Infomation +++++
	setup_fnames = glob.glob(path + "*/Exp*_setup.csv")
	df_setup     = pd.concat([pd.read_csv(sfn, index_col=0).T for sfn in setup_fnames], sort=True)
	
	# +++++ the final model results +++++
	mres_fnames = glob.glob(path + "*/Exp*_Results.csv")
	df_mres = pd.concat([fix_results(mrfn) for mrfn in mres_fnames], sort=True)
	df_mres["TotalTime"]  = df_mres.TotalTime / pd.to_timedelta(1, unit='m')
	df_mres, keys = Experiment_name(df_mres, df_setup, var = "experiment")


	# ========= Load in the observations ==========
	OvP_fnames = glob.glob(path + "*/Exp*_OBSvsPREDICTED.csv")
	df_OvsP    = pd.concat([load_OBS(ofn) for ofn in OvP_fnames], sort=True)
	
	gclass     = glob.glob(path + "*/Exp*_OBSvsPREDICTEDClas_y_test.csv")
	df_clest   = pd.concat([load_OBS(mrfn) for mrfn in gclass], sort=True)

	branch     = glob.glob(path + "*/Exp*_BranchItteration.csv")
	df_branch  = pd.concat([load_OBS(mrfn) for mrfn in branch], sort=True)
	# breakpoint()
	splts = np.arange(-1, 1.05, 0.05)
	splts[ 0] = -1.00001
	splts[-1] = 1.00001


	confusion_plots(path, df_mres, df_setup, df_OvsP, keys, experiments=[334],  ncol = 4, 
		inc_class=False, split=splts, sumtxt="", annot=True, zline=True)
	breakpoint()

# ==============================================================================
def confusion_plots(path, df_mres, df_setup, df_OvsP, keys, experiments=None,  ncol = 4, 
	inc_class=False, split=None, sumtxt="", annot=True, zline=True):
	"""Function to create and plot the confusion matrix"""
	# ========== Look to see if experiments have been passed ==========
	# This it to allow only a subset to be looked at 
	if experiments is None:
		experiments = df_OvsP.experiment.unique() #all experiments
	# ========== Check if i needed to pull out the splits ==========
	if split is None:
		splitstr = df_setup.splits.loc["Exp200"]
		split = []
		for sp in splitstr.split(", "):
			if not sp == "":
				split.append(float(sp))

	# ========== Get the observed values ==========
	df_class = df_OvsP.copy()
	expsize  = len(split) -1 # df_class.experiment.unique().size
	df_class["Observed"]  =  pd.cut(df_class["Observed"], split, labels=np.arange(expsize))
	df_class["Estimated"] =  pd.cut(df_class["Estimated"], split, labels=np.arange(expsize))

	# ========== Pull out the classification only accuracy ==========
	expr   = OrderedDict()
	cl_on  = OrderedDict() #dict to hold the classification only 
	for num, expn in enumerate(experiments):
		if expn // 100 == 1 or expn // 100 == 3:
			expr[expn] = keys[expn]
		elif expn // 100 == 2:
			expr[expn] = keys[expn]
			if inc_class:

				# +++++ load in the classification only +++++
				OvP_fn = glob.glob(path + "%d/Exp%d*_OBSvsPREDICTEDClas_y_test.csv"% (expn,  expn))
				df_OvP  = pd.concat([load_OBS(ofn) for ofn in OvP_fn])

				df_F =  df_class[df_class.experiment == expn].copy()
				for vr in df_F.version.unique():
					dfe = (df_OvP[df_OvP.version == vr]).reindex(df_F[df_F.version==vr].index)
					df_F.loc[df_F.version==vr, "Estimated"] = dfe.class_est
				# +++++ Check and see i haven't added any null vals +++++
				if (df_F.isnull()).any().any():
					breakpoint()
				# +++++ Store the values +++++
				cl_on[expn+0.1] = df_F
				expr[ expn+0.1] = keys[expn]+"_Class"
		else:
			breakpoint()
	# ========== Create the figure ==========
	plt.rcParams.update({'figure.subplot.top' : 0.90 })
	plt.rcParams.update({'figure.subplot.bottom' : 0.10 })
	plt.rcParams.update({'figure.subplot.right' : 0.95 })
	plt.rcParams.update({'figure.subplot.left' : 0.05 })
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':8})
	font = {'family' : 'normal',
	        'weight' : 'bold', #,
	        'size'   : 8}
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	try:
		sptsze = split.size
	except:
		sptsze = len(split)

	fig= plt.figure(
		figsize=(19,11),
		num=("Normalised Confusion Matrix " + sumtxt), 
		dpi=130)#, constrained_layout=True)figsize=(17,12),

	# fig, axes = plt.subplots(np.ceil( expsize / ncol).astype(int), ncol, 
	#  	figsize=(16,9),sharex=True, sharey=True,	num=("Normalised Confusion Matrix "), dpi=130)
	# for ax, exp in zip(axes.flat, df_class.experiment.unique()):
	axs = []


	# ========== Loop over ach experiment ==========
	for num, exp in enumerate(expr):
		ax = fig.add_subplot(np.ceil( len(expr) / ncol).astype(int), ncol, num+1, label=exp)
			# sharex=True, sharey=True, label =exp)
		# ========== Pull out the data for each experiment ==========
		if (exp % 1) == 0.:
			df_c = df_class[df_class.experiment == exp]
		else: 
			df_c = cl_on[exp]
		
		if any(df_c["Estimated"].isnull()):
			# warn.warn(str(df_c["Estimated"].isnull().sum())+ " of the estimated Values were NaN")
			df_c = df_c[~df_c.Estimated.isnull()]
		
		df_c.sort_values("Observed", axis=0, ascending=True, inplace=True)
		print(exp, sklMet.accuracy_score(df_c["Observed"], df_c["Estimated"]))
			# breakpoint()
		
		# ========== Calculate the confusion matrix ==========
		try:
			cMat  = sklMet.confusion_matrix(
				df_c["Observed"], df_c["Estimated"], 
				labels=df_c["Observed"].cat.categories).astype(int) 
			cCor  = np.tile(df_c.groupby("Observed").count()["Estimated"].values.astype(float), (cMat.shape[0], 1)).T
			# breakpoint()
			# print(cMat.shape)
			conM =  ( cMat/cCor).T
			conM[np.logical_and((cCor == 0), (cMat==0)).T] = 0.
			# if (np.isnan(conM)).any():
			# 	breakpoint()
			df_cm = pd.DataFrame(conM, index = [int(i) for i in np.arange(expsize)],
			                  columns = [int(i) for i in np.arange(expsize)])
		except Exception as er:
			print(str(er))
			breakpoint()
		# breakpoint()
		if annot:
			ann = df_cm.round(3)
		else:
			ann = False
		sns.heatmap(df_cm, annot=ann, vmin=0, vmax=1, ax = ax, cbar=False,  square=True)
		ax.plot(np.arange(expsize+1), np.arange(expsize+1), "w", alpha=0.5)
		plt.title(expr[exp])
		# ========== fix the labels +++++
		if (sptsze > 10):
			# +++++ The location of the ticks +++++
			interval = int(np.floor(sptsze/10))
			location = np.arange(0, sptsze, interval)
			# +++++ The new values +++++
			values = np.round(np.linspace(-1., 1, location.size), 2)
			ax.set_xticks(location)
			ax.set_xticklabels(values)
			ax.set_yticks(location)
			ax.set_yticklabels(values)
			# ========== Add the cross hairs ==========
			if zline:

				ax.axvline(location[values == 0][0], alpha =0.25, linestyle="--", c="grey")
				ax.axhline(location[values == 0][0], alpha =0.25, linestyle="--", c="grey")
		else:
			# ========== Calculate the zero lines ==========
			if zline:
				# ========== Calculate the zero line location ==========
				((x0),) = np.where(np.array(split) < 0)
				x0a = x0[-1]
				((x1),) = np.where(np.array(split) >=0)
				x1a = x1[0]
				zeroP =  x0a + (0.-split[x0a])/(split[x1a]-split[x0a])
				# ========== Add the cross hairs ==========
				ax.axvline(x=zeroP, ymin=0.1, alpha =0.25, linestyle="--", c="w")
				ax.axhline(y=zeroP, xmax=0.9, alpha =0.25, linestyle="--", c="w")

			# +++++ fix the values +++++
			location = np.arange(0., sptsze)#+1
			location[ 0] += 0.00001
			location[-1] -= 0.00001
			ax.set_xticks(location)
			ax.set_xticklabels(np.round(split, 2), rotation=90)
			ax.set_yticks(location)
			ax.set_yticklabels(np.round(split, 2), rotation=0)



	plt.tight_layout()

	# ========== Save tthe plot ==========
	# print("starting save at:", pd.Timestamp.now())
	# fnout = path+ "plots/BM03_Normalised_Confusion_Matrix_" + sumtxt
	# for ext in [".pdf", ".png"]:
	# 	plt.savefig(fnout+ext, dpi=130)
	
	# plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
	# 	__title__, __version__,  __author__, pd.Timestamp.now())
	# gitinfo = cf.gitmetadata()
	# cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()
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
				nreakpoint()
		except KeyError:
			# Places with missing regions
			df_in["%s_siteinc"  % region] = 0.
			df_in["%s_sitefrac" % region] = 0.
		# 	# df_in["%s_siteinc" % region] = df_in["%s_siteinc" % region].astype(float)
		# 	df_in["%s_sitefrac" % region] = (df_in["%s_siteinc" % region] / Rcounts[region])
	return df_in
# ==============================================================================

if __name__ == '__main__':
	main()