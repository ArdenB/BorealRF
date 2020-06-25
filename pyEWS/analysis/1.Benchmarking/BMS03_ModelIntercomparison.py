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

__title__ = "Comparing Biomass prediction frameworks"
__author__ = "Arden Burrell"
__version__ = "v1.0(23.06.2020)"
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
	keys = {}
	for cat in df_mres["experiment"].unique():
		# =========== Setup the names ============
		nm = "%d.%s" % (cat, df_setup[df_setup.Code.astype(int) == int(cat)].name.values[0])
		keys[cat] = nm
		df_mres["experiment"].replace({cat:nm}, inplace=True)

	df_mres["experiment"] = df_mres["experiment"].astype('category')

	# +++++ the estimated vs observed values +++++
	OvP_fnames = glob.glob(path + "*/Exp*_OBSvsPREDICTED.csv")
	df_OvsP    = pd.concat([load_OBS(ofn) for ofn in OvP_fnames])
	
	confusion_plots(path, df_mres, df_setup, df_OvsP, keys, sumtxt="7Quantiles")#, inc_class=True)
	Region_plots(path, df_mres, df_setup, df_OvsP, keys, ncol = 4, sumtxt="")
	scatter_plots(path, df_mres, df_setup, df_OvsP, keys, ncol = 4, sumtxt="")
	# ========== Create the confusion matrix plots ==========
	confusion_plots(path, df_mres, df_setup, df_OvsP, keys, 
		split = [-1.00001, 0.0, 1.00001], sumtxt="SplitatO")

	Main_plots(path, df_mres, df_setup, df_OvsP, keys)

	warn.warn("To do: Implement some form of variable importance plots")
	breakpoint()

# ==============================================================================
def Region_plots(path, df_mres, df_setup, df_OvsP, keys, ncol = 4, sumtxt=""):
	# ========== Create the figure ==========
	plt.rcParams.update({'figure.subplot.right' : 0.85 })
	plt.rcParams.update({'figure.subplot.left' : 0.05 })
	plt.rcParams.update({'axes.titleweight':"bold"})
	font = {'family' : 'normal',
	        'weight' : 'bold', #,
	        'size'   : 10}
	mpl.rc('font', **font)
	sns.set_style("whitegrid")

	warn.warn("this is very much a rushed figure to show brendan")
	warn.warn("This is broken, have pached a temp fix that will need to be removed")
	regions = []
	for col in df_mres.columns:
		if col [-4:] == "frac":
			re = col[:2]
			# ========== This is a temp fix I will need to recalculate the values ==========
			dfc           = df_mres[["experiment", col]].reset_index(drop=True).rename(
				{col:"SitePer"}, axis=1)
			
			dfc["Region"] = re
			regions.append(dfc)

	# ========== Stack the regions ==========
	df = pd.concat(regions)

	fig, ax= plt.subplots(
		figsize=(19,10),
		num=("Regional Site Fraction " + sumtxt), 
		dpi=130)

	sns.barplot(x="experiment", y="SitePer", hue="Region", data=df, ax=ax)
	ax.set_xlabel("")
	ax.set_ylabel("% of sites")
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
	plt.tight_layout()

	# ========== Save the plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = path+ "plots/BM03_Regional_SiteFrac_Model_Comparison" + sumtxt
	for ext in [".pdf", ".png"]:
		plt.savefig(fnout+ext, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel Site Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()


def scatter_plots(path, df_mres, df_setup, df_OvsP, keys, ncol = 4, sumtxt=""):
	# ========== Create the figure ==========
	plt.rcParams.update({'figure.subplot.right' : 0.85 })
	plt.rcParams.update({'figure.subplot.left' : 0.05 })
	plt.rcParams.update({'axes.titleweight':"bold"})
	font = {'family' : 'normal',
	        'weight' : 'bold', #,
	        'size'   : 10}
	mpl.rc('font', **font)
	sns.set_style("whitegrid")

	warn.warn("this is very much a rushed figure to show brendan")

	sns.relplot(x="Observed", y="Estimated", col = "experiment",
		col_wrap=4, data=df_OvsP, ci=None)

	print("starting save at:", pd.Timestamp.now())
	fnout = path+ "plots/BM03_Model_scatter" + sumtxt
	for ext in [".pdf", ".png"]:
		plt.savefig(fnout+ext, dpi=130)
	
	plotinfo = "PLOT INFO: General Multimodel Obs vs Pred Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()


def confusion_plots(path, df_mres, df_setup, df_OvsP, keys, ncol = 4, 
	inc_class=False, split=None, sumtxt=""):
	"""Function to create and plot the confusion matrix"""
	
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
	for num, expn in enumerate(df_class.experiment.unique()):
		if expn // 100 == 1:
			expr[expn] = keys[expn]
		else:
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
	
	# ========== Create the figure ==========
	plt.rcParams.update({'figure.subplot.right' : 0.85 })
	plt.rcParams.update({'figure.subplot.left' : 0.05 })
	plt.rcParams.update({'axes.titleweight':"bold"})
	font = {'family' : 'normal',
	        'weight' : 'bold', #,
	        'size'   : 10}
	mpl.rc('font', **font)
	sns.set_style("whitegrid")

	fig= plt.figure(
		figsize=(19,11),
		num=("Normalised Confusion Matrix " + sumtxt), 
		dpi=130)#, constrained_layout=True)figsize=(17,12),

	# fig, axes = plt.subplots(np.ceil( expsize / ncol).astype(int), ncol, 
	#  	figsize=(16,9),sharex=True, sharey=True,	num=("Normalised Confusion Matrix "), dpi=130)
	# ========== Loop over ach experiment ==========
	# for ax, exp in zip(axes.flat, df_class.experiment.unique()):
	axs = []
	for num, exp in enumerate(expr):
		ax = fig.add_subplot(np.ceil( len(expr) / ncol).astype(int), ncol, num+1, label=exp)
			# sharex=True, sharey=True, label =exp)
		# ========== Pull out the data for each experiment ==========
		if (exp % 1) == 0.:
			df_c = df_class[df_class.experiment == exp]
		else: 
			df_c = cl_on[exp]
		
		print(exp, sklMet.accuracy_score(df_c["Observed"], df_c["Estimated"]))
		
		# ========== Calculate the confusion matrix ==========
		conM =  (sklMet.confusion_matrix(df_c["Observed"], df_c["Estimated"]) / 
			df_c.groupby("Observed").count()["Estimated"].values)
		df_cm = pd.DataFrame(conM, index = [int(i) for i in np.arange(expsize)],
		                  columns = [int(i) for i in np.arange(expsize)])
		sns.heatmap(df_cm, annot=True, vmin=0, vmax=1, ax = ax, cbar=False,  square=True)
		ax.plot(np.arange(expsize+1), np.arange(expsize+1), "w", alpha=0.5)
		plt.title(expr[exp])
		# ax.set_aspect("equal")
		# ax = sklMet.ConfusionMatrixDisplay(conM, np.arange(expsize))
		# print(" should be able to pass ax here and solve this")
		# ax.plot()
	plt.tight_layout()

	# ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = path+ "plots/BM03_Normalised_Confusion_Matrix_" + sumtxt
	for ext in [".pdf", ".png"]:
		plt.savefig(fnout+ext, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()

def Main_plots(path, df_mres, df_setup, df_OvsP, keys, ncol = 4, sumtxt=""):
	"""
	function to produce some basic performance plots 
	"""
	font = ({
		'family' : 'normal',
		'weight' : 'bold', 
		'size'   : 11})
	axes = ({
		'titlesize':11,
		'labelweight':'bold'
		})


	sns.set_style("whitegrid")
	mpl.rc('font', **font)
	mpl.rc('axes', **axes)

	# ========== Plot Broad summary  ==========
	fig, (ax1, ax2, ax3) = plt.subplots(
		1, 3, figsize=(20,8),num=("General Summary"), dpi=130)
	# +++++ R2 plot +++++
	sns.barplot(y="R2", x="experiment", data=df_mres, ci="sd", ax=ax1)
	ax1.set_xlabel("")
	ax1.set_ylabel(r"$R^{2}$")
	ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')
	ax1.set(ylim=(0., 1.))
	# +++++ time taken plot +++++
	sns.barplot(y="TotalTime", x="experiment", data=df_mres, ci="sd", ax=ax2)
	ax2.set_xlabel("")
	ax2.set_ylabel(r"$\Delta$t (min)")
	ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, horizontalalignment='right')
	# +++++ R2 plot +++++
	sns.barplot(y="fractrows", x="experiment", data=df_mres, ci="sd", ax=ax3)
	ax3.set_xlabel("")
	ax3.set_ylabel("% of sites")
	ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, horizontalalignment='right')
	ax3.set(ylim=(0., 1.))
	
	plt.tight_layout()
	# ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = path+ "plots/BM03_General_Model_Comparison" + sumtxt
	for ext in [".pdf", ".png"]:
		plt.savefig(fnout+ext, dpi=130)
	
	plotinfo = "PLOT INFO: General Multimodel Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()

	# .tick_params(axis='x', labelrotation= )
	breakpoint()
	# sns.boxplot(y="R2", x="experiment", data=df_mres)
	sns.swarmplot(y="R2", x="experiment", data=df_mres)
	plt.show()

	sns.scatterplot(y="TotalTime", x="R2", hue="experiment", data=df_mres)
	plt.show()

	sns.scatterplot(y="R2", x="version", hue="experiment", data=df_mres)
	plt.show()

	sns.lineplot(y="R2", x="version", hue="experiment", data=df_mres)
	plt.show()

	breakpoint()

# ==============================================================================
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
			# df_in["%s_siteinc" % region] = df_in["%s_siteinc" % region].astype(float)
			df_in["%s_sitefrac" % region] = (df_in["%s_siteinc" % region] / Rcounts[region])
		except KeyError:
			# Places with missing regions
			df_in["%s_siteinc"  % region] = 0.
			df_in["%s_sitefrac" % region] = 0.
	return df_in

if __name__ == '__main__':
	main()
