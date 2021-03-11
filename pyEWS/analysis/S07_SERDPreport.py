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

__title__ = "Making SERDP plots"
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
import palettable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import string



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
			else:
				nm = "%d.%s" % (cat, df_setup[df_setup.Code.astype(int) == int(cat)].name.values[0])
		except Exception as er:
			print(str(er))
			breakpoint()
		keys[cat] = nm
		df_mres["experiment"].replace({cat:nm}, inplace=True)

	df_mres["experiment"] = df_mres["experiment"].astype('category')



	OvP_fnames = glob.glob(path + "*/Exp*_OBSvsPREDICTED.csv")
	df_OvsP    = pd.concat([load_OBS(ofn) for ofn in OvP_fnames])

	gclass     = glob.glob(path + "*/Exp*_OBSvsPREDICTEDClas_y_test.csv")
	df_clest   = pd.concat([load_OBS(mrfn) for mrfn in gclass])
	# ========== Create the confusion matrix plots ==========

	twostageplots(path, df_mres, df_setup, df_OvsP, df_clest, keys, 
		experiments=[201, 202, 204], ref=[100, 120])

	confusion_plots(path, df_mres, df_setup, df_OvsP, keys, sumtxt="7Quantiles", ncol = 5, 
		experiments=[104, 103, 100, 102, 101, 120, 200, 201, 202, 204])
	
	# breakpoint()
	splts = np.arange(-1, 1.05, 0.050)
	splts[ 0] = -1.00001
	splts[-1] = 1.00001

	confusion_plots(path, df_mres, df_setup, df_OvsP, keys,
		split = splts, sumtxt="JustConfusion", annot=False, 
		experiments=[100, 120, 200, 201, 202, 204])

	confusion_plots(path, df_mres, df_setup, df_OvsP, keys, ncol = 4,
		split = splts, sumtxt="JustConfusion1", annot=False, 
		experiments=[102, 103, 120, 200, 201, 202, 204])

	twostageplots(path, df_mres, df_setup, df_OvsP, df_clest, keys, 
		experiments=[201, 202, 204], ref=[100])


	confusion_plots(path, df_mres, df_setup, df_OvsP, keys, 
		split = splts, sumtxt="SplitEqualDist", annot=False, plttyp = ["bplot"])

	RegionalPerformance(path, df_mres, df_setup, df_OvsP, df_clest, keys)

# ==============================================================================
def RegionalPerformance(path, df_mres, df_setup, df_OvsP, df_clest, keys, ncol = 3, 
	inc_class=False, experiments=[104, 103, 100, 102, 101, 200, 201, 202, 204],
	fig=None, axs=None, ref=None, plttyp = ["boxplot", "KDE"], sumtxt="RegionPerformance", ROI="CAFI"):
	# pass
	# ========== work out the correct regions ==========
	folder = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/"
	# ========== Look for a region file ==========
	region_fn = folder + "TTS_sites_and_regions.csv"
	df_site = pd.read_csv(region_fn, index_col=0)

	# ========== Pull the region infomation ==========
	vi_fn = "./EWS_package/data/models/input_data/vi_df_all_V2.csv"
	vi_df = pd.read_csv( vi_fn, index_col=0).reset_index()
	vi_df = vi_df.merge(df_site, on="site").set_index("index")

	df_OD = OrderedDict()
	df_ls = []
	print("looking at site data")
	for num, expn in enumerate(experiments):
		reg = []
		df_sub = df_OvsP[df_OvsP.experiment == expn].copy()
		for index, row in df_sub.iterrows():
			reg.append(vi_df.loc[index].Region)
		# +++++ Add the regions +++++
		df_sub["Region"] = reg
		df_sub["Region"] = df_sub.Region.astype('category')

		# +++++ Subset by region +++++
		dfr = df_sub[df_sub["Region"] == ROI]
		df_ls.append(dfr)
		OD = OrderedDict()
		if not dfr.empty:
			r2   = sklMet.r2_score(dfr["Observed"], dfr["Estimated"])
			# if r2< 0:
			# 	r2 = 0.
			RMSE = sklMet.mean_squared_error(dfr["Observed"], dfr["Estimated"], squared=False)
			OD["experiment"] = keys[expn]
			OD["R2"]         = r2
			OD["RMSE"]       = RMSE
			OD["NumTest"]    = dfr.shape[0]/10.
			OD["NumTot"]     = df_mres[df_mres.experiment == keys[expn]]["%s_siteinc"%ROI].mean()

			df_OD[num] = OD
		else:
			warn.warn("no data here")
			breakpoint()
	df = pd.DataFrame(df_OD).T

	# ========== Create the figure ==========
	plt.rcParams.update({'figure.subplot.top' : 0.99 })
	plt.rcParams.update({'figure.subplot.bottom' : 0.05 })
	plt.rcParams.update({'figure.subplot.right' : 0.98 })
	plt.rcParams.update({'figure.subplot.left' : 0.10 })
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':8})
	font = {'family' : 'normal',
	        'weight' : 'bold', #,
	        'size'   : 8}
	mpl.rc('font', **font)
	sns.set_style("whitegrid")

	# ========== Plot Broad summary  ==========
	fig, (ax1, ax2, ax3) = plt.subplots(
		3, 1, figsize=(9,10),num=("General Summary"), dpi=130)


	# +++++ R2 plot +++++
	sns.barplot(y="R2", x="experiment", data=df, ax=ax1)
	ax1.set_xlabel("")
	ax1.set_ylabel(r"$R^{2}$", weight="bold")
	ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')
	# ax1.set(ylim=(-1., 1.))
	
	# Timeplot
	sns.barplot(y="RMSE", x="experiment", data=df, ax=ax2)
	ax2.set_xlabel("")
	ax2.set_ylabel(r"RMSE", weight="bold")
	ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, horizontalalignment='right')

	sns.barplot(y="NumTot", x="experiment", data=df, ax=ax3)
	ax3.set_xlabel("")
	ax3.set_ylabel("# %s Sites" % ROI, weight="bold")
	ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, horizontalalignment='right')
	plt.tight_layout()

	print("starting save at:", pd.Timestamp.now())
	fnout = path+ "plots/SERDP_REGION_" + sumtxt
	for ext in [".pdf", ".png"]:
		plt.savefig(fnout+ext, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel Region Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()

	breakpoint()
	ipdb.set_trace()

	df_OP = pd.concat(df_ls)
	# ===== Change the keys =====
	for expr in df_OP.experiment.unique():df_OP.replace({expr:keys[expr]}, inplace=True)


	sns.relplot(x="Observed", y="Estimated", col = "experiment",
		col_wrap=3, data=df_OP, ci=None, facet_kws={"ylim":(-1., 1.), "xlim":(-1., 1.)})
	# sns.plt.ylim(0, 20)
	# sns.plt.xlim(0, None)

	print("starting save at:", pd.Timestamp.now())
	fnout = path+ "plots/SERDP_REGION_xyscatter_" + sumtxt
	for ext in [".pdf", ".png"]:
		plt.savefig(fnout+ext, dpi=130)

	plotinfo = "PLOT INFO: Multimodel Region Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()

	breakpoint()
	ipdb.set_trace()




def twostageplots(path, df_mres, df_setup, df_OvsP, df_clest, keys, ncol = 3, 
	inc_class=False, experiments=[201, 202, 204],
	fig=None, axs=None, ref=None, plttyp = ["boxplot", "KDE"], sumtxt="KDEandBoxplot"):
	alf = dict(enumerate(string.ascii_lowercase, 1))
	 # ========== Create the figure ==========
	plt.rcParams.update({'figure.subplot.top' : 0.99 })
	plt.rcParams.update({'figure.subplot.bottom' : 0.05 })
	plt.rcParams.update({'figure.subplot.right' : 0.98 })
	plt.rcParams.update({'figure.subplot.left' : 0.05 })
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':8})
	font = {'family' : 'normal',
	        'weight' : 'bold', #,
	        'size'   : 8}
	mpl.rc('font', **font)
	sns.set_style("whitegrid")

	alf = dict(enumerate(string.ascii_lowercase, 1))

	fig= plt.figure(
		figsize=(16,11),
		num=("Twostage Plots"), 
		dpi=130)

	gs = mpl.gridspec.GridSpec(len(plttyp), ncol, figure=fig)

	ax = fig.add_subplot(gs[0, :])

	dflist = []
	ks = []
	for expn in experiments+ ref:
		# breakpoint()
		dfs = df_OvsP[df_OvsP.experiment == expn]
		dfs["experiment"] =  keys[expn]
		dflist.append(dfs) 

		ks.append(keys[expn])

	df = pd.concat(dflist)
	# df_score["experiment"] = df_score.experiment.astype('object')
	df["experiment"] = df.experiment.astype('category')#.cat.reorder_categories(ks)

	df["ObsCat"] = [vls.left for vls in pd.cut(df.Observed, 8)]
	df["ObsCat"].replace(df["ObsCat"].min(), -1.0, inplace=True)
	df["Residual"] = df.Estimated - df.Observed
	sns.boxplot(y="Residual", x="ObsCat", hue="experiment", data=df, ax=ax)
	ax.set_xlabel("Observed", weight='bold')
	ax.set_ylabel("Residual", weight='bold')
	ax.text(-0.035, 1.0, "a)", transform=ax.transAxes, 
				size=8, weight='bold')#, zorder=106)


	# ====== Make the KDE ==========
	hexcbar = palettable.colorbrewer.qualitative.Set1_9.hex_colors
	for num, expn in enumerate(experiments):
		txt = "%s)" % alf[num+2]
		splitstr = df_setup.splits.loc["Exp%d" %expn]
		split = []
		for sp in splitstr.split(", "):
			if not sp == "":
				split.append(float(sp))
		# print(np.ceil((num+1)/ncol).astype(int), ncol)
		ax1 = fig.add_subplot(gs[1, num])
		plt.title(keys[expn])

		ax1.set_xlim(-1, 1)
		ax1.set_ylim(0, 10)
		# ax.set(xlim=(10, 40))
		# ========== Creat the data ==========
		dfOP = df_OvsP[df_OvsP.experiment == expn]
		dfCC = df_clest[df_clest.experiment == expn]

		# ========== merge ==========
		dfM = pd.merge(dfOP.reset_index(), dfCC.reset_index(), how="inner")
		
		# ========== Make one per class ==========
		for cnu, gcl in enumerate(np.sort(dfCC.class_est.unique())):
			sns.kdeplot(dfM[dfM.class_est == gcl].Observed, shade=True, ax=ax1, 
				color=hexcbar[cnu], label= "Class%d:%.02f to %.02f"%(cnu,split[cnu], split[cnu+1]))
		ax1.set_xlabel("Observed", weight='bold')
		ax1.set_ylabel("Kernal Density", weight='bold')
		# ========== vertical class lines ==========
		for vln in split[1:-1]:
			ax1.axvline(vln, alpha =1, linestyle="--", c="grey")
		ax1.text(-0.13, 1.03, txt, transform=ax1.transAxes, 
					size=8, weight='bold')#, zorder=106)

	
	print("starting save at:", pd.Timestamp.now())
	fnout = path+ "plots/SERDP_TWOSTAGE_" + sumtxt
	for ext in [".pdf", ".png"]:
		plt.savefig(fnout+ext, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()
	breakpoint()

def barplots(path, df_mres, df_setup, df_OvsP, keys, ncol = 3, 
	inc_class=False, experiments=[104, 103, 100, 102, 101],
	fig=None, axs=None):
	
	dflist = []
	ks = []
	for expn in experiments:
		dflist.append(df_mres[df_mres.experiment == keys[expn]]) 
		ks.append(keys[expn])

	df_score = pd.concat(dflist)
	df_score["experiment"] = df_score.experiment.astype('object')
	df_score["experiment"] = df_score.experiment.astype('category').cat.reorder_categories(ks)

	if fig is None:
		font = ({
			'family' : 'normal',
			'weight' : 'bold', 
			'size'   : 8})
		axes = ({
			'titlesize':8,
			'labelweight':'bold'
			})

		plt.rcParams.update({'figure.subplot.top' : 0.90 })
		plt.rcParams.update({'figure.subplot.bottom' : 0.10 })
		plt.rcParams.update({'figure.subplot.right' : 0.90 })
		plt.rcParams.update({'figure.subplot.left' : 0.15 })
		plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':8})
		font = {'family' : 'normal',
		        'weight' : 'bold', #,
		        'size'   : 8}
		mpl.rc('font', **font)
		sns.set_style("whitegrid")	
		mpl.rc('font', **font)
		mpl.rc('axes', **axes)


		# ========== Plot Broad summary  ==========
		fig, (ax1, ax2, ax3) = plt.subplots(
			1, 3, figsize=(16,6),num=("General Summary"), dpi=130)
	else:
		ax1 = fig.add_subplot(2, ncol, 4)
		ax2 = fig.add_subplot(2, ncol, 5)
		ax3 = fig.add_subplot(2, ncol, 6)
		# breakpoint()

	# +++++ R2 plot +++++
	sns.barplot(y="R2", x="experiment", data=df_score, ci="sd", ax=ax1)
	ax1.set_xlabel("")
	ax1.set_ylabel(r"$R^{2}$")
	ax1.set_xticklabels(ax1.get_xticklabels())#, rotation=45, horizontalalignment='right')
	ax1.set(ylim=(0., 1.))
	
	# Timeplot
	sns.barplot(y="TotalTime", x="experiment", data=df_score, ci="sd", ax=ax2)
	ax2.set_xlabel("")
	ax2.set_ylabel(r"$\Delta$t (min)")
	ax2.set_xticklabels(ax2.get_xticklabels())#, rotation=45, horizontalalignment='right')

	# Do the reaction of region 
	regions = []
	for col, re in zip(['fractrows', "CAFI_sitefrac"], ["Total", "Alaska"]):
		# re = col[:2]
		# ========== This is a temp fix I will need to recalculate the values ==========
		dfc           = df_score[["experiment", col]].reset_index(drop=True).rename(
			{col:"SitePer"}, axis=1)
		
		dfc["Region"] = re
		regions.append(dfc)

	# ========== Stack the regions ==========
	df = pd.concat(regions)
	sns.barplot(x="experiment", y="SitePer", hue="Region", data=df, ax=ax3, palette=sns.color_palette("Paired"))
	ax3.set_xlabel("")
	ax3.set_ylabel("% of sites")
	ax3.set_xticklabels(ax3.get_xticklabels())#, rotation=45, horizontalalignment='right')
	ax3.set(ylim=(0., 1.))

	# ========== Add some Lab ==========
	for ax, txt in zip([ax1, ax2, ax3], ["d)", "e)", "f)"]):
		ax.text(-0.15, 1.01, txt, transform=ax.transAxes, 
					size=8, weight='bold')#, zorder=106)
		axs.append(ax)

	# if not mapdet.text is None:

	
	# plt.show()
	# breakpoint()


def confusion_plots(path, df_mres, df_setup, df_OvsP, keys, ncol = 3, 
	inc_class=False, split=None, sumtxt="", annot=True, zline=True, 
	experiments=[104, 100, 101], plttyp=[]):
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
	for num, expn in enumerate(experiments):
		expr[expn] = keys[expn]
		if expn // 100 == 1 and inc_class:
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
	plt.rcParams.update({'figure.subplot.top' : 0.99 })
	plt.rcParams.update({'figure.subplot.bottom' : 0.05 })
	plt.rcParams.update({'figure.subplot.right' : 0.98 })
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

	alf = dict(enumerate(string.ascii_lowercase, 1))

	nrows = int(np.ceil(len(experiments)/float(ncol)) + len(plttyp))

	fig= plt.figure(
		figsize=((5*ncol+1),(5*nrows+1)),
		num=("Normalised Confusion Matrix " + sumtxt), 
		dpi=130)#, constrained_layout=True)figsize=(17,12),

	# fig, axes = plt.subplots(np.ceil( expsize / ncol).astype(int), ncol, 
	#  	figsize=(16,9),sharex=True, sharey=True,	num=("Normalised Confusion Matrix "), dpi=130)
	# for ax, exp in zip(axes.flat, df_class.experiment.unique()):
	axs = []
	# ========= Calculate the number for rows ==========

	# ========== Loop over ach experiment ==========
	for num, exp in enumerate(expr):
		# breakpoint()np.ceil( len(expr) / ncol).astype(int)
		ax = fig.add_subplot(nrows, ncol, num+1, label=exp)
		axs.append(ax)

			# sharex=True, sharey=True, label =exp)
		# ========== Pull out the data for each experiment ==========
		if (exp % 1) == 0.:
			df_c = df_class[df_class.experiment == exp]
		else: 
			df_c = cl_on[exp]
		
		if any(df_c["Estimated"].isnull()):
			warn.warn(str(df_c["Estimated"].isnull().sum())+ " of the estimated Values were NaN")
			df_c = df_c[~df_c.Estimated.isnull()]

		print(exp, num, sklMet.accuracy_score(df_c["Observed"], df_c["Estimated"]))
		
		# ========== Calculate the confusion matrix ==========
		cMat  = sklMet.confusion_matrix(df_c["Observed"], df_c["Estimated"]).astype(float)
		cCor  = np.tile(df_c.groupby("Observed").count()["Estimated"].values.astype(float), (cMat.shape[0], 1)).T
		conM =  ( cMat/cCor).T
		df_cm = pd.DataFrame(conM, index = [int(i) for i in np.arange(expsize)],
		                  columns = [int(i) for i in np.arange(expsize)])
		# breakpoint()
		if annot:
			ann = df_cm.round(3)
		else:
			ann = False
		
		# if int(num+1) == ncol:
		plt.title("%s" % (expr[exp]), loc = "left")
		divider = make_axes_locatable(ax)
		cax1 = divider.append_axes("right", size="5%", pad=0.05)
		fg = sns.heatmap(df_cm, annot=ann, vmin=0, vmax=1, ax = ax, 
			cbar_ax = cax1,  square=True) # cbar_kws={"shrink": .5}
		txt = "%s)" % alf[num+1]
		ax.text(-0.15, 1.05, txt, transform=ax.transAxes, 
					size=8, weight='bold')#, zorder=106)

		# fig.colorbar(img1, cax=cax1)


		# cbar_ax = fig.add_axes([.905, .3, .05, .3])
			
			# breakpoint()
		# else:
		# 	fg = sns.heatmap(df_cm, annot=ann, vmin=0, vmax=1, ax = ax, cbar=False,  square=True)
		# 	cbar=False
		ax.plot(np.arange(expsize+1), np.arange(expsize+1), "w", alpha=0.5)
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
			location = np.arange(0, sptsze+1)
			location[ 0] += 0.00001
			location[-1] -= 0.00001
			ax.set_xticks(location)
			ax.set_xticklabels(np.round(split, 2), rotation=90)
			ax.set_yticks(location)
			ax.set_yticklabels(np.round(split, 2), rotation=0)

		ax.set_xlabel("Observed", weight='bold')
		# if num//ncol == 0:
		ax.set_ylabel("Predicted", weight='bold')
		# elif int(num+1) == ncol:
		# 	# cbar=True
		# 	breakpoint()
		# 	fig.colorbar(fg, ax=ax, shrink=0.6)

	if ncol > 3:
		plt.tight_layout()
	# plt.colorbar()
	if "bplot" in plttyp:
		barplots(path, df_mres, df_setup, df_OvsP, keys, fig=fig, axs=axs)

	# ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = path+ "plots/SERDP_Normalised_Confusion_Matrix_" + sumtxt
	for ext in [".pdf", ".png"]:
		plt.savefig(fnout+ext, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()

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
