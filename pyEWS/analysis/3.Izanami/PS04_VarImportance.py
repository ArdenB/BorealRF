"""
Boreal EWS PSP data anlysis 
 
Script to  make individaul psps plots  
"""

# ==============================================================================

__title__ = "Feature Importance across Runs"
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
import matplotlib.gridspec as gridspec
from collections import OrderedDict, defaultdict
import seaborn as sns
import palettable
# from numba import jit
import matplotlib.colors as mpc
from tqdm import tqdm
import pickle
from itertools import product
import ast


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
from scipy.cluster import hierarchy
# from sklearn.utils import shuffle
# from scipy.stats import spearmanr

# ==============================================================================
def main():
	# ========== Get the file names and open the files ==========
	formats = None
	path  = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"
	cf.pymkdir(path+"plots/")
	ppath = "./pyEWS/analysis/3.Izanami/Figures/PS04/"
	cf.pymkdir(ppath)
	exp = 434

	# ==========  ========== 
	corr, col_nms, corr_linkage = _getcorr(path, exp)
	df, ver, hueord = _ImpOpener(path, [exp], AddFeature=True)
	featureFigPres(exp, ppath, corr, col_nms, corr_linkage, df, ver, hueord, huex = "VariableGroup")
	breakpoint()
	
	featureFig(ppath, corr, col_nms, corr_linkage, df, ver, hueord, huex = "VariableGroup")
	breakpoint()
	# ========== the old method left for supp ==========
	expr = OrderedDict()
	# expr['DeltaBiomass']  = [402, 405]
	# expr['Delta_biomass'] = [402, 405, 406] 
	# expr["Predictors"]    = [400, 401, 402] 
	# expr['Obs_biomass']   = [401, 403, 404] 
	# expr["Complete"]      = [400, 401, 402, 403, 404, 405, 406] 

	# var  = "PermutationImportance"
	# var  = "Importance"
	for epnm in expr:
		# exps = [401, 403, 404]
		huex = "VariableGroup"#"Count"
		# ========== get the PI data ==========
		df, ver, hueord = _ImpOpener(path, expr[epnm], AddFeature=True)
		# try:
		for var in ["PermutationImportance"]:#, "FeatureImportance"
			featureplotter(df, ppath, var, expr[epnm], huex, epnm, ver, hueord)
		# except Exception as er:
		# 	warn.warn(str(er))
	breakpoint()

# ==============================================================================
def featureFigPres(exp, ppath, corr, col_nms, corr_linkage, df, ver, hueord, huex = "VariableGroup"):
	""" 
	Figure to look at features, figures in presentation format
	"""
	# ========== Setup the matplotlib params ==========
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':8, "axes.labelweight":"bold"})
	font = ({'family' : 'normal','weight' : 'bold', 'size'   : 8})
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	# map_proj = ccrs.LambertConformal(central_longitude=lons.mean(), central_latitude=lats.mean())

	# ========== Create the figure ==========
	fig, ax = plt.subplots(constrained_layout=True, figsize=(11,13))
	# g = sns.catplot(x="Variable", y="PermutationImportance", hue=huex, 	dodge=False, data=df, palette= hueord["cmap"], col="experiment",  ax=ax3)
	
	vorder = df.loc[df.Count >= 5].groupby("VariableName").mean().sort_values("PermutationImportance", ascending=False).index
	g = sns.boxplot(
		y="VariableName", x="PermutationImportance", hue=huex, 
		dodge=False, data=df.loc[df.Count >= 5], palette= hueord["cmap"], ax=ax, order=vorder)
	# g.set_xticklabels(g.get_xticklabels(),  rotation=15, horizontalalignment='right')

	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS04_PaperFig02_VarImportance_Perm_{exp}_in5" 
	for ext in [".png", ".pdf",]:
		plt.savefig(fnout+ext)#, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()


	fig, ax = plt.subplots(constrained_layout=True, figsize=(11,13))
	# g = sns.catplot(x="Variable", y="PermutationImportance", hue=huex, 	dodge=False, data=df, palette= hueord["cmap"], col="experiment",  ax=ax3)
	
	forder = df.groupby("VariableName").mean().sort_values("PermutationImportance", ascending=False).index
	g = sns.boxplot(
		y="VariableName", x="PermutationImportance", hue=huex, 
		dodge=False, data=df, palette= hueord["cmap"], ax=ax, order=forder)
	# g.set_xticklabels(g.get_xticklabels(),  rotation=15, horizontalalignment='right')

	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS04_PaperFig02_VarImportance_Perm_{exp}_full" 
	for ext in [".png", ".pdf",]:
		plt.savefig(fnout+ext)#, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()
	breakpoint()

	# ========== Create the figure ==========
	# fig  = plt.figure(constrained_layout=True, figsize=(12,18))
	# spec = gridspec.GridSpec(ncols=4, nrows=3, figure=fig)

	# +++++ Correlation matric +++++
	fig, ax = plt.subplots(constrained_layout=True, figsize=(15,14))
	_heatmap(corr, col_nms, fig, ax, cbar=True)
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS04_PaperFig02_VarImportance_correlationmatrix" 
	for ext in [".png", ".pdf",]:
		plt.savefig(fnout+ext)#, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()
	breakpoint()


	# +++++ Correlation matric +++++
	fig, ax = plt.subplots(constrained_layout=True, figsize=(27,15))
	_Network_plot(corr, corr_linkage, col_nms, fig, ax)
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS04_PaperFig02_VarImportance_linkagedendro" 
	for ext in [".png", ".pdf",]:
		plt.savefig(fnout+ext)#, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()


	# g.set(ylim=(0, 1))
	# g.set_axis_labels("", var)
	

	# ========== Create the figure ==========
	# fig, ax = plt.subplots(constrained_layout=True, figsize=(27,15))
	# # g = sns.catplot(x="Variable", y="PermutationImportance", hue=huex, 	dodge=False, data=df, palette= hueord["cmap"], col="experiment",  ax=ax3)
	# g2 = sns.boxplot(
	# 	x="VariableName", y="FeatureImportance", hue=huex, 
	# 	dodge=False, data=df.loc[df.Count >= 5], palette= hueord["cmap"], ax=ax)
	# g2.set_xticklabels(g.get_xticklabels(),  rotation=15, horizontalalignment='right')
	# 	# ========== Save tthe plot ==========
	# print("starting save at:", pd.Timestamp.now())
	# fnout = f"{ppath}PS04_PaperFig02_VarImportance" 
	# for ext in [".png", ".pdf",]:
	# 	plt.savefig(fnout+ext)#, dpi=130)
	
	# plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
	# 	__title__, __version__,  __author__, pd.Timestamp.now())
	# gitinfo = cf.gitmetadata()
	# cf.writemetadata(fnout, [plotinfo, gitinfo])
	# plt.show()
	# breakpoint()




def featureFig(ppath, corr, col_nms, corr_linkage, df, ver, hueord, huex = "VariableGroup"):
	""" Figure to look at features """
	# ========== Setup the matplotlib params ==========
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':12, "axes.labelweight":"bold"})
	font = ({'family' : 'normal','weight' : 'bold', 'size'   : 12})
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	# map_proj = ccrs.LambertConformal(central_longitude=lons.mean(), central_latitude=lats.mean())

	# ========== Create the figure ==========
	fig  = plt.figure(constrained_layout=True, figsize=(12,18))
	spec = gridspec.GridSpec(ncols=4, nrows=3, figure=fig)

	# +++++ Correlation matric +++++
	ax1  = fig.add_subplot(spec[0, :2])
	_heatmap(corr, col_nms, fig, ax1)

	# +++++ Correlation matric +++++
	ax2  = fig.add_subplot(spec[0, 2:])
	_Network_plot(corr, corr_linkage, col_nms, fig, ax2)


	# ========== Create the figure ==========
	ax3  = fig.add_subplot(spec[1, :])
	# g = sns.catplot(x="Variable", y="PermutationImportance", hue=huex, 	dodge=False, data=df, palette= hueord["cmap"], col="experiment",  ax=ax3)
	g = sns.boxplot(
		x="Variable", y="PermutationImportance", hue=huex, 
		dodge=False, data=df.loc[df.Count >= 6], palette= hueord["cmap"], ax=ax3)
	g.set_xticklabels(g.get_xticklabels(),  rotation=15, horizontalalignment='right')
	# g.set(ylim=(0, 1))
	# g.set_axis_labels("", var)
	# breakpoint()
	# ========== Create the figure ==========
	# ax4  = fig.add_subplot(spec[2, :])
	# # g = sns.catplot(x="Variable", y="PermutationImportance", hue=huex, 	dodge=False, data=df, palette= hueord["cmap"], col="experiment",  ax=ax3)
	# g2 = sns.boxplot(
	# 	x="Variable", y="FeatureImportance", hue=huex, 
	# 	dodge=False, data=df.loc[df.Count >= 6], palette= hueord["cmap"], ax=ax4)
	# g2.set_xticklabels(g.get_xticklabels(),  rotation=15, horizontalalignment='right')
		# ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS04_PaperFig02_VarImportance" 
	for ext in [".png", ".pdf",]:
		plt.savefig(fnout+ext)#, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()
	breakpoint()



def _Network_plot(corr, corr_linkage, col_nms, fig, ax, ):
	# +++++ Calculate the ward hierarchy +++++
	# ========== Build a plot ==========
	dendro  = hierarchy.dendrogram(corr_linkage, labels=col_nms, ax=ax, 
		leaf_rotation=90, color_threshold=6)
	# breakpoint()
	

def _heatmap(corr, col_nms, fig, ax, cbar=True):

	dfcorr = pd.DataFrame(corr, columns=col_nms, index=col_nms)
	cmap   = mpc.ListedColormap(palettable.cmocean.diverging.Balance_20_r.mpl_colors)
	sns.heatmap(dfcorr, center=0, square=True, 	cbar=cbar, cmap=cmap,	cbar_kws={"pad": 0.015, "shrink": .90}, ax=ax)
	# dendro_idx      = np.arange(0, len(dendro['ivl']))
	# ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
	# ax2.set_xticks(dendro_idx)
	# ax2.set_yticks(dendro_idx)
	# ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
	# ax2.set_yticklabels(dendro['ivl'])
	# fig.tight_layout()
	# plt.show()
	# ipdb.set_trace()

def featureplotter(df, ppath, var, exps, huex, epnm, ver, hueord, AddFeature=True):
	""" Function to plot the importance of features """
	# ========== Setup params ==========
	plt.rcParams.update({'axes.titleweight':"bold","axes.labelweight":"bold", 'axes.titlesize':10})
	font = {'family' : 'normal',
	        'weight' : 'bold', #,
	        'size'   : 10}
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	# plt.rcParams.update({'axes.titleweight':"bold", })
	# if AddFeature:
	# 	breakpoint()

	# ========== Create the figure ==========
	g = sns.catplot( x="Variable", y=var, hue=huex, dodge=False, data=df, 
		palette= hueord["cmap"], col="experiment",  
		col_wrap=1, sharex=False, aspect=5, height=6., kind="bar")
	g.set_xticklabels( rotation=45, horizontalalignment='right')
	g.set(ylim=(0, 1))
	g.set_axis_labels("", var)
	# ========== create a color dict ==========
	colordict = OrderedDict()
	for lab, pcolor in zip(g.legend.texts, g.legend.get_patches()):
		# breakpoint()
		colordict[lab.get_text()] = pcolor.get_facecolor()

	for exp,  ax in zip(exps, g.axes):
		for tick_label, patch in zip(ax.get_xticklabels(), ax.patches):
		    try: 
		    	cgroup = df.loc[np.logical_and(df.Variable == tick_label._text, df.experiment==exp), huex].unique()[0]
		    	tick_label.set_color(colordict[f"{cgroup}"])
		    except Exception as er:
		    	warn.warn(str(er))
		    	breakpoint()
	g.fig.suptitle(
		f'{epnm}{ f" ({ver+1} runs out of 10)" if (ver < 9.0) else ""}', 
		fontweight='bold')
	plt.tight_layout()

	# ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS04_{epnm}_{var}" 
	for ext in [".png"]:#".pdf",
		plt.savefig(fnout+ext, dpi=130)
	
	plotinfo = "PLOT INFO: Feature Importance plots made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])

	plt.show()

	# ========== Create the same fig but for the vars that i care about ==========
	g = sns.catplot( 
		x="Variable", y=var, #ci=0.95, estimator=bn.nanmedian, 
		hue=huex, dodge=False, 
		data=df.loc[df.Count >= 5], 
		col="experiment",  col_wrap=1, sharex=False, aspect=5, height=6., kind="bar")
	g.set_xticklabels( rotation=45, horizontalalignment='right')
	g.set(ylim=(0, 1))
	g.set_axis_labels("", var)
	
	# ========== create a color dict ==========
	colordict = OrderedDict()
	for lab, pcolor in zip(g.legend.texts, g.legend.get_patches()):
		# breakpoint()
		colordict[lab.get_text()] = pcolor.get_facecolor()

	for exp,  ax in zip(exps, g.axes):
		for tick_label, patch in zip(ax.get_xticklabels(), ax.patches):
		    cgroup = df.loc[np.logical_and(df.Variable == tick_label._text, df.experiment==exp), huex].unique()[0]
		    try: 
		    	tick_label.set_color(colordict[f"{cgroup}"])
		    except:
		    	breakpoint()
	g.fig.suptitle(
		f'{epnm}{ f" ({ver+1} runs out of 10)" if (ver < 9.0) else ""}', 
		fontweight='bold')
	plt.tight_layout()
	# ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS04_{epnm}_topAgreement_{var}" 
	for ext in [".png"]:#".pdf",
		plt.savefig(fnout+ext, dpi=130)
	
	plotinfo = "PLOT INFO: Feature Importance plots made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])

	plt.show()

	breakpoint()


def _getcorr(path, exp, 
	dpath = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/", 
	inheritrows=True):
	# ========== load the setup ==========
	setup            = pd.read_csv(f"{path}{exp}/Exp{exp}_setup.csv", index_col=0).T.infer_objects().reset_index()
	for va in ["window", "Nstage", "test_size", "DropNAN", "FutDist", "FullTestSize"]:
		setup[va] = setup[va].astype(float)
	for vaxs in ["dropvar", "splitvar"]:	#
		# breakpoint()
		try:
			setup[vaxs] = [ast.literal_eval(setup[vaxs].values[0])]
		except ValueError:
			continue

	setup = setup.loc[0]
	for tf in ["yTransformer", "Transformer"]:
		if np.isnan(setup[tf]):
			setup[tf] = None
		else:
			breakpoint()
	# breakpoint()
	branch  = 0
	version = 0
	if (setup["predvar"] == "lagged_biomass") or inheritrows:
		basestr = f"TTS_VI_df_AllSampleyears" 
	else:
		basestr = f"TTS_VI_df_AllSampleyears_{setup['predvar']}" 

	if not setup["FullTestSize"] is None:
		basestr += f"_{int(setup['FullTestSize']*100)}FWH"
		if setup["splitvar"] == ["site", "yrend"]:
			basestr += f"_siteyear{setup['splitmethod']}"
		elif setup["splitvar"] == "site":
			basestr += f"_site{setup['splitmethod']}"

	if setup.loc["predvar"] == "lagged_biomass":
		fnamein  = f"{dpath}VI_df_AllSampleyears.csv"
		sfnamein = f"{dpath}SiteInfo_AllSampleyears.csv"
	else:
		fnamein  = f"{dpath}VI_df_AllSampleyears_ObsBiomass.csv"
		sfnamein = f"{dpath}SiteInfo_AllSampleyears_ObsBiomass.csv"
		# bsestr = f"TTS_VI_df_AllSampleyears_{setup.loc[0, 'predvar']}" 

	# ========== load in the data ==========
	# breakpoint()
	setup["DropDist"] = True
	X_train, X_test, y_train, y_test, col_nms, loadstats, corr, df_site, dbg = bf.datasplit(
		setup.loc["predvar"], exp, version,  branch, setup, final=True, #cols_keep=ColNm, #force=True,
		vi_fn=fnamein, region_fn=sfnamein, basestr=basestr, dropvar=setup.loc["dropvar"])

	corr_linkage = hierarchy.ward(corr)
	# cluster_ids   = hierarchy.fcluster(corr_linkage, branch, criterion='distance')
	return corr, col_nms, corr_linkage



def _ImpOpener(path, exps, var = "PermutationImportance", AddFeature=False):
	"""
	Function to open the feature importance files and return them as a single 
	DataFrame"""

	# ========== Loop over the exps ==========
	df_list = []
	for exp in exps:
		fnames = sorted(glob.glob(f"{path}{exp}/Exp{exp}_*PermutationImportance.csv"))
		for ver, fn in enumerate(fnames):
			dfin = pd.read_csv( fn, index_col=0)
			if AddFeature:
				ver    = int(fn[-28:-26])
				fn_mod = f"{path}{exp}/models/XGBoost_model_exp{exp}_version{ver}.dat"
				model  = pickle.load(open(f"{fn_mod}", "rb"))
				try:
					dfin["FeatureImportance"] = model.feature_importances_
				except :
					warn.warn("This usually fails when there is an XGBoost version mismatch")
					breakpoint()
					raise ValueError
				
				# if not os.path.isfile()
				if var == "Importance":
					dfin = pd.melt(dfin, id_vars="Variable", value_vars=["PermutationImportance", "FeatureImportance"], var_name="Metric", value_name=var)
					dfin.replace({"PermutationImportance":"PI", "FeatureImportance":"FI"}, inplace=True)
				# breakpoint()
			dfin["experiment"] = exp
			dfin["version"]    = ver
			vnames = Smartrenamer(dfin.Variable.values)
			dfin["VariableName"]  = vnames.VariableGroup
			if dfin.groupby("VariableName")["version"].count().max() >1:
				dfin = dfin.groupby("VariableName").agg(
					Variable              = ('Variable', 'first'), 
					PermutationImportance = ('PermutationImportance', 'sum'),
					FeatureImportance     = ('FeatureImportance', 'sum'),
					experiment            = ('experiment', 'first'), 
					version               = ('version', 'first'), 
					).reset_index().reindex(dfin.columns, axis=1).sort_values("PermutationImportance", ascending=False)

				# warn.warn("Similarr feature in model")
				# breakpoint()
			df_list.append(dfin)
	# ========== Group them together ==========
	df = pd.concat(df_list).reset_index(drop=True)
	# ========== Calculate the counts ==========
	df["Count"] = df.groupby(["experiment", "VariableName"])[var].transform("count")
	df.sort_values(['experiment', 'Count'], ascending=[True, False], inplace=True)
	df.reset_index(drop=True,inplace=True)

	# ========== group the vartypes ==========
	sp_groups  = pd.read_csv("./EWS_package/data/raw_psp/SP_groups.csv", index_col=0)
	soils      = pd.read_csv( "./EWS_package/data/psp/modeling_data/soil_properties_aggregated.csv", index_col=0).columns.values
	permafrost = pd.read_csv("./EWS_package/data/psp/modeling_data/extract_permafrost_probs.csv", index_col=0).columns.values
	def _getgroup(VN, species=[], soils = [], permafrost=[]):
		if VN in ["biomass", "stem_density", "ObsGap", "StandAge"]:
			return "Survey"
		elif VN in ["Disturbance", "DisturbanceGap", "Burn", "BurnGap", "DistPassed"]:
			return "Disturbance"
		elif (VN.startswith("Group")) or (VN in species):
			return "Species"
		elif VN.startswith("LANDSAT"):
			return "RS VI"
		elif VN.endswith("30years"):
			return "Climate"
		elif VN in soils:
			return "Soil"
		elif VN in permafrost:
			return "Permafrost"
		else: 
			print(VN)
			breakpoint()
			return "Unknown"

	
	df["VariableGroup"] = df.Variable.apply(_getgroup, 
		species = sp_groups.scientific.values, soils=soils, permafrost=permafrost).astype("category")
	

	hueord =  cmapper(df["VariableGroup"], AddFeature, var)
	if AddFeature and var == "Importance":
		df["VariableGroup"] = pd.Series([f"{MI}-{VG}" for MI, VG in zip(df["Metric"], df["VariableGroup"])]).astype("category")
	
	# ========== set the cat order ==========
	df["VariableGroup"].cat.set_categories(hueord["HueOrder"], inplace=True)
	return df, ver, hueord

def Smartrenamer(names):
	
	df = pd.DataFrame({"Variable":names})

	sitenm     = {"biomass":"(st) Initial Biomass", "stem_density":"(st) Stem Density", "ObsGap":"(st) Observation Gap", "StandAge": "(st) Stand Age"}
	sp_groups  = pd.read_csv("./EWS_package/data/raw_psp/SP_groups.csv", index_col=0)
	soils      = pd.read_csv( "./EWS_package/data/psp/modeling_data/soil_properties_aggregated.csv", index_col=0).columns.values
	permafrost = pd.read_csv("./EWS_package/data/psp/modeling_data/extract_permafrost_probs.csv", index_col=0).columns.values
	
	def _getname(VN, sitenm=[], species=[], soils = [], permafrost=[], droptime=True):
		if VN in sitenm.keys():
			return sitenm[VN]
		elif VN in ["Disturbance", "DisturbanceGap", "Burn", "BurnGap", "DistPassed"]:
			return f"(Dis) {VN}"
		elif VN.startswith("Group"):
			VNcl = VN.split("_")
			if len(VNcl) == 3:
				return f"(sp.) GP{VNcl[1]} {VNcl[2]}"
			if len(VNcl) == 4:
				return f"(sp.) GP{VNcl[1]} {VNcl[2]} {VNcl[3]}"
			else:
				breakpoint()

		elif VN in species:
			return f"(sp.) {VN}"

		elif VN.startswith("LANDSAT"):
			if not droptime:
				breakpoint()

			# VNc = VN.split(".")[0]
			VNcl = VN.split("_")
			if not len(VNcl) in [4, 5]:
				print("Length is wrong")
			# breakpoint()

			return f"(RSVI) {VNcl[1].upper()} {VNcl[2] if not VNcl[2]=='trend' else 'Theil.'} {VNcl[3] if not VNcl[3] == 'pulse' else 'size'}"

		elif VN.endswith("30years"):
			if "DD_" in VN:
				VN = VN.replace("DD_", "DDb")
			
			if "abs_trend" in VN:
				VN = VN.replace("abs_trend", "abs. trend")
			
			VNcl = VN.split("_")
			if len(VNcl) == 3:
				return f"(Cli.) {VNcl[0]} {VNcl[1]}"
			elif len(VNcl) == 4:
				breakpoint()
				return f"(Cli.) {VNcl[0]} {VNcl[1]} {VNcl[2]}"
			elif len(VNcl) == 5:
				print(VN, "Not uunderstood")
				breakpoint()
				return f"(Cli.) {VNcl[0]} {VNcl[1]} {VNcl[3]} {VNcl[4]}"
			else:
				breakpoint()
		elif VN in soils:
			VNcl = VN.split("_")
			if not len(VNcl)==5:
				breakpoint()
			return f"(Soil) {VNcl[0]} {VNcl[3]}cm"
		elif VN in permafrost:
			return f"(PF.) {VN}"
		else: 
			print(VN)
			breakpoint()
			return "Unknown"

	df["VariableGroup"] = df.Variable.apply(_getname, sitenm=sitenm,
		species = sp_groups.scientific.values, soils=soils, permafrost=permafrost).astype("category")
	return df


# ==============================================================================
def cmapper(varlist, AddFeature, var):
	# pick the cmap
	# "Disturbance", 
	if "Disturbance" in varlist:
		warn.warn("THis is not implemented, currently no provision for disturbance")
		breakpoint()
		sys.exit()

	vaorder = ["Climate", "RS VI", "Survey", "Species","Permafrost", "Soil",]
	cmapHex = palettable.colorbrewer.qualitative.Paired_12.hex_colors
	if AddFeature and var == "Importance":
		vaorder = [f"{MI}-{VG}" for VG, MI in product(vaorder, ["FI", "PI"])]
		# breakpoint()
	else:
		cmapHex = cmapHex[1::2]
	# ========== Decide the order of stuff =========
	return ({"cmap":cmapHex, "HueOrder":vaorder})

	
# ==============================================================================
if __name__ == '__main__':
	main()