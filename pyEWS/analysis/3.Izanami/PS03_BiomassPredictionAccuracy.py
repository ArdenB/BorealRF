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

	# ========== Create the matplotlib params ==========
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':8, "axes.labelweight":"bold",})
	font = {'family' : 'normal',
	        'weight' : 'bold', #,
	        'size'   : 8}
	mpl.rc('font', **font)
	sns.set_style("whitegrid")


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

	expr = OrderedDict()
	expr['DeltaBiomass']  = [402, 405]
	expr['Delta_biomass'] = [402, 405, 406] 
	expr["Complete"]      = [400, 401, 402, 403, 404, 405, 406] 
	expr["Predictors"]    = [400, 401, 402] 
	expr['Obs_biomass']   = [401, 403, 404] 
	
	nvar = "exp"
	ci="sd"
	
	for epnm in expr:
		# ========== get the scores ==========
		df = Translator(df_setup, df_mres, keys, df_OvsP, df_clest, df_branch, expr[epnm], path)
		dfM, dfS, dfScores, exp_names = transmet(df, expr[epnm], df_mres, ppath, epnm, keys)

		# ========== Make the plts ==========
		ConfusionPlotter(df, expr[epnm], df_mres, ppath, epnm, keys, nvar, 
			ci, dfM, dfS, dfScores, exp_names)
		_overviewplots(df, expr[epnm], df_mres, ppath, epnm, keys, nvar, ci, dfM, dfS, dfScores, exp_names)
		_scatterplots(df, expr[epnm], df_mres, ppath, epnm, keys, nvar, ci, dfM, dfS, dfScores, exp_names)

	breakpoint()

# ==============================================================================
def ConfusionPlotter(df, experiments, df_mres, ppath, epnm, keys, nvar, 
	ci, dfM, dfS, dfScores, exp_names):#, split, obsvar, estvar,):
	obsvar = "ObsDelta"
	estvar = "EstDelta"
	
	# maxval = np.ceil(dfM.loc[:, obsvar].abs().max()/ 100) * 100
	maxval =  500
	minval = -500
	gap    = 10 
	# split  = np.hstack([np.min([-maxval ,-1000.]),np.arange(-400., 401, 10), np.max([1000., maxval])])
	split  = np.arange(minval, maxval+1, gap)
	dfMp = dfM.copy()
	for va  in [obsvar, estvar]:
		dfMp.loc[dfMp[va]<= minval, va] = (minval+1)
		dfMp.loc[dfMp[va]>= maxval, va] = ( maxval-1)


	ncol = np.min([4, len(experiments)]).astype(int)
	nrow = np.ceil(len(experiments)/ncol).astype(int)

	fig= plt.figure(
		figsize=(ncol*6,nrow*5),
		num=(f"Normalised Confusion Matrix - {epnm}"), 
		dpi=130)#, constrained_layout=True)figsize=(17,12),

	axs = []


	# ========== Loop over ach experiment ==========
	for num, exp in enumerate(experiments):
		ax = fig.add_subplot(np.ceil( len(experiments) / ncol).astype(int), ncol, num+1, label=exp)
		_confusion_plots(df_mres, dfMp, dfS, obsvar, estvar, keys, exp, fig, ax, split, 
			inc_class=False, sumtxt=epnm, annot=False, zline=True, num=num)


	# for exp in experiments:
		# fig, ax = plt.subplots(1, 1, figsize=(15,13))
	fig.suptitle(
		f'{epnm}{ f" ({dfM.version.max()+1} runs out of 10)" if (dfM.version.max() < 9.0) else ""}', 
		fontweight='bold')
	plt.tight_layout()
	# ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS03_{epnm}_Normalised_Confusion_Matrix" 
	for ext in [".png"]:#".pdf",
		plt.savefig(fnout+ext, dpi=130)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()
	

# ==============================================================================
def _confusion_plots(
	df_mres, dfM, dfS, obsvar, estvar,  keys, exp, fig, ax, split,
	inc_class=False, sumtxt="", annot=False, zline=True, num=0, cbar=True, mask=True):
	"""Function to create and plot the confusion matrix"""


	# ========== Get the observed and estimated values ==========
	df_c         = dfM.loc[dfM.experiment == keys[exp], [obsvar,estvar]].copy()
	expsize      = len(split) -1 # df_class.experiment.unique().size
	df_c[obsvar] = pd.cut(df_c[obsvar], split, labels=np.arange(expsize))
	df_c[estvar] = pd.cut(df_c[estvar], split, labels=np.arange(expsize))
	df_set       = dfS[dfS.experiment == keys[exp]]

	# if any((df2 <= split.min()).any()) or any((df2 >= split.max()).any()):
	# 	breapoint()
	# ========== Pull out the classification only accuracy ==========
	# expr   = OrderedDict()
	# cl_on  = OrderedDict() #dict to hold the classification only 
	# for num, expn in enumerate(experiments):
	# expr[exp] = keys[exp]

	if any(df_c[estvar].isnull()):
		breakpoint()
		df_c = df_c[~df_c[estvar].isnull()]

	if any(df_c[obsvar].isnull()):
		breakpoint()
		df_c = df_c[~df_c[obsvar].isnull()]

	try:
		sptsze = split.size
	except:
		sptsze = len(split)
	
	df_c.sort_values(obsvar, axis=0, ascending=True, inplace=True)
	print(exp, sklMet.accuracy_score(df_c[obsvar], df_c[estvar]))
	
	# ========== Calculate the confusion matrix ==========
	# \\\ confustion matrix  observed (rows), predicted (columns), then transpose and sort
	df_cm  = pd.DataFrame(
		sklMet.confusion_matrix(df_c[obsvar], df_c[estvar],  
			labels=df_c[obsvar].cat.categories,  normalize='true'),  
		index = [int(i) for i in np.arange(expsize)], 
		columns = [int(i) for i in np.arange(expsize)]).T.sort_index(ascending=False)

	cmap = mpc.ListedColormap(palettable.matplotlib.Inferno_20_r.mpl_colors)
	if mask:
		#+++++ remove 0 values +++++
		df_cm.replace(0, np.NaN, inplace=True)
		# breakpoint()

	if annot:
		ann = df_cm.round(3)
	else:
		ann = False


	g = sns.heatmap(df_cm, annot=ann, vmin=0, vmax=1, ax = ax, cbar=cbar, square=True, cmap=cmap)
	ax.plot(np.flip(np.arange(expsize+1)), np.arange(expsize+1), "w", alpha=0.5)
	# plt.title(expr[exp])
	# ========== fix the labels +++++
	if (sptsze > 10):
		# +++++ The location of the ticks +++++
		interval = int(np.floor(sptsze/10))
		location = np.arange(0, sptsze, interval)
		# +++++ The new values +++++
		values = np.round(np.linspace(split[0], split[-1], location.size))
		ax.set_xticks(location)
		ax.set_xticklabels(values, rotation=90)
		ax.set_yticks(location)
		ax.set_yticklabels(np.flip(values), rotation=0)
		# ========== Add the cross hairs ==========
		if zline:
			try:
				ax.axvline(location[values == 0][0], alpha =0.25, linestyle="--", c="grey")
				ax.axhline(location[values == 0][0], alpha =0.25, linestyle="--", c="grey")
			except Exception as er:
				print(er)
				breakpoint()
	else:
		# warn.warn("Yet to fix the ticks here")
		# breakpoint()
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
		ax.set_yticklabels(np.flip(np.round(split, 2)), rotation=0)


	ax.set_title(f"{exp}-{keys[exp]} $R^{2}$ {df_set.R2.mean()}", loc= 'left')
	ax.set_xlabel(obsvar)
	ax.set_ylabel(estvar)


	# g.tight_layout()



def _overviewplots(df, experiments, df_mres, ppath, epnm, keys, nvar, ci, dfM, dfS, dfScores, exp_names):
	# ========== Plot Broad summary  ==========
	fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, num=("General Summary"), figsize=(20,10))
	
	# +++++ time taken plot +++++
	sns.barplot(y="TotalTime", x="experiment", hue="Computer", ci=ci, data=dfS, ax=ax1, order=exp_names)
	# sns.barplot(y="TotalTime", x="experiment", data=df_set, ci="sd", ax=ax2, )
	ax1.set_xlabel("")
	ax1.set_ylabel(r"$\Delta$t (min)")
	# ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')

	# +++++ R2 plot +++++
	sns.barplot(y="R2", x="experiment",  data=dfS, ci=ci, ax=ax2, order=exp_names)
	ax2.set_xlabel("")
	ax2.set_ylabel(r"$R^{2}$")
	# ax2.set_xticklabels(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
	ax2.set(ylim=(0., 1.))

	# +++++ group rank plot +++++
	sns.barplot(y="index", x="Rank", hue="experiment", data=dfScores, ax = ax3, hue_order=exp_names)
	ax3.set_xlabel("Min abs. residual Rank")
	ax3.set_ylabel("Count")

	# +++++ Mean Absolute Error +++++
	sns.barplot(y="MAE", x="experiment", data=dfS, ci=ci, ax=ax4, order=exp_names)
	ax4.set_xlabel("")
	ax4.set_ylabel("Mean Absolute Error")
	# ax4.set_xticklabels(ax4.get_xticklabels(), rotation=30, horizontalalignment='right')
	
	# +++++ Median Absolute Error +++++
	sns.barplot(y="MedianAE", x="experiment", data=dfS, ci=ci, ax=ax5, order=exp_names)
	ax5.set_xlabel("")
	ax5.set_ylabel("Median Absolute Error")
	# ax5.set_xticklabels(ax5.get_xticklabels(), rotation=30, horizontalalignment='right')
	
	# +++++ Median Absolute Error +++++
	sns.barplot(y="BadValueCount", x="experiment", data=dfS, ci=ci, ax=ax6, order=exp_names)
	ax6.set_xlabel("")
	ax6.set_ylabel("Bad Value Count")
	if len(experiments) > 4:
		for axi in [ax1, ax2, ax4, ax5, ax6]:
			axi.set_xticklabels(axi.get_xticklabels(), rotation=10, horizontalalignment='right')


	fig.suptitle(
		f'{epnm}{ f" ({dfM.version.max()+1} runs out of 10)" if (dfM.version.max() < 9.0) else ""}', 
		fontweight='bold')
	plt.tight_layout()
	# ========== Save tthe plot ==========
	print("starting save at:", pd.Timestamp.now())
	fnout =  f"{ppath}PS03_{epnm}_General_Model_Comparison"
	for ext in [".png"]:#".pdf", 
		plt.savefig(fnout+ext)
	plotinfo = "PLOT INFO: General Multimodel Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	
	plt.show()

def _scatterplots(df, experiments, df_mres, ppath, epnm, keys, nvar, ci, dfM, dfS, dfScores, exp_names):
	# ========== Biomass scatter plot ==========
	fig = sns.relplot(data=dfM, x="Observed", y="Estimated",  col="experiment", col_wrap=np.min([4, len(experiments)])) #hue="experiment",
	fig.set(ylim=(0, 2000))
	fig.set(xlim=(0, 2000))
	fig.fig.suptitle(
		f'Biomass - {epnm}{ f" ({dfM.version.max()+1} runs out of 10)" if (dfM.version.max() < 9.0) else ""}', 
		fontweight='bold')
	fig.tight_layout()
	
	fnout =  f"{ppath}/PS03_{epnm}_Biomass"
	for ext in [".png"]:#".pdf", 
		plt.savefig(fnout+ext)
	plotinfo = "PLOT INFO: General Multimodel Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()
	
	# ========== Delta biomass Biomass scatter plot ==========
	fig = sns.relplot(data=dfM, x="ObsDelta", y="EstDelta", 
		col="experiment", col_wrap=np.min([4, len(experiments)])) # hue="experiment", 
	fig.set(ylim=(-500, 500))
	fig.set(xlim=(-500, 500))
	fig.fig.suptitle(
		f'\u0394Biomass - {epnm}{ f" ({dfM.version.max()+1} runs out of 10)" if (dfM.version.max() < 9.0) else ""}', 
		fontweight='bold')

	fig.tight_layout()
	fnout =  f"{ppath}PS03_{epnm}_DeltaBiomass"
	for ext in [".png"]:#".pdf", 
		plt.savefig(fnout+ext)
	plotinfo = "PLOT INFO: Delta Biomass Multimodel Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()

# ==============================================================================

def transmet(df, experiments, df_mres, ppath, epnm, keys, nvar = "exp", ci="sd"):
	"""Function to look at ther overall performance of the different approaches"""


	# ========== create a dataframe ==========
	dfM = df.dropna().copy()

	# ========== pull out matched runs so i can compare across runs ==========
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
				"experiment":df_mres.loc[np.logical_and(df_mres.experiment == exp, df_mres.version == ver), nvar].values[0], #exp,
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
			# print(mets)

	#  ========== Setup the dataframes ==========
	exp_names         = [keys[expn] for expn in experiments]
	dfM['experiment'] = dfM.experiment.map(keys)
	dfM["experiment"] = dfM["experiment"].astype('category')

	dfS               = pd.DataFrame(metrics).T.infer_objects()
	dfS["experiment"] = dfS["experiment"].astype('category')
	dfScores          = dfM.groupby(["experiment","Rank"]).nunique()["index"].reset_index()

	return dfM, dfS, dfScores, exp_names

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
			if type(df_setup.loc[f"Exp{exp}", "yTransformer"]) == float:

				vfunc            = np.vectorize(_delag)
				dfC["Estimated"] = vfunc(df_act["biomass"].values, df_OP["Estimated"].values)
			else:
				warn.warn("Not Implemeted Yet")
				breakpoint()
		elif pvar == 'Obs_biomass':
			if not type(df_setup.loc[f"Exp{exp}", "yTransformer"]) == float:
				for ver in dfC.version.unique().astype(int):
					ppath = f"./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/{exp}/" 
					fn_mod = f"{ppath}models/XGBoost_model_exp{exp}_version{ver}"

					setup = pickle.load(open(f"{fn_mod}_setuptransfromers.dat", "rb"))
					dfC.loc[dfC.version == ver, "Estimated"] = setup['yTransformer'].inverse_transform(dfC.loc[dfC.version == ver, "Estimated"].values.reshape(-1, 1))

		elif pvar == 'Delta_biomass':
			if not type(df_setup.loc[f"Exp{exp}", "yTransformer"]) == float:
				for ver in dfC.version.unique().astype(int):
					ppath = f"./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/{exp}/" 
					fn_mod = f"{ppath}models/XGBoost_model_exp{exp}_version{ver}"

					setup = pickle.load(open(f"{fn_mod}_setuptransfromers.dat", "rb"))
					dfC.loc[dfC.version == ver, "Estimated"] = setup['yTransformer'].inverse_transform(dfC.loc[dfC.version == ver, "Estimated"].values.reshape(-1, 1))
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
	if not  var == "experiment":
		df[var] = df["experiment"].copy()
	for cat in df["experiment"].astype(int).unique():
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
			elif cat >= 400:
				mdva = df_setup[df_setup.Code.astype(int) == cat]["predvar"][0]
				if type(mdva) == float:
					mdva = "lagged_biomass"
				
				def _TFnamer(stn, tfname):
					# Function to quickly rename transfomrs 
					if type(tfname) == float:
						return ""
					elif tfname in ["QuantileTransformer(output_distribution='normal')", "QuantileTransformer(ignore_implicit_zeros=True, output_distribution='normal')"]:
						return f" {stn}_QTn"
					else:
						breakpoint()
						return f" {stn}_UNKNOWN"


				nm = f'{mdva}{_TFnamer("Ytf", df_setup.loc[f"Exp{int(cat)}", "yTransformer"])}{_TFnamer("Xtf", df_setup.loc[f"Exp{int(cat)}", "Transformer"])}'

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