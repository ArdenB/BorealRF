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
import matplotlib.gridspec as gridspec
from collections import OrderedDict, defaultdict
import seaborn as sns
import palettable
# from numba import jit
import matplotlib.colors as mpc
from tqdm import tqdm
from itertools import product


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

	# experiments = [400]
	# ========== Create the figure ==========
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':12})
	font = {'family' : 'normal',
	        'weight' : 'bold', #,
	        'size'   : 12}
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	plt.rcParams.update({'axes.titleweight':"bold", "axes.labelweight":"bold"})

	experiments = [400, 401, 402]
	df = Translator(df_setup, df_mres, keys, df_OvsP, df_clest, df_branch, experiments, path)

	pred = OrderedDict()
	pred["DeltaBiomass"] = ({
		"obsvar":"ObsDelta",
		"estvar":"EstDelta", 
		"limits":(-300, 300),
		"Resname":"Residual"
		})
	pred["Biomass"] = ({
		"obsvar":"Observed",
		"estvar":"Estimated", 
		"limits":(0, 1000), 
		"Resname":"Residual"
		})
	# breakpoint()

	for exp, var in product(experiments, pred):
		print(exp, var)
		if var == "Biomass":
			fig, ax = plt.subplots(1, 1, figsize=(14,6))
			Temporal_predictability(ppath, [exp], df_setup, df, keys,  fig, ax, var, va=pred[var]['Resname'])	

		fig, ax = plt.subplots(1, 1, figsize=(14,6))
		pdfplot(ppath, df, exp, keys, fig, ax, pred[var]['obsvar'], pred[var]['estvar'], var, pred[var]['limits'])

	# vi_df, fcount = VIload()

	warn.warn("TThe following plots have not been adjusted for different variable types")
	breakpoint()
	
	
	for var, ylab, ylim in zip(["R2", "TotalTime", "colcount"], [r"$R^{2}$", r"$\Delta$t (min)", "# Predictor Vars."], [(0., 1.), None, None]):
		fig, ax = plt.subplots(1, 1, figsize=(15,13))
		branchplots(exp, df_mres, keys, var, ylab, ylim,  fig, ax)


	# breakpoint()
	splts = np.arange(-1, 1.05, 0.10)
	splts[ 0] = -1.00001
	splts[-1] = 1.00001

	fig, ax = plt.subplots(1, 1, figsize=(15,13))
	confusion_plots(path, df_mres, df_setup, df_OvsP, keys,  exp, fig, ax, 
		inc_class=False, split=splts, sumtxt="", annot=False, zline=True)

	splts = np.arange(-1, 1.05, 1.0)
	splts[ 0] = -1.00001
	splts[-1] = 1.00001
	fig, ax = plt.subplots(1, 1, figsize=(15,13))
	confusion_plots(path, df_mres, df_setup, df_OvsP, keys,  exp, fig, ax, 
		inc_class=False, split=splts, sumtxt="", annot=True, zline=True)


	# breakpoint()


	
	breakpoint()

# ==============================================================================
def pdfplot(ppath, df, exp, keys, fig, ax, obsvar, estvar, var, clip, single=True):
	""" Plot the probability distribution function """
	dfin = df[df.experiment == exp]
	dfin = pd.melt(dfin[[obsvar, estvar]])
	# breakpoint()
	g = sns.kdeplot(data=dfin, x="value", hue="variable", fill=True, ax=ax, clip=clip)#clip=(-1., 1.)
	ax.set_xlabel(var)
	if single:
		plt.title(f'{exp} - {keys[exp]}', fontweight='bold')
		plt.tight_layout()

		# ========== Save tthe plot ==========
		print("starting save at:", pd.Timestamp.now())
		fnout = f"{ppath}PS02_{exp}_{var}_ProbDistFunc" 
		for ext in [".png"]:#".pdf",
			plt.savefig(fnout+ext, dpi=130)
		
		plotinfo = "PLOT INFO: PDF plots made using %s:v.%s by %s, %s" % (
			__title__, __version__,  __author__, pd.Timestamp.now())
		gitinfo = cf.gitmetadata()
		cf.writemetadata(fnout, [plotinfo, gitinfo])
		plt.show()


def branchplots(exp, df_mres, keys, var, ylab, ylim,  fig, ax):

	# +++++ R2 plot +++++
	df_set = df_mres[df_mres.experiment == keys[exp]]
	# breakpoint()
	sns.barplot(y=var, x="version", data=df_set,  ax=ax, ci=None)
	ax.set_xlabel("")
	ax.set_ylabel(ylab)
	ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
	if not ylim is None:
		ax.set(ylim=ylim)
	
	plt.show()


def confusion_plots(path, df_mres, df_setup, df_OvsP, keys, exp, fig, ax,
	inc_class=False, split=None, sumtxt="", annot=False, zline=True, num=0, cbar=True, mask=True):
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
	df_set = df_mres[df_mres.experiment == keys[exp]]
	# ========== Pull out the classification only accuracy ==========
	expr   = OrderedDict()
	cl_on  = OrderedDict() #dict to hold the classification only 
	# for num, expn in enumerate(experiments):
	if (exp // 100 in [1, 3, 4]):# or exp // 100 == 3:
		expr[exp] = keys[exp]
	elif exp // 100 == 2:
		expr[exp] = keys[exp]
		if inc_class:

			# +++++ load in the classification only +++++
			OvP_fn = glob.glob(path + "%d/Exp%d*_OBSvsPREDICTEDClas_y_test.csv"% (exp,  exp))
			df_OvP  = pd.concat([load_OBS(ofn) for ofn in OvP_fn])

			df_F =  df_class[df_class.experiment == exp].copy()
			for vr in df_F.version.unique():
				dfe = (df_OvP[df_OvP.version == vr]).reindex(df_F[df_F.version==vr].index)
				df_F.loc[df_F.version==vr, "Estimated"] = dfe.class_est
			# +++++ Check and see i haven't added any null vals +++++
			if (df_F.isnull()).any().any():
				breakpoint()
			# +++++ Store the values +++++
			cl_on[exp+0.1] = df_F
			expr[ exp+0.1] = keys[exp]+"_Class"
	else:
		breakpoint()

	try:
		sptsze = split.size
	except:
		sptsze = len(split)

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
	
	# ========== Calculate the confusion matrix ==========
	# \\\ confustion matrix  observed (rows), predicted (columns), then transpose and sort
	df_cm  = pd.DataFrame(
		sklMet.confusion_matrix(df_c["Observed"], df_c["Estimated"],  labels=df_c["Observed"].cat.categories,  normalize='true'),  
		index = [int(i) for i in np.arange(expsize)], columns = [int(i) for i in np.arange(expsize)]).T.sort_index(ascending=False)

	cmap = mpc.ListedColormap(palettable.matplotlib.Inferno_20_r.mpl_colors)
	if mask:
		#+++++ remove 0 values +++++
		df_cm.replace(0, np.NaN, inplace=True)
		# breakpoint()

	if annot:
		ann = df_cm.round(3)
	else:
		ann = False
	sns.heatmap(df_cm, annot=ann, vmin=0, vmax=1, ax = ax, cbar=cbar, square=True, cmap=cmap)
	ax.plot(np.flip(np.arange(expsize+1)), np.arange(expsize+1), "w", alpha=0.5)
	# plt.title(expr[exp])
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
		ax.set_yticklabels(np.flip(values))
		# ========== Add the cross hairs ==========
		if zline:

			ax.axvline(location[values == 0][0], alpha =0.25, linestyle="--", c="grey")
			ax.axhline(location[values == 0][0], alpha =0.25, linestyle="--", c="grey")
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


	ax.set_title(f"a) {exp}-{keys[exp]} $R^{2}$ {df_set.R2.mean()}", loc= 'left')
	ax.set_xlabel("Observed")
	ax.set_ylabel("Predicted")


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

def Temporal_predictability(
	ppath, experiments, df_setup, df, keys,  fig, ax, var,
	va = "Residual", CI = "QuantileInterval", single=True):

	"""
	Function to make a figure that explores the temporal predictability. This 
	figure will only use the runs with virable windows
	"""
	if len(experiments) > 1:
		# pick only the subset that are matched 
		dfX = df.dropna()
	else:
		dfX = df.loc[df["experiment"] == experiments[0]]



	# ========== make the plot ==========
	# for va in ["AbsResidual", "residual"]:
	# 	for CI in ["SD", "QuantileInterval"]:
	print(f"{va} {CI} {pd.Timestamp.now()}")
	# Create the labels

	lab = [keys[expn] for expn in experiments]
	# ========== set up the colours and build the figure ==========
	colours = palettable.cartocolors.qualitative.Vivid_10.hex_colors
	# ========== Build the first part of the figure ==========
	if CI == "SD":
		sns.lineplot(y=va, x="ObsGap", data=dfX, 
			hue="experiment", ci="sd", ax=ax, 
			palette=colours[:len(experiments)], legend=False)
	else:
		# Use 
		sns.lineplot(y=va, x="ObsGap", data=dfX, 
			hue="experiment", ci=None, ax=ax, 
			palette=colours[:len(experiments)], legend=False)
		for expn, hue in zip(experiments, colours[:len(experiments)]) :
			df_ci = dfX[dfX.experiment == expn].groupby("ObsGap")[va].quantile([0.05, 0.95]).reset_index()
			ax.fill_between(
				df_ci[df_ci.level_1 == 0.05]["ObsGap"].values, 
				df_ci[df_ci.level_1 == 0.95][va].values, 
				df_ci[df_ci.level_1 == 0.05][va].values, alpha=0.10, color=hue)
	# ========== fix the labels ==========
	ax.set_xlabel('Years Between Observation', fontsize=12, fontweight='bold')
	ax.set_ylabel(r'Mean Residual ($\pm$ %s)' % CI, fontsize=12, fontweight='bold')
	# ========== Create hhe legend ==========
	ax.legend(title='Experiment', loc='upper right', labels=lab)
	ax.set_title(f"{var} {va} {CI}", loc= 'left')


	# ========== The second subplot ==========
	# breakpoint()
	# sns.histplot(data=df_OvsP, x="ObsGap", hue="Region",  
	# 	multiple="dodge",  ax=ax2) #palette=colours[:len(experiments)]
	# # ========== fix the labels ==========
	# ax2.set_xlabel('Years Between Observation', fontsize=8, fontweight='bold')
	# ax2.set_ylabel(f'# of Obs.', fontsize=8, fontweight='bold')
	# ========== Create hhe legend ==========
	# ax2.legend(title='Experiment', loc='upper right', labels=lab)
	# ax2.set_title(f"b) ", loc= 'left')

	if single:
		plt.tight_layout()

		# ========== Save tthe plot ==========
		print("starting save at:", pd.Timestamp.now())
		if len (experiments) == 0:
			fnout = f"{ppath}PS02_{var}_{va}_{CI}_{experiments[0]}_TemporalPred" 
		else:
			fnout = f"{ppath}PS02_{var}_{va}_{CI}_TemporalPred" 
		for ext in [".png"]:#".pdf",
			plt.savefig(fnout+ext)#, dpi=130)
		plotinfo = "PLOT INFO: PDF plots made using %s:v.%s by %s, %s" % (
			__title__, __version__,  __author__, pd.Timestamp.now())
		gitinfo = cf.gitmetadata()
		cf.writemetadata(fnout, [plotinfo, gitinfo])
		plt.show()
	# plt.show()

# ==============================================================================
def Translator(df_setup, df_mres, keys, df_OvsP, df_clest, df_branch, experiments, path):
	""" Function to transfrom different methods of calculating biomass 
	into a comperable number """
	bioMls = []
	vi_fn = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/VI_df_AllSampleyears_ObsBiomass.csv"
	vi_df  = pd.read_csv( vi_fn, index_col=0).loc[:, ['biomass', 'Obs_biomass', 'Delta_biomass','ObsGap']]
	
	# ========== Fill in the missing sites ==========
	region_fn ="./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/SiteInfo_AllSampleyears.csv"
	site_df = pd.read_csv(region_fn, index_col=0)
	# ========== Loop over each experiment ==========
	for exp in tqdm(experiments):
		pvar =  df_setup.loc[f"Exp{exp}", "predvar"]
		if type(pvar) == float:
			# deal with the places i've alread done
			pvar = "lagged_biomass"

		# +++++ pull out the observed and predicted +++++
		df_OP  = df_OvsP.loc[df_OvsP.experiment == exp]
		df_act = vi_df.iloc[df_OP.index]
		df_s   = site_df.iloc[df_OP.index]
		dfC    = df_OP.copy()
		# breakpoint()
		
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
		dfC["Observed"]      = df_act["Obs_biomass"].values
		dfC["Residual"]      = dfC["Estimated"] - dfC["Observed"]
		dfC["Original"]      = df_act["biomass"].values
		dfC["ObsDelta"]      = df_act["Delta_biomass"].values
		dfC["EstDelta"]      = dfC["Estimated"] - dfC["Original"]
		# dfC["DeltaResidual"] = dfC["EstDelta"] - dfC["ObsDelta"]
		dfC["ObsGap"]        = df_act["ObsGap"].values
		dfC["Region"]        = df_s["Region"].values
		bioMls.append(dfC)

	# ========== Convert to a dataframe ==========
	df = pd.concat(bioMls).reset_index().sort_values(["version", "index"]).reset_index(drop=True)

	# ========== Perform grouped opperations ==========
	df["Rank"]     = df.drop("Region", axis=1).abs().groupby(["version", "index"])["Residual"].rank(na_option="bottom").apply(np.floor)
	df["RunCount"] = df.drop("Region", axis=1).abs().groupby(["version", "index"])["Residual"].transform("count")
	df.loc[df["RunCount"] < len(experiments), "Rank"] = np.NaN
	
	return df
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
				nreakpoint()
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