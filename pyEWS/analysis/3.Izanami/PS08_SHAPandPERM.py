"""
Script for dealing with permutation importance and SHAP values
"""

# ==============================================================================

__title__ = "Loss variables "
__author__ = "Arden Burrell"
__version__ = "v1.0(08.06.2021)"
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

# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf
import myfunctions.benchmarkfunctions as bf

# ========== Import packages for parellelisation ==========
# import multiprocessing as mp
import xgboost as xgb
import xarray as xr
import cartopy.crs as ccrs
import dask
from dask.diagnostics import ProgressBar
from tqdm import tqdm
import shap

import cartopy.feature as cpf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ast

# ========== Import ml packages ==========
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn import metrics as sklMet
# from sklearn.utils import shuffle
# from scipy.stats import spearmanr
# from scipy.cluster import hierarchy


# ==============================================================================

def main():
	# ========== Setup the pathways ==========
	formats = None
	path  = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"
	cf.pymkdir(path+"plots/")
	ppath = "./pyEWS/analysis/3.Izanami/Figures/PS08/"
	cf.pymkdir(ppath)

	# ========== Chose the experiment ==========
	exps      = [434, 424]
	_ImpOpener(path, ppath, exps)


def _ImpOpener(path, ppath, exps, var = "PermutationImportance", AddFeature=False, 
	textsize=14, plotSHAP=True, plotind=False):
	"""
	Function to open the feature importance files and return them as a single 
	DataFrame"""
	sns.set_style("whitegrid")
	font = ({'weight' : 'bold', 'size'   : textsize})
	mpl.rc('font', **font)
	plt.rcParams.update({'axes.titleweight':"bold", 
		"axes.labelweight":"bold", 'axes.titlesize':textsize, 'axes.titlelocation': 'left',}) 

	# ========== Loop over the exps ==========
	
	for exp in exps:
		SHAPlst = []
		df_list = []
		

		fnames = sorted(glob.glob(f"{path}{exp}/Exp{exp}_*PermutationImportance.csv"))

		for ver, fn in enumerate(fnames):
			print(ver)

			# ========== load the model ==========
			dfin = pd.read_csv( fn, index_col=0)
			dfin["experiment"] = exp
			dfin["version"]    = ver
			vnames = Smartrenamer(dfin.Variable.values)
			dfin["VariableName"]  = vnames.VariableGroup
			df_list.append(dfin)

			if (not ver == 2) or (not plotSHAP):
				warn.warn("Skipping SHAP values to start")
				continue

			fn_mod = f"{path}{exp}/models/XGBoost_model_exp{exp}_version{ver}.dat"
			model  = pickle.load(open(f"{fn_mod}", "rb"))
			ColNm = dfin["Variable"].values
			X_train, X_test, y_train, y_test, col_nms, loadstats, corr, df_site, dbg = _getdata(path, exp, ColNm)

			# ========== calculate the SHAP values ==========
			explainer   = shap.TreeExplainer(model)
			shap_values = explainer.shap_values(X_test)
			SHAPlst.append(shap_values)

			# ========== Make the relevant explainer plots ==========
			shap.summary_plot(shap_values, X_test, feature_names= [vn for vn in vnames.VariableGroup],  
				max_display=20, plot_size=(15, 13), show=False)
			# Get the current figure and axes objects.
			fig, ax = plt.gcf(), plt.gca()
			plt.tight_layout()

			# ========== Save the plot ==========
			print("starting save at:", pd.Timestamp.now())
			fnout = f"{ppath}PS08_{exp}_{var}_SHAPsummary" 
			for ext in [".png"]:#".pdf",
				plt.savefig(fnout+ext, dpi=130)
			
			plotinfo = "PLOT INFO: SHAP plots made using %s:v.%s by %s, %s" % (
				__title__, __version__,  __author__, pd.Timestamp.now())
			gitinfo = cf.gitmetadata()
			cf.writemetadata(fnout, [plotinfo, gitinfo])
			plt.show()
			
			X_test2 = X_test.copy()
			X_test2.columns = dfin["VariableName"].values.tolist()
			
			fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(9.5,12))
			for ax, varsig in zip([ax1, ax2, ax3, ax4], dfin.sort_values("PermutationImportance", ascending=False)["VariableName"][:4]):
				shap.dependence_plot(varsig, shap_values, X_test2, ax=ax, show=False)
			# shap.dependence_plot("ObsGap", shap_values, X_test, ax=ax2, show=False)
			plt.tight_layout()
			# ========== Save the plot ==========
			print("starting save at:", pd.Timestamp.now())
			fnout = f"{ppath}PS08_{exp}_{var}_SHAPpartialDependance" 
			for ext in [".png"]:#".pdf",
				plt.savefig(fnout+ext, dpi=130)
			plt.show()
			breakpoint()
			if plotind:

				explainer2   = shap.Explainer(model, X_test2)
				shap_values2 = explainer2(X_test2)
				shap.plots.waterfall(shap_values2[0])
				shap.plots.waterfall(shap_values2[1000], show=False)
				plt.tight_layout()
				print("starting save at:", pd.Timestamp.now())
				fnout = f"{ppath}PS08_{exp}_{var}_SHAPexamplepixel" 
				for ext in [".png"]:#".pdf",
					plt.savefig(fnout+ext, dpi=130)
				plt.show()
				breakpoint()


		# ========== Group them together ==========
		df = pd.concat(df_list).reset_index(drop=True)
		df["Count"] = df.groupby(["experiment", "VariableName"])[var].transform("count")

		print(df["VariableName"].unique().size)
		
		breakpoint()

	"""
	IDEA:
	Combine SPAP Plot
	-  norm all the predictors by columns so i can group and simplify variables
	"""
	# df = pd.melt(df.drop(var, axis=1) , 
	# 	id_vars=['Variable','experiment','version', 'Count'], value_vars=['Loss', 'Gain'],
	# 	var_name='ChangeDirection', value_name=var)
	
	# # ========== Calculate the counts ==========
	# # df.sort_values(['experiment', 'Count'], ascending=[True, False], inplace=True)
	# # df.reset_index(drop=True,inplace=True)
	# df["ChangeDirection"] = df["ChangeDirection"].astype("category").cat.set_categories(["Loss", "Gain"])

	# fnout = f"{path}{exp}/Exp{exp}_PermutationImportance_lossgain_PS08.csv"
	# df.to_csv(fnout)

	# g = sns.boxplot(x="Variable", y="PermutationImportance", hue="ChangeDirection", data=df.loc[df.Count >= 5])
	# plt.show()


	breakpoint()

# ==============================================================================


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


def _getdata(path, exp, ColNm,
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
	X_train, X_test, y_train, y_test, col_nms, loadstats, corr, df_site, dbg = bf.datasplit(
		setup.loc["predvar"], exp, version,  branch, setup,  cols_keep=ColNm, final=True, #force=True,
		vi_fn=fnamein, region_fn=sfnamein, basestr=basestr, dropvar=setup.loc["dropvar"], column_retuner=True)

	return X_train, X_test, y_train, y_test, col_nms, loadstats, corr, df_site, dbg

# ==============================================================================
if __name__ == '__main__':
	main()