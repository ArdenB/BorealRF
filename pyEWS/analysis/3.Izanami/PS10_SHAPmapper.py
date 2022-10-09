"""
Script for dealing with permutation importance and SHAP values
"""

# ==============================================================================

__title__ = "SHAP mapper"
__author__ = "Arden Burrell"
__version__ = "v1.0(13.09.2022)"
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
import matplotlib.colors as colors

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
	ppath = "./pyEWS/analysis/3.Izanami/Figures/PS10/"
	cf.pymkdir(ppath)

	# ========== Chose the experiment ==========
	exps      = [434, 424]
	for exp in exps:
		shapdf, obs, perm = _ImpOpener(path, ppath, exp)
		# var       = '(st) Initial Biomass'

		# varls     = ['(st) Initial Biomass', "(Cli.) TD mean"]
		# varls =  shapdf.iloc[:, 2:-2].abs().mean().sort_values(ascending=False).index.to_list() [:20]
		varls = perm.index.to_list()#[20:]# + ['(Cli.) MWMT trend']
		# breakpoint()
		for num,  var in enumerate(varls): 
			# if num < 20:
				# continue
			print(var)
			ds = gridder(exp, shapdf, obs, var)
			mapmaker(ppath, exp, var, ds, num)
			# sys.exit()
		breakpoint()


	breakpoint()
def mapmaker(ppath, exp, var, ds, num, textsize=14, 
	lats = np.arange(  42,  70.1,  0.5), lons = np.arange(-170, -50.1,  0.5)):
	""" Function to make dual panel maps with the observed and SHAP 
	values
	"""

	# ========== Create the figure ==========
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':textsize})
	font = ({'weight' : 'bold', 'size'   : textsize})
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	plt.rcParams.update({'axes.titleweight':"bold", "axes.labelweight":"bold"})

	# ========== Create the mapp projection ==========
	map_proj = ccrs.LambertConformal(central_longitude=lons.mean(), central_latitude=lats.mean())

	# ========== Create the figure ==========
	fig  = plt.figure(constrained_layout=True, figsize=(14, 12))
	spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)#, width_ratios=[11,1,11,1])

	# ========== make the obs value plot ========== 
	ax1 = fig.add_subplot(spec[0, 0], projection= map_proj)
	_simplemapper(ds, var, fig, ax1, map_proj, SHAP=False)


	# ========== make the SHAP value plot ==========
	ax2 = fig.add_subplot(spec[1, 0], projection= map_proj)
	_simplemapper(ds, var, fig, ax2, map_proj, SHAP=True)

	print("starting save at:", pd.Timestamp.now())
	# ========== create the short name by remving things that might break a save ==========
	svar = var.translate({ord('('):None, ord(')'):None,ord(' '):None, ord('.'):None, })
	fnout = f"{ppath}PS10_Suppfigure_SHAPmap_{exp}_rank{num:02}_{svar}" 
	for ext in [".png", ".pdf"]:#".pdf",
		plt.savefig(fnout+ext)#, dpi=130)
	
	plotinfo = "PLOT INFO: Paper figure made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()
	# breakpoint()

def _simplemapper(ds, var, fig, ax, map_proj, SHAP=False,
	lats = np.arange(  42,  70.1,  0.5), lons = np.arange(-170, -50.1,  0.5)):
	# ========== pull in the var specif plot params ==========

	# ========== Make the actual plot ==========
	if SHAP:
		f = ds[f"Mean SHAP {var}"].plot(
			x="longitude", y="latitude", #col="time", col_wrap=2, 
			transform=ccrs.PlateCarree(), 
			cbar_kwargs={"pad": 0.015, "shrink":0.65,},
			ax=ax)
		ax.set_title(f"b) {var} SHAP", loc= 'left')
	else:
		plotky = keywords(var)
		f = ds[f"Mean {var}"].plot(
			x="longitude", y="latitude", #col="time", col_wrap=2, 
			transform=ccrs.PlateCarree(), 
			cbar_kwargs =plotky["cbar_kwargs"][0],
			vmin        =plotky["vmin"],
			vmax        =plotky["vmax"],
			ax=ax)
		ax.set_title(f"a) {var}", loc= 'left')
	
	ax.set_extent([lons.min()+15, lons.max()-3, lats.min()-3, lats.max()-6])
	ax.gridlines(alpha=0.5)
	ax.stock_img()

	coast = cpf.GSHHSFeature(scale="intermediate")
	ax.add_feature(cpf.LAND, facecolor='dimgrey', alpha=1, zorder=0)
	ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
	ax.add_feature(coast, zorder=101, alpha=0.5)
	ax.add_feature(cpf.BORDERS, linestyle='--', zorder=104)

	provinc_bodr = cpf.NaturalEarthFeature(category='cultural', 
		name='admin_1_states_provinces_lines', scale='50m', facecolor='none', edgecolor='k')
	ax.add_feature(provinc_bodr, linestyle='--', linewidth=0.6, edgecolor="k", zorder=105)

def keywords(var):
	# pass
	plotky = OrderedDict()
	plotky["cbar_kwargs"] ={"pad": 0.015, "shrink":0.65, },
	plotky["vmin"]        = None
	plotky["vmax"]        = None
	# "extend":"max"
	if var == "(st) Initial Biomass":
		plotky["vmax"] = 400
		plotky["cbar_kwargs"][0]["extend"]="max"
	elif var == "(st) Stem Density":
		plotky["vmax"] = 4000
		plotky["cbar_kwargs"][0]["extend"]="max"
	elif var == "(Cli.) MSP mean":
		plotky["vmax"] = 1000
		plotky["cbar_kwargs"][0]["extend"]="max"



	return plotky

		

def gridder(exp, shapdf, obs, var, 
	lats = np.arange(  42,  70.1,  0.5), lons = np.arange(-170, -50.1,  0.5)):
	"""
	function to take the input files and return two netcdf files with the SHAP
	and observed data value
	"""
	# ========== pull out the variable from the dataframes ==========
	dashap = shapdf.groupby(["latitude", "longitude"])[var].mean().to_xarray().sortby("latitude", ascending=False)
	daobs  = obs.groupby(["latitude", "longitude"])[var].mean().to_xarray().sortby("latitude", ascending=False)

	# ========== Add units ==========
	dashap.attrs["units"] = r"t $ha^{-1}$"
	# breakpoint()
	# dscount  = dfC.groupby(["time","latitude", "longitude", "Version"])[var].count().to_xarray().sortby("latitude", ascending=False)

	ds = xr.Dataset({
		f"Mean SHAP {var}":dashap, 
		f"Mean {var}":daobs})
	return ds

def _ImpOpener(path, ppath, exp, var = "PermutationImportance", AddFeature=False, 
	textsize=14, plotSHAP=True, plotind=False, smodel=False, force=False):
	"""
	Function to open the feature importance files and return them as a single 
	DataFrame"""
	sns.set_style("whitegrid")
	font = ({'weight' : 'bold', 'size'   : textsize})
	mpl.rc('font', **font)
	plt.rcParams.update({'axes.titleweight':"bold", 
		"axes.labelweight":"bold", 'axes.titlesize':textsize, 'axes.titlelocation': 'left',}) 

	# ========== Loop over the exps ==========
	
	# for exp in exps:
	fno = f"{ppath}Exp{exp}_SHAPvalues.csv"
	fns = f"{ppath}Exp{exp}_OBSvalues.csv"
	fnp = f"{ppath}Exp{exp}_PERMvalues.csv"

	if not all([os.path.isfile(fn) for fn in [fno, fns, fnp]]) and not force:
		# breakpoint()
		SHAPlst = []
		df_list = []
		ypredl  = []
		X_testl = []
		expcted = []
		

		fnames = sorted(glob.glob(f"{path}{exp}/Exp{exp}_*PermutationImportance.csv"))

		for ver, fn in enumerate(fnames):
			print(f"exp {exp} ver {ver}")

			# ========== load the model ==========
			dfin = pd.read_csv( fn, index_col=0)
			dfin["experiment"] = exp
			dfin["version"]    = ver
			vnames = Smartrenamer(dfin.Variable.values)
			dfin["VariableName"]  = vnames.VariableGroup
			df_list.append(dfin)
			# breakpoint()


			if smodel:
				if (not ver == 2) or (not plotSHAP):
					warn.warn(f"Skipping SHAP in ver:{ver}")
					continue

			fn_mod = f"{path}{exp}/models/XGBoost_model_exp{exp}_version{ver}.dat"
			model  = pickle.load(open(f"{fn_mod}", "rb"))
			ColNm  = dfin["Variable"].values
			try:
				X_train, X_test, y_train, y_test, col_nms, loadstats, corr, df_site, dbg = _getdata(path, exp, ColNm)
			except Exception as err:
				print(str(err))
				breakpoint()
				break
			y_pred = model.predict(X_test)
			ypredl.append(y_pred)

			# ========== calculate the SHAP values ==========
			explainer   = shap.TreeExplainer(model)
			expcted.append(explainer.expected_value)

			shap_values = explainer.shap_values(X_test)
			# ===== add in the lat and lon =====
			clnm = ["Longitude", "Latitude"]+vnames.VariableGroup.tolist()
			LL = np.hstack((df_site.loc[X_test.index, ["Longitude", "Latitude"]].values, shap_values))

			dfr = pd.DataFrame(LL, columns=clnm)
			dfx = X_test.set_axis(vnames.VariableGroup.tolist(), axis=1)

			dfx = pd.concat([df_site.loc[X_test.index, ["Longitude", "Latitude"]], dfx], axis=1)
			
			# +++++ Find places with unique columns +++++
			if not  vnames.VariableGroup.unique().shape[0] == vnames.VariableGroup.shape[0]: 
				uq, ct = np.unique(vnames.VariableGroup.tolist(), return_counts=True)
				if np.sum(ct > 1)>1:
					breakpoint()
				
				dfr = dfr.loc[:, ~dfr.columns.duplicated()]
				dfx = dfx.loc[:, ~dfx.columns.duplicated()]
				# # dfr[uq[ct>1]] = dfr[uq[ct>1]].mean(axis=1)
				# dfr[uq[ct>1]].mean(axis=1)

			# breakpoint()
			SHAPlst.append(dfr)
			X_testl.append(dfx)

		result  = pd.concat(SHAPlst, ignore_index=True, sort=False)
		X_tt    = pd.concat(X_testl, ignore_index=True, sort=False)

		# ========== make a perm importance rank order ==========
		permdf  = pd.concat(df_list, ignore_index=True, sort=False).drop("experiment", axis=1)
		perm = permdf.groupby(["VariableName", "version"]).sum().reset_index().groupby(["VariableName"]).mean()#
		perm = perm.sort_values("PermutationImportance", ascending=False)#.drop("version", axis=1).reset_index()
		# y_preds = np.hstack(ypredl)#.tolist()

		# breakpoint()
		result.to_csv(fno)
		X_tt.to_csv(fns)
		perm.to_csv(fnp)

	else:
		result = pd.read_csv(fno, index_col=0)
		X_tt   = pd.read_csv(fns, index_col=0)
		perm   = pd.read_csv(fnp, index_col=0)

	# ========== Simple lons and lats ========== 
	lons = np.arange(-170, -50.1,  0.5)
	lats = np.arange(  42,  70.1,  0.5)

	for df in [result, X_tt]:
		df["longitude"] = pd.cut(df["Longitude"], lons, labels=bn.move_mean(lons, 2)[1:])
		df["latitude"]  = pd.cut(df["Latitude" ], lats, labels=bn.move_mean(lats, 2)[1:])

	return result, X_tt, perm
		# breakpoint()




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
				# breakpoint()
				VN = VN.replace("abs_trend", "trend")
			
			# if "_trend_" in VN:
			# 	breakpoint()
			
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