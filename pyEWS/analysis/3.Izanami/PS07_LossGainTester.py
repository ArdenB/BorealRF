"""
Loss Gain tester
"""

# ==============================================================================

__title__ = "Script to compare Loss prediction performance"
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

import cartopy.feature as cpf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.linear_model import LinearRegression, TheilSenRegressor

# ========== Import ml packages ==========
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.inspection import permutation_importance
from sklearn import metrics as sklMet
import string
# from sklearn.utils import shuffle
# from scipy.stats import spearmanr
# from scipy.cluster import hierarchy


# ==============================================================================

def main():
	# ========== Setup the pathways ==========
	formats = None
	path  = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"
	cf.pymkdir(path+"plots/")
	ppath = "./pyEWS/analysis/3.Izanami/Figures/PS07/"
	cf.pymkdir(ppath)
	expname = {434:"Site", 424:"Endpoint"}
	os.environ["CARTOPY_USER_BACKGROUNDS"] = "./data/Background/"

	# ========== Simple lons and lats ========== 
	lons = np.arange(-170, -50.1,  0.5)
	lats = np.arange(  42,  70.1,  0.5)
	FWH = True
	# ========== Chose the experiment ==========
	exps      = [434, 424]
	out    = []
	dfls   = []
	dflist = []
	dsdict = {}
	for exp in exps: #, False
		# ========== Pull out the prediction ==========
		dfout, dfl = fpred(path, exp, FWH=FWH)#, bins=[0,4,5,6,9,10,11,14,15,16,19, 20, 40])
		out.append(dfout)
		# breakpoint()
		dfl["GainLoss"] = (dfl.Observed >=0).astype(float) - (dfl.Observed <0).astype(float)
		dfls.append(dfl)
		
		df  = futpred(path, exp, [2020], lats, lons)
		dflist.append(df)

		# ========== Convert to a dataarray ==========
		ds = gridder(path, exp, [2020], df, lats, lons)
		dsdict[exp] = ds

	dfp    = pd.concat(dflist)
	dfloss = pd.concat(dfls)
	dfloss["Ensemble"] = dfloss["experiment"].replace(expname)
	# breakpoint()


	df_LS, ds_LS, metric = lsloader(dfp, 2020, metric='ftrend')
	agreement(path, ppath, dfout, dfl, FWH, df, ds, lats, lons, df_LS, ds_LS, dfp, dsdict, dfloss,)
	breakpoint()
	# try:
	FigureAgree(path, ppath, dfout, df_LS, ds_LS, dfp, dsdict, dfloss, lats, lons, metric)
	# except Exception as e:
	# 	print(str(er))
	# 	breakpoint()






	# ========== Plot the loss performance ==========
	losspotter(path, ppath, dfout, dfl, FWH)
	breakpoint()


# ==============================================================================
def FigureAgree(path, ppath, dfout, df_LS, ds_LS, dfp, dsdict, dfloss, lats, lons, metric,
	lgfs = 12, loc='lower right', ltitle = "Obs. AGB Change"):
	"""
	Make the final plot for the publication that has both models 
	"""


	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':16, "axes.labelweight":"bold", 'axes.titlelocation': 'left'})
	font = {'family' : 'normal',
	        'weight' : 'bold', #,
	        'size'   : 16}
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	os.environ["CARTOPY_USER_BACKGROUNDS"] = "./data/Background/"

	cmap = mpc.ListedColormap(palettable.colorbrewer.diverging.PiYG_11.mpl_colors)
	

	# ========== Create the mapp projection ==========
	map_proj = ccrs.LambertConformal(central_longitude=lons.mean(), central_latitude=lats.mean())


	fig  = plt.figure(constrained_layout=True, figsize=(10,13))

	spec = gridspec.GridSpec(ncols=2, nrows=4, figure=fig)
	ax0 = fig.add_subplot(spec[0, :])
	sns.lineplot(y="GainLoss", x="ModAgree", hue="Ensemble", data=dfloss, ci=99, ax=ax0)
	ax0.set_yticks(np.arange(-1, 1.1, 0.2))
	ax0.set_xticks(np.arange(-1, 1.1, 0.2))
	ax0.set_title(f"a)", loc= 'left')
	ax0.set_ylabel("Obs. AGB Gain and Loss")
	ax0.set_xlabel("Ensemble Agreement")
	ax0.legend(loc=loc, fontsize=lgfs, title_fontsize=lgfs)# title=ltitle,

	#breakpoint()
	ax1 = fig.add_subplot(spec[1, :], projection= map_proj)
	f = ds_LS["sitesInc"].isel(time=0).plot(
		x="longitude", y="latitude", 
		transform=ccrs.PlateCarree(), 
		cbar_kwargs={"pad": 0.015, "shrink":0.85, "label": r"$\Delta$NDVImax Direction"},
		cmap=cmap, #size =8,
		ax=ax1
		)
	# ========== create the second axiss ==========
	ax2 = fig.add_subplot(spec[2, :], projection= map_proj)
	f = dsdict[434]["EnsembleDirection"].isel(time=0).mean(dim="Version").plot(
		x="longitude", y="latitude", 
		transform=ccrs.PlateCarree(), 
		cbar_kwargs={"pad": 0.015, "shrink":0.85, "label": r"$\Delta$AGB Direction"},
		cmap=cmap, #size =8,
		ax=ax2
		)

	ax3 = fig.add_subplot(spec[3, :], projection= map_proj)
	f = dsdict[424]["EnsembleDirection"].isel(time=0).mean(dim="Version").plot(
		x="longitude", y="latitude", 
		transform=ccrs.PlateCarree(), 
		cbar_kwargs={"pad": 0.015, "shrink":0.85, "label": r"$\Delta$AGB Direction"},
		cmap=cmap, #size =8,
		ax=ax3
		)
	
	for ax, tit in zip([ax1, ax2, ax3], ["b) Landsat", "c) SWE", "d) EWE"]):
		ax.set_extent([lons.min()+15, lons.max()-3, lats.min()-5, lats.max()-10])
		ax.background_img(name='BM', resolution='low')
		ax.gridlines()
		coast = cpf.GSHHSFeature(scale="intermediate")
		ax.add_feature(cpf.LAND, facecolor='dimgrey', alpha=1, zorder=0)
		ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
		ax.add_feature(coast, zorder=101, alpha=0.5)
		ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
		ax.add_feature(cpf.RIVERS, zorder=104)
		ax.add_feature(cpf.BORDERS, linestyle='--', zorder=102)
		ax.set_title(f"{tit}", loc= 'left')

	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS07_PaperFig05_EnsenbleAgreement_{metric}" 
	for ext in [".png",".pdf"]:
		plt.savefig(fnout+ext, dpi=300)
	
	plotinfo = "PLOT INFO: Ensemble Agreement made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()
	breakpoint()


def lsloader(dfp, year, metric = "ndvi.max"):
	"""
	Function 
	"""
	lsdf = pd.read_csv("./data/NDVI/6.Landsat/lsat_nam_psps_ndvi_timeseries_20210628.csv")
	fnout = f"./data/NDVI/6.Landsat/LS_ProcesessedtoPSP_{metric}.csv"
	if not metric in lsdf.columns:
		submet = "ndvi.max"
	else:
		submet = metric
	if not os.path.isfile(fnout):
	# breakpoint()
		dfpt = dfp.groupby(["Plot_ID", "year"]).first().reset_index()[['Plot_ID', 'Longitude', 'Latitude', 'Region', 'year', 'time', 'longitude', 'latitude']].dropna()
		OGndvi = np.zeros(dfpt.Plot_ID.size)
		CTndvi = np.zeros(dfpt.Plot_ID.size)
		
		OGndvi [:] = np.nan
		CTndvi [:] = np.nan
		
		for (num, (index, row)) in tqdm(enumerate(dfpt.iterrows()), total=dfpt.shape[0]):
			try:
				if row.year<1984:
					continue

				sub = lsdf.loc[lsdf.site == row.Plot_ID, ["year",submet]]
				if (not year in sub.year.values) or (not row.year in sub.year.values):
					continue

				if submet == metric:
					OGndvi[num] = sub.loc[sub.year == row.year, metric].values
					CTndvi[num] = sub.loc[sub.year == year, metric].values
				else:
					if metric == "Trend":
						x = sub[sub.year >= row.year].year.values
						X = x[:, np.newaxis]
						y = sub[sub.year >= row.year][submet].values
					elif metric == "ftrend":
						x = sub.year.values
						X = x[:, np.newaxis]
						y = sub[submet].values
					elif metric == "f7trend":
						x = sub[sub.year >= 2000].year.values
						X = x[:, np.newaxis]
						y = sub[sub.year >= 2000][submet].values
					reg = TheilSenRegressor().fit(X, y)
					OGndvi[num] = reg.coef_[0]
					CTndvi[num] = reg.score(X, y)

					# breakpoint()
			except Exception as er:
				warn.warn(str(er))
				breakpoint()

		if submet == metric:
			dfpt["Initial NDVI"] = OGndvi
			dfpt["Final NDVI"]   = CTndvi
			dfpt["Delta NDVI"]   = OGndvi-CTndvi
			dfpt.dropna(inplace=True)
			dfpt["ChangeDirection"] = ((dfpt["Delta NDVI"]> 0)*2 -1)
		else:
			dfpt["NDVI Trend"] = OGndvi
			dfpt["R2 score"]   = CTndvi

			dfpt.dropna(inplace=True)
			dfpt["ChangeDirection"] = ((dfpt["NDVI Trend"]> 0)*2 -1)
			# breakpoint()

		dfpt.to_csv(fnout)
	else:
		dfpt = pd.read_csv(fnout, index_col=0)
	

	dscount  = dfpt.groupby(["time","latitude", "longitude"])["ChangeDirection"].count().to_xarray().sortby("latitude", ascending=False)
	dscount  = dscount.where(dscount>0)
	dsp      = dfpt.loc[dfpt["ChangeDirection"]> 0].groupby(["time","latitude", "longitude",])["ChangeDirection"].count().to_xarray().sortby("latitude", ascending=False)
	dsn      = dfpt.loc[dfpt["ChangeDirection"]<=0].groupby(["time","latitude", "longitude",])["ChangeDirection"].count().to_xarray().sortby("latitude", ascending=False)
	dspos    = (dsp-dsn)/dscount
	dspos.attrs["units"] = "Fraction of sites Increasing"

	ds = xr.Dataset({
		"sites":dscount, 
		"sitesInc":dspos, 
		})
		# f"Mean{var}":dsmean, 
		# f"Median{var}":dsmedian, 
		# f"AnnualMeanBiomass":dsannual,
		# "EnsembleDirection":dschange
	
	return dfpt, ds, metric




def agreement(path, ppath, dfout, dfl, FWH, df, ds, lats, lons,  
	df_LS, ds_LS, dfp, dsdict, dfloss,  modnum=10, lgfs = 12, loc='lower right', ltitle = "Obs. AGB Change"):

	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':16, "axes.labelweight":"bold", 'axes.titlelocation': 'left'})
	font = {'family' : 'normal',
	        'weight' : 'bold', #,
	        'size'   : 16}
	mpl.rc('font', **font)
	sns.set_style("whitegrid")

	cmap = mpc.ListedColormap(palettable.colorbrewer.diverging.PiYG_11.mpl_colors)
	cbkw = {"pad": 0.015, "shrink":0.85, "label": r"$\Delta$AGB Direction"}

	# ========== Create the mapp projection ==========
	# map_proj = ccrs.LambertConformal(central_longitude=lons.mean(), central_latitude=lats.mean())
	dfloss.replace({"Site":"SWE", "Endpoint":"IWE"}, inplace=True)

	# ========== Create the figure ==========
	fig  = plt.figure(constrained_layout=True, figsize=(12,4))
	spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

	ax0 = fig.add_subplot(spec[0, :])
	# Safe_2
	# palettable.cartocolors.qualitative.Safe_2.hex_colors
	sns.lineplot(y="GainLoss", x="ModAgree", hue="Ensemble", data=dfloss, 
		ci=99, ax=ax0, palette= palettable.cartocolors.qualitative.Safe_2.hex_colors)
	# ========== hack workaround to fix alpha ==========
	ax0.collections[-1].set_alpha(0.3)
	ax0.collections[-2].set_alpha(0.5)
	ax0.set_yticks(np.arange(-1, 1.1, 0.2))
	ax0.set_xticks(np.arange(-1, 1.1, 0.2))
	# ax0.set_title(f"a)", loc= 'left')
	ax0.set_ylabel("Obs. AGB Gain and Loss")
	ax0.set_xlabel("Ensemble Agreement")
	ax0.legend(loc=loc, fontsize=lgfs, title_fontsize=lgfs)# title=ltitle,
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS07_PaperFig05_EnsenbleStats_agreement" 
	for ext in [".png", ".pdf",]:
		plt.savefig(fnout+ext, dpi=300)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()
	# breakpoint()


	# ========== Create the figure ==========
	fig  = plt.figure(constrained_layout=True, figsize=(12,2*4))
	spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)

	# ax0 = fig.add_subplot(spec[0, :])

	# breakpoint()
	dfloss["AGB Direction"] = (dfloss["Observed"] > 0).replace({True:"Gain", False:"Loss"})
	# dfloss["Ens. Direction"] = dfloss["Ensemble"] + " Ens. " + dfloss["Gain"].replace({True:"Gain", False:"Loss"})
	for nx, ves in enumerate(["SWE", "IWE"]):
		ax1 = fig.add_subplot(spec[nx, :])
		sns.lineplot(y="ModScore", x="ObsGap", hue="AGB Direction", data=dfloss[dfloss.Ensemble == ves], ci=99, ax=ax1)
		ax1.set_yticks(np.arange(0, 1.01, 0.1))
		ax1.set_xticks(np.arange(0, 31, 5))

		ax1.set_title(f"{string.ascii_lowercase[nx]}) {ves}", loc= 'left')
		ax1.set_ylabel("Direction Accuracy")
		ax1.set_xlabel("Observation Gap")
		ax1.legend(loc=loc, fontsize=lgfs, title_fontsize=lgfs, title= r"Obs. $\Delta$AGB")#
		
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS07_PaperFig05_EnsenbleStats_agreementmulti" 
	for ext in [".png", ".pdf",]:
		plt.savefig(fnout+ext, dpi=300)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()
	breakpoint()

	# ax1 = fig.add_subplot(spec[1, :], projection= map_proj)
	# f = dsdict[434]["EnsembleDirection"].isel(time=0).mean(dim="Version").plot(
	# 	x="longitude", y="latitude", 
	# 	transform=ccrs.PlateCarree(), 
	# 	cbar_kwargs=cbkw,
	# 	cmap=cmap, #size =8,
	# 	ax=ax1
	# 	)
	# ax1.set_extent([lons.min()+15, lons.max()-3, lats.min()-5, lats.max()-10])
	# ax1.background_img(name='BM', resolution='low')
	# # ax1.set_extent([lons.min()+10, lons.max()-5, lats.min()-13, lats.max()])
	# ax1.gridlines()
	# coast = cpf.GSHHSFeature(scale="intermediate")
	# ax1.add_feature(cpf.LAND, facecolor='dimgrey', alpha=1, zorder=0)
	# ax1.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
	# ax1.add_feature(coast, zorder=101, alpha=0.5)
	# ax1.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
	# ax1.add_feature(cpf.RIVERS, zorder=104)
	# ax1.add_feature(cpf.BORDERS, linestyle='--', zorder=102)
	# ax1.set_title(f"b) SWE", loc= 'left')


	


def losspotter(path, ppath, dfout, dfl, FWH, modnum=10):
	"""
	plots the loss performance
	"""
	# ========== Create the matplotlib params ==========
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':16, "axes.labelweight":"bold", 'axes.titlelocation': 'left'})
	font = {'family' : 'normal',
	        'weight' : 'bold', #,
	        'size'   : 16}
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	# dfl["Obsloss"] = dfl["Obsloss"].astype(float)

	dfl["percMLoss"] = dfl["ModLossN"]/modnum
	dfl.rename(columns={"Obsloss":"LossProb", "GapGroup":"PredictionWindow","GapQuant":"QPredictionWindow"}, inplace=True)
	# fig = sns.lineplot(y="LossProb", x="ModLossN", hue="PredictionWindow", data=dfl)#, ci="sd")
	# # dfg = dfl.groupby(["GapGroup", "ModLossN"])["LossProb"].mean().reset_index()
	# # breakpoint()
	# fig.set_yticks(np.arange(0, 1.1, 0.1))
	# fig.set_xticks(np.arange(0, 10, 1))
	# plt.show()
	

	# fig = sns.lineplot(y="LossProb", x="ModLossN", hue="QPredictionWindow", data=dfl)
	# fig.set_yticks(np.arange(0, 1.1, 0.1))
	# fig.set_xticks(np.arange(0, 10, 1))
	# plt.show()
	# ============ Model Counts ==========
	fig, ax = plt.subplots(constrained_layout=True, figsize=(13,11))
	sns.lineplot(y="LossProb", x="percMLoss", data=dfl, ci=99, ax=ax)
	ax.set_yticks(np.arange(0, 1.1, 0.1))
	ax.set_xticks(np.arange(0, 1.1, 0.1))
	ax.set_xlabel(f"% Models Predicting Loss")
	ax.set_ylabel(f"% of sites where loss is observed")
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS07_PaperFig02_EnsenbleStats_fract" 
	for ext in [".png", ".pdf",]:
		plt.savefig(fnout+ext, dpi=300)
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()


	# ============ Model Counts ==========
	fig = sns.lineplot(y="LossProb", x="ModLossN", data=dfl, ci=99)
	fig.set_yticks(np.arange(0, 1.1, 0.1))
	fig.set_xticks(np.arange(0, 10, 1))
	fig.set_xlabel(f"no. Models Predicting Loss")
	fig.set_ylabel(f"% of sites where loss is observed")
	print("starting save at:", pd.Timestamp.now())
	fnout = f"{ppath}PS07_PaperFig02_EnsenbleStats" 
	for ext in [".png", ".pdf",]:
		try:
			plt.savefig(fnout+ext, dpi=300)
		except Exception as err:
			warn.warn(str(err))
	
	plotinfo = "PLOT INFO: Multimodel confusion plots Comparioson made using %s:v.%s by %s, %s" % (
		__title__, __version__,  __author__, pd.Timestamp.now())
	gitinfo = cf.gitmetadata()
	cf.writemetadata(fnout, [plotinfo, gitinfo])
	plt.show()

	breakpoint()
# ==============================================================================

def fpred(path, exp, fpath = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/", 
	nanthresh=0.5, drop=True, bins=[0, 5, 10, 15, 20, 40], qbins=8, FWH=True):
	"""
	function to predict future biomass
	args:
	path:	str to files
	exp:	in of experiment
	years:  list of years to predict 
	"""
	# warn.warn("\nTo DO: Implemnt obsgap filtering")
	
	OvP_fnames = glob.glob(path + f"{exp}/Exp{exp}*_OBSvsPREDICTED.csv")
	df_OvsP    = pd.concat([load_OBS(ofn) for ofn in OvP_fnames], sort=True).reset_index()

	dfpm    = df_OvsP.pivot(index='index', columns='version', values='Estimated').dropna()
	# dfob    = df_OvsP.pivot(index='index', columns='version', values='Observed').dropna()
	
	# ========== Load the variables ==========
	setup   = pd.read_csv(f"{path}{exp}/Exp{exp}_setup.csv", index_col=0)
	site_df = pd.read_csv(f"{fpath}SiteInfo_AllSampleyears_ObsBiomass.csv", index_col=0)
	vi_df   = pd.read_csv(f"{fpath}VI_df_AllSampleyears_ObsBiomass.csv", index_col=0)
	pvar    = setup.loc["predvar"].values[0]

	if type(pvar) == float:
		# deal with the places i've alread done
		pvar = "lagged_biomass"


	# ========== Loop over the model versions
	est_list = []

	for ver in tqdm(range(10)):

		fn_mod = f"{path}{exp}/models/XGBoost_model_exp{exp}_version{ver}.dat"
		if not os.path.isfile(fn_mod):
			# missing file
			continue
		# ========== Load the run specific params ==========
		model  = pickle.load(open(f"{fn_mod}", "rb"))
		fname  = glob.glob(f"{path}{exp}/Exp{exp}_*_vers{ver:02}_PermutationImportance.csv")
		feat   = pd.read_csv(fname[0], index_col=0)["Variable"].values

		# ========== Make a dataframe ==========
		dfout = site_df.loc[:, ["Plot_ID", "Longitude", "Latitude", "Region", "year"]].copy()
		dfout["Version"]  = ver
		dfout["Original"] = vi_df["biomass"].values
		dfout["Observed"] = vi_df[pvar]

		dfoutC = dfout.copy()

		# ========== pull out the variables and apply transfors ==========
		dfX = vi_df.loc[:, feat].copy()
		if not type(setup.loc["Transformer"].values[0]) == float:
			warn.warn("Not implemented yet")
			breakpoint()

		# ========== Pull out a NaN mask ==========
		nanmask = ((dfX.isnull().sum(axis=1) / dfX.shape[1] ) > nanthresh).values


		# ========== Perform the prediction ==========
		est = model.predict(dfX.values)
		est[nanmask] = np.NaN
		# breakpoint()
		if not type(setup.loc["yTransformer"].values[0]) == float:
			warn.warn("Not implemented yet")
			breakpoint()

		# ========== Convert to common forms ==========
		if pvar == "lagged_biomass":
			breakpoint()
		elif pvar == 'Delta_biomass':
			dfoutC[f"Biomass"]      = vi_df["biomass"].values + est
			dfoutC[f"DeltaBiomass"] = est
			# breakpoint()
		elif pvar == 'Obs_biomass':
			# dfout[f"BIO_{yr}"]   = est
			# dfout[f"DELTA_{yr}"] = est - vi_df["biomass"].values
			dfoutC[f"Biomass"]      = est
			dfoutC[f"DeltaBiomass"] = est - vi_df["biomass"].values
		
		dfoutC["ObsGap"] = dfX["ObsGap"] 
		dfoutC["CorrectDirection"] = (dfoutC[f"Observed"]>0) == (dfoutC[f"DeltaBiomass"]>0)
		# breakpoint()
		# dfoutC.loc[(dfoutC.time.dt.year - dfoutC.year) > maxdelta, ['Biomass', 'DeltaBiomass']] = np.NaN
		est_list.append(dfoutC)


	dft = pd.concat(est_list).dropna().reset_index().rename(columns={'index': 'rownum'})
	dft["PredCount"] = dft.groupby("rownum")['DeltaBiomass'].transform('count')

	if drop:
		dft = dft.loc[dft["PredCount"] == dft["PredCount"].max(), ].reset_index(drop=True)
		# breakpoint()
		# dfoutC = dfoutC.loc[dfoutC.year > (yr - maxdelta)].dropna()
	if FWH:
		dft.set_index("rownum", inplace=True)
		dft = dft.loc[dfpm.index].reset_index()
	
	# ========== Pull in some landsat trend results ==========
	# fnout = f"./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/{exp}/Exp{exp}_MODISandLANDSATndvi_delta.csv"
	# dfin = pd.read_csv(fnout, index_col=0)
	# dfi = dfin.groupby("Plot_ID").first()

	# breakpoint()

	# ========== get the number of models that are cexreasing ==========
	dftest = dft.loc[:, ["ObsGap", "rownum", "Observed", 'DeltaBiomass', "CorrectDirection"]].copy()
	# dftest["BiomassLoss"]
	# dftest["LossModNum"]
	# breakpoint()
	dfl = dftest.groupby("rownum").agg(
		ObsGap   = ('ObsGap', 'mean'), 
		Observed = ('Observed', 'mean'),  	
		Obsloss  = ('Observed', lambda x: float((x.mean() < 0))), 
		ModLossN = ("DeltaBiomass", lambda x: (x < 0).sum()), 
		ModAgree = ("DeltaBiomass", lambda x: ((x >= 0).sum() - 5)/5), 
		ModMean  = ('DeltaBiomass', 'mean'),  
		ModScore = ('CorrectDirection', 'mean'),  
		)
	dfl["GapGroup"] = pd.cut(dfl["ObsGap"], bins)#, include_lowest=True)#, labels=np.arange(5, 45, 5))
	dfl["GapQuant"] = pd.qcut(dfl["ObsGap"], qbins, duplicates ="drop")
	dfl["experiment"] = exp
	dft["experiment"] = exp
	return dft, dfl

def load_OBS(ofn):
	df_in = pd.read_csv(ofn, index_col=0)
	df_in["experiment"] = int(ofn.split("/")[-2])
	df_in["experiment"] = df_in["experiment"].astype("category")
	df_in["version"]    = float(ofn.split("_vers")[-1][:2])
	return df_in


def gridder(path, exp, years, df, lats, lons, var = "DeltaBiomass"):

	# ========== Setup params ==========
	# plt.rcParams.update({'axes.titleweight':"bold","axes.labelweight":"bold", 'axes.titlesize':10})
	# font = {'family' : 'normal',
	#         'weight' : 'bold', #,
	#         'size'   : 10}
	# mpl.rc('font', **font)
	# sns.set_style("whitegrid")
	""" Function to convert the points into a grid """
	# ========== Copy the df so i can export multiple grids ==========
	dfC = df.copy()#.dropna()

	# breakpoint()

	if var == 'DeltaBiomass':
		dfC["AnnualBiomass"] = dfC[var] / dfC["ObsGap"]
	else:
		breakpoint()


	# ========== Convert the different measures into xarray formats ==========
	dscount  = dfC.groupby(["time","latitude", "longitude", "Version"])[var].count().to_xarray().sortby("latitude", ascending=False)
	dscount  = dscount.where(dscount>0)
	dsp      = dfC.loc[dfC["DeltaBiomass"]> 0].groupby(["time","latitude", "longitude", "Version"])[var].count().to_xarray().sortby("latitude", ascending=False)
	dsn      = dfC.loc[dfC["DeltaBiomass"]<=0].groupby(["time","latitude", "longitude", "Version"])[var].count().to_xarray().sortby("latitude", ascending=False)
	dspos    = (dsp-dsn)/dscount
	dspos.attrs["units"] = "Fraction of sites Increasing"
	
	dsmean   = dfC.groupby(["time","latitude", "longitude", "Version"])[var].mean().to_xarray().sortby("latitude", ascending=False)
	dsmedian = dfC.groupby(["time","latitude", "longitude", "Version"])[var].median().to_xarray().sortby("latitude", ascending=False)
	dsannual = dfC.groupby(["time","latitude", "longitude", "Version"])["AnnualBiomass"].mean().to_xarray().sortby("latitude", ascending=False)
	dschange = dfC.groupby(["time","latitude", "longitude", "Version"])["ModelAgreement"].mean().to_xarray().sortby("latitude", ascending=False)
	# dfC2     
	# breakpoint()
	# ========== Convert the different measures into xarray formats ==========
	ds = xr.Dataset({
		"sites":dscount, 
		"sitesInc":dspos, 
		f"Mean{var}":dsmean, 
		f"Median{var}":dsmedian, 
		f"AnnualMeanBiomass":dsannual,
		"EnsembleDirection":dschange})
	# breakpoint()
	return ds
	


def futpred(path, exp, years, lats, lons, var = "DeltaBiomass",
	fpath    = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/", 
	maxdelta = 30):
	"""
	function to predict future biomass
	args:
	path:	str to files
	exp:	in of experiment
	years:  list of years to predict 
	"""
	# warn.warn("\nTo DO: Implemnt obsgap filtering")
	# ========== Load the variables ==========
	site_df = pd.read_csv(f"{fpath}SiteInfo_AllSampleyears_FutureBiomass.csv", index_col=0)
	vi_df   = pd.read_csv(f"{fpath}VI_df_AllSampleyears_FutureBiomass.csv", index_col=0)
	setup   = pd.read_csv(f"{path}{exp}/Exp{exp}_setup.csv", index_col=0)
	pvar    = setup.loc["predvar"].values[0]
	if type(pvar) == float:
		# deal with the places i've alread done
		pvar = "lagged_biomass"

	# ========== Loop over the model versions
	est_list = []
	for ver in tqdm(range(10)):
		fn_mod = f"{path}{exp}/models/XGBoost_model_exp{exp}_version{ver}.dat"
		if not os.path.isfile(fn_mod):
			# missing file
			continue
		# ========== Load the run specific params ==========
		model  = pickle.load(open(f"{fn_mod}", "rb"))
		fname  = glob.glob(f"{path}{exp}/Exp{exp}_*_vers{ver:02}_PermutationImportance.csv")
		feat   = pd.read_csv(fname[0], index_col=0)["Variable"].values

		# ========== Make a dataframe ==========
		dfout = site_df.loc[:, ["Plot_ID", "Longitude", "Latitude", "Region", "year"]].copy()
		dfout["Version"]  = ver
		dfout["Original"] = vi_df["biomass"].values

		# ========== Make a filter for bad latitudes ==========
		# breakpoint()
		# dfout.loc[dfout["Longitude"] == 0, ["Longitude", "Latitude"]] = np.NaN
		# dfout.loc[dfout["Longitude"] < -180, ["Longitude", "Latitude"]] = np.NaN
		# dfout.loc[dfout["Longitude"] >  180, ["Longitude", "Latitude"]] = np.NaN
		# dfout.loc[dfout["Latitude"] <= 0, ["Longitude", "Latitude"]] = np.NaN
		# dfout.loc[dfout["Latitude"] >  90, ["Longitude", "Latitude"]] = np.NaN

		# breakpoint()
		for yr in years:
			dfoutC = dfout.copy()
			# ========== Check for missing columns ==========
			fcheck = []
			for ft in feat:	
				fcheck.append(ft not in vi_df.columns)

			if any(fcheck):
				print("Fixing missing columns")
				vi_dfX = pd.read_csv(f"{fpath}VI_df_AllSampleyears.csv", index_col=0)
				for clnm in feat[fcheck]:
					vi_df[clnm] = vi_dfX.loc[:, ["site", clnm]].groupby("site").median().loc[vi_df.site]
				vi_df.to_csv(f"{fpath}VI_df_AllSampleyears_FutureBiomass.csv")
			# ========== pull out the variables and apply transfors ==========
			dfX = vi_df.loc[:, feat].copy()
			# ========== pull out the variables and apply transfors ==========
			# try:
			# 	dfX = vi_df.loc[:, feat].copy()	
			# except Exception as err:
			# 	warn.warn(str(err))
			# 	# vi_dfo = pd.read_csv(f"{fpath}VI_df_AllSampleyears_ObsBiomass.csv", index_col=0)
			# 	breakpoint()
			if not type(setup.loc["Transformer"].values[0]) == float:
				warn.warn("Not implemented yet")
				breakpoint()

			# ========== calculate the obsgap ==========
			if "ObsGap" in feat:
				dfX["ObsGap"] = yr - site_df["year"].values

			# ========== Perform the prediction ==========
			est = model.predict(dfX.values)
			if not type(setup.loc["yTransformer"].values[0]) == float:
				warn.warn("Not implemented yet")
				breakpoint()

			# ========== Convert to common forms ==========
			if pvar == "lagged_biomass":
				breakpoint()
			elif pvar == 'Delta_biomass':
				dfoutC[f"Biomass"]      = vi_df["biomass"].values + est
				dfoutC[f"DeltaBiomass"] = est
				# breakpoint()
			elif pvar == 'Obs_biomass':
				# dfout[f"BIO_{yr}"]   = est
				# dfout[f"DELTA_{yr}"] = est - vi_df["biomass"].values
				dfoutC[f"Biomass"]      = est
				dfoutC[f"DeltaBiomass"] = est - vi_df["biomass"].values
			
			dfoutC["time"] = pd.Timestamp(f"{yr}-12-31")
			dfoutC.loc[(dfoutC.time.dt.year - dfoutC.year) > maxdelta, ['Biomass', 'DeltaBiomass']] = np.NaN
			# dfoutC[f"DeltaBiomass"] /=(dfoutC.time.dt.year - dfoutC.year)
			est_list.append(dfoutC)

	df = pd.concat(est_list)
	df["longitude"] = pd.cut(df["Longitude"], lons, labels=bn.move_mean(lons, 2)[1:])
	df["latitude"]  = pd.cut(df["Latitude" ], lats, labels=bn.move_mean(lats, 2)[1:])
	df["ObsGap"]    = df.time.dt.year - df.year
	df = df[df.ObsGap <= 30]

	# ========== do the direction stuff ==========
	dft = df[[var, "Plot_ID", "year", "ObsGap"]].copy()
	dft[var] = dft[var]>=0
	df["ModelAgreement"] = (dft.groupby(["Plot_ID", "year", "ObsGap"]).transform("sum") - 5) / 5
	df["experiment"] = exp
	# breakpoint()
	return df




# ==============================================================================

if __name__ == '__main__':
	main()