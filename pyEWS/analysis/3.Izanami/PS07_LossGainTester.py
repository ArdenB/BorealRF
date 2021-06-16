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
	# ========== Setup the pathways ==========
	formats = None
	path  = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"
	cf.pymkdir(path+"plots/")
	ppath = "./pyEWS/analysis/3.Izanami/Figures/PS06/"
	cf.pymkdir(ppath)

	# ========== Chose the experiment ==========
	exp      = 402
	# ========== Pull out the prediction ==========
	dfout, dfl = fpred(path, exp)#, bins=[0,4,5,6,9,10,11,14,15,16,19, 20, 40])

	# ========== Plot the loss performance ==========
	losspotter(path, ppath, dfout, dfl)


# ==============================================================================
def losspotter(path, ppath, dfout, dfl):
	"""
	plots the loss performance
	"""
	# ========== Create the matplotlib params ==========
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':12, "axes.labelweight":"bold",})
	font = {'family' : 'normal',
	        'weight' : 'bold', #,
	        'size'   : 12}
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	# dfl["Obsloss"] = dfl["Obsloss"].astype(float)

	dfl.rename(columns={"Obsloss":"LossProb", "GapGroup":"PredictionWindow","GapQuant":"QPredictionWindow"}, inplace=True)
	fig = sns.lineplot(y="LossProb", x="ModLossN", hue="PredictionWindow", data=dfl)#, ci="sd")
	# dfg = dfl.groupby(["GapGroup", "ModLossN"])["LossProb"].mean().reset_index()
	# breakpoint()
	fig.set_yticks(np.arange(0, 1.1, 0.1))
	fig.set_xticks(np.arange(0, 10, 1))
	plt.show()
	

	fig = sns.lineplot(y="LossProb", x="ModLossN", hue="QPredictionWindow", data=dfl)
	fig.set_yticks(np.arange(0, 1.1, 0.1))
	fig.set_xticks(np.arange(0, 10, 1))
	breakpoint()
# ==============================================================================
def fpred(path, exp, fpath    = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/ModDataset/", 
	nanthresh=0.5, drop=True, bins=[0, 5, 10, 15, 20, 40], qbins=8):
	"""
	function to predict future biomass
	args:
	path:	str to files
	exp:	in of experiment
	years:  list of years to predict 
	"""
	# warn.warn("\nTo DO: Implemnt obsgap filtering")
	# ========== Load the variables ==========
	site_df = pd.read_csv(f"{fpath}SiteInfo_AllSampleyears_ObsBiomass.csv", index_col=0)
	vi_df   = pd.read_csv(f"{fpath}VI_df_AllSampleyears_ObsBiomass.csv", index_col=0)
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
		# dfoutC.loc[(dfoutC.time.dt.year - dfoutC.year) > maxdelta, ['Biomass', 'DeltaBiomass']] = np.NaN
		est_list.append(dfoutC)


	dft = pd.concat(est_list).dropna().reset_index().rename(columns={'index': 'rownum'})
	dft["PredCount"] = dft.groupby("rownum")['DeltaBiomass'].transform('count')

	if drop:
		dfout = dft.loc[dft["PredCount"] == dft["PredCount"].max(), ].reset_index(drop=True)
		# breakpoint()
		# dfoutC = dfoutC.loc[dfoutC.year > (yr - maxdelta)].dropna()
	# ========== get the number of models that are cexreasing ==========
	dftest = dft.loc[:, ["ObsGap", "rownum", "Observed", 'DeltaBiomass']].copy()
	# dftest["BiomassLoss"]
	# dftest["LossModNum"]
	# breakpoint()
	dfl = dftest.groupby("rownum").agg(
		ObsGap   = ('ObsGap', 'mean'), 
		Observed = ('Observed', 'mean'),  	
		Obsloss  = ('Observed', lambda x: float((x.mean() < 0))), 
		ModLossN = ("DeltaBiomass", lambda x: (x < 0).sum()), 
		)
	dfl["GapGroup"] = pd.cut(dfl["ObsGap"], bins)#, include_lowest=True)#, labels=np.arange(5, 45, 5))
	dfl["GapQuant"] = pd.qcut(dfl["ObsGap"], qbins, duplicates ="drop")
	return dft, dfl


# ==============================================================================

if __name__ == '__main__':
	main()