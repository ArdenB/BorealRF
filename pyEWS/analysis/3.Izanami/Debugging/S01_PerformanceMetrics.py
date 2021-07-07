"""
Script goal, 

Interrogate the drop in performance 
"""

# ==============================================================================

__title__ = "One Stage XGboost Dataset manipulations"
__author__ = "Arden Burrell"
__version__ = "v1.0(7.07.2021)"
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
# import ipdb
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
import seaborn as sns
import pickle

# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf
import myfunctions.benchmarkfunctions as bf

# ========== Import packages for parellelisation ==========
# import multiprocessing as mp

# ========== Import ml packages ==========
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import QuantileTransformer
from sklearn import metrics as sklMet
from sklearn.utils import shuffle
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
import xgboost as xgb
from tqdm import tqdm

print("seaborn version : ", sns.__version__)
# print("xgb version : ", xgb.__version__)
# breakpoint()


# ==============================================================================
def main():

	# ========= Load in the observations ==========
	path  = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"
	cols  =  ['experiment','version','R2', 'TotalTime', 'fractrows', 'colcount',]
	fn  = './pyEWS/experiments/3.ModelBenchmarking/1.Datasets/TTS_VI_df_AllSampleyears_10FWH_TTSlookup.csv'
	dfk        = pd.read_csv(fn, index_col=0)
	# breakpoint()
	# ========== Experiment 1 - Subsetting of data ==========
	# experiments = {410:"410 - FWH Sites", 415:"415 - FWH-PredYear"}
	# MMT = True

	# # +++++ PULL OUT THE INDEXS +++++
	# indFW = dfk.index[dfk.iloc[:, 0] == 3].values


	# OpenDatasets("Witholding method", path, cols, experiments, indFW)

	# ========== Experiment 2 - Size of multimode full witheld ==========
	experiments  = {410:"410 - 30% testsize", 412:"412 - 20% test size"}
	indFW = dfk.index[dfk.iloc[:, 0] == 2].values
	# breakpoint()
	OpenDatasets("Test Dataset Size", path, cols, experiments, indFW)
	breakpoint()
	# MMT  = False
	# pair = True
	# breakpoint()
	# dfrx = dfr.groupby(["SiteFWH", "winner"]).count()["Observed"].reset_index()
	# plt.show()

	# breakpoint()
	# dfr['count_max'] = dfr.groupby(['index', "Version"])['Residual'].abs().transform(min)
	# df_mres = pd.concat([pd.read_csv(mrfn, index_col=0).T.loc[:, cols] for mrfn in perfn], sort=True)
	# sns.barplot(y = "Observed", x = "SiteFWH", hue="winner", data=dfrx)
	# plt.show()

	
	breakpoint()




	# ========== Experiment 3 - Future Disturbance ==========
	exp = [413, 410, 414]
	MMT  = False
	pair = True



# ==============================================================================
def OpenDatasets(name, path, cols, experiments, indFW):
		# reslist = []
	# res = OrderedDict()re
	res   = []
	perfn = []
	for ver in range(10):
		try:
			dflist = [load_OBS(glob.glob(f"{path}{exp}/Exp{exp}*_vers0{ver}_OBSvsPREDICTED.csv")[0]) for exp in experiments]
				
		except Exception as e:
			print(f"At least one of the files of v. {ver} is misssing")
			continue
		indx = dflist[0].index
		for dfl in 	dflist:
			indx = indx.intersection(dfl.index)
				#.values
		indx = indx.values
		# breakpoint()
		# OvP_fn1 =
		# OvP_fn2 = glob.glob(f"{path}{exp[1]}/Exp{exp[1]}*_vers0{ver}_OBSvsPREDICTED.csv")

		# df1 = load_OBS(OvP_fn1[0])
		# df2 = load_OBS(OvP_fn2[0])
		
		for ex, dfpp in zip(experiments, dflist):
			# perfn.append()
			dfx =pd.read_csv(glob.glob(f"{path}{ex}/Exp{ex}*_vers0{ver}_Results.csv")[0], index_col=0).T.loc[:, cols]
			ts  = pd.Timedelta(dfx["TotalTime"].values[0])
			dfx = dfx.apply(pd.to_numeric, errors='ignore')
			dfx["experiment"] = experiments[ex]
			dfx['TotalTime']  = ts

			dfx['MAE']  = sklMet.mean_absolute_error(dfpp["Observed"], dfpp["Estimated"])
			dfx['RMSE'] = np.sqrt(sklMet.mean_squared_error(dfpp["Observed"], dfpp["Estimated"]))
			perfn.append(dfx.copy())
		
		for ind in tqdm(indx):
			valx = np.argmin(np.abs([df.loc[ind,"Residual"] for df in dflist]))
			# breakpoint()
			for nu, (ex, dfpp) in enumerate(zip(experiments, dflist)):
				# if not ind in indFW:
				# 	continue
				# breakpoint()
				res.append({
					"index":ind, 
					"experiment":experiments[ex], 
					"Version":ver, 
					"Observed":dfpp.loc[ind, "Observed"].copy(), 
					"Estimated":dfpp.loc[ind, "Estimated"].copy(),
					"Residual":dfpp.loc[ind, "Residual"].copy(), 
					"CorrectDir":(dfpp.loc[ind, "Observed"]<0) == (dfpp.loc[ind, "Estimated"] < 0).copy(),
					"SiteFWH":ind in indFW, 
					"winner":nu == valx, 
					# "winner":exp[np.argmin(np.abs([df1.loc[ind, "Residual"], df2.loc[ind, "Residual"]]))]
					})
		# dfra  = pd.DataFrame(res).T
		# dfra.groupby("")
		# breakpoint()

	dfr  = pd.DataFrame(res)#.T.reset_index()
	dfp = pd.concat(perfn)
	plotmaker(name, experiments, dfr, dfp)
	breakpoint()

def plotmaker(name, exp, dfr, dfp):
	font = ({
		'weight' : 'bold', 
		'size'   : 9})
	axes = ({
		'titlesize':9,
		'labelweight':'bold'
		})


	sns.set_style("whitegrid")
	mpl.rc('font', **font)
	mpl.rc('axes', **axes)
	# plt.subplots(,)

	# ========== Plot Broad summary  ==========
	fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
		2, 3, figsize=(15,10),num=(name), dpi=130)
	# +++++ R2 plot +++++
	sns.barplot(y="R2", x="experiment", data=dfp, ci=0.95, ax=ax1)#, order=exp_names)
	ax1.set_xlabel("")
	ax1.set_ylabel(r"$R^{2}$")
	# ax1.set_xticklabels(ax1.get_xticklabels(), rotation=10, horizontalalignment='right')
	ax1.set(ylim=(0., 1.))
	
	# +++++ time taken plot +++++
	sns.barplot(y="MAE", x="experiment", data=dfp, ci=0.95, ax=ax2)
	ax2.set_xlabel("")
	ax2.set_ylabel(r"Mean Absolute Error")
	# ax2.set_xticklabels(ax2.get_xticklabels(), rotation=10, horizontalalignment='right')

	sns.barplot(y="RMSE", x="experiment", data=dfp, ci=0.95, ax=ax3)
	ax3.set_xlabel("")
	ax3.set_ylabel(r"RMSE")
	# ax2.set_xticklabels(ax3.get_xticklabels(), rotation=10, horizontalalignment='right')

	dfrx = dfr.groupby(["experiment","SiteFWH"])["winner", "CorrectDir"].mean().reset_index()
	# breakpoint()
	
	# # +++++ site fraction plot +++++
	sns.barplot(y="winner",  hue="experiment", x="SiteFWH", data=dfrx, ax=ax4)
	ax4.set_xlabel("")
	ax4.set_ylabel("Best prediction %")
	# ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, horizontalalignment='right')
	# ax3.set(ylim=(0., 1.))
	sns.barplot(y="CorrectDir",  hue="experiment", x="SiteFWH", data=dfrx, ax=ax5)
	ax5.set_xlabel("")
	ax5.set_ylabel("Correct Direction %")


	def absmean(x):	return np.mean(np.abs(x))
	dfrr = dfr.groupby(["experiment", "Version","SiteFWH"])["Residual"].apply(absmean).reset_index()
	sns.barplot(y="Residual",  hue="experiment", x="SiteFWH", data=dfrr, ax=ax6)
	ax6.set_xlabel("")
	ax6.set_ylabel("Grouped Mean Absolute Residual")
	# .mean()
	# ax4.set_xticklabels(ax3.get_xticklabels(), rotation=30, horizontalalignment='right')
	# ax4.set(ylim=(0., np.ceil(df_set.itterrows.max()/1000)*1000))
	plt.tight_layout()
	plt.show()
	# breakpoint()


def load_OBS(ofn):
	df_in = pd.read_csv(ofn, index_col=0)
	df_in["Residual"] =df_in["Estimated"] - df_in["Observed"]
	df_in["experiment"] = int(ofn.split("/")[-2])
	df_in["experiment"] = df_in["experiment"].astype("category")
	df_in["version"]    = float(ofn.split("_vers")[-1][:2])
	return df_in

# ==============================================================================
if __name__ == '__main__':
	main()