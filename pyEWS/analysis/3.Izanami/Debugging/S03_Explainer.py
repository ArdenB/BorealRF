"""
Script goal, 

Funxtion to explain the strange model behaviour
"""

# ==============================================================================

__title__ = "XGBoost data manupliation"
__author__ = "Arden Burrell"
__version__ = "v1.0(p.07.2021)"
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
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
import cudf

print("seaborn version : ", sns.__version__)
print("xgb version : ", xgb.__version__)
# breakpoint()


# ==============================================================================
def main():
	# ========= make some paths ==========
	dpath = "./pyEWS/experiments/3.ModelBenchmarking/1.Datasets/"
	path  = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/"
	opath = "./pyEWS/experiments/3.ModelBenchmarking/2.ModelResults/Debugging/"
	cf.pymkdir(opath)
	force = False
	# ========== Load in the data ==========
	fnames = glob.glob(f"{opath}DBG*.csv")
	df     = pd.concat([pd.read_csv(fn, index_col=0) for fn in fnames])

	# ========== Compare validation and test sets in models ==========
	testsetscore(path, sitern=416, siteyrn=417)

	# ========== test method Method validation ==========
	benchmarkvalidation(path, df, sitern=416, siteyrn=417)

	# ========== Do the R2 fall within the random sorting range ==========
	matchedrandom(path, df)

	# ========== Test size ==========
	Testsize(path, df, metric="R2")
	
	# ========== Future Disturbance ==========
	FutureDisturbance(path, df, metric="R2")

	# ========== NaN Fraction ==========
	# NanFraction(path, df, metric="R2")

	# ========== Test out optimisation ==========

# ==============================================================================
def NanFraction(path, df, metric="R2"):
	"""
	Check and see how the runs compare to random sampling
	"""
	stack = []
	for splitvar in ["Site", "SiteYF"]:
		# ========== subset the df to the test set ==========
		l1 = np.logical_and(df.group=="Rand", df.sptname==splitvar)
		l2 = np.logical_and(df.test_size==0.3, df.FutDist==20)
		df_sub = df.loc[np.logical_and(l1, l2)]#["testname", "expn", "R2"]
		# breakpoint()
		# ========== Make the indivdual plots ==========
		ax = sns.boxplot(y=metric, x="DropNAN", data=df_sub, color='.8')
		ax = sns.stripplot(y=metric, x="DropNAN", data=df_sub)
		plt.show()
		stack.append(df_sub.copy())
	
	dfs = pd.concat(stack)
	ax = sns.boxplot(y=metric, x="DropNAN", hue="sptname", data=dfs)#, color='.8')
	# ax = sns.stripplot(y=metric, hue="sptname", x="FutDist", data=dfs)
	plt.show()
	sns.barplot(y="obsnum", x="DropNAN", hue="sptname", data=dfs)
	plt.show()

# ==============================================================================
def FutureDisturbance(path, df, metric="R2"):
	"""
	Check and see how the runs compare to random sampling
	"""
	stack = []
	for splitvar in ["Site", "SiteYF"]:
		# ========== subset the df to the test set ==========
		l1 = np.logical_and(df.group=="Rand", df.sptname==splitvar)
		l2 = np.logical_and(df.test_size==0.3, df.DropNAN==0.5)
		df_sub = df.loc[np.logical_and(l1, l2)]#["testname", "expn", "R2"]
		# breakpoint()
		# ========== Make the indivdual plots ==========
		ax = sns.boxplot(y=metric, x="FutDist", data=df_sub, color='.8')
		ax = sns.stripplot(y=metric, x="FutDist", data=df_sub)
		plt.show()
		stack.append(df_sub.copy())
	
	dfs = pd.concat(stack)
	ax = sns.boxplot(y=metric, x="FutDist", hue="sptname", data=dfs)#, color='.8')
	# ax = sns.stripplot(y=metric, hue="sptname", x="FutDist", data=dfs)
	plt.show()
	sns.barplot(y="obsnum", x="FutDist", hue="sptname", data=dfs)
	plt.show()
	# ADD some form of action here to look at the number of sites that are veing included 



# ==============================================================================
def Testsize(path, df, metric="R2"):
	"""
	Check and see how the runs compare to random sampling
	"""
	stack = []
	for splitvar in ["Site", "SiteYF"]:
		# ========== subset the df to the test set ==========
		l1 = np.logical_and(df.group=="Rand", df.sptname==splitvar)
		l2 = np.logical_and(df.FutDist==20, df.DropNAN==0.5)
		df_sub = df.loc[np.logical_and(l1, l2)]#["testname", "expn", "R2"]
		# breakpoint()
		# ========== Make the indivdual plots ==========
		# ax = sns.boxplot(y=metric, x="group", data=df_sub, color='.8')
		ax = sns.stripplot(y=metric, x="test_size", data=df_sub)
		plt.show()
		stack.append(df_sub.copy())
	
	dfs = pd.concat(stack)
	# ax = sns.boxplot(y=metric, x="sptname", data=dfs, color='.8')
	ax = sns.stripplot(y=metric, hue="sptname", x="test_size", data=dfs)
	plt.show()
	# breakpoint()



# ==============================================================================
def matchedrandom(path, df, metric="R2"):
	"""
	Check and see how the runs compare to random sampling
	"""
	stack = []
	for splitvar in ["Site", "SiteYF"]:
		# ========== subset the df to the test set ==========
		l1 = np.logical_and(df.test_size==0.3, df.sptname==splitvar)
		l2 = np.logical_and(df.FutDist==20, df.DropNAN==0.5)
		df_sub = df.loc[np.logical_and(l1, l2)]#["testname", "expn", "R2"]

		# ========== Make the indivdual plots ==========
		ax = sns.boxplot(y=metric, x="group", data=df_sub, color='.8')
		ax = sns.stripplot(y=metric, x="group", data=df_sub)
		plt.show()
		stack.append(df_sub.copy())
	
	dfs = pd.concat(stack)
	ax = sns.boxplot(y=metric, x="sptname", data=dfs, color='.8')
	ax = sns.stripplot(y=metric, x="sptname", hue="group", data=dfs)
	plt.show()
	breakpoint()

# ==============================================================================
def benchmarkvalidation(path, df, sitern=416, siteyrn=417):
	"""
	Prove my results match the performance i observed in the full experiments
	"""
	cols =  ['experiment','version','R2']
	# ========== setup the runs and open the files =========
	
	# ========== subset the df to the test set ==========
	for rn, splitvar in zip([sitern, siteyrn], ["Site", "SiteYF"]):
		# pull out the nex experiment set 
		df_sub = df.loc[np.logical_and(df.group=="Test", df.sptname==splitvar), ["testname", "expn", "R2"]]
		# pull out the old set
		dfx = pd.concat([pd.read_csv(fnx, index_col=0).T for fnx in glob.glob(f"{path}{rn}/Exp{rn}*_Results.csv")]).loc[:, cols]
		dfx.rename({"experiment":"testname", "version":"expn"}, axis=1, inplace=True)
		dft = pd.concat([dfx, df_sub])
		dft = dft.apply(pd.to_numeric, errors='ignore')

		sns.barplot(y="R2", x="expn", hue="testname", data=dft)

		plt.show()
	breakpoint()


def testsetscore(path, sitern=416, siteyrn=417):

	# ========== setup the runs and open the files =========
	breakpoint()


if __name__ == '__main__':
	main()