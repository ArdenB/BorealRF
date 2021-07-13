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
import matplotlib.gridspec as gridspec
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
import statsmodels.api as sm

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
	benchmarkvalidation(path, df.loc[~df.preclean.astype(bool)], sitern=416, siteyrn=417)

	# ========== Do the R2 fall within the random sorting range ==========
	matchedrandom(path, df.loc[~df.preclean.astype(bool)])
	
	# ========== Explain the gaps ==========
	# This need more work to explain 
	gapexplainer(dpath, path, df, sitern=416, siteyrn=417)

	# ========== Do the R2 fall within the random sorting range ==========
	matchedrandom(path, df, hue="preclean")

	# ========== Future Disturbance ==========
	FutureDisturbance(path, df.loc[df.preclean.astype(bool)], metric="R2")



	# ========== Test size ==========
	Testsize(path, df.loc[df.preclean.astype(bool)], metric="R2")
	
	breakpoint()

	# ========== NaN Fraction ==========
	# NanFraction(path, df, metric="R2")

	# ========== Test out optimisation ==========
# ==============================================================================
def gapexplainer(dpath, path, df, sitern=416, siteyrn=417):
	"""
	function to try and determine why there is a gap between my performance metrics
	"""
	# ========== load in the X an Y variables ==========
	y, X = datasubset(path, dpath, FutDist=20, DropNAN=0.5)
	cols =  ['experiment','version','R2', 'FWH:R2']
	
	# ========== load in the number coding ==========
	dfk = OrderedDict()
	dfk["Site"]   = pd.read_csv(f'{dpath}TTS_VI_df_AllSampleyears_10FWH_TTSlookup.csv', index_col=0)
	dfk["SiteYF"] = pd.read_csv(f'{dpath}TTS_VI_df_AllSampleyears_10FWH_siteyear_TTSlookup.csv', index_col=0)

	relist = []
	sclist = []
	# ========== loop over the splot endpoints ==========
	for rn, splitvar in (zip([sitern, siteyrn], ["Site", "SiteYF"])):
		fns = glob.glob(f"{path}{rn}/Exp{rn}*_BranchItteration.csv")
		# .iloc[:-1,]
		flist = []

		for ver, fnx in enumerate(fns):
			dfx = pd.read_csv(fnx, index_col=0).loc[:, cols].reset_index(drop=True).reset_index()
			dfx.rename({"R2":"Validation", "FWH:R2":"Test"}, axis=1, inplace=True)
			dfm = pd.melt(dfx, id_vars=['experiment', 'version', 'index'], value_vars=["Validation","Test"]).set_index("index").loc[dfx.index.max() - 1]
			flist.append(dfm.copy())
		
		dfl                = pd.concat(flist).reset_index(drop=True)
		dfl["splitvar"]    = splitvar
		dfl, df_exp        = lookuptable(dfk[splitvar], dfl, y, X)
		df_exp["splitvar"] = splitvar
		relist.append(dfl)
		sclist.append(df_exp)
		
	
	dfr  = pd.concat(relist)
	dfp = pd.concat(sclist) 
	
	dfrx = dfr.drop(["experiment", "version", "variable"], axis=1)
	# ]]]]] Z score to make plotting easy ]]]]]
	# dfrx.loc[:, dfp.index.unique().values] = dfrx[dfp.index.unique().values].apply(sp.stats.zscore)
	
	# dfrm = pd.melt(dfrx, id_vars=['splitvar', "R2"], value_vars=dfp.index.unique().values)
	# row1 = dfp.index.unique().values[:6]
	# row2 = dfp.index.unique().values[6:]
	# fig  = plt.figure(constrained_layout=True)#, figsize=(16,20))
	# spec = gridspec.GridSpec(ncols=6, nrows=2, figure=fig,) #width_ratios=[5,1,5,5], height_ratios=[5, 10, 5]
	# for col, (rn1, rn2) in enumerate(zip(row1, row2)):
	# 	ax1 = fig.add_subplot(spec[0, col])
	# 	ax2 = fig.add_subplot(spec[1, col])

	# 	for row_var, ax in zip([rn1, rn2], [ax1, ax2]):
	# 		breakpoint()

	# ========== Plot the metrics ==========
	sns.barplot(y="R2", x="index",hue="splitvar",  data=dfp.reset_index())
	plt.show()

	# breakpoint()


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
		df_sub = df.loc[np.logical_and(l1, l2)].copy()#["testname", "expn", "R2"]
		df_sub.loc[:, 'FutDist'] =  df_sub['FutDist'].round(decimals=2).values
		# breakpoint()
		# ========== Make the indivdual plots ==========
		ax = sns.boxplot(y=metric, x="FutDist", data=df_sub, color='.8')
		ax = sns.stripplot(y=metric, x="FutDist", data=df_sub)
		plt.show()
		# ax = sns.boxplot(y=metric, x="FutDist", hue="preclean", data=df_sub, color='.8')
		# ax = sns.stripplot(y=metric, x="FutDist", hue="preclean", data=df_sub, dodge=True)
		# plt.show()
		# breakpoint()
		stack.append(df_sub.copy())
	
	dfs = pd.concat(stack)
	ax  = sns.boxplot(y=metric, x="FutDist", hue="sptname", data=dfs)#, color='.8')
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
		df_sub.loc[:, 'test_size'] =  df_sub['test_size'].round(decimals=2).values
		# breakpoint()
		# ========== Make the indivdual plots ==========
		# ax = sns.boxplot(y=metric, x="group", data=df_sub, color='.8')
		# breakpoint()
		ax = sns.stripplot(y=metric, x="test_size", data=df_sub)
		plt.show()
		stack.append(df_sub.copy())
	
	dfs = pd.concat(stack)
	# ax = sns.boxplot(y=metric, x="sptname", data=dfs, color='.8')
	ax = sns.stripplot(y=metric, hue="sptname", x="test_size", data=dfs, dodge=True)
	plt.show()
	# breakpoint()



# ==============================================================================
def matchedrandom(path, df, metric="R2", hue=None):
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
		ax = sns.boxplot(y=metric, x="group", hue=hue, data=df_sub, color='.8')
		ax = sns.stripplot(y=metric, x="group", hue=hue, data=df_sub, dodge=True)
		plt.show()
		# breakpoint()
		stack.append(df_sub.copy())
	
	dfs = pd.concat(stack)
	if hue is None:
		va = "sptname"
		order= ["Site", "SiteYF"]
	elif hue == "preclean":
		va      = "preclean_sptname"
		dfs[va] = "fix"+(dfs["preclean"].astype(str))+"_" + dfs["sptname"]  
		order   = ["fixFalse_Site", "fixFalse_SiteYF", "fixTrue_Site", "fixTrue_SiteYF"]
	# ax = sns.catplot(y=metric, x="sptname", data=dfs, color='.8', col="preclean", kind="box")
	# ax = sns.catplot(y=metric, x="sptname", hue="group",data=dfs, col="preclean", kind="strip")
	ax = sns.boxplot(y=metric, x=va, order=order, data=dfs, color='.8')
	ax = sns.stripplot(y=metric, x=va, order=order, hue="group", data=dfs)
	plt.show()
	# breakpoint()

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
		dfx = pd.concat([pd.read_csv(fnx, index_col=0).T for fnx in glob.glob(f"{path}{rn}/Exp{rn}*_Results.csv")]).loc[:, cols]

		plt.show()
	# breakpoint()


def testsetscore(path, sitern=416, siteyrn=417):

	# ========== setup the runs and open the files =========
	cols =  ['experiment','version','R2', 'FWH:R2']
	for rn, splitvar in zip([sitern, siteyrn], ["Site", "SiteYF"]):
		fns = glob.glob(f"{path}{rn}/Exp{rn}*_BranchItteration.csv")
		# .iloc[:-1,]
		dfx = pd.concat([pd.read_csv(fnx, index_col=0).reset_index() for fnx in fns]).loc[:, cols].reset_index()
		dfx.rename({"R2":"Validation", "FWH:R2":"Test"}, axis=1, inplace=True)
		# pd.wide_to_long(dfx, stubnames='R2', i=['experiment', 'version'], j='R2')
		dfm = pd.melt(dfx, id_vars=['experiment', 'version', 'index'], value_vars=["Validation","Test"])
		# ========== Make the plot ==========
		
		sns.relplot(
		    data=dfm, x="index", y="value", col="version",
		    hue="variable", kind="line",col_wrap=5)
		plt.show()
		# breakpoint()
		# Exp416_OneStageXGBOOST_AllGap_Debug_Sitesplit_vers00_BranchItteration.csv
	# breakpoint()

# ==============================================================================

def datasubset(path, dpath, FutDist=20, DropNAN=0.5):

	"""
	Function to do the splitting and return the datasets
	"""
	
	# ========== load in the stuff used by every run ==========
	# Datasets
	vi_df   = pd.read_csv(f"{dpath}ModDataset/VI_df_AllSampleyears_ObsBiomass.csv", index_col=0)
	df_site = pd.read_csv(f"{dpath}ModDataset/SiteInfo_AllSampleyears_ObsBiomass.csv", index_col=0)
	df_site.rename({"Plot_ID":"site"}, axis=1, inplace=True)
	# Column names, THis was chose an s model with not too many vars

	colnm   = pd.read_csv(
		f"{path}411/Exp411_OneStageXGBOOST_AllGap_50perNA_PermutationImp_RFECV_FINAL_Delta_biomass_altsplit_vers04_PermutationImportance_RFECVfeature.csv", index_col=0)#.index.values
	colnm   = colnm.loc[colnm.InFinal].index.values
	# colnm   = colnm.index.values
	predvar = "Delta_biomass"
	# ========== pull out the X values ==========
	X = vi_df.loc[:, colnm].copy()		
	nanccal = X.isnull().sum(axis=1)<= DropNAN
	distcal = df_site.BurnFut + df_site.DisturbanceFut
	distcal.where(distcal<=100., 100, inplace=True)
	# +++++ make a sing dist and nan layer +++++
	dist = (distcal <= FutDist) & nanccal
	X = X.loc[dist]

	y = vi_df.loc[X.index, predvar].copy() 
	if bn.anynan(y):
		X = X.loc[~y.isnull()]
		y = y.loc[~y.isnull()]
	
	return y, X

def lookuptable(dfk, dfl, y, X):
	"""
	dfk:	dataframe
		coded lookup table

	"""

	kys = ({
		"Validation":{"in_train":[0], "in_test":[1]},
		"Test":{"in_train":[0], "in_test":[2, 3]},
		})
	ind = y.index.values


	# ========== Total stats ==========
	dfl = dfl.rename({"value":"R2"}, axis=1)
	dfl["Tot_MedianY"]   = y.median()
	dfl["Tot_FracLossY"] = (y<0).mean()
	dfl["Tot_5thperY"]   = np.percentile(y, 5)
	dfl["Tot_95thperY"]  = np.percentile(y, 95)
	dfl["Tot_negOutliersY"] = (y < -200).sum()
	dfl["Tot_posOutliersY"] = (y >  200).sum()
	dfl["Tot_posRatOutliersY"] = ((y/X["ObsGap"]) >  20).sum()
	
	# ========== loop over the dataframes columns ==========
	for index, row in dfl.iterrows():
		nx = row.version
		ftrain = dfk.loc[ind, str(nx)]. apply(_findcords, test=kys[row.variable]['in_train'])
		ftest  = dfk.loc[ind, str(nx)]. apply(_findcords, test=kys[row.variable]['in_test'])
		
		# \\\ Statscal \\\
		# ========== Total stats ==========
		dfl.loc[index, "Sub_MedianY"]   = y.loc[ftest].median()
		dfl.loc[index, "Sub_MeanY"  ]   = y.loc[ftest].mean()
		dfl.loc[index, "Sub_MinY"]      = y.loc[ftest].min()
		dfl.loc[index, "Sub_MaxY"]      = y.loc[ftest].max()
		dfl.loc[index, "Sub_FracLossY"] = (y.loc[ftest]<0).mean()
		dfl.loc[index, "Sub_1stperY"]   = np.percentile(y.loc[ftest], 1)
		dfl.loc[index, "Sub_5thperY"]   = np.percentile(y.loc[ftest], 5)
		dfl.loc[index, "Sub_95thperY"]  = np.percentile(y.loc[ftest], 95)
		dfl.loc[index, "Sub_99thperY"]  = np.percentile(y.loc[ftest], 99)
		# dfl.loc[index, "Sub_OutliersY"] = (
		# 	((y.loc[ftest] > (y.mean() + (3*y.std()))).mean()) + 
		# 	((y.loc[ftest] < (y.mean() - (3*y.std()))).mean()))

		dfl.loc[index, "Sub_negOutlegiersY"] = (y.loc[ftest] < -200).mean()
		dfl.loc[index, "Sub_posOutliersY"] = (y.loc[ftest] >  200).mean()
		dfl.loc[index, "Sub_posRatOutliersY"] = ((y.loc[ftest]/X.loc[ftest, "ObsGap"]) >  20).mean()
		# assert np.logical_xor(ftrain.values,ftest.values).all()

		# ptrainl.append(ftrain.loc[ftrain.values].index.values)
		# ptestl.append(ftest.loc[ftest.values].index.values)
	vnames = dfl.columns.values[[clnm.startswith("Sub_")  for clnm in dfl.columns.values]]
	
	Yn = dfl["R2"]
	expl = OrderedDict()
	for vn in vnames:
		Xn = dfl[vn]
		Xn = sm.add_constant(Xn)
		model = sm.OLS(Yn,Xn)
		results = model.fit()
		# print(f'{vn} R2: {results.rsquared} pvalue: {results.pvalues[vn]} slope: {results.params[vn]}')
		expl[vn] = ({
			"R2"    : np.round(results.rsquared, decimals=8),
			"pvalue": np.round(results.pvalues[vn], decimals=8),
			"slope" : np.round(results.params[vn], decimals=8) 
			})
	# breakpoint()
	return dfl, pd.DataFrame(expl).T

def _findcords(x, test):
	# Function check if x is in different arrays

	## I might need to use an xor here
	return x in test 

# ==============================================================================
# ==============================================================================
if __name__ == '__main__':
	main()