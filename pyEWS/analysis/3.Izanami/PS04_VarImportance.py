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
	ppath = "./pyEWS/analysis/3.Izanami/Figures/PS04/"
	cf.pymkdir(ppath)
	
	
	exps = [401, 403, 404]
	var  = "PermutationImportance"
	huex = "VariableGroup"#"Count"
	# ========== get the PI data ==========
	df = _ImpOpener(path, exps)

	featureplotter(df, ppath, var, exps, huex)
	breakpoint()

# ==============================================================================
def featureplotter(df, ppath, var, exps, huex):
	""" Function to plot the importance of features """
	# ========== Setup params ==========
	plt.rcParams.update({'axes.titleweight':"bold","axes.labelweight":"bold", 'axes.titlesize':8})
	font = {'family' : 'normal',
	        'weight' : 'bold', #,
	        'size'   : 8}
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	# plt.rcParams.update({'axes.titleweight':"bold", })

	# ========== Create the figure ==========
	g = sns.catplot( 
		x="Variable", y=var, #ci=0.95, estimator=bn.nanmedian, 
		hue=huex, dodge=False, 
		data=df, 
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
	plt.tight_layout()
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
	plt.tight_layout()
	plt.show()

	breakpoint()

def _ImpOpener(path, exps, var = "PermutationImportance"):
	"""
	Function to open the feature importance files and return them as a single 
	DataFrame"""

	# ========== Loop over the exps ==========
	df_list = []
	for exp in exps:
		fnames = sorted(glob.glob(f"{path}{exp}/Exp{exp}_*PermutationImportance.csv"))
		for ver, fn in enumerate(fnames):
			dfin = pd.read_csv( fn, index_col=0)
			dfin["experiment"] = exp
			dfin["version"]    = ver
			df_list.append(dfin)
	# ========== Group them together ==========
	df = pd.concat(df_list).reset_index(drop=True)
	# ========== Calculate the counts ==========
	df["Count"] = df.groupby(["experiment", "Variable"])[var].transform("count")
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
	# breakpoint()
	return df

# ==============================================================================
if __name__ == '__main__':
	main()