"""
Script goal, 

To learn about the existing dataset through the use of Random Forest
	- Open the datasets the Sol used in his classification 
	- Perform a basic random forest on them
	- Bonus Goal, test out some CUDA 

Relevent existing R code
	./code/analysis/modeling/build_model/rf_class......
"""

# ==============================================================================

__title__ = "Random Forest Testing"
__author__ = "Arden Burrell"
__version__ = "v1.0(18.03.2020)"
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
import xarray as xr
import bottleneck as bn
import scipy as sp
import glob
import shutil
import time
import ipdb
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict

# ========== Import packages for parellelisation ==========
# import multiprocessing as mp

# ========== Import ml packages ==========
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics as sklMet
from scipy.stats import spearmanr
from scipy.cluster import hierarchy


# ========== Import Cuda Package packages ==========
warn.warn("add a cuda state variable here")
# import cupy
# import cudf
# import cuml
# from cuml import RandomForestRegressor as cuRFR
# model_selection.train_test_split

# ==============================================================================
def main():
	# ========== These are all from Sol ===========
	region      = 'all' 
	num_quants  = 7 
	window      = 10
	predict_var = 'biomass'
	ntree       = 500
	test_size   = 0.2

	# ========== open and import dataframes ===========
	X, y, col_nms = df_proccessing(window, mode="Sol")

	# ========== Do the feature selection ===========
	Importance = []
	for VarImp in [None, "base", "recursive"]:
		print(VarImp)
		Importance.append(feature_selection(
			X, y,  col_nms, ntree, test_size, 
			cores=-1, SelMethod="hierarchial",  
			var_importance=VarImp))
	# +++++ make a single df that contains the spped and accuracy of the methods +++++
	imp_df = pd.concat(Importance).reset_index()
	imp_df.groupby("FeatureSelection").plot.bar(subplots=True)

	ipdb.set_trace()
	import seaborn as sns
	sns.catplot(x="index", y="R2", data=imp_df, kind="bar")



# ==============================================================================
# ==============================================================================
# ============================= Feature selection ==============================
# ==============================================================================

def feature_selection(X, y,  col_nms, ntree, test_size, cores=-1,  SelMethod = None, var_importance=None):
	"""
	Function to perform recursive feature selection
	"""

	warn.warn("In the future pull this out into its own function \n \n ")
	if SelMethod is None:
		# Test
		time,  r2, feature_imp  = skl_rf_regression(Xset, yset,  col_nms, ntree, test_size, cores=cores)
		pass
	elif SelMethod == "hierarchial":
		"""
		This is a very basic variable importance testing. I'm stealing this code from:
		# Permutation Importance with Multicollinear or Correlated Features
		# https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
		I am also improving the implemetation a little:

		This approach uses a clustering approach from the method above,
		but instead of just picking the first value in each cluster, I've
		modified it to use the most important variable in each cluster 
		"""
		# ========== Setup the inital clustering ==========
		# +++++ Build a spearman table +++++
		corr         = spearmanr(X).correlation
		# +++++ Calculate the ward hierarchy +++++
		corr_linkage = hierarchy.ward(corr)
		# +++++ make the experiment name +++++
		if var_importance is None:
			experiment = SelMethod
		else:
			experiment = SelMethod + var_importance

		plot = False
		if plot:
			# ========== Build a plot ==========
			fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
			dendro          = hierarchy.dendrogram(corr_linkage, labels=X.columns.values, ax=ax1, leaf_rotation=90)
			dendro_idx      = np.arange(0, len(dendro['ivl']))

			ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
			ax2.set_xticks(dendro_idx)
			ax2.set_yticks(dendro_idx)
			ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
			ax2.set_yticklabels(dendro['ivl'])
			fig.tight_layout()
			plt.show()
			ipdb.set_trace()

		# ========== Create a dictionary so i can store performance metrics ==========
		perf  = OrderedDict()
		ColNm = X.columns.values #all the names of the columns
		
		# ========== Create the branches i want to loop over ==========
		tpbrnch = 80
		interv  = 10
		singbr  = 40
		warn.warn("The testing space is shurnk for development, remove this and change values later")
		# test_branchs = np.hstack([np.arange(singbr), np.arange(singbr, tpbrnch, interv)])
		# test_branchs = np.arange(0, tpbrnch, interv)
		test_branchs = np.arange(singbr)

		# ========== Setup the inital clustering ==========
		for br_num, branch in enumerate(test_branchs):
			print("starting Random forest with hierarchical clustering level %d at:" % branch, pd.Timestamp.now())

			# ========== Random forest regression ===========
			time,  r2, feature_imp  = skl_rf_regression(X[ColNm], y,  col_nms, ntree, test_size, cores=cores, verbose=False)
			# tcuda, r2cuda = cuda_rf_regression( X, y,  col_nms, ntree, test_size)
			
			# ========== Store the base feature importance ===========
			if br_num == 0:
				BaseFeatImp = feature_imp
				cum_time    = time
				base_time   = time
			else:
				if var_importance is None:
					cum_time  = time
				elif var_importance == "base":
					# Only consideres the base variable 
					cum_time = base_time + time
				elif var_importance == "recursive":
					cum_time = cum_time + time   

			# ========== Add the performce stats to the OrderedDict ===========
			perf["Branch%02d" % branch] = ({
				"Time":time, "TimeCumulative":cum_time,  
				"R2":r2, "NumVar":len(feature_imp)})

			# ========== Check if i need to cluster and remove data ==========
			if branch + interv >= tpbrnch:
				continue

			# ========== Performing Clustering based on the branch level ==========
			# ////////// This wont work ////////////////
			cluster_ids   = hierarchy.fcluster(corr_linkage, test_branchs[br_num], criterion='distance')
			clusID_featID = defaultdict(list)
			
			# ========== Find what variable belong in each cluster ==========
			for idx, cluster_id in enumerate(cluster_ids): clusID_featID[cluster_id].append(idx)
			
			# ========== Find the most important variable in each cluster ==========
			sel_feat = [] #Container for the featture names
			
			for clus in  clusID_featID:
				try:
					# +++++ Get the IDs, the feature importance and the name +++++
					IDs = clusID_featID[clus]
					NMs = X.columns.values[IDs]

					if var_importance is None:
						# the way documentation picked the feature, ignore importance and picked the first
						sel_feat.append(NMs[0])
						continue
					elif var_importance == "base":
						# Only consideres the base variable 
						FIs = [BaseFeatImp[fn] for fn in NMs]
					elif var_importance == "recursive":
						# +++++ drop the names that are irrelevant +++++
						# This approach will look at the most recent feature performance
						FIs = []
						for fn in NMs:
							if fn in ColNm:
								FIs.append(feature_imp[fn])
							else:
								FIs.append(np.NAN)
						# inset = [nn in ColNm for nn in NMs]
						# for nn in NMs: inset.append(nn in ColNm)
					else:
						raise ValueError("unknown var_importance passed, use None, base or recursive")
					# +++++ Find the most relevant feature +++++
					sel_feat.append(NMs[bn.nanargmax(FIs)])
				except Exception as e:
					warn.warn(str(e))
					ipdb.set_trace()
			# ========== Pull out the features that i'm going to use on the next loop ==========
			ColNm = sel_feat

		# ========== Pull out the features that i'm going to use on the next loop ==========
		perf_df = pd.DataFrame(perf).T
		perf_df["Time"]             = perf_df.Time.astype('timedelta64[s]')
		perf_df["TimeCumulative"]   = perf_df.TimeCumulative.astype('timedelta64[m]')
		perf_df["FeatureSelection"] = experiment

		return perf_df

	elif var_importance == "Sol":
		# This mode implements the exact 
		pass

# ==============================================================================
# =============================== Random Forest ================================
# ==============================================================================

def cuda_rf_regression( Xin, yin,  col_nms, ntree, test_size):
	"""
	This function is to test out the  speed of the random forest regressions using
	sklearn 
	args:
		X: 		numpy array
			data to be predicted
		y: 		numpy array 
			data to be used for prediction
		col_nms:	array
			name of the columns
	"""
	# ========== convert to a GPU dataframe ==========
	print ("Moving data to the GPU at: ", pd.Timestamp.now())
	Xgpu = cudf.DataFrame.from_pandas(X)
	ygpu = cudf.DataFrame.from_pandas(y)

	ipdb.set_trace()



def skl_rf_regression( Xin, yin,  col_nms, ntree, test_size, cores=-1, verbose=True):
	"""
	This function is to test out the  speed of the random forest regressions using
	sklearn 
	args:
		Xin: 			ndarray or pd dataframe
			data to be used for prediction
		yin: 			ndarray or pd dataframe
			data to be predicted
		col_nms:	array
			name of the columns
		cores:		int
			the number of CPU cores to use, Defualt is the total number of threads
		verbose: 	bool
			How much infomation to print
	"""
	# ========== Split the data  ==========
	X_train, X_test, y_train, y_test = train_test_split(Xin, yin, test_size=test_size)

	# ========== Setup some skl params ==========
	skl_rf_params = ({
		'n_estimators': ntree,
		'n_jobs': cores })
		# 'max_depth': 13,

	# ========== Start timing  ==========
	t0 = pd.Timestamp.now()
	print("starting sklearn random forest regression at:", t0)

	# ========== Do the RF regression training ==========
	regressor = RandomForestRegressor(**skl_rf_params)
	regressor.fit(X_train, y_train)

	# ========== Testing out of prediction ==========
	print("starting sklearn random forest prediction at:", pd.Timestamp.now())
	y_pred = regressor.predict(X_test)
	tDif = pd.Timestamp.now()-t0
	
	# ========== print all the infomation if verbose ==========
	if verbose:
		print('r squared score:',         sklMet.r2_score(y_test, y_pred))
		print('Mean Absolute Error:',     sklMet.mean_absolute_error(y_test, y_pred))
		print('Mean Squared Error:',      sklMet.mean_squared_error(y_test, y_pred))
		print('Root Mean Squared Error:', np.sqrt(sklMet.mean_squared_error(y_test, y_pred)))

		# ========== Print the time taken ==========
		for var, imp in  zip(col_nms[:-1], regressor.feature_importances_):
			print("Variable: %s Importance: %06f" % (var, imp))

	# ========== Convert Feature importance to a dictionary ==========
	FI = OrderedDict()
	for fname, f_imp in zip(Xin.columns.values, regressor.feature_importances_):
		FI[fname] = f_imp

	print("The time taken to perform the random forest regression:", tDif)
	return tDif, sklMet.r2_score(y_test, y_pred), FI


# ==============================================================================
# ============================== Data processing ===============================
# ==============================================================================

def df_proccessing(window, mode=None):
	"""
	This function opens and performs all preprocessing on the dataframes.
	These datasets were originally made using:
		./code/data_prep/create_VI_df.R
		./code/data_prep/vi_calcs/functions/modified......
		./code/data_prep/vi_calcs/full_df_calcs_loop......
	args:
		window:		int
			the max size of the values considered 
		mode:		str
			the mode of data filtering to be used. if mode= "Sol", the code is 
			designed to replicate Sols original implementation. Defualt is None
			where i use a much more simplified filtering approach

	"""
	# ============ Setup the file names ============
	# The level of correlation between datasets
	cor_fn = "./EWS_package/data/models/input_data/correlations_2019-10-09.csv"
	vi_fn  = "./EWS_package/data/models/input_data/vi_df_all_2019-10-30.csv"

	# ============ Open the variables and correlations ============
	# This may need to change when using cuda 
	vi_df  = pd.read_csv( vi_fn, index_col=0)
	cor_df = pd.read_csv(cor_fn, index_col=0)

	# ============ Filter the rows ============
	if mode is None:
		# ========== This is where i implement a much more simplified approach ==========
		warn.warn("This is a simplified implementation")
		cols_keep = np.hstack([
			["biomass", "stem_density" ], 
			vi_df.columns[np.where([cln.endswith("%d" % window) for cln in vi_df.columns])],
			["lagged_biomass"]
			])
		warn.warn("In SOls code he used zeros here")
		# ========== Drop any NA rows ==========
		data      = vi_df[cols_keep].dropna()

	elif mode == "Sol":
		# ========== This is where i implement Sols exact approach ==========
		# +++++ A container to hold the kept columns  +++++ 
		cols_keep = []
		clnm =  vi_df.columns.values
		for cn in clnm:
			# Test to see if the column is one of the ones i want to keep
			# if cn in ["biomass", "stem_density", "lagged_biomass"]:
			# 	cols_keep.append(cn)
			if cn.startswith("LANDSAT"):
				# The VI datasets, check the lengtu
				if int(cn.split("_")[-1]) <= window:
					cols_keep.append(cn)
			elif cn in ['site']:
				pass
			else:
				cols_keep.append(cn)

		# ========== Fill in any NA rows ==========
		# data      = vi_df[cols_keep].fillna(value=0)
		data      = vi_df[cols_keep].dropna()

	# =========== drop rows with 0 variance ===========
	data.drop(data.columns[data.std() == 0], axis=1, inplace=True)

	# =========== Pull out the data used for prediction ===========
	# X = data.values[:, :-1]
	X = data.drop(["lagged_biomass"], axis = 1).astype("float32")

	# =========== Pull out the data the is predicted ===========
	# y = data.values[:, -1]
	y = (data["lagged_biomass"]).astype("float32")

	# ============ Return the filtered data  ============
	return X, y, cols_keep


# ==============================================================================
# ==============================================================================

if __name__ == '__main__':
	main()