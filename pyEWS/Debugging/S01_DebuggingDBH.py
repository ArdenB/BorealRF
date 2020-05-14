# Dataset fixer 
"""
Script goal, 

Find the bug in the DBH, check every site to make sure 

The bug is insane DBH values that have a flow on effect on biomass

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
import seaborn as sns

# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf

# ==============================================================================

def main():
	# ========== setup the path ==========
	path = "./EWS_package/data/raw_psp/"
	# allv = False #look at everyvalue rather than just the site max
	allv = True #look at everyvalue rather than just the site max

	# ========== Make a dictionary ==========
	fnames = glob.glob(path+"**/checks/*.csv")

	# ========== Create a container to hold site, region and DBHmax ==========
	sitesumm = OrderedDict()
	t0       = pd.Timestamp.now()
	lines    = len(fnames)

	# ========== Loop over each site to proccess them one at a time ==========
	for num, fn in enumerate(fnames):
		# +++++ print the line number and time remaining  +++++
		_lnflick(num, lines, t0, lineflick=1000, desc="Site")


		# +++++ load the data check file +++++
		dfin = pd.read_csv(fn, index_col=0)

		# +++++ split the string to pull out the key infomation +++++
		_, _, _, _, region, _, sitecom = (fn).split("/")
		sitenm = sitecom.split("_check.csv")[0]

		# +++++ pull out the DBH values by subsetting the column names +++++
		colnames = []
		for col in dfin.columns:
			if "dbh" in col:
				colnames.append(col)

		dfsub  = dfin[colnames]
		dbhmax = bn.nanmax(dfsub.values)
		
		if allv:
			# +++++ Pull out all the DBH values +++++
			dbhval = dfsub.values.reshape([-1])

			# +++++ use loop to add avery DBH values to the OrderedDict +++++
			for obs, dbh in enumerate(dbhval):
				# +++++ Append the site infomation to the ordered dict +++++
				sitesumm["%d.%05d" % (num, obs)] = {"region":region, "site":sitenm, "DBH":dbh}

			# ipdb.set_trace()
		else:
			# +++++ Append the site infomation to the ordered dict +++++
			sitesumm[num] = {"region":region, "site":sitenm, "DBH":dbhmax}
		
		
	# ========== Convert to a dataframe ==========
	print("Starting dataframe creation at:", pd.Timestamp.now())
	df = pd.DataFrame.from_dict(sitesumm, orient="index")
	df = df.astype({"region":"str", "site":"str", "DBH":"float64",})
	# ========== make some plots and do some checks ==========
	ax = plt.subplot()
	# fig, (ax, ax2) = plt.subplots(2, 1)
	sns.boxplot(   x="region", y="DBH", data=df, ax = ax)
	# sns.violinplot(x="region", y="maxDBH", data=df, ax = ax2)
	plt.show()
	ipdb.set_trace()

# ==============================================================================
# ==============================================================================
def Filecheck():
	"""
	script to quickly check the files
	"""
	
# ==============================================================================
# ==============================================================================

def _lnflick(line, line_max, t0, lineflick=1000, desc="line"):
	"""Function to allow for quick and rreable like counts"""

	if (line % lineflick == 0):
		string = ("\r%s: %d of %d" % 	(desc, line, line_max))
		if line > 0:
			# TIME PER LINEFLICK
			lfx = (pd.Timestamp.now()-t0)/line
			lft = str((lfx*lineflick))
			trm = str(((line_max-line)*(lfx)))
			string += (" t/%d lines: %s. ETA: %s" % (
				lineflick,lft, trm) )
			
		sys.stdout.write(string)
		sys.stdout.flush()
	else:
		pass

# ==============================================================================
# ==============================================================================

if __name__ == '__main__':
	main()