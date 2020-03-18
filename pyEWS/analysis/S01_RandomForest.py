"""
Script goal, 

To learn about the existing dataset through the use of Random Forest
	- Open the datasets the Sol used in his classification 
	- Perform a basic random forest on them
	- Bonus Goal, test out some CUDA 

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
# Import packages
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
import idpb

# ==============================================================================
def main():
	pass

# ==============================================================================

def df_proccessing():
	"""
	This function opens and performs all preprocessing on the dataframes
	"""
	ipdb.set_trace()
	pass

# ==============================================================================
if __name__ == '__main__':
	main()