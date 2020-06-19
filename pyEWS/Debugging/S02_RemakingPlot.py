# Dataset fixer 
"""
Script goal, 

Find the bug in the DBH, check every site to make sure 

The bug is insane DBH values that have a flow on effect on biomass

"""

# ==============================================================================

__title__ = "Remake Plots"
__author__ = "Arden Burrell"
__version__ = "v1.0(21.05.2020)"
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


import matplotlib.pyplot as plt
import xarray as xr
import ipdb
import rasterio
# ==============================================================================

def main():
	# ===========  import the data ==========
	da = xr.open_rasterio("./EWS_package/data/plots/spatial/ARDEN_raster.tif")
	src = rasterio.open("./EWS_package/data/plots/spatial/ARDEN_raster.tif")
	# ========== Fix the data ==========
	da = da.rename({"band":"time", "x":"longitude", "y":"latitude"}) 
	da = da.where(da > -1)

	ipdb.set_trace()

	

# ==============================================================================

if __name__ == '__main__':
	main()