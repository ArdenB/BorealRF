"""
Function for print eta of long loops

"""
#==============================================================================

__title__ = "lineflick"
__author__ = "Arden Burrell"
__version__ = "1.0(14.06.2019)"
__email__ = "arden.burrell@gmail.com"

# ========== Function to keep track od time ==========
import pandas as pd 
import sys

def lineflick(line, line_max, t0, lineflick=1000):
	if (line % lineflick == 0):
		string = ("\rLine: %d of %d" % 
					(line, line_max))
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