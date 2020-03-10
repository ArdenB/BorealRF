README for 'functions' directory
12/19/19
Sol Cooperdock

Not much to say here. 'vi_calcs' is a script with a series of functions that are called by the script 'full_df_calcs_loop' above. The subdirectory 'modified_earlywarnings_package' is in turn called by 'vi_calcs'. The reason it's here and not just imported in the script is that a number of the original functions from the 'earlywarnings' package create plots which is unnecessary and slows things down. These modified version have the plots commented out.