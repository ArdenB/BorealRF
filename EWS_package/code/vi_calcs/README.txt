README for 'vi_calcs' directory
12/19/19
Sol Cooperdock

This directory has one subdirectory, 'functions', which houses scripts that the script 'full_df_calcs_loop' imports as functions.
In addition there are 4 scripts:
'df_plot_ids' is a script that needed to be made because the script 'metric_dataframe_loop' wasn't including the plot ids when it wrote out the final dataframe. In order to get everything in order, I needed to write out the ids to be read in when those dataframes were used. In the future if these scripts are used again, I would just add that into the original script so that the dataframes written out to file have the plot ids.
'full_df_calcs_loop' calculates the statistical metrics for each VI. Currently the way it's set up is so that I could run a loop in slurm and submit a different job for each VI, but it can easily be adjusted to loop in different ways. I would not write the loop within the R script however, unless you can do it in parallel because each VI takes a couple days to finish these calculations.
'landsat_vis_adjust' takes the VI data that Logan provided us and turns it into a format that everything else is in for this workflow.
'metric_dataframe_loop' takes all the calculated metrics and puts them into a single dataframe to be read in later.