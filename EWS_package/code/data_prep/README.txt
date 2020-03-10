README for 'data_prep' directory
12/19/19
Sol Cooperdock

This directory has one subdirectory ('database_consolidation', which has scripts for dealing with all the individual PSP databases) and several prep scripts.
'biomass_interpolation' creates a matrix with interpolated biomass values for years in between forest inventory dates. There are some notes in the script itself as well as in the word document in the topmost directory of this package 'EWS_methodology.doc' which describe how this interpolation is done.
'climate_trends' creates matrices for each of the ClimateNA variables as 30 year trends before each year as 30 year averages before each year.
'create_vi_df' is a timesaver script so that the database for modeling can be quickly read in before each time you want to run the model, it creates a .Rda file which has all the variables needed for modeling in one extensive dataframe. It does the same for the correlations data.
'extractBurnLandSat' determines whether a Landsat pixel that a plot is in was burned on any year according to the Canadian and Alaska Large Fire Database polygons. 
'PSP_BA_dec_ever' creates a dataframe with deciduous and evergreen fractions for each side by basal area (Richard Massey needed this).
'PSP_boreal_fraction' is much like the above, but it creates dataframes that indicate the fraction of trees at each site that were boreal species by biomass, basal area, and stem count. I chose which species were boreal, there can be some debate about others.
'sp_comp_dataframe' is similar to 'biomass_interpolation' except that it is interpolating the fraction of trees at each site that were each species. This creates over 100 separate dataframes that correspond to each species. I didn't do the complicated interpolation that I did above because it doesn't have quite as large of an impact (because we aren't directly building the model on this information).
'survey_years' consolidates dataframes from each province that tell when each site was inventoried.