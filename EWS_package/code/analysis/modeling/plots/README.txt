README for 'plots' directory
12/19/19
Sol Cooperdock

Within this directory are two R scripts which will produce plots. 'rf_class_final_biomass_nointerp_PD_plots' is just a chunk taken from the actual modeling script that was taken out because it really slowed down the script when trying to make partial dependence plots. This script will take your final model and create partial dependence plots for both the classification model and the regression model. 'use_model_spatial_plots' makes a series of spatial plots. It can show the pixel-by-pixel observed changes in biomass, observed changes in ndvi, as well as predicted changes in biomass according to the final model.