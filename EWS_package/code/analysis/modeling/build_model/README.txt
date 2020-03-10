These scripts build the actual model. They are the final step.

'rf_class_final_biomass_nointerp.R' creates the classification model and 'rf_final_afterclass_whilerunning_biomass_nointerp_loadrda.R' creates the individual regression models.

These are separate scripts to speed up the process. They can create models trained on all sites, boreal sites, or only ABoVE sites and they produce comparisons of model fit on several different spatial regions. The way it is set up right now is to run 100 classification models and simultaneously run 100 regression models. However, each time a new classification model is better than previous ones, a new set of 100 regression models are started until 100 regression models are run on the best classification model of the 100.