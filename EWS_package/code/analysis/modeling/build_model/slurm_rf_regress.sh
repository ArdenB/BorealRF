#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50G
#SBATCH --mail-type=ALL
#SBATCH --job-name=reg_all
R --slave -f rf_final_afterclass_whilerunning_biomass_nointerp_loadrda.R 
