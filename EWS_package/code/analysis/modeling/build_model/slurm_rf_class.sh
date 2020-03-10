#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50G
#SBATCH --mail-type=ALL
#SBATCH --job-name=cl_all
R --slave -f rf_class_final_biomass_nointerp.R 
