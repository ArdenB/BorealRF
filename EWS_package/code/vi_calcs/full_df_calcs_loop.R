#####
# scooperdock@whrc.org
# Modified from Kylen's scripts
# Calculate NDVI Metrics for all years of all sites using range of time windows.
# Store results in CSVs for later analysis.
#####

#### Clear environment and load packages ####
rm(list=ls(all=TRUE))
require(doParallel)
require(gplots)
require(earlywarnings)
require(plyr)

#### Set parameters ####

# Run on NASA?
NASAcomp_run = 1

# Which VI data sets to include in analysis, can edit to remove or add data sets,just using Landsat
data_set_vec=c("LANDSAT") 

# For each dataset, which analyses to include
analysis_list = list(
  "GIMMS" = c("trend","pulse","ddj","ews"),
  "MODIS" = c("ddj","pulse","trend"),
  "MODISTERRA" = c("trend","pulse","ddj"),
  "LANDSAT" = c("ddj","ews","trend","pulse")
)


#This takes an argument from a batch job scheduler. The way the script is written is so that I can input a vector of all the VIs
#along with the script path and it will submit this script to be run for each VI. Alternatively you can just create a vector with
#the names of the VI or VIs that you want to run as below, it will loop through however many elements VIs has in it.
args <- commandArgs(trailingOnly = T)
VIs = args
print(VIs)

#VIs = c("SATVI")



# Sub metrics. This is only really used for ddj and ews, so it's a bit inefficient to have it for both but oh well...
sub_metrics = list(
  "trend" = c("trend"),
  "pulse" = c("pulse"),
  "ews" = c("ar1","sd","sk","kurt","cv","densratio","acf1"),
  "ddj" = c("TotVar.t","Diff2.t","Lamda.t","S2.t")
)

# Set timewindow start. 
tl_start = 0

# Time lag lengths for VI trend, pulse, ews, and ddj, separated by data set. Creates lists for each of these variables which include all 
#numbers from the first digit to the last. This will be used to create the csvs of different time lags
tl_list = list(
  # GIMMS, 
  "GIMMS" = list(
    "trend" = 5:20,
    "pulse" = 5:20,
    "ews" = 5:20,
    "ddj"  = 5:20
  ),
  "MODIS" = list(
    "trend" = 5:13,
    "pulse" = 5:13,
    "ews" = 5:13,
    "ddj"  = 5:13
  ),
  "MODISTERRA" = list(
    "trend" = 5:13,
    "pulse" = 5:13,
    "ews" = 5:13,
    "ddj"  = 5:13
  ),
  "LANDSAT" = list(
    "trend" = 5:20,
    "pulse" = 5:20,
    "ews" = 5:20,
    "ddj"  = 5:20
  ),
  "LANDSATMAX" = list(
    "trend" = 5:20,
    "pulse" = 5:20,
    "ews" = 5:20,
    "ddj"  = 5:20
  )
)

#I think this is deprecated, but I was too nervous to delete it
pix_select_list = list(
  "GIMMS" = "1pixa",
  "MODIS" = "1pixa",
  "MODISTERRA" = "1pixa",
  "LANDSAT" = "3x3a",
  "LANDSATMAX" = "3x3a"
)

# Null value
null_val = -32768

# Set outpath
outpath = "scooperdock/EWS/data/vi_metrics/"

#### Some quick variable definition based on the inputs #### I commented these for now because I'm just going to use the path above. 
#### I may need to use this section of code again
# Set path lead based on whether we're on lcomp or not:
if(NASAcomp_run==1) {
  leadpath = "/att/nobackup/scooperd/"
} else {
  leadpath = "/Volumes/"
}
outpath = paste0(leadpath,outpath)

# Source vicalcs function
source("/att/nobackup/scooperd/EWS_package/code/vi_calcs/functions/vi_calcs.R")


# Set full y_range
y_range = 1981:2017

# Set rule for missing years. 
miss_rule = c(25,2)

#### Functions ####

# Clean up data frames
clean_vi_df = function(df) {
  
  # Eliminate null vals
  df[df == null_val] = NA
  
  
  elim_cols = which(colnames(df) %in% c("region","ref")) #Deletes columns "region" and "ref"
  if(length(elim_cols>0)) {
    df = df[ ,-elim_cols]
  }
  
  # Match up all column names
  add_years = y_range[!(paste0("X",y_range) %in% colnames(df))] 
  if(length(add_years)>0) {
    df[,paste0("X",add_years)] = NA
  }
  
  # Ensure column order is set properly
  df = df[,paste0("X",y_range)]
  
  #if((data_set=="LANDSAT")) {
  #  df[,paste0("X",y_range[1]:cafi_landsat_ystart)] = NA
  #}
  
  return(df)
}

#### VI Data Prep and Processing ####
for(data_set in data_set_vec) {
  for(VI in VIs){
    # Read in vi data
    raw_vi_df = read.csv(paste0(leadpath,"scooperdock/EWS/data/VIs/",data_set,"/full_",data_set,"_",VI,"_median.csv"),row.names="X")
    
    dirs = paste0(leadpath,'scooperdock/EWS/data/vi_metrics/',data_set,'/median/',VI,'/',analysis_list[[data_set]])
    for(d in dirs){
      dir.create(d,recursive = T)
    }
    
    # Clean up dataframe
    vi_df = clean_vi_df(raw_vi_df) #Uses the previously defined function "clean_vi_df" to remove some columns
    
    # Run Calculation
    output_cube = vi_df_calc() #Uses function from vi_calcs to run the calculations
    
    outpath_input = paste(outpath,data_set,VI,sep="/")
    
    # Write to csv
    write_vi_dfs(output_cube,outpath_input) #Uses function from vi_calcs to write the output_cube data
    #into a csv file in the folder of outpath/data_set
  }
}


