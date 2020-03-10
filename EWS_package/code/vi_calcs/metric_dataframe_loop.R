#####
# scooperdock@whrc.org
# Modified from Kylen's scripts
# Make heatmaps with time window on the x-axis and mortality threshold on the Y.
#####


#### Clear environment and load packages ####
rm(list=ls(all=TRUE))
require(doParallel)
require(gplots)
require(earlywarnings)
require(plyr)

#### Set parameters ####
# Run on lcomp?
NASAcomp_run = T


# Which VI data sets to include in analysis
data_set_vec=c("LANDSAT")

# For each dataset, which to analysis to include in LM?
analysis_list = list(
    "GIMMS" = c("trend","pulse","ddj","ews"),
    "MODIS" = c("trend","pulse","ddj"),
    "MODISTERRA" = c("trend","pulse","ddj"),
    "LANDSAT" = c("trend","pulse","ddj","ews")
   )

# Sub metrics. This is only really used for ddj and ews, so it's a bit inefficient to have it for both but oh well...
sub_metrics = list(
  "trend" = c("trend"),
  "pulse" = c("pulse"),
  "ews" = c("ar1","sd","kurt","densratio","cv","acf1","sk"),
  "ddj" = c("S2.t","Diff2.t","Lamda.t","TotVar.t")
)


stat_list = list(
  "GIMMS" = c("mean"),
  "MODIS" = c("mean","median"),
  "MODISTERRA" = c("mean","median"),
  "LANDSAT" = c("mean","max")
)

# Run through different flags. 0 means only best quality data. 1 means good quality but possibly with some issues and 3
# means use all data.
flag_list = list(
  "GIMMS" = c(3),
  "MODIS" = c(0,1,3),
  "MODISTERRA" = c(0,1,3),
  "LANDSAT" = c(3)
)

args <- commandArgs(trailingOnly = T)
VIs = args
print(VIs)

# For cafi only
shift_list = list(
  "trend" = .5,
  "pulse"= 0,
  "ews" = 0,
  "ddj" = 0
)

# Set timelag start
tl_start = 1

# Time lag lengths for VI trend, pulse, ews, and ddj, separated by data set
tl_list = list(
  # GIMMS
  "GIMMS" = list(
    "trend" = 5:20,
    "pulse" = 5:20,
    "ews" = 5:20,
    "ddj"  = 5:20
  ),
  # MODIS
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
  )
)
# 3x3 or 1pixel selection?
pix_select_list = list(
  "GIMMS" = "1pixa",
  "MODIS" = "1pixa",
  "MODISTERRA" = "1pixa",
  "LANDSAT" = "3x3a"
)

# Null value
null_val = -32768

#### Some quick variable definition based on the inputs ####
#Set path lead based on whether we're on lcomp or not:
if(NASAcomp_run) {
  leadpath = "/att/nobackup/scooperd/"
} else {
  leadpath = "/Volumes/"
}

# Source heatmap functions
source("/home/scooperd/r_scripts/code/analysis/functions/create_heatmap_sol.R")
#source("H:/scooperd/r_scripts/code/analysis/functions/create_heatmap_sol.R")
# Set full y_range
y_range = 1981:2017


# Set rule for missing years. 
miss_rule = c(25,2)

#### Functions ####


clean_df = function(in_df) {
  
  # Eliminate null vals
  in_df[in_df == null_val] = NA
  
  add_sites = sites[!(sites %in% rownames(in_df))]
  if(length(add_sites)>0){
    in_df[add_sites,] = NA
  }
  
  # Match up all column names
  add_years = y_range[!(paste0("X",y_range) %in% colnames(in_df))]
  if(length(add_years)>0) {
    in_df[,paste0("X",add_years)] = NA
  }
  
  # Ensure column order is set properly
  in_df = in_df[,paste0("X",y_range)]
  in_df = in_df[sites,]
  
  #Arrange rownames and remove sites we have no observations from
  in_df = in_df[sites,]
  
  return(in_df)
}

# Shift vi results for trend
shift_vi = function(in_df,shift_fraction) {
  temp_shift_df = round(shift_df*shift_fraction,0)
  new_df = in_df
  for(r in 1:dim(in_df)[1]) {
    for(y in y_range) {
      if(!is.na(temp_shift_df[r,paste0("X",y)])) {
        new_y = y-temp_shift_df[r,paste0("X",y)]
        if(new_y>=min(y_range)){
          new_df[r,paste0("X",y)] = in_df[r,paste0("X",new_y)]
        }
      }
    }
  }
  return(new_df)
}



#Read in lengths of time since last survey
raw_shift_df = read.csv(paste0(leadpath,"scooperdock/EWS/data/psp/surv_interval_filled.csv"))
raw_shift_df = arrange(raw_shift_df,X)
rownames(raw_shift_df) = raw_shift_df[,"X"]
sites = rownames(raw_shift_df)
shift_df = clean_df(raw_shift_df)
adj_YT = sites[grep("11_",sites)]
for(i in 1:length(adj_YT)){
  adj_YT[i] = paste0("11_",as.numeric(strsplit(adj_YT[i],"_")[[1]][2]))
}
sites[grep("11_",sites)] = adj_YT

obs = shift_df
for (c in 1:dim(obs)[2]){
  for (r in 1:dim(obs)[1]){
    obs[r,c] = paste0(colnames(obs)[c],"_",rownames(obs)[r])
  }
}

mort_vi_df = data.frame("obs" = unlist(obs))
      #### Read and Process VI Results ####
      # Populate with premade csvs
for(data_set in data_set_vec) {#Loops through data sets
  for(VI in VIs){
    
    for(an_type in analysis_list[[data_set]]) {#Loop through each analysis
      for(metric in sub_metrics[[an_type]]) {#Loop through each metric
        for(tl_length in tl_list[[data_set]][[an_type]]) {#Loop through each timelag
          temp_df = read.csv(paste0(leadpath,"scooperdock/EWS/data/vi_metrics/",data_set,"/",VI,"/",an_type,"/",metric,"_tlength",tl_length,".csv"),row.names=1)
          names = read.csv(paste0(leadpath,"scooperdock/EWS/data/vi_metrics/",VI,"_plot_IDs.csv"),row.names = 'X',stringsAsFactors = F)
          temp_df = temp_df[names$X.1,]
          rownames(temp_df) = names$Plot_ID
          temp_df = clean_df(temp_df)
          
          #Shift trend results to midpoint
          #temp_df = shift_vi(temp_df,shift_list[[an_type]])
          
          mort_vi_df[paste(data_set,VI,an_type,metric,tl_length,sep="_")] = unlist(temp_df)
          
        }
        #print(paste(data_set,an_type,metric,stat,"flag:",flag,"finished!",sep = " "))
      }
      
      
    }

  }
}
write.csv(mort_vi_df,paste0(leadpath,"scooperdock/EWS/data/vi_metrics/metric_dataframe_",VI,"_noshift.csv"))
