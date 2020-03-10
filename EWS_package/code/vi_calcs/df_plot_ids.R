#####
# In case plot IDs are not written for the the metrics
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

# Which VI data sets to include in analysis, can edit to remove or add data sets
data_set_vec=c("LANDSAT") #<- this is all of them


if(NASAcomp_run==1) {
  leadpath = "/att/nobackup/scooperd/"
} else {
  leadpath = "/Volumes/"
}

VIs = c('ndvi','msi','ndii','satvi','ndvsi','nirv','nbr','ndwi','tvfc','psri')

#### VI Data Prep and Processing ####
for(data_set in data_set_vec) {
  for(VI in VIs){
    # Read in vi data
    raw_vi_df = read.csv(paste0(leadpath,"scooperdock/EWS/data/VIs/",data_set,"/full_",data_set,"_",VI,"_median.csv"))
    out = raw_vi_df[,c('X','Plot_ID')]
    
    write.csv(out,paste0(leadpath,"scooperdock/EWS/data/vi_metrics/",VI,"_plot_IDs.csv"))
    # dirs = paste0(leadpath,'scooperdock/EWS/data/vi_metrics/',data_set,'/median/',VI,'/',analysis_list[[data_set]])
    # for(d in dirs){
    #   dir.create(d,recursive = T)
    # }
    # 
    # # Clean up dataframe
    # vi_df = clean_vi_df(raw_vi_df) #Uses the previously defined function "clean_vi_df" to remove some columns
    # 
    # # Run Calculation
    # output_cube = vi_df_calc() #Uses function from vi_calcs to run the calculations
    # 
    # outpath_input = paste(outpath,data_set,VI,sep="/")
    # 
    # # Write to csv
    # write_vi_dfs(output_cube,outpath_input) #Uses function from vi_calcs to write the output_cube data
    # #into a csv file in the folder of outpath/data_set
  }
}


