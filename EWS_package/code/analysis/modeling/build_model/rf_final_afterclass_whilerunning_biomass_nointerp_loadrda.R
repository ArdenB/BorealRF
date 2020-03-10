
library(dplyr)
library(plyr)
library(randomForest)
library(ggplot2)
library(caret)
library(doParallel)
library(car)
library(cowplot)
library(gridExtra)
library(data.table)
library(sp)
library(raster)
library(rgdal)


y_range = 1981:2017
null_val = -32768
leadpath = "/att/nobackup/scooperd/EWS_package/"

#How many quantiles
num_quants = 7

#For argument input from slurm script
args <- commandArgs(trailingOnly = T)
region = args
print(region)

#Can also directly input region
region='all'

#Which VIs do we want to try? 
VIs = c('ndvi','psri','ndii','ndvsi','msi','nirv','ndwi','nbr','satvi','tvfc')

#### Functions ####


# Clean up data frames
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
  
  # Ensure column and row order is set properly
  in_df = in_df[,paste0("X",y_range)]
  in_df = in_df[sites,]
  
  return(in_df)
}


get_density = function(x,y,...){
  dens <- MASS::kde2d(x,y,...)
  ix <- findInterval(x,dens$x)
  iy <- findInterval(y,dens$y)
  ii <- cbind(ix,iy)
  return(dens$z[ii])
}

set_quants = function(vect,quantiles){
  out = vector(length = length(vect))
  quantiles[1] = quantiles[1] - 0.1
  for(i in 2:length(quantiles)){
    index = which(vect>quantiles[i-1]&vect<=quantiles[i])
    out[index] = i-1
  }
  return(out)
}

####Read in a few dataframes
#Interval between measurents
raw_shift_df = read.csv(paste0(leadpath,"data/psp/modeling_data/surv_interval_filled.csv"))
raw_shift_df = arrange(raw_shift_df,X)
rownames(raw_shift_df) = raw_shift_df[,"X"]
sites = rownames(raw_shift_df)
shift_df = clean_df(raw_shift_df)

#GPS coordinates of all sites
site_loc = read.csv(paste0(leadpath,"data/raw_psp/All_sites_101218.csv"),row.names = 'Plot_ID')
#YT site names need adjusting because they were read in differently in two different places
adj_YT = site_loc[grep("11_",rownames(site_loc)),]
for(i in 1:nrow(adj_YT)){
  rownames(adj_YT)[i] = paste0("11_",sprintf("%03.0f",as.numeric(strsplit(rownames(adj_YT)[i],"_")[[1]][2])))
}
rownames(site_loc)[grep("11_",rownames(site_loc))] = rownames(adj_YT)
site_loc = site_loc[sites,]


#Read in ecoregions shape and determine what sites are in each ecoregion
ecoregions = readOGR(paste0(leadpath,"data/ancillary_files/Ecoregions/NA_CEC_Eco_Level3.shp"))
ecoregions = spTransform(ecoregions,CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
site_loc_sp = site_loc[!is.na(site_loc$Longitude),]
site_loc_sp = SpatialPointsDataFrame(coords = site_loc_sp[,2:3],data = data.frame(rownames(site_loc_sp)),proj4string = CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
#site_loc_sp = raster::intersect(site_loc_sp,above)
site_regions = vector(length = length(site_loc_sp))
names(site_regions) = site_loc_sp$rownames.site_loc_sp.
ecoregions = raster::intersect(ecoregions,site_loc_sp)
regions = unique(ecoregions$NA_L2NAME)
for(i in regions){
  intersection = raster::intersect(site_loc_sp,ecoregions[ecoregions$NA_L2NAME==i,])
  site_regions[intersection$rownames.site_loc_sp.] = i
}


#Read in boreal species fraction dataframe and determine which sites have >50% boreal species
bor_frac = read.csv(paste0(leadpath,"data/psp/databases/PSP_boreal_frac_biomass_v1.csv"),row.names = 'X')
bor_frac = rowMeans(bor_frac,na.rm=T)
bor_frac = bor_frac[sites]
bor_over50 = names(bor_frac)[bor_frac>0.5]

#Read in above shape and determine which sites are in the above domain
above = readOGR(paste0(leadpath,"data/ancillary_files/ABoVE_Study_Domain_Final/ABoVE_Study_Domain.shp"))
above = spTransform(above,CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
site_loc_sp = site_loc[!is.na(site_loc$Longitude),]
site_loc_sp = SpatialPointsDataFrame(coords = site_loc_sp[,2:3],data = data.frame(rownames(site_loc_sp)),proj4string = CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
site_loc_sp = raster::intersect(site_loc_sp,above)
above_loc = site_loc_sp@data$rownames.site_loc_sp.

#Same for above core domain
above_core = above[above$Region=='Core Region',]
site_loc_sp = raster::intersect(site_loc_sp,above_core)
above_core_loc = site_loc_sp@data$rownames.site_loc_sp.

#Load in vi_df and correlations
load(paste0(leadpath,"data/models/input_data/vi_df_all_2019-10-30.Rda"))
load(paste0(leadpath,"data/models/input_data/correlations_2019-10-09.Rda"))

#Determine quantile steps
quant = quantile(vi_df$lagged_biomass,seq(0,1,length.out = num_quants+1),na.rm=T)

#Remove certain sites depending on what region we're modeling
if(region=='boreal'){
  vi_df[!vi_df$site %in% bor_over50,] = NA
}else if(region=='above'){
  vi_df[!vi_df$site %in% above_loc,] = NA
}else if(region=='above_core'){
  vi_df[!vi_df$site %in% above_core_loc,] = NA
}

#Create a vector that relates to the length of time windows for each variable in vi_df (only valid for VIs)
windows = colnames(vi_df)
for(i in 1:length(windows)){
  win = strsplit(windows[i],"_")[[1]]
  windows[i] = as.numeric(win[length(win)])
}
windows = as.numeric(windows)

#find folder and most recently created classification model (or manually input another date if you want an earlier model)
folder = paste0(leadpath,"data/models/rf_models/",region,"/",paste0(VIs,collapse="_"),"/")
day = max(list.files(folder))
##day = '2019-09-16'
folder1 = paste0(folder,day)

#Wait for at least one classification model to finish before starting regression (only matters if running this script while running class script)
while(!file.exists(paste0(folder1,"/final_model.rds"))){
  print('zzz...')
  Sys.sleep(300)
}

#If regression script is restarted, this will determine how many iterations of 100 regression models have been started and start at the next one
regress_folders = list.files(paste0(folder,day))
regress_folders = regress_folders[grep('regress',regress_folders)]
class_runs = max(as.numeric(gsub("([a-z._]+)","",regress_folders)),na.rm=T)
if(class_runs==-Inf){
  class_runs=0
}

#Similarly to above, if the script was restarted and it had been running regressions in the regress_with_regions folder, this will
#rename that folder and increment the above value by one so that we can start a new regression run
if(dir.exists(paste0(folder1,"/regress_with_regions/"))){
  mod_time = file.info(paste0(folder1,"/final_model.rds"))$mtime
  if(file.info(paste0(folder1,"/regress_with_regions/"))$mtime<mod_time){
    file.rename(paste0(folder1,"/regress_with_regions/"),paste0(folder1,"/regress_",class_runs))
    class_runs = class_runs+1
    
  }
}

###Functions for producing faceted regression plots and statistics for different regions 
eco_cv_function = function(data,eco){
  if(eco=='all'){
    df = data.frame(data$predicted,data$observed,data$set)
    
  }else{
    df = data.frame(data$predicted[data$ecoregion==eco],data$observed[data$ecoregion==eco],data$set[data$ecoregion==eco])
  }
  colnames(df) = c('predicted','observed','set')
  df = df[!is.na(df$observed),]
  if(nrow(df[df$set=='Withheld sites',])>10){
    stats = data.frame(c('Training sites','Withheld sites'),matrix(nrow=2,ncol = 2))
    colnames(stats) =c('set','cv_r2','n')
    
    df$density = get_density(df$observed,df$predicted,n=1000)
    stats$cv_r2[stats$set=='Withheld sites'] = 1-(sum((df$observed[df$set=='Withheld sites']-df$predicted[df$set=='Withheld sites'])^2)/sum((mean(df$observed[df$set=='Withheld sites'])-df$observed[df$set=='Withheld sites'])^2))
    stats$cv_r2[stats$set=='Training sites'] = 1-(sum((df$observed[df$set=='Training sites']-df$predicted[df$set=='Training sites'])^2)/sum((mean(df$observed[df$set=='Training sites'])-df$observed[df$set=='Training sites'])^2))
    stats$n[stats$set=='Withheld sites'] = sum(df$set=='Withheld sites')
    stats$n[stats$set=='Training sites'] = sum(df$set=='Training sites')
    plot = ggplot(df) + geom_point(aes(x=observed,y=predicted,color = density)) + scale_color_gradientn(colors = c('yellow','yellow2','orangered','red','darkred')) +
      theme(panel.background = element_blank(),legend.position = 'none') + geom_abline(linetype = 2,slope = 1) +xlab ('Observed values') + ylab('Predicted values') +
      scale_x_continuous(limits = c(-1,1)) + scale_y_continuous(limits = c(-1,1)) + ggtitle(eco) +
      annotate('text', label = paste0('CV R2 = ',round(stats$cv_r2,2)),x = -0.9,y=1) + annotate('text', label = paste0('n = ',stats$n),x = -0.9,y=0.9)
    #labels = c(train = "Model training", test = "Withheld")
    return(plot + facet_grid(.~set)) 
    
  }
}

eco_stat_func = function(data,eco){
  if(eco=='all'){
    df = data.frame(data$predicted,data$observed,data$set)
    
  }else{
    df = data.frame(data$predicted[data$ecoregion==eco],data$observed[data$ecoregion==eco],data$set[data$ecoregion==eco])
  }
  colnames(df) = c('predicted','observed','set')
  df = df[!is.na(df$observed),]
  
  stats = data.frame(matrix(nrow=2,ncol = 2))
  colnames(stats) =c('cv_r2','n')
  rownames(stats) = c('Train','Test')
  
  
  stats['Test','cv_r2'] = 1-(sum((df$observed[df$set=='Withheld sites']-df$predicted[df$set=='Withheld sites'])^2)/sum((mean(df$observed[df$set=='Withheld sites'])-df$observed[df$set=='Withheld sites'])^2))
  stats['Train','cv_r2'] = 1-(sum((df$observed[df$set=='Training sites']-df$predicted[df$set=='Training sites'])^2)/sum((mean(df$observed[df$set=='Training sites'])-df$observed[df$set=='Training sites'])^2))
  stats['Test','n'] = sum(df$set=='Withheld sites')
  stats['Train','n'] = sum(df$set=='Training sites')
  
  return(stats)
}

pr_cv_func = function(data,pr){
  df = data.frame(data$predicted[grep(paste0(pr,"_"),data$site)],data$observed[grep(paste0(pr,"_"),data$site)],data$set[grep(paste0(pr,"_"),data$site)])
  colnames(df) = c('predicted','observed','set')
  df = df[!is.na(df$observed),]
  if(nrow(df[df$set=='Withheld sites',])>10){
    stats = data.frame(c('Training sites','Withheld sites'),matrix(nrow=2,ncol = 2))
    colnames(stats) =c('set','cv_r2','n')
    
    df$density = get_density(df$observed,df$predicted,n=1000)
    stats$cv_r2[stats$set=='Withheld sites'] = 1-(sum((df$observed[df$set=='Withheld sites']-df$predicted[df$set=='Withheld sites'])^2)/sum((mean(df$observed[df$set=='Withheld sites'])-df$observed[df$set=='Withheld sites'])^2))
    stats$cv_r2[stats$set=='Training sites'] = 1-(sum((df$observed[df$set=='Training sites']-df$predicted[df$set=='Training sites'])^2)/sum((mean(df$observed[df$set=='Training sites'])-df$observed[df$set=='Training sites'])^2))
    stats$n[stats$set=='Withheld sites'] = sum(df$set=='Withheld sites')
    stats$n[stats$set=='Training sites'] = sum(df$set=='Training sites')
    plot = ggplot(df) + geom_point(aes(x=observed,y=predicted,color = density)) + scale_color_gradientn(colors = c('yellow','yellow2','orangered','red','darkred')) +
      theme(panel.background = element_blank(),legend.position = 'none') + geom_abline(linetype = 2,slope = 1) +xlab ('Observed values') + ylab('Predicted values') +
      scale_x_continuous(limits = c(-1,1)) + scale_y_continuous(limits = c(-1,1)) + ggtitle(names(provs)[provs==pr]) +
      annotate('text', label = paste0('CV R2 = ',round(stats$cv_r2,2)),x = -0.9,y=1) + annotate('text', label = paste0('n = ',stats$n),x = -0.9,y=0.9)
    
    return(plot + facet_grid(.~set))
  }
}

pr_stat_func = function(data,pr){
  df = data.frame(data$predicted[grep(paste0(pr,"_"),data$site)],data$observed[grep(paste0(pr,"_"),data$site)],data$set[grep(paste0(pr,"_"),data$site)])
  colnames(df) = c('predicted','observed','set')
  df = df[!is.na(df$observed),]
  
  stats = data.frame(matrix(nrow=2,ncol = 2))
  colnames(stats) =c('cv_r2','n')
  rownames(stats) = c('Train','Test')
  
  
  stats['Test','cv_r2'] = 1-(sum((df$observed[df$set=='Withheld sites']-df$predicted[df$set=='Withheld sites'])^2)/sum((mean(df$observed[df$set=='Withheld sites'])-df$observed[df$set=='Withheld sites'])^2))
  stats['Train','cv_r2'] = 1-(sum((df$observed[df$set=='Training sites']-df$predicted[df$set=='Training sites'])^2)/sum((mean(df$observed[df$set=='Training sites'])-df$observed[df$set=='Training sites'])^2))
  stats['Test','n'] = sum(df$set=='Withheld sites')
  stats['Train','n'] = sum(df$set=='Training sites')
  
  return(stats)
}

major_reg_cv_function = function(data,reg){
  if(reg=='boreal'){
    data[!data$site %in% bor_over50,] = NA
  }else if(reg=='above'){
    data[!data$site %in% above_loc,] = NA
  }else if(reg=='above_core'){
    data[!data$site %in% above_core_loc,] = NA
  }
  df = data.frame(data$predicted,data$observed,data$set)
  colnames(df) = c('predicted','observed','set')
  df = df[!is.na(df$observed),]
  if(nrow(df[df$set=='Withheld sites',])>10){
    stats = data.frame(c('Training sites','Withheld sites'),matrix(nrow=2,ncol = 2))
    colnames(stats) =c('set','cv_r2','n')
    
    df$density = get_density(df$observed,df$predicted,n=1000)
    stats$cv_r2[stats$set=='Withheld sites'] = 1-(sum((df$observed[df$set=='Withheld sites']-df$predicted[df$set=='Withheld sites'])^2)/sum((mean(df$observed[df$set=='Withheld sites'])-df$observed[df$set=='Withheld sites'])^2))
    stats$cv_r2[stats$set=='Training sites'] = 1-(sum((df$observed[df$set=='Training sites']-df$predicted[df$set=='Training sites'])^2)/sum((mean(df$observed[df$set=='Training sites'])-df$observed[df$set=='Training sites'])^2))
    stats$n[stats$set=='Withheld sites'] = sum(df$set=='Withheld sites')
    stats$n[stats$set=='Training sites'] = sum(df$set=='Training sites')
    plot = ggplot(df) + geom_point(aes(x=observed,y=predicted,color = density)) + scale_color_gradientn(colors = c('yellow','yellow2','orangered','red','darkred')) +
      theme(panel.background = element_blank(),legend.position = 'none') + geom_abline(linetype = 2,slope = 1) +xlab ('Observed values') + ylab('Predicted values') +
      scale_x_continuous(limits = c(-1,1)) + scale_y_continuous(limits = c(-1,1)) + ggtitle(reg) +
      annotate('text', label = paste0('CV R2 = ',round(stats$cv_r2,2)),x = -0.9,y=1) + annotate('text', label = paste0('n = ',stats$n),x = -0.9,y=0.9)
    
    return(plot + facet_grid(.~set)) 
    
  }
}

major_reg_stat_func = function(data,reg){
  if(reg=='boreal'){
    data[!data$site %in% bor_over50,] = NA
  }else if(reg=='above'){
    data[!data$site %in% above_loc,] = NA
  }else if(reg=='above_core'){
    data[!data$site %in% above_core_loc,] = NA
  }
  df = data.frame(data$predicted,data$observed,data$set)
  colnames(df) = c('predicted','observed','set')
  df = df[!is.na(df$observed),]
  
  stats = data.frame(matrix(nrow=2,ncol = 2))
  colnames(stats) =c('cv_r2','n')
  rownames(stats) = c('Train','Test')
  
  
  stats['Test','cv_r2'] = 1-(sum((df$observed[df$set=='Withheld sites']-df$predicted[df$set=='Withheld sites'])^2)/sum((mean(df$observed[df$set=='Withheld sites'])-df$observed[df$set=='Withheld sites'])^2))
  stats['Train','cv_r2'] = 1-(sum((df$observed[df$set=='Training sites']-df$predicted[df$set=='Training sites'])^2)/sum((mean(df$observed[df$set=='Training sites'])-df$observed[df$set=='Training sites'])^2))
  stats['Test','n'] = sum(df$set=='Withheld sites')
  stats['Train','n'] = sum(df$set=='Training sites')
  
  return(stats)
}

sp_cv_func = function(data,sp){
  sp_vect = dom_sp[,sp]
  sp_vect = sp_vect[rownames(data)]
  df = data[sp_vect==1,]
  #df = data.frame(data$predicted[sp_vect==1],data$observed[sp_vect==1],data$set[sp_vect==1])
  colnames(df) = c('predicted','observed','set')
  if(nrow(df[df$set=='Withheld sites',])>10){
    stats = data.frame(c('Training sites','Withheld sites'),matrix(nrow=2,ncol = 2))
    colnames(stats) =c('set','cv_r2','n')
    sp_lab = substring(sp,12)
    sp_lab = gsub("."," ",sp_lab,fixed=T)
    df$density = get_density(df$observed,df$predicted,n=1000)
    stats$cv_r2[stats$set=='Withheld sites'] = 1-(sum((df$observed[df$set=='Withheld sites']-df$predicted[df$set=='Withheld sites'])^2)/sum((mean(df$observed[df$set=='Withheld sites'])-df$observed[df$set=='Withheld sites'])^2))
    stats$cv_r2[stats$set=='Training sites'] = 1-(sum((df$observed[df$set=='Training sites']-df$predicted[df$set=='Training sites'])^2)/sum((mean(df$observed[df$set=='Training sites'])-df$observed[df$set=='Training sites'])^2))
    stats$n[stats$set=='Withheld sites'] = sum(df$set=='Withheld sites')
    stats$n[stats$set=='Training sites'] = sum(df$set=='Training sites')
    plot = ggplot() + geom_point(data = df,aes(x=observed,y=predicted,color = density)) + scale_color_gradientn(colors = c('yellow','yellow2','orangered','red','darkred')) +
      theme(panel.background = element_blank(),legend.position = 'none') + geom_abline(linetype = 2,slope = 1) +xlab ('Observed values') + ylab('Predicted values') +
      scale_x_continuous(limits = c(-1,1)) + scale_y_continuous(limits = c(-1,1)) + ggtitle(sp_lab) +
      geom_text(data = stats,aes(label=paste0('CV R2 = ',round(cv_r2,2)),x = -0.9, y = 1)) + geom_text(data=stats,aes(label = paste0('n = ',n),x = -0.9,y=0.9))
    
    return(plot + facet_grid(.~set))
  }
}

#Needed to run the above for provinces
provs = c(1:9,11:13)
names(provs) = c('BC','AB','MB','SK','ON','QC','NL','NB','NS','YT','NT','AK')

#Define which major regions to run the above functions for depending on which region the model is trained on
if(region=='all'){
  major_regs = c('all','boreal','above','above_core')
}else if(region=='boreal'){
  major_regs = c('boreal','above','above_core')
}else{
  major_regs = c('above','above_core')
}

species = colnames(vi_df)[grep('scientific',colnames(vi_df))]
species_df = vi_df[,species]

dom_sp = matrix(nrow = nrow(species_df),ncol = 20)
dom_sp[] = 0
colnames(dom_sp) = colnames(species_df)

for(n in 1:nrow(dom_sp)){
  mx = max(species_df[n,],na.rm=T)
  sp = colnames(species_df)[species_df[n,]==mx]
  dom_sp[n,sp] = 1
}
rownames(dom_sp) = rownames(species_df)

#Iterate
for (class_run in (class_runs+1):100){

  #Read in classification model and get some info
  class_model = readRDS(paste0(folder1,"/final_model.rds"))
  mod_time = file.info(paste0(folder1,"/final_model.rds"))$mtime
  models = read.delim(paste0(folder1,"/models.txt"),sep=" ",header=T,stringsAsFactors = F)
  final_model_inf = models[which(models$kappa==max(models$kappa)),]
  
 
  
  #Create the necessary folders
  folder = paste0(folder1,"/regress_with_regions/")
  dir.create(folder,recursive = T)
  
  DP_folder = paste0(folder,'density_plots/')
  dir.create(DP_folder,recursive = T)
  
  PD_folder = paste0(folder,'PD_plots/')
  dir.create(PD_folder,recursive = T) 
  
  regress_folder = paste0(folder,'regression_plots/')
  dir.create(regress_folder,recursive = T) 
  
  imp_folder = paste0(folder,'importance_plots/')
  dir.create(imp_folder,recursive = T)
  
  spat_folder = paste0(folder,'spatial_plots/')
  dir.create(spat_folder,recursive = T)
  
  test_folder = paste0(folder,'test_data/')
  dir.create(test_folder,recursive = T)
  
  use_plot_folder = paste0(folder,'/use_plots/')
  dir.create(use_plot_folder,recursive = T) 
  
  eco_plot_folder = paste0(folder,'/use_plots/eco_plots/')
  dir.create(eco_plot_folder,recursive = T) 
  
  pr_plot_folder = paste0(folder,'/use_plots/pr_plots/')
  dir.create(pr_plot_folder,recursive = T)
  
  reg_plot_folder = paste0(folder,'/use_plots/reg_plots/')
  dir.create(reg_plot_folder,recursive = T)
  
  sp_plot_folder = paste0(folder,'/use_plots/sp_plots/')
  dir.create(sp_plot_folder,recursive = T)
  
  model_folder = paste0(folder,'models/')
  dir.create(model_folder,recursive = T)
  
  #create textfiles for outputting model info
  outfile = paste0(folder,"models.txt")
  reg_outfile = paste0(folder,"regional_models.txt")
  
  #If this is a new run, add a header to the outfiles and create filler variables for cvr2 values
  if(!file.exists(outfile)){
    cat(c('run_number','which_model','terms','model_R2',"CV_R2",'n'),file=outfile)
    cat(c('\n'),file=outfile, append=T)

    cat(c('run_number','region','Train_cv','Train_n',"Test_cv",'Test_n'),file=reg_outfile)
    cat(c('\n'),file=reg_outfile, append=T)
    topcv1 = -10
    topcv2 = -10
    topcv3 = -10
    topcv4 = -10
    topcv5 = -10
    topcv6 = -10
    topcv7 = -10
    topcv_final = -10
  }else{ #If the script was restarted, read in cvr2 values from outfiles
    reg_models = read.delim(paste0(folder,"/models.txt"),sep=" ",header=T,stringsAsFactors = F)
    topcv1 = reg_models$CV_R2[reg_models$which_model==1]
    topcv1 = max(topcv1)
    topcv2 = reg_models$CV_R2[reg_models$which_model==2]
    topcv2 = max(topcv2)
    topcv3 = reg_models$CV_R2[reg_models$which_model==3]
    topcv3 = max(topcv3)
    topcv4 = reg_models$CV_R2[reg_models$which_model==4]
    topcv4 = max(topcv4)
    topcv5 = reg_models$CV_R2[reg_models$which_model==5]
    topcv5 = max(topcv5)
    topcv6 = reg_models$CV_R2[reg_models$which_model==6]
    topcv6 = max(topcv6)
    topcv7 = reg_models$CV_R2[reg_models$which_model==7]
    topcv7 = max(topcv7)
    topcv_final = reg_models$CV_R2[reg_models$which_model=='combined']
    topcv_final = max(topcv_final)
  }
  
  #Again, if the script was restarted, figure out where it was restarted
  run_vect = list.files(spat_folder)
  run_vect = max(as.numeric(gsub("([a-z._]+)","",run_vect)))
  if(!is.finite(run_vect)){
    run_vect=0
  }
  run_vect = (as.numeric(run_vect)+1):100

  #only run  iterations if <100 models have been run already
  if(run_vect[1]!=101){
    
    
    
    
    #Run interations
    for(run in run_vect){
      
      #j is a holdover from old tests, it refers the maximum length (in years) timeseries window to use for VIs. I settled on 10
      j=10
      
      #prep data for classification using the class model and predict
      data = vi_df[!is.na(vi_df$lagged_biomass),]
      predictors = rownames(class_model$importance)
      data = data[,predictors]
      test_data = data[rowSums(is.na(data))==0,]
      predicted_class = predict(class_model,newdata = test_data)

      #Determine which predictors are less than the maximum time window
      predictors = colnames(vi_df)[is.na(windows)|windows<=j]
      
      #Take data again, this time with all variables
      data = vi_df[!is.na(vi_df$lagged_biomass),]
      data = data[,predictors]
      data = data[rownames(test_data),]
      
      #Create another dataframe for histograms
      hist_df = data.frame(data$lagged_biomass,predicted_class)

      #Create overlapping historgrams for each class
      r=1
      for(i in 1:length(unique(predicted_class))){
        for(k in 1:length(unique(predicted_class))){
          plot_df = hist_df[as.numeric(hist_df$predicted_class)==i|as.numeric(hist_df$predicted_class)==k,]
          meds = ddply(plot_df,'predicted_class',summarise,grp.mean=median(data.lagged_biomass))
          qs = ddply(plot_df,'predicted_class',summarise,grp.sd=quantile(data.lagged_biomass,seq(0,1,0.005),na.rm=T))
          bounds = data.frame(unique(plot_df$predicted_class),cbind(quant[as.numeric(unique(plot_df$predicted_class))],quant[as.numeric(unique(plot_df$predicted_class))+1]))
          qs95 = data.frame(qs[c(6,196,207,397),])
          qs_overlap = data.frame(qs$grp.sd[1:201],rev(qs$grp.sd[202:402]))
          qs_overlap$intersect = qs_overlap$qs.grp.sd.1.201.>qs_overlap$rev.qs.grp.sd.202.402..
          intersect = max(which(qs_overlap$intersect==FALSE))
          thresh_round = round(mean(colMeans(qs_overlap[c(intersect,intersect+1),c(1,2)])),3)
          perc = (201-intersect)*0.005
          colnames(bounds) = c('class','xmin','xmax')
          bounds = arrange(bounds,xmax)
          p = ggplot() + geom_rect(data= bounds, aes(xmin = xmin,xmax = xmax,ymin = 0, ymax = Inf,fill = class),alpha = 0.2) + 
            geom_density(data = plot_df,aes(x=data.lagged_biomass,fill=predicted_class),alpha=.3) + theme(legend.position = c(0.1,0.8),axis.title = element_blank()) +
            labs(fill = 'Class')  #+ geom_vline(data = mu,aes(xintercept = grp.mean,color = predicted_class),linetype='dashed')
          build = ggplot_build(p)
          
          p = p + annotate('text', label = paste0('Overlap threshold = ',thresh_round,'\nPercent overlap = ',perc*100,'%'),x = mean(build$data[[2]]$x),y=max(build$data[[2]]$density))
          ggsave(paste0(DP_folder,"density_plot_overlap_x=",k,"_y=",i,'_',run,".png"),p,width=6,height=6,units='in')
          
        }
      }
      
      #Create some vectors for later
      predicted_final = vector()
      test_final = vector()
      predicted_cv_final = vector()
      test_cv_final = vector()
      train_vect_final = vector()
      train_predict = vector()
      train_obs = vector()
      
      #Run regressions for each quantile
      for(b in 1:num_quants){
        
        #Take data that is expected to be in this class an prep
        this_data = data[predicted_class==b,]
        predictors = colnames(vi_df)[is.na(windows)|windows<=j]
        predictors = predictors[!predictors %in% c('site','lagged_biomass')]
        steps = 50
        importance_vect = vector()
        all_corr = unique(correlations$y[correlations$x %in% predictors])
        
        #Run random forest interations, removing all variables correlated to the most important variable (which has not
        #been defined as the most important at previous steps) at each step until no more correlations exist
        while (sum(predictors %in% all_corr)>0){  
          temp_data = this_data[,predictors]
          mass_vect = this_data$lagged_biomass
          mass_vect = mass_vect[rowSums(is.na(temp_data))==0]
          temp_data = temp_data[rowSums(is.na(temp_data))==0,]
          rf_model = randomForest(temp_data,mass_vect,importance = T,do.trace = T, ntree = steps)
          varImpPlot(rf_model,sort=T,type=1,scale=F)
          imp = data.frame(rf_model$importance,rownames(rf_model$importance))
          non_imp = as.character(imp[imp$X.IncMSE<=0,3])
          
          imp = imp[!imp$rownames.rf_model.importance. %in% importance_vect,]
          
          most_imp = as.character(arrange(imp,desc(X.IncMSE))[1,3])
          importance_vect = c(importance_vect,most_imp)
          corr = correlations$y[correlations$x==most_imp]
          predictors = predictors[!predictors %in% corr]
          predictors = predictors[!predictors %in% non_imp]
          all_corr = predictors[predictors %in% unique(correlations$y[correlations$x %in% predictors])]
        }
        
        imp_vect = unique(c(importance_vect,predictors))
        
        steps = 500
        
        #prep data for final rf model, taking only one observation per site
        temp_data = this_data[,c(imp_vect,'site','lagged_biomass')]
        temp_data = temp_data[rowSums(is.na(temp_data))==0,]
        temp_data2 = temp_data
        temp_data = temp_data[tapply(1:nrow(temp_data),temp_data$site,some,1),]
        mass_vect = temp_data$lagged_biomass
        
        #Take a sample of sites for training, and withhold some for testing
        sample = sample(unique(temp_data$site),length(unique(temp_data$site))*.7)
        train_data = temp_data[temp_data$site %in% sample,]
        train_vect = mass_vect[temp_data$site %in% sample]
        test_data = temp_data[!temp_data$site %in% sample,]
        test_vect = mass_vect[!temp_data$site %in% sample]
        train_data = train_data[,imp_vect]
        test_data = test_data[,imp_vect]
        
        #Create model with training data
        rf_model = randomForest(train_data,train_vect,importance = T,do.trace = T, ntree = steps)
        
        png(filename = paste0(imp_folder,'imp_plot_',b,'_',run,'.png'),width=600,height=600)
        varImpPlot(rf_model,sort=T,type=1,scale=F)
        dev.off()
        
        
        train_vect_final = c(train_vect_final,rownames(train_data))
      
       #Write out the training data for later use 
        write.csv(train_data,paste0(PD_folder,'train_data_',b,'_',run,'.csv'))
        #Could create partial dependence plots, but it slows the script down, so there's a separate script for that
        
  #      for(i in 1:nrow(imp)){
  #        if(!is.na(imp$rownames.rf_model.importance.[i])){
  #          
  #          part_plot = partialPlot(rf_model,pred.data = train_data,x.var=as.character(imp$rownames.rf_model.importance.[i]),
  #                                  xlab = as.character(imp$rownames.rf_model.importance.[i]),main = paste0('Partial Dependence on ',as.character(imp$rownames.rf_model.importance.[i])),
  #                                  ylim = c(quant[b],quant[b+1]))
  #          
  #          
  #          dens_df = train_data[,as.character(imp$rownames.rf_model.importance.[i])]
  #          dens_plot = ggplot() + geom_density(data = data.frame(dens_df),aes(x=dens_df),alpha=.3)
  #          dens_plot_build = ggplot_build(dens_plot)
  #          dens_plot_df = data.frame(dens_plot_build$data[[1]]$x,dens_plot_build$data[[1]]$density)
  #          colnames(dens_plot_df) = c('x','y')
  #          
  #          max_y = max(dens_plot_df$y)
  #          
  #          dep_plot = ggplot() + geom_line(data = as.data.frame(part_plot), aes(x=x,y=y)) + geom_line(data = dens_plot_df, aes(x=x,y=y*(quant[b+1]-quant[b])/max_y+quant[b],color='red'))+
  #            scale_y_continuous(position = "left",name = 'Partial dependence', sec.axis = sec_axis(~(.-quant[b])*max_y/(quant[b+1]-quant[b]),name = "Density")) + 
  #            theme(axis.text.y.right = element_text(color = 'red'),axis.title.y.right = element_text(color='red'),legend.position = 'none') +
  #            xlab(as.character(imp$rownames.rf_model.importance.[i]))
  #          
  #          
  #          ggsave(filename = paste0(PD_folder,'PD_plot_',b,'_',as.character(imp$rownames.rf_model.importance.[i]),'_',run,'.png'),
  #                 plot = dep_plot, width = 6, height = 6, units ='in')
  #        }
  #      }
  #      
        #Predict the values for the whitheld data and create a plot
        predicted = predict(rf_model,newdata = test_data)
        df = data.frame(predicted,test_vect)
        df$density = get_density(df$test_vect,df$predicted,n=1000)
        cv_r2 = 1-(sum((df$test_vect-df$predicted)^2)/sum((mean(df$test_vect)-df$test_vect)^2))
        plot = ggplot(df) + geom_point(aes(x=test_vect,y=predicted,color = density)) + scale_color_gradientn(colors = c('yellow','yellow2','orangered','red','darkred')) +
          theme(panel.background = element_blank(),legend.position = 'none') + geom_abline(linetype = 2,slope = 1) +xlab ('Observed values') + ylab('Predicted values') +
          scale_x_continuous(limits = c(-1,1)) + scale_y_continuous(limits = c(-1,1)) + ggtitle(paste0('Maximum metric window length: ',j)) + 
          annotate('text', label = paste0('Model R2 = ',round(rf_model$rsq[500],2)),x = -0.9,y=1) + annotate('text', label = paste0('CV R2 = ',round(cv_r2,2)),x = -0.9,y=0.9)
        ggsave(paste0(regress_folder,"plot_",b,"_",run,".png"),plot,width=8,height=6,units='in')
       
        predicted_final = c(predicted_final,predicted)
        test_final = c(test_final,test_vect)
        
        importance_vect = paste0(imp_vect,collapse=",")
        importance_vect = gsub(" ","_",importance_vect)
        
        #DO the same as above, but add back in all observations from withheld sites
        test_data = temp_data2[!temp_data2$site %in% sample,]
        test_vect = temp_data2$lagged_biomass[!temp_data2$site %in% sample]
        test_data = test_data[,imp_vect]
        
        predicted = predict(rf_model,newdata = test_data)
        df = data.frame(predicted,test_vect)
        df$density = get_density(df$test_vect,df$predicted,n=1000)
        cv_r2 = 1-(sum((df$test_vect-df$predicted)^2)/sum((mean(df$test_vect)-df$test_vect)^2))
        plot = ggplot(df) + geom_point(aes(x=test_vect,y=predicted,color = density)) + scale_color_gradientn(colors = c('yellow','yellow2','orangered','red','darkred')) +
          theme(panel.background = element_blank(),legend.position = 'none') + geom_abline(linetype = 2,slope = 1) +xlab ('Observed values') + ylab('Predicted values') +
          scale_x_continuous(limits = c(-1,1)) + scale_y_continuous(limits = c(-1,1)) + ggtitle(paste0('Maximum metric window length: ',j)) + 
          annotate('text', label = paste0('Model R2 = ',round(rf_model$rsq[500],2)),x = -0.9,y=1) + annotate('text', label = paste0('CV R2 = ',round(cv_r2,2)),x = -0.9,y=0.9)
        ggsave(paste0(regress_folder,"plot_all_cv_",b,"_",run,".png"),plot,width=8,height=6,units='in')
        
        out = c(run,b,importance_vect,rf_model$rsq[steps],cv_r2,length(test_vect))
        cat(out,file=outfile,append=T)
        cat(c('\n'),file=outfile, append=T)
        
        predicted_cv_final = c(predicted_cv_final,predicted)
        test_cv_final = c(test_cv_final,test_vect)
        
        train_predict = c(train_predict,predict(rf_model,newdata = train_data))
        train_obs = c(train_obs,train_vect)
        
        #If this is the best fit model, save it as such
        if(cv_r2>eval(parse(text = paste0("topcv",b)))){
          assign(paste0("topcv",b),cv_r2)
          saveRDS(rf_model,paste0(model_folder,"best_model",b,".rds"))
          assign(paste0('best_predicted_',b),predicted)
          assign(paste0('best_test_',b),test_vect)
        }
        saveRDS(rf_model,paste0(model_folder,"model_",b,"_",run,".rds"))
        assign(paste0('rf_model_',b),rf_model)
        
      }
      
      # predicted = c(predicted_1,predicted_2,predicted_3,predicted_4,predicted_5,predicted_6,predicted_7,predicted_8,predicted_9,predicted_10)
      # test_vect = c(test_vect_1,test_vect_2,test_vect_3,test_vect_4,test_vect_5,test_vect_6,test_vect_7,test_vect_8,test_vect_9,test_vect_10)
      
      df = data.frame(predicted_final,test_final)
      df$density = get_density(df$test_final,df$predicted_final,n=1000)
      cv_r2 = 1-(sum((df$test_final-df$predicted_final)^2)/sum((mean(df$test_final)-df$test_final)^2))
      plot = ggplot(df) + geom_point(aes(x=test_final,y=predicted_final,color = density)) + scale_color_gradientn(colors = c('yellow','yellow2','orangered','red','darkred')) +
        theme(panel.background = element_blank(),legend.position = 'none') + geom_abline(linetype = 2,slope = 1) +xlab ('Observed values') + ylab('Predicted values') +
        scale_x_continuous(limits = c(-1,1)) + scale_y_continuous(limits = c(-1,1)) + ggtitle(paste0('Maximum metric window length: ',j)) +
        annotate('text', label = paste0('CV R2 = ',round(cv_r2,2)),x = -0.9,y=1)
      ggsave(paste0(regress_folder,"plot_combined_",run,".png"),plot,width=8,height=6,units='in')
      
      
      
      
      # predicted = c(predicted_all_cv_1,predicted_all_cv_2,predicted_all_cv_3,predicted_all_cv_4,predicted_all_cv_5,predicted_all_cv_6,predicted_all_cv_7,predicted_all_cv_8,predicted_all_cv_9,predicted_all_cv_10)
      # test_vect = c(test_vect_all_cv_1,test_vect_all_cv_2,test_vect_all_cv_3,test_vect_all_cv_4,test_vect_all_cv_5,test_vect_all_cv_6,test_vect_all_cv_7,test_vect_all_cv_8,test_vect_all_cv_9,test_vect_all_cv_10)
      
      df = data.frame(predicted_cv_final,test_cv_final)
      df$density = get_density(df$test_cv_final,df$predicted_cv_final,n=1000)
      cv_r2 = 1-(sum((df$test_cv_final-df$predicted_cv_final)^2)/sum((mean(df$test_cv_final)-df$test_cv_final)^2))
      plot = ggplot(df) + geom_point(aes(x=test_cv_final,y=predicted_cv_final,color = density)) + scale_color_gradientn(colors = c('yellow','yellow2','orangered','red','darkred')) +
        theme(panel.background = element_blank(),legend.position = 'none') + geom_abline(linetype = 2,slope = 1) +xlab ('Observed values') + ylab('Predicted values') +
        scale_x_continuous(limits = c(-1,1)) + scale_y_continuous(limits = c(-1,1)) + ggtitle(paste0('Maximum metric window length: ',j)) +
        annotate('text', label = paste0('CV R2 = ',round(cv_r2,2)),x = -0.9,y=1)
      ggsave(paste0(regress_folder,"plot_all_cv_combined",run,".png"),plot,width=8,height=6,units='in')
      
      
      
      out = c(run,'combined',NA,NA,cv_r2,length(test_cv_final))
      cat(out,file=outfile,append=T)
      cat(c('\n'),file=outfile, append=T)
      
      #Create some spatial plots
      #df_res is difference between observed and predicted
      df_res = df$test_cv_final-df$predicted_cv_final
      df_sites = substring(rownames(df),6)
      df_years = substring(rownames(df),2,5)
      df_locs = site_loc[as.numeric(df_sites),]
      df_sp = SpatialPointsDataFrame(coords = df_locs[,2:3],data = data.frame(abs(df_res)),proj4string = CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
      
      
      political = readOGR(paste0(leadpath,'data/ancillary_files/political_shapes/bound_p/boundary_p_v2.shp'))
      political = spTransform(political,CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
      political@data$id = rownames(political@data)
      fortified = fortify(political, region = 'id')
      political_DF = merge(fortified,political@data, by = 'id')
      political_DF = political_DF[political_DF$long<0,]
     
      
      train_vect_final = substring(train_vect_final,6)
      train_locs = site_loc[as.numeric(train_vect_final),]
      train_locs$dens = get_density(train_locs$Longitude,train_locs$Latitude,n=1000)
      
      
      spat_g_plot = ggplot() + geom_polygon(data = political_DF,mapping = aes(x=long,y=lat,group = group)) + 
        geom_point(data = train_locs, aes(x=Longitude,y=Latitude,color=dens)) + 
        scale_color_gradientn(colors = c('yellow','yellow2','orangered','red','darkred')) + theme(legend.position = 'none')
      ggsave(paste0(spat_folder,"train_data_",run,".png"),spat_g_plot,width=8,height=6,units='in')
      
      df_ext = extent(-155,-50,40,70)
      grid_size = 1
      
      #set crs (crs of points)
      crs = "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"
      
      template = raster(df_ext,res = grid_size)
      crs(template) = crs
      
      rast = rasterize(df_sp,template,field='abs.df_res.',fun=mean,na.rm=T)
      
      
      png(file=paste0(spat_folder,"accuracy_",run,".png"),width = 8, height = 6,units='in',res=600) 
      plot(political,xlim=c(-180,-50))
      plot(rast,col=colorRampPalette(c('green','red'))(n=100),add=T)
      dev.off()
      
      
      test_out = data.frame(rownames(site_loc)[as.numeric(df_sites)],df_years,test_cv_final)
      write.csv(test_out,paste0(test_folder,"test_data_",run,".csv"))
      
      df = df[,1:2]
      colnames(df) = c('predicted','observed')
      df$set = 'Withheld sites'
      df2 = data.frame(train_predict,train_obs)
      colnames(df2) = c('predicted','observed')
      df2$set = 'Training sites'
      df = rbind(df,df2)
      
      data_sites = substring(rownames(df),6)
      data_sites = rownames(site_loc)[as.numeric(data_sites)]
      df$site = data_sites
      
      
      #Create regional faceted plots
      for(j in 1:length(data_sites)){
        df$ecoregion[j] = site_regions[data_sites[j]]
      }
      
      
      eco_regs = unique(df$ecoregion)
      eco_regs = eco_regs[eco_regs!='FALSE']
      #eco_regs = c('all',eco_regs)
      
      for(eco in eco_regs){
        eco_plot = eco_cv_function(df,eco)
        if(!is.null(eco_plot)){
          ggsave(paste0(eco_plot_folder,"eco_cv_",eco,"_",run,".png"),eco_plot,width=12,height=6,units='in')
        }
        eco_stats = eco_stat_func(df,eco)
        out = c(run,eco,eco_stats['Train','cv_r2'],eco_stats['Train','n'],eco_stats['Test','cv_r2'],eco_stats['Test','n'])
        cat(out,file=reg_outfile,append=T)
        cat(c('\n'),file=reg_outfile, append=T)
      }
      
      for(pr in provs){
        pr_plot = pr_cv_func(df,pr)
        if(!is.null(pr_plot)){
          ggsave(paste0(pr_plot_folder,"pr_cv_",names(provs)[provs==pr],"_",run,".png"),pr_plot,width=12,height=6,units='in')
        }
        pr_stats = pr_stat_func(df,pr)
        out = c(run,names(provs)[provs==pr],pr_stats['Train','cv_r2'],pr_stats['Train','n'],pr_stats['Test','cv_r2'],pr_stats['Test','n'])
        cat(out,file=reg_outfile,append=T)
        cat(c('\n'),file=reg_outfile, append=T)
      }
      for(reg in major_regs){
        reg_plot = major_reg_cv_function(df,reg)
        if(!is.null(reg_plot)){
          ggsave(paste0(reg_plot_folder,"reg_cv_",reg,"_",run,".png"),reg_plot,width=12,height=6,units='in')
        }
        reg_stats = major_reg_stat_func(df,reg)
        out = c(run,reg,reg_stats['Train','cv_r2'],reg_stats['Train','n'],reg_stats['Test','cv_r2'],reg_stats['Test','n'])
        cat(out,file=reg_outfile,append=T)
        cat(c('\n'),file=reg_outfile, append=T)
      }
      
      for(sp in colnames(dom_sp)){
        sp_plot = sp_cv_func(df,sp)
        if(!is.null(sp_plot)){
          ggsave(paste0(sp_plot_folder,"sp_cv_",sp,"_",run,".png"),sp_plot,width=12,height=6,units='in')
        }
      }
      if(cv_r2>topcv_final){
        topcv_final = cv_r2
        #saveRDS(rf_model,paste0(folder,"final_model_combined.rds"))
        saveRDS(rf_model_1,paste0(folder,"final_model_1.rds"))
        saveRDS(rf_model_2,paste0(folder,"final_model_2.rds"))
        saveRDS(rf_model_3,paste0(folder,"final_model_3.rds"))
        saveRDS(rf_model_4,paste0(folder,"final_model_4.rds"))
        saveRDS(rf_model_5,paste0(folder,"final_model_5.rds"))
        saveRDS(rf_model_6,paste0(folder,"final_model_6.rds"))
        saveRDS(rf_model_7,paste0(folder,"final_model_7.rds"))
      }
      
      if(all(c(exists('best_predicted_1'),exists('best_predicted_2'),exists('best_predicted_3'),exists('best_predicted_4'),
               exists('best_predicted_5'),exists('best_predicted_6'),exists('best_predicted_7')))){
        predicted_best = c(best_predicted_1,best_predicted_2,best_predicted_3,best_predicted_4,best_predicted_5,best_predicted_6,best_predicted_7)
        test_best = c(best_test_1,best_test_2,best_test_3,best_test_4,best_test_5,best_test_6,best_test_7)
        
        df = data.frame(predicted_best,test_best)
        df$density = get_density(df$test_best,df$predicted_best,n=1000)
        cv_r2 = 1-(sum((df$test_best-df$predicted_best)^2)/sum((mean(df$test_best)-df$test_best)^2))
        plot = ggplot(df) + geom_point(aes(x=test_best,y=predicted_best,color = density)) + scale_color_gradientn(colors = c('yellow','yellow2','orangered','red','darkred')) +
          theme(panel.background = element_blank(),legend.position = 'none') + geom_abline(linetype = 2,slope = 1) +xlab ('Observed values') + ylab('Predicted values') +
          scale_x_continuous(limits = c(-1,1)) + scale_y_continuous(limits = c(-1,1)) + ggtitle(paste0('Maximum metric window length: ',j)) +
          annotate('text', label = paste0('CV R2 = ',round(cv_r2,2)),x = -0.9,y=1)
        ggsave(paste0(regress_folder,"plot_all_cv_best_combined.png"),plot,width=8,height=6,units='in')
      }
      #If the classification model has changed since the beginning of this run, break the loop and start a new one
      if(file.info(paste0(folder1,"/final_model.rds"))$mtime!=mod_time){
        
        if(exists('test_best')){
          out = c(NA,'best_combined',NA,NA,cv_r2,length(test_best))
          cat(out,file=outfile,append=T)
          cat(c('\n'),file=outfile, append=T)
        }
        file.rename(folder,paste0(folder1,"/regress_",class_run))
      break
      }
    }
    #If we made it all the way to 100 runs, check to make sure the class model is finished. If not, wait until it is
    if (run==100){
      models = read.delim(paste0(folder1,"/models.txt"),sep=" ",header=T,stringsAsFactors = F)
      while(max(models$run_number)!=100&file.info(paste0(folder1,"/final_model.rds"))$mtime==mod_time){
        print('zzz...')
        Sys.sleep(600)
        models = read.delim(paste0(folder1,"/models.txt"),sep=" ",header=T,stringsAsFactors = F)
      }
      if(exists('test_best')){
        out = c(NA,'best_combined',NA,NA,cv_r2,length(test_best))
        cat(out,file=outfile,append=T)
        cat(c('\n'),file=outfile, append=T)
      }
      if(file.info(paste0(folder1,"/final_model.rds"))$mtime!=mod_time){
        file.rename(folder,paste0(folder1,"/regress_",class_run))
      }
    }
    if(file.info(paste0(folder1,"/final_model.rds"))$mtime==mod_time){
      break
    }
  }
}

#ggplot() + geom_tile(data = rast,aes(fill = factor(value)))
