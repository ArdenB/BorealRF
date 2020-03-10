
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

args <- commandArgs(trailingOnly = T)
#print(args)
#args = strsplit(args,"")[[1]]
# VIs = args[1]
# region = args[2]
# VIs = strsplit(VIs,",")[[1]]
# print(region)
# print(VIs)
region=args
print(region)
region="all"

#today = Sys.Date()
today = '2019-11-15'
num_quants = 7

VIs = c('ndvi','psri','ndii','ndvsi','msi','nirv','ndwi','nbr','satvi','tvfc')
# folder = paste0(leadpath,"scooperdock/EWS/data/rf_models/biomass_nointerp/",region,"/",paste0(VIs,collapse="_"),"/")
# today = max(list.files(folder))
# if(!dir.exists(folder)){
#   today = Sys.Date()
# }

folder = paste0(leadpath,"data/models/rf_models/",region,"/",paste0(VIs,collapse="_"),"/",today,"/")
#dir.create(folder,recursive = T)
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

# #Read in lengths of time since last survey
# raw_shift_df = read.csv(paste0(leadpath,"data/psp/modeling_data/surv_interval_filled.csv"))
# raw_shift_df = arrange(raw_shift_df,X)
# rownames(raw_shift_df) = raw_shift_df[,"X"]
# sites = rownames(raw_shift_df)
# shift_df = clean_df(raw_shift_df)
# 
# site_loc = read.csv(paste0(leadpath,"data/raw_psp/All_sites_101218.csv"),row.names = 'Plot_ID')
# adj_YT = site_loc[grep("11_",rownames(site_loc)),]
# for(i in 1:nrow(adj_YT)){
#   rownames(adj_YT)[i] = paste0("11_",sprintf("%03.0f",as.numeric(strsplit(rownames(adj_YT)[i],"_")[[1]][2])))
# }
# rownames(site_loc)[grep("11_",rownames(site_loc))] = rownames(adj_YT)
# site_loc = site_loc[sites,]




PD_folder = paste0(folder,'PD_plots/')


model_folder = paste0(folder,'models/')


outfile = paste0(folder,"models.txt")
models = read.delim(outfile,sep=" ",header=T,stringsAsFactors = F)

# bor_frac = read.csv(paste0(leadpath,"data/psp/databases/PSP_boreal_frac_biomass_v1.csv"),row.names = 'X')
# bor_frac = rowMeans(bor_frac,na.rm=T)
# bor_frac = bor_frac[sites]
# bor_over50 = names(bor_frac)[bor_frac>0.5]
# 
# above = readOGR(paste0(leadpath,"data/ancillary_files/ABoVE_Study_Domain_Final/ABoVE_Study_Domain.shp"))
# above = spTransform(above,CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
# site_loc_sp = site_loc[!is.na(site_loc$Longitude),]
# 
# site_loc_sp = SpatialPointsDataFrame(coords = site_loc_sp[,2:3],data = data.frame(rownames(site_loc_sp)),proj4string = CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
# site_loc_sp = raster::intersect(site_loc_sp,above)
# 
# above_loc = site_loc_sp@data$rownames.site_loc_sp.
# 
# above_core = above[above$Region=='Core Region',]
# site_loc_sp = raster::intersect(site_loc_sp,above_core)
# above_core_loc = site_loc_sp@data$rownames.site_loc_sp.
# 
load(paste0(leadpath,"data/models/input_data/vi_df_all_2019-10-30.Rda"))
# load(paste0(leadpath,"data/models/input_data/correlations_2019-10-09.Rda"))

 quant = quantile(vi_df$lagged_biomass,seq(0,1,length.out = num_quants+1),na.rm=T)
# 
# if(region=='boreal'){
#   vi_df[!vi_df$site %in% bor_over50,] = NA
# }else if(region=='above'){
#   vi_df[!vi_df$site %in% above_loc,] = NA
# }else if(region=='above_core'){
#   vi_df[!vi_df$site %in% above_core_loc,] = NA
# }
# 
# vi_df$class = set_quants(vi_df$lagged_biomass,quant)
# windows = colnames(vi_df)
# for(i in 1:length(windows)){
#   win = strsplit(windows[i],"_")[[1]]
#   windows[i] = as.numeric(win[length(win)])
# }
# windows = as.numeric(windows)

final_model_inf = models[which(models$kappa==max(models$kappa)),]
rf_model = readRDS(paste0(folder,"/final_model.rds"))
imp = data.frame(rf_model$importance,rownames(rf_model$importance))
imp = arrange(imp,desc(MeanDecreaseAccuracy))
imp = imp[1:25,]
rownames(imp) = 1:25

train_data = read.csv(paste0(PD_folder,'train_data_',final_model_inf$run_number,'.csv'))


for(r in 1:num_quants){

 partplot = partialPlot(rf_model,pred.data = train_data,x.var='biomass',which.class = r)
 max_vect = max(partplot$y)
 min_vect = min(partplot$y)
 for(i in 1:nrow(imp)){
   if(!is.na(imp$rownames.rf_model.importance.[i])){


     part_plot = partialPlot(rf_model,pred.data = train_data,x.var=as.character(imp$rownames.rf_model.importance.[i]),
                             xlab = as.character(imp$rownames.rf_model.importance.[i]),main = NULL,
                             which.class = r,ylim = c(min_vect,max_vect))
     dens_df = train_data[,as.character(imp$rownames.rf_model.importance.[i])]
     dens_plot = ggplot() + geom_density(data = data.frame(dens_df),aes(x=dens_df),alpha=.3)
     dens_plot_build = ggplot_build(dens_plot)
     dens_plot_df = data.frame(dens_plot_build$data[[1]]$x,dens_plot_build$data[[1]]$density)
     colnames(dens_plot_df) = c('x','y')

     max_y = max(dens_plot_df$y)


     dep_plot = ggplot() + geom_line(data = as.data.frame(part_plot), aes(x=x,y=y)) + geom_line(data = dens_plot_df, aes(x=x,y=y*(max_vect-min_vect)/max_y+min_vect,color='red'))+
       scale_y_continuous(position = "left",name = 'Partial dependence',limits = c(min_vect,max_vect), sec.axis = sec_axis(~(.-min_vect)*max_y/(max_vect-min_vect),name = "Density")) +
       theme(axis.text.y.right = element_text(color = 'red'),axis.title.y.right = element_text(color='red'),legend.position = 'none') +
       xlab(as.character(imp$rownames.rf_model.importance.[i])) + theme(panel.background = element_blank())

     ggsave(filename = paste0(PD_folder,'PD_plot_class=',r,"_",as.character(imp$rownames.rf_model.importance.[i]),"_",final_model_inf$run_number,'.png'),
            plot = dep_plot, width = 6, height = 6, units ='in')

   }
 }
}
      

regress_folder = paste0(folder,'regress_with_regions/')
PD_folder = paste0(regress_folder,'PD_plots/')
regress_models = read.delim(paste0(regress_folder,'models.txt'),sep=" ",header=T,stringsAsFactors = F)
regress_models = regress_models[regress_models$which_model=='combined',]
regress_models = arrange(regress_models,CV_R2)
regress_model_inf = regress_models[95,]

for(b in 1:num_quants){
  rf_model = readRDS(paste0(regress_folder,'/models/model_',b,'_',regress_model_inf$run_number,".rds"))
  
  imp = data.frame(rf_model$importance,rownames(rf_model$importance))
  imp = arrange(imp,desc(X.IncMSE))
  imp = imp[1:25,]
  rownames(imp) = 1:25

  train_data = read.csv(paste0(PD_folder,'train_data_',b,'_',regress_model_inf$run_number,'.csv'))

   for(i in 1:nrow(imp)){
     if(!is.na(imp$rownames.rf_model.importance.[i])){
  
       part_plot = partialPlot(rf_model,pred.data = train_data,x.var=as.character(imp$rownames.rf_model.importance.[i]),
                               xlab = as.character(imp$rownames.rf_model.importance.[i]),main = paste0('Partial Dependence on ',as.character(imp$rownames.rf_model.importance.[i])),
                               ylim = c(quant[b],quant[b+1]))
  
  
       dens_df = train_data[,as.character(imp$rownames.rf_model.importance.[i])]
       dens_plot = ggplot() + geom_density(data = data.frame(dens_df),aes(x=dens_df),alpha=.3)
       dens_plot_build = ggplot_build(dens_plot)
       dens_plot_df = data.frame(dens_plot_build$data[[1]]$x,dens_plot_build$data[[1]]$density)
       colnames(dens_plot_df) = c('x','y')
  
       max_y = max(dens_plot_df$y)
  
       dep_plot = ggplot() + geom_line(data = as.data.frame(part_plot), aes(x=x,y=y)) + geom_line(data = dens_plot_df, aes(x=x,y=y*(quant[b+1]-quant[b])/max_y+quant[b],color='red'))+
         scale_y_continuous(position = "left",name = 'Partial dependence', sec.axis = sec_axis(~(.-quant[b])*max_y/(quant[b+1]-quant[b]),name = "Density")) +
         theme(axis.text.y.right = element_text(color = 'red'),axis.title.y.right = element_text(color='red'),legend.position = 'none') +
         xlab(as.character(imp$rownames.rf_model.importance.[i]))
  
  
       ggsave(filename = paste0(PD_folder,'PD_plot_',b,'_',as.character(imp$rownames.rf_model.importance.[i]),'_',regress_model_inf$run_number,'.png'),
              plot = dep_plot, width = 6, height = 6, units ='in')
     }
   }
      
}
      
  