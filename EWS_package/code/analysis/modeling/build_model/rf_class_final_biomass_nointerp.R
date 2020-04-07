
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

#For argument input from slurm script
args <- commandArgs(trailingOnly = T)
region = args
print(region)

#Can also directly input region
region='all'

#Set date
today = Sys.Date()
# today = '2019-08-30'

#How many quantiles?
num_quants = 7

#Which VIS to use?
# Logan Berner method, as B
VIs = c('ndvi','psri','ndii','ndvsi','msi','nirv','ndwi','nbr','satvi','tvfc')

#Create the folder based on the above
folder = paste0(leadpath,"data/models/rf_models/",region,"/",paste0(VIs,collapse="_"),"/",today,"/")
dir.create(folder,recursive = T)

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

#Create some folders

CM_folder = paste0(folder,'confusion_matrices/')
dir.create(CM_folder,recursive = T) 

PD_folder = paste0(folder,'PD_plots/')
dir.create(PD_folder,recursive = T) 

imp_folder = paste0(folder,'importance_plots/')
dir.create(imp_folder,recursive = T) 

spat_folder = paste0(folder,'spatial_plots/')
dir.create(spat_folder,recursive = T)

model_folder = paste0(folder,'models/')
dir.create(model_folder,recursive = T)

#Create a text file for inputting modeling info
outfile = paste0(folder,"models.txt")

#If this is a new run, 
if(!file.exists(outfile)){
  #add a header to the outfile
  cat(c('run_number','terms','train_err',"kappa",'n'),file=outfile)
  cat(c('\n'),file=outfile, append=T)
  #Set number of runs to do and enter dummy variable of topkappa
  run_vect = 1:100
  topkappa = 0
}else{#If it isn't a new run
  #this will figure out what run its on and restart there.
  run_vect = list.files(spat_folder)
  run_vect = max(as.numeric(gsub("([a-z._]+)","",run_vect)))
  if(!is.finite(run_vect)){
    run_vect=0
  }
  run_vect = (as.numeric(run_vect)+1):100
  #and read in the highest kappa value
  topkappa = max(as.numeric(read.delim(paste0(folder,"/models_10.txt"),sep=" ",header=T,stringsAsFactors = F)$kappa))
}




#Read in boreal species fraction dataframe and determine which sites have >50% boreal species
bor_frac = read.csv(paste0(leadpath,"data/psp/databases/PSP_boreal_frac_biomass.csv"),row.names = 'X')
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
# USED IN:
# determing the coresponding variables that are used in covariance
# Predictor Variables read in (Soil, Climate, VI)
# VI is the 0- to 5, 0 - 10 year trend
#   Basic Linear regression
load(paste0(leadpath,"data/models/input_data/vi_df_all_2019-10-30.Rda")) #Written out by 


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

#Set classes for each observation
vi_df$class = set_quants(vi_df$lagged_biomass,quant)

#Create a vector that relates to the length of time windows for each variable in vi_df (only valid for VIs)
windows = colnames(vi_df)
for(i in 1:length(windows)){
  win = strsplit(windows[i],"_")[[1]]
  windows[i] = as.numeric(win[length(win)])
}
windows = as.numeric(windows)

#Run iterations
for(run in run_vect){
  
    
    #Holdover from when I was testing different length max windows, set to 10 for now.
    for (j in c(10)){
      
      #Prep data
      data = vi_df[!is.na(vi_df$lagged_biomass),]
      predictors = colnames(data)[is.na(windows)|windows<=j]
      predictors = predictors[!predictors %in% c('site','lagged_biomass','class')]
      steps = 50
      importance_vect = 'biomass'
      all_corr = unique(correlations$y[correlations$x %in% predictors])
      
      # ========== This is where the variable importance is determined ==========
      #Run random forest interations, removing all variables correlated to the most important variable (which has not
      #been defined as the most important at previous steps) at each step until no more correlations exist
      while (sum(predictors %in% all_corr)>0){  
        temp_data = data[,predictors]
        mass_vect = data$class
        mass_vect = mass_vect[rowSums(is.na(temp_data))==0]
        temp_data = temp_data[rowSums(is.na(temp_data))==0,]
        rf_model = randomForest(temp_data,as.factor(mass_vect),importance = T,do.trace = T, ntree = steps)
        varImpPlot(rf_model,sort=T,type=1,scale=F) # AB: The plot uses the good importance metric ??but the actual chose variables use the bad??
        imp = data.frame(rf_model$importance,rownames(rf_model$importance))
        non_imp = as.character(imp[imp$MeanDecreaseAccuracy<=0,'rownames.rf_model.importance.'])
        
        imp = imp[-which(rownames(imp) %in% importance_vect),] #AB: drop the important vectors tha have already been checked
        
        most_imp = as.character(arrange(imp,desc(MeanDecreaseAccuracy))[1,'rownames.rf_model.importance.']) #AB: find most important left
        importance_vect = c(importance_vect,most_imp) #AB:add that most important to a list
        corr = correlations$y[correlations$x==most_imp]
        predictors = predictors[!predictors %in% corr]
        predictors = predictors[!predictors %in% non_imp]
        all_corr = predictors[predictors %in% unique(correlations$y[correlations$x %in% predictors])]
      }
      steps = 500
      importance_vect = unique(c(importance_vect,predictors))
      
      # ========== This is where the final model is made ==========
      #Prep data for final rf_model
      temp_data = data[,c(importance_vect,'site','lagged_biomass','class')]
      temp_data = temp_data[rowSums(is.na(temp_data))==0,]
      mass_vect = temp_data$class
 
      #Take sample for training and withhold some sites
      sample = sample(unique(temp_data$site),length(unique(temp_data$site))*.7)
      train_data = temp_data[temp_data$site %in% sample,]
      train_vect = mass_vect[temp_data$site %in% sample]
      test_data = temp_data[!temp_data$site %in% sample,]
      test_vect = mass_vect[!temp_data$site %in% sample]
      train_data = train_data[,importance_vect]
      test_data = test_data[,importance_vect]
      rf_model = randomForest(train_data,as.factor(train_vect),importance = T,do.trace = T, ntree = steps)
      error_rate = rf_model$err.rate[500]
      
      
      png(filename = paste0(imp_folder,'imp_plot_class_',run,'.png'),width=600,height=600)
      varImpPlot(rf_model,sort=T,type=1,scale=F)
      dev.off()
      
      #Save training data for the future
      write.csv(train_data,paste0(PD_folder,'train_data_',run,'.csv'))
      
      #Create an error matrix plot
      col_mat = matrix(nrow = num_quants,ncol = num_quants)
      col_mat[] = 1
      
      predicted = predict(rf_model,newdata = test_data)
      df = data.frame(predicted,test_vect)
      df_table = table(df)
      err_table1 = df_table/rowSums(df_table)
      
      err_table2 = df_table/t(col_mat*colSums(df_table))
      PPV_df = data.frame(err_table1)
      df_df = as.data.frame(df_table)
      NPV_df = data.frame(err_table2)
      
      gplot_plot = ggplot(PPV_df, aes(x = test_vect,y = predicted, fill = Freq)) + geom_raster() + scale_fill_gradientn(colors = c('white','orangered','red','darkred')) +
        geom_text(aes(label = round(Freq,2))) + ylab('Predicted bin') + theme(legend.position = 'none') + xlab('Observed bin') +
        ggtitle(paste0('Predictive precision, max window = ',j)) + scale_y_discrete(limits = as.character(num_quants:1))
      
      gplot_plot2 = ggplot(NPV_df, aes(x = test_vect,y = predicted, fill = Freq)) + geom_raster() + scale_fill_gradientn(colors = c('white','orangered','red','darkred')) +
        geom_text(aes(label = round(Freq,2))) + ylab('Predicted bin') + theme(legend.position = 'none') + xlab('Observed bin') +
        ggtitle("Probability of detection") + scale_y_discrete(limits = as.character(num_quants:1))
      
      
      sensitivity = rowSums(diag(num_quants)*df_table)/colSums(df_table)
      specificity = vector(length = num_quants)
      for(i in 1:num_quants){
        specificity[i] = sum(df_table[(1:num_quants)[-i],(1:num_quants)[-i]])/(sum(df_table[(1:num_quants)[-i],(1:num_quants)[-i]])+sum(df_table[i,(1:num_quants)[-i]]))
      }
      informedness = sensitivity + specificity - 1
      
      
      sens_df = data.frame('sensitivity',as.character(1:num_quants),sensitivity)
      spec_df = data.frame('specificity',as.character(1:num_quants),specificity)
      inf_df = data.frame('informedness',as.character(1:num_quants),sensitivity+specificity-1)
      colnames(sens_df) = c('x','y','data')
      colnames(spec_df) = c('x','y','data')
      colnames(inf_df) = c('x','y','data')
      
      stat_df = rbind(sens_df,spec_df,inf_df)
      
      gplot_plot3 = ggplot(stat_df, aes(x = x,y = y, fill = data)) + geom_raster() + scale_fill_gradientn(colors = c('white','orangered','red','darkred')) +
        geom_text(aes(label = round(data,2))) + xlab('Statistic') + ylab('Bin') + theme(legend.position = 'none') + ggtitle('Informedness matrix') +
        scale_y_discrete(limits = as.character(c(num_quants:1))) + theme(axis.text.x = element_text(size=10))
     
      grid = rbind(c(1,1,1,1,2,2,2,2,3,3,3),c(1,1,1,1,2,2,2,2,3,3,3),c(1,1,1,1,2,2,2,2,3,3,3),c(1,1,1,1,2,2,2,2,3,3,3))
      
      arranged = grid.arrange(gplot_plot,gplot_plot2,gplot_plot3,layout_matrix=grid)
      
      
      ggsave(paste0(CM_folder,"confusion_matrix_",run,".png"),arranged,width=12,height=6,units='in')
      
      cm=confusionMatrix(predicted,as.factor(test_vect))
      kappa=cm$overall[2]
      
      #If this model has best kappa, save as final model
      if(kappa>topkappa){
        final_model = rf_model
        topkappa = kappa
        saveRDS(final_model,paste0(folder,"final_model.rds"))
      }
      saveRDS(rf_model,paste0(model_folder,"model_",run,".rds"))
      
      #Write data to text file
      importance_vect = paste0(importance_vect,collapse=",")
      importance_vect = gsub(" ","_",importance_vect)
      out = c(run,importance_vect,error_rate,kappa,length(test_vect))
      cat(out,file=outfile,append=T)
      cat(c('\n'),file=outfile, append=T)
      
      #Can create partial dependence plots, but it makes this script a lot slower, so I built it into a separate script where 
      #you can read in the read in the model results are create partial dependence plots
      # imp = data.frame(rf_model$importance,rownames(rf_model$importance))
      # imp = arrange(imp,desc(MeanDecreaseAccuracy))
      # imp = imp[1:25,]
      # rownames(imp) = 1:25
      # 
      # 
     # for(r in 1:num_quants){
     # 
     #   partplot = partialPlot(rf_model,pred.data = train_data,x.var='biomass',which.class = r)
     #   max_vect = max(partplot$y)
     #   min_vect = min(partplot$y)
     #   for(i in 1:nrow(imp)){
     #     if(!is.na(imp$rownames.rf_model.importance.[i])){
     # 
     # 
     #       part_plot = partialPlot(rf_model,pred.data = train_data,x.var=as.character(imp$rownames.rf_model.importance.[i]),
     #                               xlab = as.character(imp$rownames.rf_model.importance.[i]),main = NULL,
     #                               which.class = r,ylim = c(min_vect,max_vect))
     #       dens_df = train_data[,as.character(imp$rownames.rf_model.importance.[i])]
     #       dens_plot = ggplot() + geom_density(data = data.frame(dens_df),aes(x=dens_df),alpha=.3)
     #       dens_plot_build = ggplot_build(dens_plot)
     #       dens_plot_df = data.frame(dens_plot_build$data[[1]]$x,dens_plot_build$data[[1]]$density)
     #       colnames(dens_plot_df) = c('x','y')
     # 
     #       max_y = max(dens_plot_df$y)
     # 
     # 
     #       dep_plot = ggplot() + geom_line(data = as.data.frame(part_plot), aes(x=x,y=y)) + geom_line(data = dens_plot_df, aes(x=x,y=y*(max_vect-min_vect)/max_y+min_vect,color='red'))+
     #         scale_y_continuous(position = "left",name = 'Partial dependence',limits = c(min_vect,max_vect), sec.axis = sec_axis(~(.-min_vect)*max_y/(max_vect-min_vect),name = "Density")) +
     #         theme(axis.text.y.right = element_text(color = 'red'),axis.title.y.right = element_text(color='red'),legend.position = 'none') +
     #         xlab(as.character(imp$rownames.rf_model.importance.[i])) + theme(panel.background = element_blank())
     # 
     #       ggsave(filename = paste0(PD_folder,'PD_plot_class=',r,"_",as.character(imp$rownames.rf_model.importance.[i]),"_",run,'.png'),
     #              plot = dep_plot, width = 6, height = 6, units ='in')
     # 
     #     }
     #   }
     # }
      
      
      ##Create some spatial plots
      
      #Need to adjust the political shapefile a bit to use in ggplot
      political = readOGR(paste0(leadpath,'data/ancillary_files/political_shapes/bound_p/boundary_p_v2.shp'))
      political = spTransform(political,CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
      political@data$id = rownames(political@data)
      fortified = fortify(political, region = 'id')
      political_DF = merge(fortified,political@data, by = 'id')
      political_DF = political_DF[political_DF$long<0,]
      
      
      #Figure out what points were used for training
      train_vect_final = substring(rownames(train_data),6)
      train_locs = site_loc[as.numeric(train_vect_final),]
      #print(train_locs)
      train_locs$dens = get_density(train_locs$Longitude,train_locs$Latitude,n=1000)
      
      #Make a plot showing spatial density of training plots
      spat_g_plot = ggplot() + geom_polygon(data = political_DF,mapping = aes(x=long,y=lat,group = group)) + 
        geom_point(data = train_locs, aes(x=Longitude,y=Latitude,color=dens)) + 
        scale_color_gradientn(colors = c('yellow','yellow2','orangered','red','darkred')) + theme(legend.position = 'none')
      ggsave(paste0(spat_folder,"train_data_",run,".png"),spat_g_plot,width=8,height=6,units='in')
      
      #now an accuracy plot
      #df_res is just whether the prediction is accurate
      df_res = as.numeric(df$predicted==df$test_vect)
      #Find out which plots used for testing
      df_sites = substring(rownames(test_data),6)
      df_locs = site_loc[as.numeric(df_sites),]
      df_sp = SpatialPointsDataFrame(coords = df_locs[,2:3],data = data.frame(df_res),proj4string = CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
      
      df_ext = extent(-155,-50,40,70)
      grid_size = 1
      
      #set crs (crs of points)
      crs = "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"
      
      template = raster(df_ext,res = grid_size)
      crs(template) = crs
      
      #rasterize points into template raster
      rast = rasterize(df_sp,template,field='df_res',fun=mean,na.rm=T)
      
      
      png(file=paste0(spat_folder,"accuracy_",run,".png"),width = 8, height = 6,units='in',res=600) 
      plot(political,xlim=c(-180,-50))
      plot(rast,col=colorRampPalette(c('red','green'))(n=100),add=T)
      
      dev.off()
    }
    
  }


