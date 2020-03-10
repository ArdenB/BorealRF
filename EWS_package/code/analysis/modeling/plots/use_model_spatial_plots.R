library(gdalUtils)
library(dplyr)
library(rgdal)
library(randomForest)
library(ggplot2)



leadpath = "/att/nobackup/scooperd/"
model_date = '2019-11-15'
y_range = 1981:2017
null_val = -32768
VIs = c('ndvi','psri','ndii','ndvsi','msi','nirv','ndwi','nbr','satvi','tvfc')
region = 'all'
folder = paste0(leadpath,"/EWS_package/data/models/rf_models/",region,"/",paste0(VIs,collapse="_"),"/",model_date,"/")


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

#####Create Ju & Masek trends raster#######
#Commented out because it's already done, just read it in
#Also, now that this script is on cloudops and not adapt, the links below will be broken.


# #Read in each Ju & Masek tile, get rid of invalid values (>0.009, <-0.009) and write to a new folder
# Ju_Masek_folder = 'R:/data/ORNL/ABoVE_Archive/daac.ornl.gov/daacdata/above/Vegetation_greenness_trend/data/'
# trend_files = list.files(Ju_Masek_folder)
# valid_trends = trend_files[1:(length(trend_files)/2)]
# for(f in 1:length(valid_trends)){
#   
#   rast = raster(paste0(Ju_Masek_folder,valid_trends[f]))
#   if(!file.exists(paste0(leadpath,'EWS_package/data/plots/rasters/masked_trends/',valid_trends[f]))){
#     rast[values(rast)>0.009] = NA
#     rast[values(rast)<(-0.009)] = NA
#     
#     writeRaster(rast,paste0(leadpath,'EWS_package/data/plots/rasters/masked_trends/',valid_trends[f]))
#     
#   }
#   print(f)
# }
# #Mosaic rasters together
# mosaic_rasters(gdalfile = paste0(leadpath,'EWS_package/data/plots/rasters/masked_trends/',valid_trends),dst_dataset = paste0(leadpath,'EWS_package/data/plots/rasters/masked_trends/merged_Ju_Masek_all.tif'),of = 'GTiff')
# 
# #Read in mosaic and aggregate, I chose a 30 km cell for this
# Ju_Masek_rast = raster(paste0(leadpath,'EWS_package/data/plots/rasters/masked_trends/merged_Ju_Masek_all.tif'))
# Ju_Masek_agg = aggregate(Ju_Masek_rast,fact=1000,fun=mean,na.rm=T)

#Read in aggregated raster
Ju_Masek_agg = raster(paste0(leadpath,'EWS_package/data/plots/rasters/masked_trends/agg_Ju_Masek.tif'))

#Create a template using the aggregated raster so everything is on the same grid
Ju_Masek_crs = Ju_Masek_agg@crs
df_ext = extent(Ju_Masek_agg)
grid_size = 30000
template = raster(df_ext,res = grid_size)
crs(template) = Ju_Masek_crs

#Read in political shape and transform to same grid
political = readOGR('/att/nobackup/scooperd/scooperdock/Combustion/political_shapes/bound_p/boundary_p_v2.shp')
political = spTransform(political,Ju_Masek_crs)

######Prep vi_df#######

#Read in lengths of time since last survey
raw_shift_df = read.csv(paste0(leadpath,"EWS_package/data/psp/modeling_data/surv_interval_filled.csv"))
raw_shift_df = arrange(raw_shift_df,X)
rownames(raw_shift_df) = raw_shift_df[,"X"]
sites = rownames(raw_shift_df)

site_loc = read.csv(paste0(leadpath,"EWS_package/data/raw_psp/All_sites_101218.csv"),row.names = 'Plot_ID')
adj_YT = site_loc[grep("11_",rownames(site_loc)),]
for(i in 1:nrow(adj_YT)){
  rownames(adj_YT)[i] = paste0("11_",sprintf("%03.0f",as.numeric(strsplit(rownames(adj_YT)[i],"_")[[1]][2])))
}
rownames(site_loc)[grep("11_",rownames(site_loc))] = rownames(adj_YT)
site_loc = site_loc[sites,]


raw_stem_df = read.csv(paste0(leadpath,"EWS_package/data/psp/modeling_data/stem_dens_all_interpolated.csv"),row.names = 'X')
stem_df = clean_df(raw_stem_df)
stem_df[stem_df<0] = NA

load(paste0(leadpath,"EWS_package/data/models/input_data/vi_df_all_interp_2019-11-22.Rda"))
vi_df$stem_density = unlist(stem_df)




#####Read in model stuff#######

class_model = readRDS(paste0(folder,"/final_model.rds"))
regress_folder = paste0(folder,'regress_with_regions/')
regress_models = read.delim(paste0(regress_folder,'models.txt'),sep=" ",header=T,stringsAsFactors = F)
regress_models = regress_models[regress_models$which_model=='combined',]
regress_models = arrange(regress_models,CV_R2)
#THis is taking the 95th percentile regression model (to prevent major outliers by taking the best one)
regress_model_inf = regress_models[95,]
r_model_1 = readRDS(paste0(regress_folder,'/models/model_1_',regress_model_inf$run_number,".rds"))
r_model_2 = readRDS(paste0(regress_folder,'/models/model_2_',regress_model_inf$run_number,".rds"))
r_model_3 = readRDS(paste0(regress_folder,'/models/model_3_',regress_model_inf$run_number,".rds"))
r_model_4 = readRDS(paste0(regress_folder,'/models/model_4_',regress_model_inf$run_number,".rds"))
r_model_5 = readRDS(paste0(regress_folder,'/models/model_5_',regress_model_inf$run_number,".rds"))
r_model_6 = readRDS(paste0(regress_folder,'/models/model_6_',regress_model_inf$run_number,".rds"))
r_model_7 = readRDS(paste0(regress_folder,'/models/model_7_',regress_model_inf$run_number,".rds"))

#Create a bunch of matrices for later
biomass = matrix(nrow = 24690,ncol = length(y_range))
colnames(biomass) = y_range
rownames(biomass) = sites
biomass_predicted = biomass
percent_change = biomass
biomass_compare = biomass

#####Run predictions #######
r=1
for(i in 1:length(y_range)){
  df = vi_df[r:(r+24689),]
  colnames(df) = gsub("[ /-]",".",colnames(df))
  rownames(df) = sites
  biomass[,as.character(y_range[i])] = df$biomass
  df_class = df[,rownames(class_model$importance)]
  df_class = df_class[rowSums(is.na(df_class))==0,]
  if(nrow(df_class)>0){
    predicted_class = predict(class_model,df_class)
    df_regress = df[rownames(df_class),]
    df_1 = df_regress[predicted_class==1,]
    df_1 = df_1[,rownames(r_model_1$importance)]
    df_1 = df_1[rowSums(is.na(df_1))==0,]
    df_2 = df_regress[predicted_class==2,]
    df_2 = df_2[,rownames(r_model_2$importance)]
    df_2 = df_2[rowSums(is.na(df_2))==0,]
    df_3 = df_regress[predicted_class==3,]
    df_3 = df_3[,rownames(r_model_3$importance)]
    df_3 = df_3[rowSums(is.na(df_3))==0,]
    df_4 = df_regress[predicted_class==4,]
    df_4 = df_4[,rownames(r_model_4$importance)]
    df_4 = df_4[rowSums(is.na(df_4))==0,]
    df_5 = df_regress[predicted_class==5,]
    df_5 = df_5[,rownames(r_model_5$importance)]
    df_5 = df_5[rowSums(is.na(df_5))==0,]
    df_6 = df_regress[predicted_class==6,]
    df_6 = df_6[,rownames(r_model_6$importance)]
    df_6 = df_6[rowSums(is.na(df_6))==0,]
    df_7 = df_regress[predicted_class==7,]
    df_7 = df_7[,rownames(r_model_7$importance)]
    df_7 = df_7[rowSums(is.na(df_7))==0,]
    predict_1 = predict(r_model_1,df_1)
    predict_2 = predict(r_model_2,df_2)
    predict_3 = predict(r_model_3,df_3)
    predict_4 = predict(r_model_4,df_4)
    predict_5 = predict(r_model_5,df_5)
    predict_6 = predict(r_model_6,df_6)
    predict_7 = predict(r_model_7,df_7)
    predicted_mag = c(predict_1,predict_2,predict_3,predict_4,predict_5,predict_6,predict_7)
    biomass_present = df[names(predicted_mag),'biomass']
    biomass_future = (predicted_mag*biomass_present+biomass_present)/(1-predicted_mag)
    biomass_predicted[names(biomass_future),as.character(y_range[i])] = biomass_future
    percent_change[names(predicted_mag),as.character(y_range[i])] = predicted_mag
  }
  r = r+24690
  #assign(paste0('vi_df',y_range[i]),df)
  
}

#Then take observed changes
lagged_mass_df = biomass
lagged_mass_df[] = NA
remove_cols = tail(1:ncol(biomass),10)*-1
for(j in (1:(ncol(biomass)))[remove_cols]){
  lagged_mass_df[,j] = (biomass[,j+10] - biomass[,j])/(biomass[,j+10] + biomass[,j])
}

#And ndvi trends
ndvi_df = read.csv(paste0(leadpath,"EWS_package/data/VIs/consolidated/full_LANDSAT_ndvi_median.csv"),row.names = 'Plot_ID')
ndvi_df = ndvi_df[sites,]
ndvi_df = ndvi_df[,-1]
ndvi_df[ndvi_df==null_val] = NA
trend_ndvi_df = data.frame()

for(j in c(1985,1995,2005)){
  for(r in 1:nrow(ndvi_df)){
    if(sum(!is.na(as.numeric(ndvi_df[r,paste0('X',(j+1):as.character(j+10))])))>1){
      trend_ndvi_df[r,as.character(j)] = summary(lm(as.numeric(ndvi_df[r,paste0('X',(j+1):as.character(j+10))])~c(1:10)))$coefficients[2]
    }
  }
}

colnames(biomass_predicted) = 1991:2027
plot(unlist(biomass[,as.character(1991:2017)]),unlist(biomass_predicted[,as.character(1991:2017)]))
summary(lm(as.vector(biomass[,as.character(1991:2017)])~as.vector(biomass_predicted[,as.character(1991:2017)])))



#create a color palette for plotting
my_palette = colorRampPalette(c("red", "white","green"))(n = 299)

max_ndvi = max(abs(quantile(trend_ndvi_df,c(0.01,0.99),na.rm=T)),na.rm=T)

#Make some 10 year plots
for(y in c(1985,1995,2005,2015)){
  obs_df_year = lagged_mass_df[,as.character(y)]
  obs_df_year = obs_df_year[!is.na(obs_df_year)]
  pred_df_year = percent_change[,as.character(y)]
  pred_df_year = pred_df_year[!is.na(pred_df_year)]
  
  
  
  
  
  if(y!=2015){
    obs_locs = site_loc[names(obs_df_year),]
    obs_sp = SpatialPointsDataFrame(coords = obs_locs[,2:3],data = data.frame(obs_df_year),proj4string = CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
    obs_sp = spTransform(obs_sp,Ju_Masek_crs)
    obs_rast = rasterize(obs_sp,template,field='obs_df_year',fun=mean,na.rm=T)
    pdf(file=paste0(leadpath,'EWS_package/data/plots/spatial/observed_',y,'to',y+10,'.pdf'),width = 16, height = 12) 
    plot(political,col='gray',main = paste0('Observed percent change ',y,' to ',y+10))
    plot(obs_rast,col=my_palette,zlim = c(-1,1),add=T)
    dev.off()
    
    ndvi_df_year = trend_ndvi_df[,as.character(y)]
    names(ndvi_df_year) = sites
    ndvi_df_year = ndvi_df_year[!is.na(ndvi_df_year)]
    ndvi_locs = site_loc[names(ndvi_df_year),]
    ndvi_sp = SpatialPointsDataFrame(coords = ndvi_locs[,2:3],data = data.frame(ndvi_df_year),proj4string = CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
    ndvi_sp = spTransform(ndvi_sp,Ju_Masek_crs)
    ndvi_rast = rasterize(ndvi_sp,template,field='ndvi_df_year',fun=mean,na.rm=T)
    pdf(file=paste0(leadpath,'EWS_package/data/plots/spatial/ndvi_',y,'to',y+10,'.pdf'),width = 16, height = 12)
    plot(political,col='gray',main = paste0('NDVI trend ',y,' to ',y+10))
    plot(ndvi_rast,col=my_palette,zlim=c(-max_ndvi,max_ndvi),add=T)
    dev.off()
  }
  if(y!=1985){
    pred_locs = site_loc[names(pred_df_year),]
    pred_sp = SpatialPointsDataFrame(coords = pred_locs[,2:3],data = data.frame(pred_df_year),proj4string = CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
    pred_sp = spTransform(pred_sp,Ju_Masek_crs)
    pred_rast = rasterize(pred_sp,template,field='pred_df_year',fun=mean,na.rm=T)
    pdf(file=paste0(leadpath,'EWS_package/data/plots/spatial/predicted_',y,'to',y+10,'.pdf'),width = 16, height = 12)
    plot(political,col='gray',main = paste0('Predicted percent change ',y,' to ',y+10))
    plot(pred_rast,col=my_palette,zlim = c(-1,1),add=T)
    dev.off()
  }
}

#Now create a predicted biomass change df for the years 1984-2012 (to match up with the Ju and Masek era)
#What this does is takes any observed data, interpolates it and then predicts from the last year of good data
#in all cases, if data extends past 2002, I predict from the year 2002, so we have at least on prediction for 
#each site
r=1
df_years = list()
y_range = 1981:2012
for(i in 1:length(y_range)){
  df = vi_df[r:(r+24689),]
  colnames(df) = gsub("[ /-]",".",colnames(df))
  rownames(df) = sites
  df_years[[as.character(y_range[i])]] = df
  r = r+24690
  print(paste0(y_range[i]," ",sum(!is.na(df$biomass))))
}
y_range = 1981:2002
trees = 2083:2138
for(i in 1:length(y_range)){
  df = df_years[[as.character(y_range[i])]]
  
  biomass_compare[,as.character(y_range[i])] = df$biomass
  print(paste0(y_range[i]," ",sum(!is.na(df$biomass))))
  #Because I'm using predicted biomass values for future predictions, I need to also predict species composition values,
  #Otherwise those values will be NA because they are based on having a valid biomass value. Instead of actually building
  #another model to do this, I'm just taking the last valid composition value
  if(i!=1){
    
    for (t in trees){
      na_sites = rownames(df)[is.na(df[,t])]
      df[na_sites,t] = df_years[[as.character(y_range[i]-1)]][na_sites,t]
      df_years[[as.character(y_range[i])]][na_sites,t] = df_years[[as.character(y_range[i]-1)]][na_sites,t]
    }
  }
  df_class = df[,rownames(class_model$importance)]
  df_class = df_class[rowSums(is.na(df_class))==0,]
  if(nrow(df_class)>0){
    predicted_class = predict(class_model,df_class)
    df_regress = df[rownames(df_class),]
    df_1 = df_regress[predicted_class==1,]
    df_1 = df_1[,rownames(r_model_1$importance)]
    df_1 = df_1[rowSums(is.na(df_1))==0,]
    df_2 = df_regress[predicted_class==2,]
    df_2 = df_2[,rownames(r_model_2$importance)]
    df_2 = df_2[rowSums(is.na(df_2))==0,]
    df_3 = df_regress[predicted_class==3,]
    df_3 = df_3[,rownames(r_model_3$importance)]
    df_3 = df_3[rowSums(is.na(df_3))==0,]
    df_4 = df_regress[predicted_class==4,]
    df_4 = df_4[,rownames(r_model_4$importance)]
    df_4 = df_4[rowSums(is.na(df_4))==0,]
    df_5 = df_regress[predicted_class==5,]
    df_5 = df_5[,rownames(r_model_5$importance)]
    df_5 = df_5[rowSums(is.na(df_5))==0,]
    df_6 = df_regress[predicted_class==6,]
    df_6 = df_6[,rownames(r_model_6$importance)]
    df_6 = df_6[rowSums(is.na(df_6))==0,]
    df_7 = df_regress[predicted_class==7,]
    df_7 = df_7[,rownames(r_model_7$importance)]
    df_7 = df_7[rowSums(is.na(df_7))==0,]
    predict_1 = predict(r_model_1,df_1)
    predict_2 = predict(r_model_2,df_2)
    predict_3 = predict(r_model_3,df_3)
    predict_4 = predict(r_model_4,df_4)
    predict_5 = predict(r_model_5,df_5)
    predict_6 = predict(r_model_6,df_6)
    predict_7 = predict(r_model_7,df_7)
    predicted_mag = c(predict_1,predict_2,predict_3,predict_4,predict_5,predict_6,predict_7)
    biomass_present = df[names(predicted_mag),'biomass']
    biomass_future = (predicted_mag*biomass_present+biomass_present)/(1-predicted_mag)
    if(y_range[i]!=2002){
      non_nas = rownames(df_years[[as.character(y_range[i]+10)]])[!is.na(df_years[[as.character(y_range[i]+10)]]$biomass)]
      non_nas = non_nas[non_nas %in% names(biomass_future)]
      biomass_future = biomass_future[!names(biomass_future) %in% non_nas]
      df_years[[as.character(y_range[i]+10)]][names(biomass_future),'biomass'] = biomass_future
    }else{
      df_years[[as.character(y_range[i]+10)]][names(biomass_future),'biomass'] = biomass_future
    }
    biomass_predicted[names(biomass_future),as.character(y_range[i])] = biomass_future
    percent_change[names(predicted_mag),as.character(y_range[i])] = predicted_mag
  }
  r = r+24690
  #assign(paste0('vi_df',y_range[i]),df)
  
}
for(y in 2003:2012){
  df = df_years[[as.character(y)]]
  biomass_compare[,as.character(y)] = df$biomass
  print(paste0(y," ",sum(!is.na(df$biomass))))
}

# for(y in 1981:2012){
#   row = df_years[[as.character(y)]]['1_01001 G000505',]
#   row = row[,rownames(class_model$importance)]
#   print(y)
#   print(row)
# }

#Now determine the linear trends over the time frame, same as Ju and Masek
trends = vector(length = nrow(biomass_compare))
names(trends) = rownames(biomass_compare)

for(i in 1:length(trends)){
  row = biomass_compare[i,as.character(1984:2012)]
  if(sum(!is.na(row))>1){
    trends[i] = summary(lm(1:29~row))$coefficients[2]
  }else{
    trends[i] = NA
  }
}


trends = trends[!is.na(trends)]
trend_locs = site_loc[names(trends),]
trend_sp = SpatialPointsDataFrame(coords = trend_locs[,2:3],data = data.frame(trends),proj4string = CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
trend_sp = spTransform(trend_sp,Ju_Masek_crs)
trend_rast = rasterize(trend_sp,template,field='trends',fun=mean,na.rm=T)
pdf(file=paste0(leadpath,'EWS_package/data/plots/spatial/predicted_1984to2012.pdf'),width = 16, height = 12)
plot(political,col='gray',main = paste0('Predicted biomass trend (Mg/ha/year) 1984 to 2012'))
plot(trend_rast,col=my_palette,zlim = c(-2,2),add=T)
dev.off()

#Mask the Ju & Masek trends by the prediction trends so we are only looking at pixels where I have data
Ju_Masek_masked = mask(Ju_Masek_agg,trend_rast)
pdf(file=paste0(leadpath,'EWS_package/data/plots/spatial/ndvi_1984to2012.pdf'),width = 16, height = 12)
plot(political,col='gray',main = paste0('NDVI trend (Ju & Masek) 1984 to 2012'))
plot(Ju_Masek_masked,col=my_palette,add=T)
dev.off()




######FOr species plots#######
load(paste0(leadpath,"EWS_package/data/models/input_data/vi_df_all_2019-10-30.Rda"))

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
# class_inf = read.delim(paste0(folder,'models.txt'),sep=" ",header=T,stringsAsFactors = F)
# class_inf = class_inf[which(class_inf$kappa==max(class_inf$kappa)),]
# class_train = read.csv(paste0(folder,'PD_plots/train_data_',class_inf$run_number,'.csv'))
test_data = read.csv(paste0(folder,'regress_with_regions/test_data/test_data_',regress_model_inf$run_number,'.csv'))

# test_df = vi_df
# test_df = test_df[test_df$site %in% test_data$rownames.site_loc..as.numeric.df_sites..,]
test_df = vi_df[,rownames(class_model$importance)]
test_df = test_df[rowSums(is.na(test_df))==0,]

predicted_class = predict(class_model,test_df)

train_sites = vector()
for(i in 1:7){
  train_data = read.csv(paste0(folder,'regress_with_regions/PD_plots/train_data_',i,'_',regress_model_inf$run_number,'.csv'))
  train_sites = c(train_sites,as.character(train_data$X))
}

df_regress = vi_df[rownames(test_df),]
df_1 = df_regress[predicted_class==1,]
df_1 = df_1[,rownames(r_model_1$importance)]
df_1 = df_1[rowSums(is.na(df_1))==0,]
df_2 = df_regress[predicted_class==2,]
df_2 = df_2[,rownames(r_model_2$importance)]
df_2 = df_2[rowSums(is.na(df_2))==0,]
df_3 = df_regress[predicted_class==3,]
df_3 = df_3[,rownames(r_model_3$importance)]
df_3 = df_3[rowSums(is.na(df_3))==0,]
df_4 = df_regress[predicted_class==4,]
df_4 = df_4[,rownames(r_model_4$importance)]
df_4 = df_4[rowSums(is.na(df_4))==0,]
df_5 = df_regress[predicted_class==5,]
df_5 = df_5[,rownames(r_model_5$importance)]
df_5 = df_5[rowSums(is.na(df_5))==0,]
df_6 = df_regress[predicted_class==6,]
df_6 = df_6[,rownames(r_model_6$importance)]
df_6 = df_6[rowSums(is.na(df_6))==0,]
df_7 = df_regress[predicted_class==7,]
df_7 = df_7[,rownames(r_model_7$importance)]
df_7 = df_7[rowSums(is.na(df_7))==0,]
predict_1 = predict(r_model_1,df_1)
predict_2 = predict(r_model_2,df_2)
predict_3 = predict(r_model_3,df_3)
predict_4 = predict(r_model_4,df_4)
predict_5 = predict(r_model_5,df_5)
predict_6 = predict(r_model_6,df_6)
predict_7 = predict(r_model_7,df_7)
predicted_mag = c(predict_1,predict_2,predict_3,predict_4,predict_5,predict_6,predict_7)
observed_mag = vi_df[names(predicted_mag),'lagged_biomass']

df = data.frame(predicted_mag,observed_mag)
colnames(df) = c('predicted','observed')
# df1 = df[!rownames(df) %in% train_sites,]
# df1$set = 'Withheld sites'
# df2 = df[rownames(df) %in% train_sites,]
# df2$set = 'Training sites'
df[!rownames(df) %in% train_sites,'set'] = 'Withheld sites'
df[rownames(df) %in% train_sites,'set'] = 'Training sites'
#df = rbind(df1,df2)

# df2 = data.frame(train_predict,train_obs)
# colnames(df2) = c('predicted','observed')
# df2$set = 'Training sites'
# df = rbind(df,df2)

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

for(sp in colnames(dom_sp)){
  sp_plot = sp_cv_func(df,sp)
  if(!is.null(sp_plot)){
    ggsave(paste0(leadpath,'EWS_package/data/plots/species/sp_cv_',sp,".png"),sp_plot,width=12,height=6,units='in')
  }
}
