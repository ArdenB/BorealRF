# This creates the corellations and the RDA files
library(dplyr)
library(plyr)

library(data.table)

#Define variables
y_range = 1981:2017
null_val = -32768
leadpath = "/att/nobackup/scooperd/"
burn_win = 50
burn_th = 10

#Which VIs do we want to try? Order is important, should be in this sequence: NDVI,MSI,NDII,SATVI,NDVSI,PSRI,TWFC
VIs = c('ndvi','psri','ndii','ndvsi','msi','nirv','ndwi','nbr','satvi','tvfc')


####Functions######

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

# Process burn df
#Creates a df with 0s for all years that are within 'burn_win' year after  a fire that burned >'burn_th'% of the pixel that contains the site
#burn_win and burn_th are defined above
process_burn = function(in_df) {
  burn_site_list = as.factor(rownames(in_df))
  burn_windows = lapply(y_range,function(x) ((x-burn_win):x))
  unburned_df = as.data.frame(sapply(burn_windows,function(x) rowSums(in_df[,paste0("X",as.character(x)[paste0("X",as.character(x)) %in% colnames(in_df)])]>burn_th))==0)
  colnames(unburned_df) = c(paste0("X",y_range))
  unburned_df[unburned_df==0]=NA
  return(unburned_df)
}

#Process damage df
#Same as above but for other disturbances we decided to remove
process_dam = function(in_df){
  in_df<-in_df[,-1]
  in_df[is.na(in_df)]=0
  burn_windows = lapply(y_range,function(x) ((x-burn_win):x))
  undamaged_df<- data.frame(sapply(burn_windows,function(x) rowSums(in_df[,paste0("X",as.character(x)[paste0("X",as.character(x)) %in% colnames(in_df)])]>burn_th))==0)
  colnames(undamaged_df) = c(paste0("X",y_range))
  undamaged_df = clean_df(undamaged_df)
  undamaged_df[is.na(undamaged_df)] = 1
  undamaged_df=undamaged_df*1
  undamaged_df[undamaged_df==0]=NA
  return(undamaged_df)
}


#Read in lengths of time since last survey
raw_shift_df = read.csv(paste0(leadpath,"scooperdock/EWS/data/psp/surv_interval_filled.csv"))
raw_shift_df = arrange(raw_shift_df,X)
rownames(raw_shift_df) = raw_shift_df[,"X"]
sites = rownames(raw_shift_df)
shift_df = clean_df(raw_shift_df)

#Read in site locations
site_loc = read.csv(paste0(leadpath,"scooperdock/EWS/data/raw_psp/All_sites_101218.csv"),row.names = 'Plot_ID')
#There's a mismatch in the names for some Yukon sites, so this fixes it
adj_YT = site_loc[grep("11_",rownames(site_loc)),]
for(i in 1:nrow(adj_YT)){
  rownames(adj_YT)[i] = paste0("11_",sprintf("%03.0f",as.numeric(strsplit(rownames(adj_YT)[i],"_")[[1]][2])))
}
rownames(site_loc)[grep("11_",rownames(site_loc))] = rownames(adj_YT)
site_loc = site_loc[sites,]

#Read in survey dates
surv_date = read.csv(paste0(leadpath,"scooperdock/EWS/data/psp/surv_date_matrix.csv"))
surv_date = arrange(surv_date,X)
rownames(surv_date) = surv_date[,"X"]
surv_date = clean_df(surv_date)
#Create a lagged dataframe that indicates when there is a measurement 1- years after a date. This is used to only take values 
#that have been interpolated using a year 10 value from an actual measurement
surv_date_lag = data.frame(surv_date[,-c(1:10)],matrix(nrow = nrow(surv_date),ncol = 10))
colnames(surv_date_lag) = colnames(surv_date)

shift_df[!is.na(surv_date)] = 1

#Read in biomass dataframes
raw_mass_df = read.csv(paste0(leadpath,"scooperdock/EWS/data/psp/biomass_interpolated_w_over_10years.csv"),row.names = 'X')
mass_df = clean_df(raw_mass_df)
mass_df[mass_df<0] = NA

#I use the fully interploted dataframe for the values of the actual measurements, doesn't have to be this way, but it was there
raw_mass_df = read.csv(paste0(leadpath,"scooperdock/EWS/data/psp/biomass_interpolated.csv"),row.names = 'X')
mass_df2 = clean_df(raw_mass_df)

#Read in stem density
raw_stem_df = read.csv(paste0(leadpath,"scooperdock/EWS/data/psp/stem_dens_interpolated_w_over_10years.csv"),row.names = 'X')
stem_df = clean_df(raw_stem_df)
stem_df[stem_df<0] = NA

# ========== Used for Remoing sites that have undergone disturbance ==========
#Read in damage and burn dataframes
raw_damaged_df<-read.csv(paste0(leadpath,"scooperdock/EWS/data/psp/damage_flags.csv"),stringsAsFactors = FALSE,row.names = "X")
raw_damaged_df[is.na(raw_damaged_df)] = 0
raw_damaged_df = raw_damaged_df*100
undamaged_df = process_dam(raw_damaged_df)

raw_burn_df = read.csv(paste0(leadpath,"scooperdock/EWS/data/fire/LANDSAT_fire.csv"),row.names="Plot_ID",stringsAsFactors = F)
raw_burn_df = raw_burn_df[,-1]
#As above, there are some issues with names in Yukon
adj_YT = raw_burn_df[grep("11_",rownames(raw_burn_df)),]
for(i in 1:nrow(adj_YT)){
  rownames(adj_YT)[i] = paste0("11_",sprintf("%03.0f",as.numeric(strsplit(rownames(adj_YT)[i],"_")[[1]][2])))
}
rownames(raw_burn_df)[grep("11_",rownames(raw_burn_df))] = rownames(adj_YT)
unburned_df = clean_df(process_burn(raw_burn_df))

#Remove sites that have disturbances
untainted_df = undamaged_df*unburned_df
mass_df = mass_df*untainted_df
mass_df2 = mass_df2*untainted_df
stem_df = stem_df*untainted_df

#THis is kind of a holdover, the maximum interval allowed. Since I switched interpolation method, it isn't really necessary
int = 10
keep_max_interval=shift_df
keep_max_interval[keep_max_interval>int] = NA
keep_max_interval[!is.na(keep_max_interval)]=1
mass_df2 = mass_df2*keep_max_interval

#Create the database
vi_df = data.frame('biomass' = unlist(mass_df))
vi_df$stem_density = unlist(stem_df)

#Add in all the vis
for (VI in VIs){
  df = data.frame(fread(paste0(leadpath,"scooperdock/EWS/data/vi_metrics/metric_dataframe_",VI,"_noshift.csv")))
  rownames(df) = df[,1]
  df = df[,c(-1,-2)]
  print(paste0(VI,' read: ',Sys.time()))
  vi_df = data.frame(vi_df,df)
}

# Read in species compositions
# Table used to read species groups and time, different group types
LUT = read.csv(paste0(leadpath,"scooperdock/EWS/data/raw_psp/SP_LUT.csv"),stringsAsFactors = F)
sp_groups = read.csv(paste0(leadpath,"scooperdock/EWS/data/raw_psp/SP_groups.csv"),stringsAsFactors = F) 

sp_out_df = data.frame('site' = rep(sites,37))
rows = vector()
for(i in 1:151){
  if(file.exists(paste0(leadpath,"scooperdock/EWS/data/psp/comp_interp_",i,".csv"))){
    raw_sp_df = read.csv(paste0(leadpath,"scooperdock/EWS/data/psp/comp_interp_",i,".csv"),stringsAsFactors = F,row.names = 'X')
    sp_df = clean_df(raw_sp_df)
    sp_df[!is.na(mass_df)&is.na(sp_df)] = 0
    sp = LUT$scientific[LUT$ID==i]
    sp_out_df[,sp] = unlist(sp_df)
    rows[LUT$scientific[LUT$ID==i]] = nrow(raw_sp_df)
  }
}

#Also add up into my somewhat arbitrary groupds I created, and only take species with >1500 observations
sp_groups = sp_groups[sp_groups$scientific %in% colnames(sp_out_df),]
sp_rows = 1500
for(gr in colnames(sp_groups)[2:6]){
  groups = unique(sp_groups[,gr])
  if(gr=='scientific'){
    groups = groups[groups %in% names(rows)[rows>sp_rows]]
  }
  for (g in groups){
    sp = sp_groups$scientific[sp_groups[,gr]==g]
    if(length(sp)==1){
      vi_df[,paste0(gr,"_",g)] = sp_out_df[,sp]
    }else{
      vi_df[,paste0(gr,"_",g)] = rowSums(sp_out_df[,sp],na.rm=T)
    }
  }
}

#Add in soil chracteristics
soils = read.csv(paste0(leadpath,"scooperdock/EWS/data/psp/soils/soil_properties_aggregated.csv"))            
rownames(soils) = soils$prop_vals.rownames.samp_loc.
adj_YT = soils[grep("11_",rownames(soils)),]
for(i in 1:nrow(adj_YT)){
  rownames(adj_YT)[i] = paste0("11_",sprintf("%03.0f",as.numeric(strsplit(rownames(adj_YT)[i],"_")[[1]][2])))
}
rownames(soils)[grep("11_",rownames(soils))] = rownames(adj_YT)
soils = soils[sites,]
soils = soils[,c(-1,-2)]

#Take only 30cm info
soils = soils[,c(2,5,8,11,14,17,20,22)]


num_cols = ncol(vi_df)

for(i in 1:ncol(soils)){
  vi_df = data.frame(vi_df,soils[,i])
}

colnames(vi_df)[(num_cols+1):(num_cols+i)] = colnames(soils)

#Add in climate df
all_climate = read.csv(paste0(leadpath,"scooperdock/EWS/data/psp/Climate/1951-2018/climate_df_30years.csv"),row.names = 'X')
#there were some statisitics that I decided afterwards that didn't make sense, like relative trends for a number of variables,
#so I created absolute trends and am removing the relative trends here. 
remove_clim = c('MAR_mean_30years','MAR_trend_30years','MAR_abs_trend_30years','MAT_trend_30years','MWMT_trend_30years',
                'MCMT_trend_30years','TD_trend_30years','FFP_trend_30years','EXT_trend_30years','EMT_trend_30years',
                'eFFP_trend_30years','DD5_trend_30years','DD18_trend_30years','DD_18_trend_30years','DD_0_trend_30years',
                'bFFP_trend_30years','RH_trend_30years','NFFD_trend_30years')
all_climate = all_climate[,!colnames(all_climate) %in% remove_clim]

vi_df = data.frame(vi_df,all_climate)

#Add in permafrost data
permafrost = read.csv(paste0(leadpath,"scooperdock/EWS/data/psp/permafrost/extract_permafrost_probs.csv"))
rownames(permafrost) = permafrost$rownames.samp_loc.
permafrost = permafrost[,c(-1,-2)]
adj_YT = permafrost[grep("11_",rownames(permafrost)),]
for(i in 1:nrow(adj_YT)){
  rownames(adj_YT)[i] = paste0("11_",sprintf("%03.0f",as.numeric(strsplit(rownames(adj_YT)[i],"_")[[1]][2])))
}
rownames(permafrost)[grep("11_",rownames(permafrost))] = rownames(adj_YT)
permafrost = permafrost[sites,]
permafrost[is.na(permafrost)] = 0

vi_df = data.frame(vi_df,permafrost)
vi_df$site = sites


#set lag time (as in prediction interval), this is a holdover from back when I was exploring different intervals, shouldn't need to be changed
l = 10

#Create a lagged mass_df, which calculates how much a site's biomass has changed over 10 years, this is what we're modeling
lagged_mass_df = mass_df
lagged_mass_df[] = NA
remove_cols = tail(1:ncol(mass_df),l)*-1
for(j in (1:(ncol(mass_df)))[remove_cols]){
  lagged_mass_df[,j] = (mass_df2[,j+l] - mass_df[,j])/(mass_df2[,j+l] + mass_df[,j])
}
#remove sites that don't have an actual measurment at year 10
lagged_mass_df = lagged_mass_df*surv_date_lag
lagged_mass_df = unlist(lagged_mass_df)
lagged_mass_df[!is.finite(lagged_mass_df)] = NA
vi_df$lagged_biomass = lagged_mass_df


vi_df = vi_df[!is.na(vi_df$lagged_biomass),]

save(vi_df,file = paste0(leadpath,"scooperdock/EWS/data/vi_metrics/vi_df_all_",today,".Rda"))

#Read in correlations csv
VI_corr = read.csv(paste0(leadpath,"scooperdock/EWS/data/linear_models/all_vi_corr_100219.csv"),stringsAsFactors = F)
soils_corr = read.csv(paste0(leadpath,"scooperdock/EWS/data/linear_models/soils_corr_042419.csv"),stringsAsFactors = F)
climate_corr = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/linear_models/climate_corr_082319.csv")
sp_corr = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/linear_models/sp_groups_corr_082019.csv")
#add then all together. Note: the last two vectors added are just there so that if one of the permafrost datasets comes out as important,
#the other will be removed because I've set them as correlated to each other.
correlations = rbind(VI_corr,soils_corr,climate_corr,sp_corr,c(NA,'Obu2019','Gruber',1),c(NA,'Gruber','Obu2019',1))
correlations$rho = as.numeric(correlations$rho)
rho_limit = 0.5
#only take correlations with an absolute value over my rho limit (0.5)
correlations = correlations[abs(correlations$rho)>rho_limit,]

save(correlations,file = paste0(leadpath,"scooperdock/EWS/data/linear_models/correlations_",today,".Rda"))

