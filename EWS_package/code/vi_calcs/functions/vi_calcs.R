#####
# scooperdock@whrc.org
# Modified from Kylen's script
# Functions to compute trend, pulse, ews, and ddj metrics for matrix of NDVI time series.
# Each function takes a time series of NDVI values and a vector of time window lengths to compute the metrics for.
# Returns time series of rolling window NDVI metrics.
#####

#### Load Reqs ####
require(earlywarnings)
source("/att/nobackup/scooperd/EWS_package/code/vi_calcs/functions/modified_earlywarnings_package/generic_ews_noplot.R")
source("/att/nobackup/scooperd/EWS_package/code/vi_calcs/functions/modified_earlywarnings_package/ddjnonparam_ews_noplot.R")
# source("H:/scooperd/r_scripts/code/analysis/functions/modified_earlywarnings_package/generic_ews_noplot.R")
# source("H:/scooperd/r_scripts/code/analysis/functions/modified_earlywarnings_package/ddjnonparam_ews_noplot.R")

#### Simple Organization Functions ####
# Column names to numerical years
nty = function(name_vec) {
  return(as.numeric(sapply(strsplit(name_vec,"X"), "[[", 2)))
}

# Numerical years to column names
ytn = function(year) {
  return(paste0("X",year))
}

#### Calculation Functions ####
find_leadtrail_NAs = function(vi_non_na) {

  leadtrail_nas = c()
  if(!vi_non_na[1]) {#If the first value is NA
    first_valid = which(vi_non_na)[1] #then first_valid=whichever the first datapoint with a value is
    leadtrail_nas = c(leadtrail_nas,1:(first_valid-1)) #and leadtrail_nas=the columns to start with NAs 
  }
  if(!vi_non_na[length(vi_non_na)]) {#If the last value is NA
    last_valid = which(vi_non_na)[length(which(vi_non_na))] #then last_valid=the last datapoint with a value
    leadtrail_nas = c(leadtrail_nas,(last_valid+1):length(vi_non_na)) #and leadtrail_nas=columns that end with NAs
  }
  return(leadtrail_nas)
  
}


trend_calc = function(vi,timelags) {
  output = list("trend" = list())
  
  # Which VI Exist
  vi_exists = !is.na(vi)
  
  for(tl in timelags) {
    #Output
    output_vec = vi
    output_vec[] = NA
    
    for(y in (y_range[1]+tl):tail(y_range,1)) {#for a year in y_range that is "tl" years after the start of y_range
      pred = (y-tl):(y-tl_start) #year - time lag : year - start year
      resp = as.numeric(vi[ytn(pred)]) #Takes NDVI values for the years in pred
      resp_exists = !is.na(resp) #Returns TRUE if there is data in each year
      
      if((100*(sum(resp_exists))/(1+tl-tl_start))>=(100-miss_rule[1]) # Missing rule 1, if greater than or equal to 
         #75% of values exist in "pred" (when miss_rule[1]=25)
         & !(grepl(paste(rep(FALSE,miss_rule[2]),collapse=";"),paste(resp_exists,collapse=";")))  # Missing rule 2
         #and if less than 2 consecutive datapoints are missing (when miss_rule[2]=2)
      ) { 
        output_vec[ytn(y)] = (100*coefficients(lm(resp~pred))[[2]]/mean(resp,na.rm=T))#Unsure exactly why this calculation is the one made
      }
      
    }
    output[["trend"]][[as.character(tl)]] = output_vec
  }
  return(output)
}

pulse_calc = function(vi,timelags) {
  output = list("pulse" = list())
  
  # Find which vi exist
  vi_exists = !is.na(vi) #Determines if vi data exists, returns false for missing
  
  smooth_vals = loess(as.numeric(vi)~y_range,na.action="na.exclude")$fitted #Creates a smoothed curve of vi data
  
  anom_vi_list = vi[vi_exists] - smooth_vals #determines difference between actual vi data and smoothed data
  
  zval_list = (anom_vi_list - mean(anom_vi_list))/sd(anom_vi_list) #This is how you calculate z_score for the anomoly
  
  zval_in_years = vi #Creates a vector of same size as vi, with years as column headers
  zval_in_years[] = NA #Fills in default as NA
  zval_in_years[names(vi_exists[,vi_exists])] = zval_list #sets the data to z_val_list, minus those missing in vi_exists
  
  for(tl in timelags) {
    #Output, create vector with years as headers and NA for data
    output_vec = vi
    output_vec[] = NA
    
    for(y in (y_range[1]+tl):tail(y_range,1)) {#for a year in y_range that is "tl" years after the start of y_range
      pred = (y-tl):(y-tl_start) #year - time lag : year - start year
      resp = as.numeric(zval_in_years[ytn(pred)]) #Takes z values for the years in pred
      resp_exists = !is.na(resp) #Returns TRUE if there is data in each year
      
      if((100*(sum(resp_exists))/(1+tl-tl_start))>=(100-miss_rule[1]) # Missing rule 1
         & !(grepl(paste(rep(FALSE,miss_rule[2]),collapse=";"),paste(resp_exists,collapse=";"))) # Missing rule 2
      ) { 
        output_vec[ytn(y)] = min(resp,na.rm=T) #inputs the minimum z-value for each year over the range used
      }
    }
    
    output[["pulse"]][[as.character(tl)]] = output_vec
  }
  return(output)
}

ews_calc = function(vi,timelags) {
  output = list()
  for(met in sub_metrics[["ews"]]) {
    output[[met]] = list()
  }
  
  # Find which vi exist
  vi_exists = !is.na(vi)
  
  # Find the leading and trailing NAs
  leadtrail_na = find_leadtrail_NAs(vi_exists)
    
  vi_old = vi
  if(length(leadtrail_na)>0) {#If there are NAs to start or end, remove them
    vi = vi[-leadtrail_na]
  }

  y_range_temp = nty(colnames(vi)) #creates column names of y_range_temp that correspond to years of vi
  y_range_index = 1:length(y_range_temp) #creates vector 1:length(y_range_temp)
  
  #If length of vi is less than 3 greater than the timelag, the densratio calc gets caught. This limits the timelags to loop through.
  time = timelags[timelags<length(vi)-4] #4 for tl_start of 0.
  
  for (tl in time) {
    # Find invalid time indices
    invalid_y = c()
    
    for(yi in (y_range_index[1]+tl-1):tail(y_range_index,1)) {#Note: this is using absolute time since start of measurements
      #ie: 1 through 29 as opposed to 1984 through 2012
      pred = (yi-tl):(yi-tl_start)
      resp_exists = (!is.na(vi))[pred]
      if(!(
        (100*(sum(resp_exists))/(1+tl-tl_start))>=(100-miss_rule[1]) # Missing rule 1
        & !(grepl(paste(rep(FALSE,miss_rule[2]),collapse=";"),paste(resp_exists,collapse=";")))  # Missing rule 2
      )) {
        invalid_y = c(invalid_y,yi)#If this time does not follow the two above missing rules, then it is labeled an invalid_y
      }
    }
    
    ##Only run calc if there are valid years
    if(!length((y_range_index[1]+tl-1):tail(y_range_index,1))==length(invalid_y)){
      ews_out = generic_ews_noplot(data.frame(as.numeric(vi)),detrending="loess",interpolate=T,winsize=100*(length(tl_start:tl)-1)/length(y_range_index))
      ews_out[unlist(ews_out['timeindex']) %in% invalid_y,-1] = NA #replace all rows corresponding to invalid_y with NA
    }else{#If no valid years, create empty dataframe
      ews_out = data.frame((y_range_index[1]+tl-1):tail(y_range_index,1),NA,NA,NA,NA,NA,NA,NA,NA)
      colnames(ews_out) = c('timeindex','ar1','sd','sk','kurt','cv','returnrate','densratio','acf1')
    }
    for(met in sub_metrics[["ews"]]) {
      output[[met]][[as.character(tl)]] = vi_old
      output[[met]][[as.character(tl)]][] = NA
      output[[met]][[as.character(tl)]][ytn(y_range_temp[unlist(ews_out['timeindex'])])] = ews_out[,met]
    }
  }
  return(output)
}

ddj_calc = function(vi,timelags) {
  output = list()
  for(met in sub_metrics[["ddj"]]) {
    output[[met]] = list()
  }
  
  # Find which vi exist
  vi_exists = !is.na(vi)
  
  # Find the leading and trailing NAs
  leadtrail_na = find_leadtrail_NAs(vi_exists)
  
  vi_old = vi
  if(length(leadtrail_na)>0) {
    vi = vi[-leadtrail_na]
  }
  
  y_range_temp = nty(colnames(vi))
  y_range_index = 1:length(y_range_temp)
  
  for (tl in timelags) {
    
    # Find valid time indices
    valid_y = c()
    if(tl<=tail(y_range_index,1)) {
      for(yi in (tl):tail(y_range_index,1)) {
        pred = (yi-tl):(yi-tl_start)
        resp_exists = !is.na(vi)[pred]
        if(((100*(sum(resp_exists))/(1+tl-tl_start))>=(100-miss_rule[1]) # Missing rule 1
            & !(grepl(paste(rep(FALSE,miss_rule[2]),collapse=";"),paste(resp_exists,collapse=";")))) # Missing rule 2
        ) {
          valid_y = c(valid_y,yi)
        }
      }
    }
    
    ddj_out = ddjnonparam_ews_noplot(data.frame(as.numeric(vi)),interpolate=T)
    
    for(met in sub_metrics[["ddj"]]) {
      output[[met]][[as.character(tl)]] = vi_old
      output[[met]][[as.character(tl)]][] = NA
      for(yi in valid_y) {
        output[[met]][[as.character(tl)]][ytn(y_range_temp[yi])] = max(ddj_out[[met]][(yi-tl):(yi-tl_start)],na.rm=T)
      }
    }
  }
  return(output)
}


#### Main Run Function ####
vi_df_calc = function() {
  
  
  func_list = list(
    "trend" = trend_calc,
    "pulse" = pulse_calc,
    "ews" = ews_calc,
    "ddj" = ddj_calc
  )
  
  #### Populating output ####
  # Make output list
  output_list = list()
  
  # Make template dataframe, creates dataframe filled with NAs with dimensions and names of vi_df
  blank_df = data.frame(matrix(NA,nrow=dim(vi_df)[1],ncol=dim(vi_df)[2]),row.names = rownames(vi_df))
  colnames(blank_df) = colnames(vi_df)
  
  # Populate with dataframes, creates a list of dataframes, all pertaining to the analyses and submetrics, each identically populated with blank_df
  for(an_type in analysis_list[[data_set]]) {  #?Sets each value in "analysis_list" as variable "an_type"?
    output_list[[an_type]] = list()  #?Adds a list to dataframe corresponding to an_type
    for(met in sub_metrics[[an_type]]) {#Ditto to above, but for the submetrics
      output_list[[an_type]][[met]] = list()
      for(tl_length in tl_list[[data_set]][[an_type]]) {
        output_list[[an_type]][[met]][[as.character(tl_length)]] = blank_df #populates the current list with blank_df
      }
    }
  }
  
  # Loop over rows and run analysis on each
  for(an_type in analysis_list[[data_set]]) {
    for(r in 1:dim(vi_df)[1]) {
      vi_series = vi_df[r,]
      # Only go forward if there are non-nas in the time series
      if(sum(!is.na(vi_series))<=2) {
        print(vi_series)
      } else{
        temp_out = func_list[[an_type]](vi_series,tl_list[[data_set]][[an_type]])
        for(met in sub_metrics[[an_type]]) {
          for(tl_length in tl_list[[data_set]][[an_type]]) {
            if(!is.null(temp_out[[met]][[as.character(tl_length)]])){
              output_list[[an_type]][[met]][[as.character(tl_length)]][r,] = temp_out[[met]][[as.character(tl_length)]]
            }
          }
        }
      }
      print(paste0(an_type,r))
    }
    print(paste0(data_set, " ", an_type," analysis done"))
    timestamp()
  }
  return(output_list)
}

#### Write Out to CSV Function ####
write_vi_dfs = function(df_list,outpath_lead) {
  
  # Need to do a nested sapply to write everything out to csv
  sapply(names(df_list), 
         function(x) sapply(names(df_list[[x]]),
                            function(y) sapply(names(df_list[[x]][[y]]), 
                                               function(z) write.csv(df_list[[x]][[y]][[z]],file=paste0(outpath_lead,"/",x,"/",y,"_tlength",z,".csv"))
                            )))
  print("Done!")
}

