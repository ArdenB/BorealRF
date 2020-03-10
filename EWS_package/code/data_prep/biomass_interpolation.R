###A script to interpolate biomass values, using a couple different methods.

leadpath = 'I:/ppl/scooperd/'

LUT = read.csv(paste0(leadpath,"scooperdock/EWS/data/raw_psp/SP_LUT.csv"),stringsAsFactors = F)
surveys = read.csv(paste0(leadpath,"scooperdock/EWS/data/psp/survey_dates.csv"),row.names = "X",stringsAsFactors = F)
all_mort = read.csv(paste0(leadpath,"scooperdock/EWS/data/psp/PSP_total_changes.csv"),row.names = "X")


years = min(surveys,na.rm = T):max(surveys,na.rm = T)

interp_biomass = matrix(nrow = dim(all_mort)[1],ncol = length(years))
rownames(interp_biomass) = rownames(all_mort)
colnames(interp_biomass) = years

###This first method simply interpolates linearly between all observed values, I chose to go a different route for
###the final product, but this is still a useful database to have for other uses.

for (n in 1:dim(interp_biomass)[1]){
  print(n)
  #find survey years for this site
  yrs = surveys[rownames(interp_biomass)[n],!is.na(surveys[rownames(interp_biomass)[n],])]
  #check to make sure there's nothing wonky, ie observations after 2017
  if(any(yrs>2017)){
    print(rownames(interp_biomass)[n])
  }
  #If there is more than one observation, go ahead with interpolation
  if (length(yrs)>1){
    #grab biomass values for each year from database
    for (i in 1:length(yrs)){
      interp_biomass[n,as.character(yrs[i])] = all_mort[n,paste0("live_mass_t",i)]
    }
    #interpolate
    nums = which(!is.na(interp_biomass[n,]))
    interpolated = approx(years,as.numeric(interp_biomass[n,]),n=nums[length(nums)]-nums[1]+1)
    interp_biomass[n,as.character(interpolated$x)] = interpolated$y
  }
}

 write.csv(interp_biomass,"/att/nobackup/scooperd/scooperdock/EWS/data/psp/biomass_interpolated.csv")
 
 
 #This method is more complex. Basically I decided there was an issue with trying to model using interpolated biomass values
 #as both a before and after measurement, but I wasn't willing to only use sites that had exactly 10 years between measurements,
 #so I chose to only use biomass change measurements that had an actual observation at year 10. This way I was interpolating the
 #initial biomass value at times, but never the final one. I did the same for stem density. If the survey interval is >10 years,
 #instead of using a value that is partially dependent on the year 10 measurement value (because the biomass 10 years before 
 #would have used that value as an interpolation endpoint), I used the previous two points to estimate the starting value. 
 
 biomass = matrix(nrow = dim(all_mort)[1],ncol = length(years))
 rownames(biomass) = rownames(all_mort)
 colnames(biomass) = years
 #also create an interval database for later use
 intervals = biomass
 stem_dens = biomass
 
 for (n in 1:dim(biomass)[1]){
   print(n)
   #This first section is the same as above
   yrs = surveys[rownames(biomass)[n],!is.na(surveys[rownames(biomass)[n],])]
   if(any(yrs>2017)){
     print(rownames(biomass)[n])
   }
   if (length(yrs)>1){
     for (i in 1:length(yrs)){
       biomass[n,as.character(yrs[i])] = all_mort[n,paste0("live_mass_t",i)]
       stem_dens[n,as.character(yrs[i])] = all_mort[n,paste0("live_N_t",i)]
     }
     nums = which(!is.na(biomass[n,]))
     for(j in 1:(length(nums)-1)){
       intervals[n,(nums[j]+1):nums[j+1]] = nums[j+1] - nums[j]
     }
     ints = intervals[n,]
     if(all(ints<=10,na.rm=T)){#If all intervals are less than or equal to 10, just interpolate
       interpolated = approx(years,as.numeric(biomass[n,]),n=nums[length(nums)]-nums[1]+1)
       biomass[n,as.character(interpolated$x)] = interpolated$y
       interpolated = approx(years,as.numeric(stem_dens[n,]),n=nums[length(nums)]-nums[1]+1)
       stem_dens[n,as.character(interpolated$x)] = interpolated$y
     }else{#Otherwise do something different for each interval
       if(length(nums)>2){
         for(j in 2:length(nums)){
           if(ints[nums[j]]<=10){#If interval is less than or equal to 10 years, interpolate
             interpolated = approx(as.numeric(names(nums[j-1])):as.numeric(names(nums[j])),as.numeric(biomass[n,nums[j-1]:nums[j]]),n=ints[nums[j]]+1)
             biomass[n,as.character(interpolated$x)] = interpolated$y
             interpolated = approx(as.numeric(names(nums[j-1])):as.numeric(names(nums[j])),as.numeric(stem_dens[n,nums[j-1]:nums[j]]),n=ints[nums[j]]+1)
             stem_dens[n,as.character(interpolated$x)] = interpolated$y
           }else if(ints[nums[j]]<=20){#If interval is greater than 10, but less than or equal to 20, use trend between previous 2 measurements to interpolat
             if(j>2){ #Can't do this if it's one of the first 2 measurements
               eq = lm(biomass[n,c(nums[j-2],nums[j-1])]~c(nums[j-2],nums[j-1]))
               biomass[n,nums[j]-10] = coefficients(eq)[1] + (nums[j]-10)*coefficients(eq)[2]
               eq = lm(stem_dens[n,c(nums[j-2],nums[j-1])]~c(nums[j-2],nums[j-1]))
               stem_dens[n,nums[j]-10] = coefficients(eq)[1] + (nums[j]-10)*coefficients(eq)[2]
             }
           }
         }
       }
     }
     
   }
 }
 
 #Write it out, you might note that all values were interpolated, not just those that were 10 years before a measurement.
 #That step is actually added in the create_vi_df script.
 write.csv(biomass,paste0(leadpath,"scooperdock/EWS/data/psp/biomass_interpolated_w_over_10years.csv"))
 write.csv(stem_dens,paste0(leadpath,"scooperdock/EWS/data/psp/stem_dens_interpolated_w_over_10years.csv"))
 write.csv(intervals,paste0(leadpath,"scooperdock/EWS/data/psp/surv_interval_filled.csv"))
 
 ##For just a matrix of years each site was measured
 
 surv_year = matrix(nrow = dim(surveys)[1],ncol = length(years))
 rownames(surv_year) = rownames(surv_year)
 colnames(surv_year) = years
 
 for (n in 1:dim(surveys)[1]){
   yrs = surveys[n,!is.na(surveys[n,])]
   if(length(yrs)>0){
     for (i in 1:length(yrs)){
       
       surv_year[n,as.character(yrs[i])] = 1
     }
   }
   
   
 }
 
 write.csv(surv_year,"/att/nobackup/scooperd/scooperdock/EWS/data/psp/surv_date_matrix.csv")
 