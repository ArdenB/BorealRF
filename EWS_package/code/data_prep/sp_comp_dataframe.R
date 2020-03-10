library(dplyr)


LUT = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SP_LUT.csv",stringsAsFactors = F)
surveys = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/psp/survey_dates.csv",row.names = "X",stringsAsFactors = F)
sp_ID = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/psp/PSP_sp_IDs.csv",row.names = "X",stringsAsFactors = F)
sp_comp_mass = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/psp/PSP_sp_comp_mass.csv",row.names = "X",stringsAsFactors = F)


years = min(surveys,na.rm = T):max(surveys,na.rm = T)
IDs = LUT$ID
sp_ID_unlist = unlist(sp_ID)


###For composition
col_sp_comp_list = vector()
for(i in 1:ncol(sp_comp_mass)){
  col_sp_comp_list[[i]] = strsplit(colnames(sp_comp_mass)[i],"_")[[1]][1]
}

for (sp in IDs){
  print(sp)
  this_sp = which(sp_ID_unlist==sp)
  indexes = names(sp_ID_unlist[this_sp])
  if (length(indexes)>0){
    for(i in 1:length(indexes)){
      indexes[i] = paste0(strsplit(names(sp_ID_unlist[this_sp])[i],'')[[1]][1:4],collapse = '')
      names(indexes)[i] = paste0(strsplit(names(sp_ID_unlist[this_sp])[i],'')[[1]][5:length(strsplit(names(sp_ID_unlist[this_sp])[i],'')[[1]])],collapse = '')
    }
    this_sp_sites = rownames(sp_ID[as.numeric(names(indexes)),])
    this_sp_comp_mass = sp_comp_mass[this_sp_sites,]
    
    
    this_sp_comp_mass_df = data.frame()
    
    indexes = indexes[!is.na(this_sp_comp_mass[,1])]
    #indexes_live = indexes[!is.na(this_sp_live_N[,1])]
    if (length(indexes)>0){ 
      this_sp_comp_mass = this_sp_comp_mass[!is.na(this_sp_comp_mass[,1]),]
      
      sites = rownames(this_sp_comp_mass)
      for(i in 1:length(indexes)){
        this_site_comp = this_sp_comp_mass[i,col_sp_comp_list %in% indexes[i]]
        for(n in 1:length(this_site_comp)){
          this_sp_comp_mass_df[sites[i],strsplit(colnames(this_site_comp)[n],"_")[[1]][4]] = this_site_comp[n]
    
        }
        
      }
      comp_biomass = matrix(nrow = dim(this_sp_comp_mass_df)[1],ncol = length(years))
      rownames(comp_biomass) = rownames(this_sp_comp_mass_df)
      colnames(comp_biomass) = years
      interp = comp_biomass
      for (n in 1:dim(comp_biomass)[1]){
        yrs = surveys[rownames(comp_biomass)[n],!is.na(surveys[rownames(comp_biomass)[n],])]
        if(any(yrs>2017)){
          print(rownames(comp_biomass)[n])
        }
        if (length(yrs)>1){
          for (i in 1:(length(yrs))){
            comp_biomass[n,as.character(yrs[i])] = this_sp_comp_mass_df[n,i]
            
          }
          nums = which(!is.na(comp_biomass[n,]))
          if(length(nums)>1){
            interpolated = approx(years,as.numeric(comp_biomass[n,]),n=nums[length(nums)]-nums[1]+1)
            interp[n,as.character(interpolated$x)] = interpolated$y
          }else{
            interp[n,] = comp_biomass[n,]
          }
        }
      }
      write.csv(comp_biomass,paste0("/att/nobackup/scooperd/scooperdock/EWS/data/psp/comp_biomass_",sp,".csv"))
      write.csv(interp,paste0("/att/nobackup/scooperd/scooperdock/EWS/data/psp/comp_interp_",sp,".csv"))
    }
  }
}


