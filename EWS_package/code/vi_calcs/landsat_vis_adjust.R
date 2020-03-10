library(dplyr)

#Read in VIs
landsat = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/VIs/raw/lsat_nam_psp_data/lsat_vi_at_nam_psp_20190927.csv",stringsAsFactors = F)


years = 1984:2017

#Get names
vi_names = colnames(landsat)[3:11]
#Create an empty array
vis = array(NA,dim=c(length(unique(landsat$site)),length(years),length(vi_names)),dimnames = list(unique(landsat$site),years,vi_names))


years = landsat$year
sites = landsat$site

#Get min and max satvi values for tvfc calcs
min_satvi = min(landsat$satvi.ja.med,na.rm=T)
max_satvi = max(landsat$satvi.ja.med,na.rm=T)

#Create a list of all the valid years for each site
year_list = unstack(landsat,years~sites)
for (vit in vi_names){
  #Create list of valid vi values for each site
  this_vi = unstack(landsat,landsat[,vit]~sites)
  for(i in 1:length(this_vi)){
    #line up vi values with correct years for each
    vis[names(this_vi)[i],as.character(year_list[[i]]),vit] = this_vi[[i]]
  }
}

#Check for errors
for(vi in vi_names){
  print(paste0(vi,": ",sum(is.na(vis[,,vi]))))
  print(paste0(vi,": ",sum(vis[,,vi]>(1))))
}

#Convert NAs
vis[is.na(vis)] = -32768

#Output
for (vi in vi_names){
  out = data.frame(rownames(vis[,,vi]),vis[,,vi])
  colnames(out) = c("Plot_ID",1984:2017)
  out = arrange(out, Plot_ID)


  write.csv(out,paste0("/att/nobackup/scooperd/scooperdock/EWS/data/VIs/LANDSAT/full_LANDSAT_",strsplit(vi,".",fixed=T)[[1]][1],"_median.csv"))
  #Calculate tvfc
  if(vi = 'satvi.ja.med'){
    out[out==-32768] = NA
    out[,as.character(1984:2017)] = (out[,as.character(1984:2017)] - min_satvi)/(max_satvi - min_satvi)*100
    out[,as.character(1984:2017)][is.na(out[,as.character(1984:2017)])] = -32768
    write.csv(out,paste0("/att/nobackup/scooperd/scooperdock/EWS/data/VIs/LANDSAT/full_LANDSAT_tvfc_median.csv"))
    
  }
}

