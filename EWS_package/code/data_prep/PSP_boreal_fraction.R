
library(dplyr)


surveys = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/psp/survey_dates.csv",row.names = "X")


provinces = rbind(c(1,"BC",9),
                  c(2,"AB",9.1),
                  c(3,"SK",7.1),
                  c(4,"MB",7.1),
                  c(5,"ON",9),
                  c(6,"QC",9),
                  c(7,"NL",8),
                  c(8,"NB",8),
                  c(9,"NS",9.1),
                  c(11,"YT",7.5),
                  c(12,"NWT",6.7),
                  c(13,"CAFI",7.5)
)
#Which species are boreal?
boreal = c(1,4,5,8,10,21,42,45,48,64,69,114)

boreal_frac_biomass = data.frame()
boreal_frac_BA = data.frame()
boreal_frac_N = data.frame()



for (i in 1:dim(provinces)[1]){
  files =list.files(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/",provinces[i,2],"/PSP/biomass/"))
  for(n in files){
    biomass = read.csv(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/",provinces[i,2],"/PSP/biomass/",n),row.names = "X",stringsAsFactors = F)
    if(sum(!is.na(biomass))>0){
      basal = read.csv(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/",provinces[i,2],"/PSP/basalarea/",n),row.names = "X",stringsAsFactors = F)
      measurements = dim(basal)[2]
      sp = as.matrix(read.csv(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/",provinces[i,2],"/checks/",gsub(".csv","",n),"_check.csv"),row.names = "X",stringsAsFactors = F)[,c(1:measurements)])
      for (k in 1:measurements){
        boreal_biomass = biomass[sp[,k] %in% boreal,paste0("total_t",k)]
        #nonboreal_biomass = biomass[sp[,paste0("sp_t",k)] %in% bor_non[["nonboreal"]],paste0("t",k)]
        boreal_basal = basal[sp[,k] %in% boreal,paste0("t",k)]
        #nonboreal_basal = basal[sp[,paste0("sp_t",k)] %in% bor_non[["nonboreal"]],paste0("t",k)]
        #nonboreal_N = basal[sp[,paste0("sp_t",k)] %in% bor_non[["nonboreal"]],paste0("t",k)]
        boreal_frac_biomass[paste0(provinces[i,1],"_",gsub(".csv","",n)),paste0("t",k)] = sum(boreal_biomass,na.rm =T)/sum(biomass[,paste0("total_t",k)],na.rm=T)
        boreal_frac_BA[paste0(provinces[i,1],"_",gsub(".csv","",n)),paste0("t",k)] = sum(boreal_basal,na.rm =T)/sum(basal[,paste0("t",k)],na.rm=T)
        boreal_frac_N[paste0(provinces[i,1],"_",gsub(".csv","",n)),paste0("t",k)] = sum(!is.na(boreal_biomass))/sum(!is.na(biomass[,paste0("total_t",k)]))
      }
    }
  }
  print(paste0(provinces[i,2]," done"))
}

write.csv(boreal_frac_biomass,"/att/nobackup/scooperd/scooperdock/EWS/data/psp/PSP_boreal_frac_biomass.csv")
write.csv(boreal_frac_BA,"/att/nobackup/scooperd/scooperdock/EWS/data/psp/PSP_boreal_frac_BA.csv")
write.csv(boreal_frac_N,"/att/nobackup/scooperd/scooperdock/EWS/data/psp/PSP_boreal_frac_N.csv")
