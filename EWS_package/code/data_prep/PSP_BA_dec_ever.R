
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

dec_con = list(
  "deciduous" = c(1,2,3,4,5,6,7,8,17,20,21,22,23,24,26,27,28,29,32,34,37,42,43,55,56,57,58,59,60,61,62,70,71,72,73,74,75,76,77,78,79,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,107,108,109,110,111,112,113,114,115,117,118,119,120,121,122,123,124,125,126,127,129,131,132,133,134,135,136,137,140,141,146,147,148,149,150),
  "evergreen" = c(9,10,11,12,13,14,15,16,18,19,25,30,31,33,35,36,38,39,40,41,44,45,46,47,48,49,50,51,52,53,54,63,64,65,66,67,68,69,81,82,105,106,116,117,128,130,138,139,142,143,144,145,151)
)


BA_dec_ever = data.frame()
recruit = data.frame()
recruit_N = data.frame()
mort = data.frame()
mort_N = data.frame()
living_wood = data.frame()
living_wood_N = data.frame()
sp_comp_mass = data.frame()
sp_mort_mass = data.frame()
sp_recruit_mass = data.frame()
sp_growth = data.frame()
sp_live_mass = data.frame()
sp_comp_N = data.frame()
sp_mort_N = data.frame()
sp_recruit_N = data.frame()
sp_ID = data.frame()
sp_live_N = data.frame()


for (i in 1:dim(provinces)[1]){
  files = list.files(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/",provinces[i,2],"/checks/"))
  for(n in files){
    df = read.csv(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/",provinces[i,2],"/checks/",n),row.names = "X",stringsAsFactors = F)
    measurements = (dim(df)[2]-1)/3
    basal = matrix(nrow = dim(df)[1],ncol = measurements)
    rownames(basal) = rownames(df)
    colnames(basal) = paste0("t",1:measurements)
    print(n)
    for (j in 1:dim(basal)[1]){
      for(k in 1:measurements){
        if(!is.na(df[j,paste0("status_t",k)]) & !is.na(df[j,paste0("dbh_t",k)]) & as.numeric(df[j,paste0("dbh_t",k)])>=as.numeric(provinces[i,3])){
          if (is.na(df[j,paste0("sp_t",k)])){
            df[j,paste0("sp_t",k)] = 80
          }
          basal[j,k]=(as.numeric(df[j,paste0("dbh_t",k)])/2)^2*pi/10000 #Basal area in m2
        }
      }
    }
    if(sum(!is.na(basal))>0){
      write.csv(basal,paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/",provinces[i,2],"/PSP/basalarea/",gsub("_check.csv","",n),".csv"))
      #Calculate decidous and evergreen BA for each plot at each time step (in m2/ha)
      for (k in 1:measurements){
        decid = basal[df[,paste0("sp_t",k)] %in% dec_con[["deciduous"]],paste0("t",k)]
        ever = basal[df[,paste0("sp_t",k)] %in% dec_con[["evergreen"]],paste0("t",k)]
        BA_dec_ever[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("deciduous_BA_t",k)] = sum(decid,na.rm =T)/unique(df$plot_size)
        BA_dec_ever[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("evergreen_BA_t",k)] = sum(ever,na.rm =T)/unique(df$plot_size)
      }
    }
  }
  print(paste0(provinces[i,2]," done"))
}


write.csv(BA_dec_ever,"/att/nobackup/scooperd/scooperdock/EWS/data/psp/PSP_BA_dec_ever.csv")



