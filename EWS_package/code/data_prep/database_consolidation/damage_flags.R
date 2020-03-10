library(plyr)

surveys = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/psp/survey_dates.csv",row.names = "X",stringsAsFactors = F)

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

damage = data.frame()
DID = data.frame()
thresh = 0
for (i in 1:dim(provinces)[1]){
  print(provinces[i,2])
  if(file.exists(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/",provinces[i,2],"/",provinces[i,2],"_cut_percent.csv"))){
    cut = read.csv(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/",provinces[i,2],"/",provinces[i,2],"_cut_percent.csv"),row.names = "X")
    rownames(cut) = paste0(provinces[i,1],"_",rownames(cut))
    for (j in 1:dim(cut)[1]){
      print(j)
      for (n in 1:length(surveys[rownames(cut)[j],!is.na(surveys[rownames(cut)[j],])])){
        #if(cut[j,n]>thresh){
          damage[rownames(cut)[j],as.character(surveys[rownames(cut)[j],n])] = cut[j,n]
        #}
      }
    }
  }
  if(file.exists(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/",provinces[i,2],"/",provinces[i,2],"_flags_percent.csv"))){
    flags = read.csv(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/",provinces[i,2],"/",provinces[i,2],"_flags_percent.csv"),row.names = "X")
    rownames(flags) = paste0(provinces[i,1],"_",rownames(flags))
    for (j in 1:dim(flags)[1]){
      print(j)
      #if(any(!is.na(surveys[rownames(flags)[j],]))){
        for (n in 1:length(surveys[rownames(flags)[j],!is.na(surveys[rownames(flags)[j],])])){
          if(!is.na(empty(damage[rownames(flags)[j],as.character(surveys[rownames(flags)[j],n])]))){
          #if(any(flags[j,paste0(c("Fire_t","Human_t","Other_t"),n)]>thresh)){
            damage[rownames(flags)[j],as.character(surveys[rownames(flags)[j],n])] = max(flags[j,paste0(c("Fire_t","Human_t","Other_t"),n)],na.rm=T)
          }else{
            damage[rownames(flags)[j],as.character(surveys[rownames(flags)[j],n])] = max(cbind(flags[j,paste0(c("Fire_t","Human_t","Other_t"),n)],damage[rownames(flags)[j],as.character(surveys[rownames(flags)[j],n])]),na.rm=T)
          }
          #if(any(flags[j,paste0(c("Disease.pathogen_t","Insect_t","Drought_t"),n)]>thresh)){
            DID[rownames(flags)[j],as.character(surveys[rownames(flags)[j],n])] = max(flags[j,paste0(c("Disease.pathogen_t","Insect_t","Drought_t"),n)],na.rm=T)
          #}
        #}
      }
    }
  }
}
years = min(colnames(damage)):max(colnames(damage))
add_col = years[!years %in% colnames(damage)]
damage[,as.character(add_col)] = NA
damage = damage[,as.character(years)]

years = min(colnames(DID)):max(colnames(DID))
add_col = years[!years %in% colnames(DID)]
DID[,as.character(add_col)] = NA
DID = DID[,as.character(years)]

write.csv(damage,"/att/nobackup/scooperd/scooperdock/EWS/data/psp/damage_flags.csv")
write.csv(DID,"/att/nobackup/scooperd/scooperdock/EWS/data/psp/DID_flags.csv")