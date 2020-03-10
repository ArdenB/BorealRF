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

cut_percent = data.frame()
dist_percent = data.frame()

for (i in 1:dim(provinces)[1]){
  if(file.exists(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/",provinces[i,2],"/",provinces[i,2],"_cut_percent.csv"))){
    cut = read.csv(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/",provinces[i,2],"/",provinces[i,2],"_cut_percent.csv"),row.names = "X")
    rownames(cut) = paste0(provinces[i,1],"_",rownames(cut))
    cut_percent[rownames(cut),] = NA
    add_cols = colnames(cut)[!colnames(cut) %in% colnames(cut_percent)]
    if(length(add_cols)>0){
      cut_percent[,add_cols] = NA
    }
    cut_percent[rownames(cut),colnames(cut)] = cut
  }
  if(file.exists(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/",provinces[i,2],"/",provinces[i,2],"_flags_percent.csv"))){
    dist = read.csv(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/",provinces[i,2],"/",provinces[i,2],"_flags_percent.csv"),row.names = "X")
    rownames(dist) = paste0(provinces[i,1],"_",rownames(dist))
    dist_percent[rownames(dist),] = NA
    add_cols = colnames(dist)[!colnames(dist) %in% colnames(dist_percent)]
    if(length(add_cols)>0){
      dist_percent[,add_cols] = NA
    }
    dist_percent[rownames(dist),colnames(dist)] = dist
  }
}


write.csv(cut_percent,"/att/nobackup/scooperd/scooperdock/EWS/data/psp/PSP_cut_percent.csv")
write.csv(dist_percent,"/att/nobackup/scooperd/scooperdock/EWS/data/psp/PSP_disturbance_percent.csv")
