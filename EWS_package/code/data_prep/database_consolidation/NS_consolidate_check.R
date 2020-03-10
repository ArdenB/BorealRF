library(readxl)
library(plyr)
library(dplyr)

LUT = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SP_LUT.csv",stringsAsFactors = F)
NS_tree <- read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NS/NSTreeQuery.txt",stringsAsFactors = F)
NS_damage = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NS/NSDamage.txt")



plot_vect = unique(NS_tree$PlotNumber)
 
#No species errors


cut_percent = matrix(ncol = 11,nrow = length(plot_vect))
rownames(cut_percent) = plot_vect
colnames(cut_percent) = paste0("t",1:11)


damage = matrix(nrow = length(plot_vect),ncol = 6*11)
rownames(damage) = plot_vect
dams = c("Disease/pathogen","Insect","Fire","Drought","Human","Other")
colnames(damage) = c(paste0(dams,"_t1"),paste0(dams,"_t2"),paste0(dams,"_t3"),paste0(dams,"_t4"),paste0(dams,"_t5"),paste0(dams,"_t6"),paste0(dams,"_t7"),paste0(dams,"_t8"),paste0(dams,"_t9"),paste0(dams,"_t10"),paste0(dams,"_t11"))
dam_codes = list(
  "Disease/pathogen" = c(3,11,41),
  "Insect" = c(4:6,10,20:26,28:30,37:40,46:48,51),
  "Fire" = c(),
  "Drought" = c(),
  "Human" = c(),
  "Other" = c(7,27,31,35,50)
)
cod_codes = list(
  "Disease/pathogen" = 5,
  "Insect" = 4,
  "Fire" = 7,
  "Drought" = c(),
  "Human" = c(),
  "Other" = c(2,3,10)
)
surveys = data.frame()
#Create consolidated csvs
for (i in plot_vect){
  temp = NS_tree[NS_tree$PlotNumber==i,]
  temp = arrange(temp,FieldSeason)
  samp_vect = unique(temp$FieldSeason)
  for (j in 1:length(samp_vect)){
    surveys[as.character(i),paste0("t_",j)] = samp_vect[j]
  }
  tree_range = unique(temp$TreeNumber)
  tree_range = sort(tree_range)
  df = matrix(nrow = length(tree_range),ncol = length(samp_vect)*2)
  sp_df = matrix(nrow = length(tree_range),ncol = length(samp_vect))
  rownames(df) = tree_range
  colnames(df) = c(paste0("dbh_t",1:length(samp_vect)),paste0("status_t",1:length(samp_vect)))
  colnames(sp_df) = paste0("sp_t",1:length(samp_vect))
  rownames(sp_df) = tree_range
  for (n in 1:length(samp_vect)){
    temp_samp = temp[temp$FieldSeason==samp_vect[n],]
    if(!identical(temp_samp$TreeNumber,tree_range)){
      add = tree_range[!tree_range %in% temp_samp$TreeNumber]
      add_mat = data.frame(matrix(ncol = dim(temp_samp)[2],nrow = length(add)))
      colnames(add_mat) = colnames(temp_samp)
      add_mat$TreeNumber = add
      temp_samp = rbind(temp_samp,add_mat)
      temp_samp = arrange(temp_samp,TreeNumber)
    }
    for (j in 1:dim(df)[1]){
      sp_df[j,paste0("sp_t",n)] = as.numeric(temp_samp$SpeciesId[temp_samp$TreeNumber==rownames(df)[j]])
      df[j,paste0("dbh_t",n)] = temp_samp$Dbh[temp_samp$TreeNumber==rownames(df)[j]]
      if(!is.na(temp_samp$TreeStatusId[temp_samp$TreeNumber==rownames(df)[j]])&(temp_samp$TreeStatusId[temp_samp$TreeNumber==rownames(df)[j]]==1|temp_samp$TreeStatusId[temp_samp$TreeNumber==rownames(df)[j]]==3)){
        df[j,paste0("status_t",n)] = "L"
      }
    }
    cut_percent[as.character(i),paste0("t",n)] = sum(temp_samp$TreeStatusId==4,na.rm=T)/sum(!is.na(temp_samp$TreeStatusId))
    cod = temp_samp$CauseOfDeathId
    flags = data.frame()
    for (j in 1:dim(temp_samp)[1]){
      flags[as.character(j),"1"] = NA
      if(!is.na(temp_samp$Id[j])&any(NS_damage$TreeMetricId==temp_samp$Id[j])){
        for (s in 1:length(NS_damage$DamageTypeId[NS_damage$TreeMetricId==temp_samp$Id[j]])){
          flags[as.character(j),as.character(s)] = NS_damage$DamageTypeId[NS_damage$TreeMetricId==temp_samp$Id[j]][s]
        }
      }
    }
    for (s in 1:length(dam_codes)){
      flags[flags %in% dam_codes[[s]]] = names(dam_codes)[s]
      cod[cod %in% cod_codes[[s]]] = names(cod_codes)[s]
    }
    flags = cbind(cod,flags)
    flags = as.matrix(flags)
    for (r in 1:dim(flags)[1]){
      flags[r,duplicated(flags[r,],incomparables = NA)] = NA
    }
    for(j in 1:length(dam_codes)){
      damage[as.character(i),paste0(names(dam_codes)[j],"_t",n)] = sum(flags==names(dam_codes)[j],na.rm=T)/sum(!is.na(temp_samp$TreeStatusId))
    }
  }
  sp_df2 = sp_df
  for (j in 1:dim(LUT)[1]){
    sp_df2[sp_df==as.numeric(LUT[j,"NS"])] = LUT[j,1]
  }
  plot_size = 0.0404
  df = data.frame(sp_df2,df,plot_size)
  write.csv(df,paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NS/checks/",i,"_check.csv"))
}
write.csv(cut_percent,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NS/NS_cut_percent.csv")
write.csv(damage,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NS/NS_flags_percent.csv")
write.csv(surveys,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NS/NS_surveys.csv")
