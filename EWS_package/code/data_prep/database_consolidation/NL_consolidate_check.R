library(readxl)
library(plyr)
library(dplyr)

LUT = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SP_LUT.csv",stringsAsFactors = F)
NL_Im_perm <- read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NL/tblImmTreesPerm.txt",stringsAsFactors = F)
NL_Im_re <- read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NL/tblImmatureTreesRemeas.txt",stringsAsFactors = F)
NL_M_perm <- read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NL/tblMatureTreesPerm.txt",stringsAsFactors = F)
NL_M_re <- read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NL/tblMatureTreesRemeas.txt",stringsAsFactors = F)
NL_meas = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NL/tblRemCrewDate.txt",stringsAsFactors = F)


NL_Im_re = NL_Im_re[,c(1:4,7,5,6,8,10,13)]
NL_M_re = NL_M_re[,c(1:7,9,11,12)]
NL_tree = rbind(NL_Im_re,NL_M_re)
#There are duplicates in plot numbers between districts. Concatenating district and plot number for unique identifier
NL_tree$PlotNumber = paste0(NL_tree$District,NL_tree$PlotNumber)
NL_meas$PlotNumber = paste0(NL_meas$District,NL_meas$PlotNumber)

NL_M_perm = NL_M_perm[,-5]
NL_perm = rbind(NL_M_perm,NL_Im_perm)
NL_perm$PlotNumber = paste0(NL_perm$District,NL_perm$PlotNumber)

species = vector(length = dim(NL_tree)[1])
for (i in 1:length(species)){
  species[i] = NL_perm$Species[NL_perm$PlotNumber==NL_tree$PlotNumber[i]&NL_perm$TreeNumber==NL_tree$TreeNumber[i]]
}
NL_tree = data.frame(NL_tree,species)

plot_vect = unique(NL_tree$PlotNumber)
 
#No species errors



damage = matrix(nrow = length(plot_vect),ncol = 6*9)
rownames(damage) = plot_vect
dams = c("Disease/pathogen","Insect","Fire","Drought","Human","Other")
colnames(damage) = c(paste0(dams,"_t1"),paste0(dams,"_t2"),paste0(dams,"_t3"),paste0(dams,"_t4"),paste0(dams,"_t5"),paste0(dams,"_t6"),paste0(dams,"_t7"),paste0(dams,"_t8"),paste0(dams,"_t9"))
dam_codes = list(
  "Disease/pathogen" = c(),
  "Insect" = c("I",1,2,3,4,5,6),
  "Fire" = "Y",
  "Drought" = c(),
  "Human" = c("M"),
  "Other" = c("B")
)

surveys = data.frame()
#Create consolidated csvs
for (i in plot_vect){
  print(i)
  temp = NL_tree[NL_tree$PlotNumber==i,]
  temp = arrange(temp,Remeasurement)
  samp_vect = unique(temp$Remeasurement)
  for (j in 1:length(samp_vect)){
    surveys[as.character(i),paste0("t_",j)] = NL_meas$Year[NL_meas$Remeasurement==samp_vect[j]&NL_meas$PlotNumber==i]
  }
  print("surv")
  tree_range = unique(temp$TreeNumber)
  tree_range = sort(tree_range)
  df = matrix(nrow = length(tree_range),ncol = length(samp_vect)*2)
  sp_df = matrix(nrow = length(tree_range),ncol = length(samp_vect))
  rownames(df) = tree_range
  colnames(df) = c(paste0("dbh_t",1:length(samp_vect)),paste0("status_t",1:length(samp_vect)))
  colnames(sp_df) = paste0("sp_t",1:length(samp_vect))
  rownames(sp_df) = tree_range
  for (n in 1:length(samp_vect)){
    print(n)
    temp_samp = temp[temp$Remeasurement==samp_vect[n],]
    if(!identical(temp_samp$TreeNumber,tree_range)){
      add = tree_range[!tree_range %in% temp_samp$TreeNumber]
      add_mat = data.frame(matrix(ncol = dim(temp_samp)[2],nrow = length(add)))
      colnames(add_mat) = colnames(temp_samp)
      add_mat$TreeNumber = add
      temp_samp = rbind(temp_samp,add_mat)
      temp_samp = arrange(temp_samp,TreeNumber)
    }
    for (j in 1:dim(df)[1]){
      print(j)
      sp_df[j,paste0("sp_t",n)] = as.numeric(temp_samp$species[temp_samp$TreeNumber==rownames(df)[j]])
      df[j,paste0("dbh_t",n)] = temp_samp$DBH[temp_samp$TreeNumber==rownames(df)[j]]
      if(!is.na(temp_samp$TreeStatus[temp_samp$TreeNumber==rownames(df)[j]])&any(temp_samp$TreeStatus[temp_samp$TreeNumber==rownames(df)[j]]==c(0,3,4,6,7,8))){
        df[j,paste0("status_t",n)] = "L"
      }
    }
    
    
    flags = as.matrix(temp_samp[,c("CauseOfDeath","Aphid")])
    for (s in 1:length(dam_codes)){
      flags[flags %in% dam_codes[[s]]] = names(dam_codes)[s]
    }
    for (r in 1:dim(flags)[1]){
      flags[r,duplicated(flags[r,],incomparables = NA)] = NA
    }
    for(j in 1:length(dam_codes)){
      damage[as.character(i),paste0(names(dam_codes)[j],"_t",n)] = sum(flags==names(dam_codes)[j],na.rm=T)/dim(flags)[1]
    }
  }
  sp_df2 = sp_df
  for (j in 1:dim(LUT)[1]){
    sp_df2[sp_df==as.numeric(LUT[j,"NL"])] = LUT[j,1]
  }
  plot_size = 0.0404
  df = data.frame(sp_df2,df,plot_size)
  write.csv(df,paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NL/checks/",i,"_check.csv"))
}

write.csv(damage,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NL/NL_flags_percent.csv")
write.csv(surveys,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NL/NL_surveys.csv")