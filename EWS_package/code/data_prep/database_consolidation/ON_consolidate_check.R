
library(dplyr)

LUT = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SP_LUT.csv",stringsAsFactors = F)
ON_tree <- read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/ON/ON_rawdatatreemsrwithht.csv",stringsAsFactors = F)
ON_surveys = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/ON/ON_surveys.csv")
ON_dataset = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/ON/ON_dataset.csv",stringsAsFactors = F)
ON_damage = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/ON/ON_damage.csv",stringsAsFactors = F)

#Filter out sites that are not PSPs or PGPs
plot_type = vector(length = dim(ON_tree)[1])
for (i in 1:length(plot_type)){
  plot_type[i] = ON_dataset[ON_dataset[,1]==ON_tree$DatasetCode[i],8]
}
ON_tree = data.frame(ON_tree,plot_type)
ON_tree = ON_tree[(ON_tree$plot_type=="PGP"|ON_tree$plot_type=="PSP"),]
#Filter out sites that are not in the surveys csv (I think this is just removing sites that have only been measured once)
#ON_tree = ON_tree[ON_tree$PlotName %in% ON_surveys$Plot_ID,]

#Correcting some errors
ON_tree["1145633",15] = 3

#Removing a plot that has every tree duplicated
ON_tree = ON_tree[ON_tree$PlotName!="FCTEM2002083PGP",]
# #Removing a plot that has trees numbered as -Inf
# ON_tree = ON_tree[ON_tree$PlotName!="GER1993003PSP",]

plotsize = list(
  "PSP" = 0.12,
  "PGP" = 0.04
)

plot_vect = unique(ON_tree$PlotName)


#Apparently there are no species errors...

# damage_code = matrix(nrow = dim(ON_tree)[1],ncol = 2)
# rownames(damage_code) = ON_tree$TreeMsrKey
damage_code = vector(mode = "numeric", length = dim(ON_tree)[1])
damage_code[damage_code==0] = NA
for (i in unique(ON_damage$TreeMsrKey)){
  damage_code[ON_tree$TreeMsrKey==i] = paste0(ON_damage$DefmCauseCode[ON_damage$TreeMsrKey==i],collapse = ",")
}
print("1")
# damage_code = damage_code[rownames(damage_code) %in% ON_tree$TreeMsrKey,]
# add_rows = ON_tree$TreeMsrKey %in% rownames(damage_code)
# add = matrix(nrow = length(add_rows), ncol = dim(damage_code)[2])
# for (i in 1:dim(damage_code)[1]){
#   add[,ON_tree$TreeMsrKey==rownames(damage_code)[i]] = damage_code[i]
# }
# add[add_rows=T,] = damage_code

ON_tree= data.frame(ON_tree,damage_code)
print("2")

cut_percent = matrix(ncol = 6,nrow = length(plot_vect))
rownames(cut_percent) = plot_vect
colnames(cut_percent) = paste0("t",1:6)

damage = matrix(nrow = length(plot_vect),ncol = 6*6)
rownames(damage) = plot_vect
dams = c("Disease/pathogen","Insect","Fire","Drought","Human","Other")
colnames(damage) = c(paste0(dams,"_t1"),paste0(dams,"_t2"),paste0(dams,"_t3"),paste0(dams,"_t4"),paste0(dams,"_t5"),paste0(dams,"_t6"))
dam_codes = list(
  "Disease/pathogen" = c(50,56:62,64:73,101:108,121,145,150,151,153,154,155,168,170,200,214,225,63,111:114,146,152,201,202,203,204,205,209),
  "Insect" = c(1:49,51,91:99,109,110,120,144,224),
  "Fire" = c(90,131,132,210),
  "Drought" = c(87),
  "Human" = c(119,133,138,156,217),
  "Other" = c(0,75:86,88,115:118,122,123,134,135,139:142,157,212,213)
)
print("3")
surveys= data.frame()
#Create consolidated csvs
for (i in plot_vect){
  print(i)
  temp = ON_tree[ON_tree$PlotName==i,]
  temp = arrange(temp,FieldSeasonYear)
  #Tree numbers are reused for PSPs, so this sets tree numbers in growth plots 2 and 3 to their number + the max from the previous plot
  if (unique(temp$plot_type)=="PSP"){
    plots = unique(temp$GrowthPlotNum)
    if(length(plots)>1){
      for(s in 2:length(plots)){
        temp$TreeNum[temp$GrowthPlotNum==plots[s]] = temp$TreeNum[temp$GrowthPlotNum==plots[s]]+max(temp$TreeNum[temp$GrowthPlotNum==plots[s-1]])
        #temp$TreeNum[temp$GrowthPlotNum==3] = temp$TreeNum[temp$GrowthPlotNum==3]+max(temp$TreeNum[temp$GrowthPlotNum==2])
      }
    }
  }
  samp_vect = unique(temp$FieldSeasonYear)
  
  for (n in 1:length(samp_vect)){
    surveys[paste0("5_",i),paste0("t",n)] = samp_vect[n]
  }
  tree_range = unique(temp$TreeNum)
  tree_range = sort(tree_range)
  df = matrix(nrow = length(tree_range),ncol = length(samp_vect)*2)
  sp_df = matrix(nrow = length(tree_range),ncol = length(samp_vect))
  rownames(df) = tree_range
  colnames(df) = c(paste0("dbh_t",1:length(samp_vect)),paste0("status_t",1:length(samp_vect)))
  colnames(sp_df) = paste0("sp_t",1:length(samp_vect))
  rownames(sp_df) = tree_range
  for (n in 1:length(samp_vect)){
    temp_samp = temp[temp$FieldSeasonYear==samp_vect[n],]
    temp_samp = arrange(temp_samp,TreeNum)
    #Removes some doubled trees
    if (i=="WWA1993002PSP"&n==1){
      temp_samp = temp_samp[-47,]
    }
    if(!identical(temp_samp$TreeNum,tree_range)){
      add = tree_range[!tree_range %in% temp_samp$TreeNum]
      add_mat = data.frame(matrix(ncol = dim(temp_samp)[2],nrow = length(add)))
      colnames(add_mat) = colnames(temp_samp)
      add_mat$TreeNum = add
      temp_samp = rbind(temp_samp,add_mat)
      temp_samp = arrange(temp_samp,TreeNum)
    }
    for (j in 1:dim(df)[1]){
      sp_df[j,paste0("sp_t",n)] = as.numeric(temp_samp$SpecCode[temp_samp$TreeNum==rownames(df)[j]])
      df[j,paste0("dbh_t",n)] = temp_samp$DBH[temp_samp$TreeNum==rownames(df)[j]]
      if(!is.na(temp_samp$TreeStatusCode[temp_samp$TreeNum==rownames(df)[j]]) & (temp_samp$TreeStatusCode[temp_samp$TreeNum==rownames(df)[j]]=="L " |temp_samp$TreeStatusCode[temp_samp$TreeNum==rownames(df)[j]]=="l "|temp_samp$TreeStatusCode[temp_samp$TreeNum==rownames(df)[j]]=="V ")){
        df[j,paste0("status_t",n)] = "L"
      }
    }
    cut_percent[as.character(i),n] = sum(temp_samp$TreeStatusCode=="C ",na.rm=T)/sum(!is.na(temp_samp$TreeStatusCode))
    
    flags = data.frame()
    for (s in 1:dim(temp_samp)[1]){
      fl = strsplit(as.character(temp_samp$damage_code[s]),",")[[1]]
      names(fl) = 1:length(fl)
      for (r in 1:length(fl)){
        flags[as.character(s),names(fl)[r]] = fl[r]
      }
    }
    for (s in 1:length(dam_codes)){
      flags[flags %in% dam_codes[[s]]] = names(dam_codes)[s]
    }
    flags = as.matrix(flags)
    for (r in 1:dim(flags)[1]){
      flags[r,duplicated(flags[r,],incomparables = NA)] = NA
    }
    for(j in 1:length(dam_codes)){
      damage[as.character(i),paste0(names(dam_codes)[j],"_t",n)] = sum(flags==names(dam_codes)[j],na.rm=T)/dim(flags)[1]   
    }
  }
  df[df=="NULL"] = NA
  sp_df2 = sp_df
  for (j in 1:dim(LUT)[1]){
    sp_df2[sp_df==as.numeric(LUT[j,"ON"])] = LUT[j,1]
  }
  plot_size = plotsize[[as.character(unique(temp$plot_type[!is.na(temp$plot_type)]))]]
  df = data.frame(sp_df2,df,plot_size)
  write.csv(df,paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/ON/checks/",i,"_check.csv"))
}
write.csv(damage,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/ON/ON_flags_percent.csv")
write.csv(surveys,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/ON/ON_surveys.csv")
write.csv(cut_percent,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/ON/ON_cut_percent.csv")