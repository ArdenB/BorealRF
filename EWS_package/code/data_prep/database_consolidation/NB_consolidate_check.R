library(readxl)
library(plyr)
library(dplyr)

LUT = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SP_LUT.csv",stringsAsFactors = F)
NB_tree <- read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NB/PSP_TREE_YIMO.txt",stringsAsFactors = F)
NB_surveys = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NB/NB_surveys.csv")
NB_plots = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NB/NB_PSP_PLOTS.csv")

colnames(NB_tree) = c("RemeasID","treenumber","species","cause","dbh","agecl","cr","treetop","woundtype","depth","dim","ltbh","conks","lean","leaderda",
                      "curpct","cumpct","thincr","lat","sampleTree","sampleTreeEstabYr","sampleTreeAge","sampleTreeHt","standingDeadTreeDecayClass",
                      "standingDeadTreeHtClass","Notes","Tstamp","Plot","MeasNum","Form","Vigor")
NB_plots_yr = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NB/PSP_PLOTS_YR.csv",stringsAsFactors = F)
NB_plots_yr[3,2] = "10001"

#This site has no date of inventory data
NB_tree = NB_tree[NB_tree$Plot!=8504,]

plot_vect = unique(NB_tree$Plot)
#Find species errors
sp_err = data.frame()
for (i in plot_vect){
  temp = NB_tree[NB_tree$Plot==i,]
  temp = arrange(temp,MeasNum)
  samp_vect = unique(temp$MeasNum)
  for (n in unique(temp$treenumber)){
    sp = temp$species[temp$treenumber==n]
    for (j in 1:length(sp)){
      names(sp)[j] = which(samp_vect==temp$MeasNum[temp$treenumber==n][j])
    }
    sp = sp[!is.na(sp)]
    if (length(unique(sp))>1){
      for (j in 1:length(sp)){
        sp_err[paste0(i,"_",n),names(sp)[j]] = sp[j]
      }
      #some of these errors are a result of reusing tree numbers, this filters out those trees
      nans = !is.na(sp_err[paste0(i,"_",n),])
      if(sum(nans,na.rm=T)!=0){
        if(!identical(which(nans),min(which(nans)):max(which(nans)))){
          sp_err = sp_err[!rownames(sp_err) %in% paste0(i,"_",n),]
        }
      }
    }else if(any(sp==996,na.rm = T)|any(sp==997,na.rm = T)|any(sp==999,na.rm = T)){
      for (j in 1:length(sp)){
        sp_err[paste0(i,"_",n),names(sp)[j]] = sp[j]
      }
    }
  }
}
print("errors found")

sp_err2=sp_err
#changing the sp IDs of the errors. This makes a few assumptions:
#1) Later identifications are more accurate
#2) If a tree was identified as one species more often than another, it is likely that species
for (i in 1:dim(sp_err)[1]){
  indices = colnames(sp_err)[which(!is.na(sp_err[i,]))]
  sp_vect = vector()
  for (n in indices){
    sp_vect[n] = sp_err[i,n]
  }
  if(length(sp_vect)==1&any(sp_vect==c(996,997,999))){
    sp_err[i,names(sp_vect)] = 999
  }else if (length(sp_vect)>1&(any(sp_vect==996,na.rm = T)|any(sp_vect==997,na.rm = T)|any(sp_vect==999,na.rm = T))){
    sp_err[i,names(sp_vect)] = sp_vect[sp_vect!=996|sp_vect!=997|sp_vect!=999]
  }else{
    if (sum(sp_vect==unique(sp_vect)[1])==sum(sp_vect==unique(sp_vect)[2])){
      sp = unique(sp_vect)[2]
      sp_err[i,names(sp_vect)] = sp
    }else if(any(duplicated(sp_vect))){
      sp = sp_vect[duplicated(sp_vect)]
      if (length(unique(sp))==1){
        sp = unique(sp)
        sp_err[i,names(sp_vect)] = sp
      }else if(any(duplicated(sp))){
        sp = sp[duplicated(sp)]
        if (length(unique(sp))==1){
          sp = unique(sp)
          sp_err[i,names(sp_vect)] = sp
        }
      }
    }
  }
}
print("changed")
err_check = vector()
for (n in 1:dim(sp_err)[1]){
  for (i in 1:6){
    err_check[i] = sp_err[n,i]
  }
  if (length(unique(err_check[is.na(err_check)]))>1){
    print(rownames(sp_err)[n])
  }
}

#correct species in NB_tree
for (i in 1:dim(sp_err)[1]){
  indices = colnames(sp_err)[which(!is.na(sp_err[i,]))]
  samp_vect = unique(NB_tree$MeasNum[NB_tree$Plot==strsplit(rownames(sp_err)[i],"_")[[1]][1]])
  for (n in indices){
    NB_tree$species[NB_tree$Plot==strsplit(rownames(sp_err)[i],"_")[[1]][1]&NB_tree$treenumber==strsplit(rownames(sp_err)[i],"_")[[1]][2]
                    &NB_tree$MeasNum==samp_vect[as.numeric(n)]] = sp_err[i,n]
  }
}
print("corrected")
#Setting NAs in cause to 0, assuming that no cause means the tree is alive
NB_tree$cause[is.na(NB_tree$cause)] = 0

cut_percent = matrix(ncol = length(unique(NB_tree$MeasNum)),nrow = length(plot_vect))
rownames(cut_percent) = plot_vect
colnames(cut_percent) = paste0("t",1:length(unique(NB_tree$MeasNum)))


damage = matrix(nrow = length(plot_vect),ncol = 6*7)
rownames(damage) = plot_vect
dams = c("Disease/pathogen","Insect","Fire","Drought","Human","Other")
colnames(damage) = c(paste0(dams,"_t1"),paste0(dams,"_t2"),paste0(dams,"_t3"),paste0(dams,"_t4"),paste0(dams,"_t5"),paste0(dams,"_t6"),paste0(dams,"_t7"))
dam_codes = list(
  "Disease/pathogen" = c("D"),
  "Insect" = c(1,"I"),
  "Fire" = c(),
  "Drought" = c(),
  "Human" = c("P","H"),
  "Other" = c(2,8,"M","A","O","U")
)
surveys = data.frame()
#Create consolidated csvs
for (i in plot_vect){
  print(i)
  temp = NB_tree[NB_tree$Plot==i,]
  temp$species[temp$species==997] = 999
  temp$species[temp$species==996] = 999
  temp = arrange(temp,MeasNum)
  samp_vect = unique(temp$MeasNum)
  for (j in 1:length(samp_vect)){
    surveys[as.character(i),paste0("t_",j)] = NB_plots_yr$MeasYr[NB_plots_yr$measNum==samp_vect[j]&NB_plots_yr$Plot==i]
  }
  tree_range = unique(temp$treenumber)
  tree_range = sort(tree_range)
  df = matrix(nrow = length(tree_range),ncol = length(samp_vect)*2)
  sp_df = matrix(nrow = length(tree_range),ncol = length(samp_vect))
  rownames(df) = tree_range
  colnames(df) = c(paste0("dbh_t",1:length(samp_vect)),paste0("status_t",1:length(samp_vect)))
  colnames(sp_df) = paste0("sp_t",1:length(samp_vect))
  rownames(sp_df) = tree_range
  for (n in 1:length(samp_vect)){
    temp_samp = temp[temp$MeasNum==samp_vect[n],]
    if(!identical(temp_samp$treenumber,tree_range)){
      add = tree_range[!tree_range %in% temp_samp$treenumber]
      add_mat = data.frame(matrix(ncol = dim(temp_samp)[2],nrow = length(add)))
      colnames(add_mat) = colnames(temp_samp)
      add_mat$treenumber = add
      temp_samp = rbind(temp_samp,add_mat)
      temp_samp = arrange(temp_samp,treenumber)
    }
    for (j in 1:dim(df)[1]){
      sp_df[j,paste0("sp_t",n)] = as.numeric(temp_samp$species[temp_samp$treenumber==rownames(df)[j]])
      df[j,paste0("dbh_t",n)] = temp_samp$dbh[temp_samp$treenumber==rownames(df)[j]]/10
      if(!is.na(temp_samp$cause[temp_samp$treenumber==rownames(df)[j]])&temp_samp$cause[temp_samp$treenumber==rownames(df)[j]]==0){
        df[j,paste0("status_t",n)] = "L"
      }
    }
    cut_percent[as.character(i),paste0("t",n)] = sum(temp_samp$cause==8,na.rm=T)/sum(!is.na(temp_samp$cause))
    flags = temp_samp[,"cause"]
    for (s in 1:length(dam_codes)){
      flags[flags %in% dam_codes[[s]]] = names(dam_codes)[s]
    }
    # for (r in 1:length(flags)){
    #   flags[r,duplicated(flags[r,],incomparables = NA)] = NA
    # }
    for(j in 1:length(dam_codes)){
      damage[as.character(i),paste0(names(dam_codes)[j],"_t",n)] = sum(flags==names(dam_codes)[j],na.rm=T)/sum(!is.na(temp_samp$cause))
    }
  }
  sp_df2 = sp_df
  for (j in 1:dim(LUT)[1]){
    sp_df2[sp_df==as.numeric(LUT[j,"NB"])] = LUT[j,1]
  }
  plot_size = NB_plots[NB_plots$Plot==i,"PlotSize"]/10000
  df = data.frame(sp_df2,df,plot_size)
 # write.csv(df,paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NB/checks/",i,"_check.csv"))
}
write.csv(cut_percent,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NB/NB_cut_percent.csv")
write.csv(damage,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NB/NB_flags_percent.csv")
write.csv(surveys,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NB/NB_surveys.csv")