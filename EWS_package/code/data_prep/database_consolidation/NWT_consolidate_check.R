
library(dplyr)

LUT = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SP_LUT.csv",stringsAsFactors = F)
NWT_Alldata <- read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NWT/NWT_Alldata.csv",stringsAsFactors = F)
NWT_Alldata = NWT_Alldata[!is.na(NWT_Alldata$OBJECTID),]

plot_vect=unique(NWT_Alldata$SAMPLE)
plot_vect = plot_vect[!is.na(plot_vect)]
NWT_Alldata$TREESPECIESCODE[NWT_Alldata$TREESPECIESCODE=="L"] = "La"

#Find species errors
sp_err = data.frame()
for (i in plot_vect){
  temp = NWT_Alldata[NWT_Alldata$SAMPLE==i,]
  temp = temp[!rowSums(!is.na(temp))==0,]
  temp = arrange(temp,INV_SURVEY_DATE)
  samp_vect = unique(temp$OBJECTID)
  for (n in 1:length(samp_vect)){
    temp$OBJECTID[temp$OBJECTID==samp_vect[n]] = paste0("t",n)
    NWT_Alldata$OBJECTID[NWT_Alldata$OBJECTID==samp_vect[n]&NWT_Alldata$SAMPLE==i] = paste0("t",n)
  }
  for (n in unique(temp$TREENUMBER)){
    sp = vector(length = length(samp_vect))
    names(sp) = samp_vect
    sp[temp$OBJECTID[temp$TREENUMBER==n]] = temp$TREESPECIESCODE[temp$TREENUMBER==n]
    if (length(unique(sp))>1){
      for (j in 1:length(sp)){
        sp_err[paste0(i,"_",n),names(sp)[j]] = sp[j]
      }
      
      f = sp_err[paste0(i,"_",n),]=="FALSE"
      if (any(f==TRUE,na.rm = T)){
        if(which(f)[1]==1|which(f)[length(which(f))]==3){
          sp_err = sp_err[!rownames(sp_err) %in% paste0(i,"_",n),]
        }
      }
    }
  }
}

#Find species errors
sp_err = data.frame()
for (i in plot_vect){
  temp = NWT_Alldata[NWT_Alldata$SAMPLE==i,]
  temp = temp[!rowSums(!is.na(temp))==0,]
  temp = arrange(temp,INV_SURVEY_DATE)
  samp_vect = unique(temp$OBJECTID)
  for (n in 1:length(samp_vect)){
    temp$OBJECTID[temp$OBJECTID==samp_vect[n]] = paste0("t",n)
    NWT_Alldata$OBJECTID[NWT_Alldata$OBJECTID==samp_vect[n]&NWT_Alldata$SAMPLE==i] = paste0("t",n)
  }
  for (n in unique(temp$TREENUMBER)){
    sp = vector(length = length(samp_vect))
    names(sp) = samp_vect
    sp[temp$OBJECTID[temp$TREENUMBER==n]] = temp$TREESPECIESCODE[temp$TREENUMBER==n]
    if (length(unique(sp))>1){
      for (j in 1:length(sp)){
        sp_err[paste0(i,"_",n),names(sp)[j]] = sp[j]
      }
      
      f = sp_err[paste0(i,"_",n),]=="FALSE"
      if (any(f==TRUE,na.rm = T)){
        if(which(f)[1]==1|which(f)[length(which(f))]==3){
          sp_err = sp_err[!rownames(sp_err) %in% paste0(i,"_",n),]
        }
      }
    }
  }
}

#changing the sp IDs of the errors. This makes a few assumptions:
#1) Later identifications are more accurate
#2) If a tree was identified as one species more often than another, it is likey that species
for (i in 1:dim(sp_err)[1]){
  indices = colnames(sp_err[i,!is.na(sp_err[i,])])
  sp_vect = vector()
  for (n in indices){
    sp_vect[n] = sp_err[i,n]
  }
  if(length(unique(sp_vect[sp_vect!="FALSE"]))==1){
    sp_err[i,names(sp_vect)] = unique(sp_vect[sp_vect!="FALSE"])
    sp_vect[names(sp_vect)] = unique(sp_vect[sp_vect!="FALSE"])
  }
  if(length(unique(sp_vect[sp_vect!="FALSE"]))>1){
    if (sum(sp_vect==unique(sp_vect[sp_vect!="FALSE"])[1])==sum(sp_vect==unique(sp_vect[sp_vect!="FALSE"])[2])){
      sp = unique(sp_vect[sp_vect!="FALSE"])[2]
      sp_err[i,names(sp_vect)] = sp
    }else if(any(duplicated(sp_vect[sp_vect!="FALSE"]))){
      sp = sp_vect[duplicated(sp_vect[sp_vect!="FALSE"])]
      if (length(unique(sp))==1){
        sp = unique(sp)
        sp_err[i,names(sp_vect)] = sp
      }
    }
  }
}

#correct species in NWT_Alldata
for (i in 1:dim(sp_err)[1]){
  indices = colnames(sp_err[i,!is.na(sp_err[i,])])
  for (n in indices){
    NWT_Alldata$TREESPECIESCODE[NWT_Alldata$SAMPLE==strsplit(rownames(sp_err)[i],"_")[[1]][1]&NWT_Alldata$TREENUMBER==strsplit(rownames(sp_err)[i],"_")[[1]][2]
                                &NWT_Alldata$OBJECTID==n] = sp_err[i,n]
  }
}

#Create consolidated csvs
for (i in plot_vect){
  temp = NWT_Alldata[NWT_Alldata$SAMPLE==i,]
  temp = temp[!rowSums(!is.na(temp))==0,]
  temp = arrange(temp,INV_SURVEY_DATE)
  #Change larch from L to La, because using L interferes with indication of live or dead
  
  samp_vect = unique(temp$OBJECTID)
  tree_range = min(unique(temp$TREENUMBER)):max(unique(temp$TREENUMBER))
  df = matrix(nrow = length(tree_range),ncol = length(samp_vect)*3)
  rownames(df) = tree_range
  colnames(df) = c(paste0("sp_t",1:length(samp_vect)),paste0("dbh_t",1:length(samp_vect)),paste0("status_t",1:length(samp_vect)))
  for (n in 1:length(samp_vect)){
    temp_samp = temp[temp$OBJECTID==samp_vect[n],]
    
    if(!identical(temp_samp$TREENUMBER,tree_range)){
      add = tree_range[!tree_range %in% temp_samp$TREENUMBER]
      add_mat = data.frame(matrix(ncol = dim(temp_samp)[2],nrow = length(add)))
      colnames(add_mat) = colnames(temp_samp)
      add_mat$TREENUMBER = add
      temp_samp = rbind(temp_samp,add_mat)
      temp_samp = arrange(temp_samp,TREENUMBER)
    }
    for (j in 1:dim(df)[1]){
      df[j,paste0("sp_t",n)] = temp_samp$TREESPECIESCODE[temp_samp$TREENUMBER==rownames(df)[j]]
      df[j,paste0("dbh_t",n)] = temp_samp$DBH_CM[temp_samp$TREENUMBER==rownames(df)[j]]
      if(!is.na(temp_samp$TREESPECIESCODE[temp_samp$TREENUMBER==rownames(df)[j]])){
        df[j,paste0("status_t",n)] = "L"
      }
    }
  }
  if (any(sub("_.*","",rownames(sp_err)) %in% i)){
    err = sp_err[sub("_.*","",rownames(sp_err)) %in% i,]
    for (j in 1:dim(err)[1]){
      indices = colnames(err[j,])[which(!is.na(err[j,]))]
      for (n in indices){
        df[strsplit(rownames(err)[j],"_")[[1]][2],paste0("sp_",n)] = err[j,n]
        #temp$DBH[temp$Tree_num==as.numeric(strsplit(rownames(err)[j],"_")[[1]][2])&temp$Year_meas==samp_vect[as.numeric(n)]] = as.numeric(err[j,n])
      }
    }
  }
  #Normalize sp codes using LUT
  for (j in 1:dim(LUT)[1]){
    df[df==LUT[j,"NT"]] = LUT[j,1]
  }
  plot_size = 0.04
  df = data.frame(df,plot_size)
  write.csv(df,paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NWT/checks/",i,"_check.csv"))
}
