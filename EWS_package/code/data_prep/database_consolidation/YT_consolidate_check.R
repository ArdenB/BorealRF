
library(dplyr)
library(readxl)

LUT = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SP_LUT.csv",stringsAsFactors = F)

sheets = sprintf("%03.0f",1:305)
files = vector()
r=1
for (n in sheets){
  if (file.exists(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/YT/PSP/PSP",n,".xlsx"))){
    files[r] = n
    r = r+1
  }
}

#Find species errors
sp_err = data.frame()
for (i in files){
  samp_vect = excel_sheets(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/YT/PSP/PSP",i,".xlsx"))
  temp = data.frame()
  
  for (k in 1:length(samp_vect)){
    temp1 = read_excel(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/YT/PSP/PSP",i,".xlsx"), sheet = k)
    temp1 = data.frame(k,temp1[,c(1,2,3,6)])
    temp = rbind(temp,temp1)
    # temp[[paste0("t",k)]] temp1 = read_excel(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/Yukon/PSP/PSP",i,".xlsx"), sheet = k)
    # temp[[paste0("t",k)]] = arrange(temp[[paste0("t",k)]],Tree_Tag_Number)
  }
  for (n in unique(temp$Tree_Tag_Number)){
    sp = temp$Species_ID[temp$Tree_Tag_Number==n]
    names(sp) = temp$k[temp$Tree_Tag_Number==n]
    sp = sp[!is.na(sp)]
    if (length(unique(sp))>1){
      for (j in 1:length(sp)){
        sp_err[paste0(i,"_",n),names(sp)[j]] = sp[j]
      }
      #some of these errors are a result of reusing tree numbers, this filters out those trees
      nans = !is.na(sp_err[paste0(i,"_",n),])
      if(!identical(which(nans),min(which(nans)):max(which(nans)))){
        sp_err = sp_err[!rownames(sp_err) %in% paste0(i,"_",n),]
      }
    }
  }
}
sp_err = sp_err[,as.character(c(1:dim(sp_err)[2]))]

sp_err2=sp_err
#changing the sp IDs of the errors. This makes a few assumptions:
#1) Later identifications are more accurate
#2) If a tree was identified as one species more often than another, it is likey that species
for (i in 1:dim(sp_err)[1]){
  indices = colnames(sp_err[i,!is.na(sp_err[i,])])
  sp_vect = vector()
  for (n in indices){
    sp_vect[n] = sp_err[i,n]
  }
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

err_check = vector()
for (n in 1:dim(sp_err)[1]){
  for (i in 1:6){
    err_check[i] = sp_err[n,i]
  }
  if (length(unique(err_check[is.na(err_check)]))>1){
    print(rownames(sp_err)[n])
  }
}


#Create consolidated csvs
for (i in files){
  samp_vect = excel_sheets(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/YT/PSP/PSP",i,".xlsx"))
  temp = data.frame()
  for (k in 1:length(samp_vect)){
    temp1 = as.data.frame(read_excel(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/YT/PSP/PSP",i,".xlsx"), sheet = k))
    temp1 = data.frame(k,temp1[,c(1,2,3,6)])
    temp = rbind(temp,temp1)
  }
  #A couple of trees IDed as something totally unique, they're dead trees so I'm just changing the names so it doesn't trip
  if(i=="052"){
    temp$Species_ID[335] = "Trembling Aspen"
  }
  if(i=="015"){
    temp$Species_ID[7] = NA
  }
  temp$Species_ID[temp$Species_ID=="Alnus Tenuifolia"]
  #Correct species on the fly
  if (any(sub("_.*","",rownames(sp_err)) %in% i)){
    err = sp_err[sub("_.*","",rownames(sp_err)) %in% i,]
    for (j in 1:dim(err)[1]){
      indices = colnames(err[j,!is.na(err[j,])])
      for (n in indices){
        temp$Species_ID[temp$Tree_Tag_Number==as.numeric(sub("*..._","",rownames(err)[j]))&temp$k==n] = err[j,n]
      }
    }
  }
  tree_range = min(unique(temp$Tree_Tag_Number)):max(unique(temp$Tree_Tag_Number))
  df = matrix(nrow = length(tree_range),ncol = length(samp_vect)*3)
  rownames(df) = tree_range
  colnames(df) = c(paste0("sp_t",1:length(samp_vect)),paste0("dbh_t",1:length(samp_vect)),paste0("status_t",1:length(samp_vect)))
  for (n in 1:length(samp_vect)){
    temp_samp = temp[temp$k==n,]
    #Removes some doubled trees
    if (i=="059"&n==2){
      temp_samp = temp_samp[c(-5,-31),]
    }else if (i=="071"&n==3){
      temp_samp = temp_samp[c(-82),]
    }else if (i=="082"&n==2){
      temp_samp = temp_samp[-79,]
    }else if (i=="083"&n==3){
      temp_samp = temp_samp[-42,]
    }else if (i=="083"&n==4){
      temp_samp = temp_samp[c(-54,-61),]
    }else if (i=="117"&n==1){
      temp_samp = temp_samp[-15,]
    }else if (i=="133"&n==4){
      temp_samp = temp_samp[c(-3,-6,-17,-21),]
    }
    if(!identical(temp_samp$Tree_Tag_Number,tree_range)){
      add = tree_range[!tree_range %in% temp_samp$Tree_Tag_Number]
      add_mat = data.frame(matrix(ncol = dim(temp_samp)[2],nrow = length(add)))
      colnames(add_mat) = colnames(temp_samp)
      add_mat$Tree_Tag_Number = add
      temp_samp = rbind(temp_samp,add_mat)
      temp_samp = arrange(temp_samp,Tree_Tag_Number)
    }
    for (j in 1:dim(df)[1]){
      df[j,paste0("sp_t",n)] = temp_samp$Species_ID[temp_samp$Tree_Tag_Number==rownames(df)[j]]
      df[j,paste0("dbh_t",n)] = temp_samp$Diameter[temp_samp$Tree_Tag_Number==rownames(df)[j]]
      if(!is.na(temp_samp$Tree_Class_ID[temp_samp$Tree_Tag_Number==rownames(df)[j]]) & (temp_samp$Tree_Class_ID[temp_samp$Tree_Tag_Number==rownames(df)[j]]=="1 - Live/healthy"|temp_samp$Tree_Class_ID[temp_samp$Tree_Tag_Number==rownames(df)[j]]=="2 - Live/unhealthy")){
        df[j,paste0("status_t",n)] = "L"
      }
      #Attempt to correct for trees mistakenly identified as dead or no status indicated
      if (n!=1){
        if(!is.na(df[j,paste0("status_t",n)]) & is.na(df[j,paste0("status_t",n-1)]) & !is.na(df[j,paste0("sp_t",n-1)]) & !is.na(df[j,paste0("dbh_t",n-1)]) & df[j,paste0("sp_t",n)]==df[j,paste0("sp_t",n-1)] & 0<as.numeric(df[j,paste0("dbh_t",n-1)]) & as.numeric(df[j,paste0("dbh_t",n-1)])<as.numeric(df[j,paste0("dbh_t",n)])){
          df[j,paste0("status_t",n-1)] = "L"
        }
      }
    }
  }
  #Normalize sp codes using LUT
  for (j in 1:dim(LUT)[1]){
    df[df==LUT[j,"YT"]] = LUT[j,1]
  }
  plot_size = 0.04
  df = data.frame(df,plot_size)
  write.csv(df,paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/YT/checks/",i,"_check.csv"))
}
