
library(dplyr)
library(readxl)

LUT = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SP_LUT.csv",stringsAsFactors = F)

files = list.files("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/AB/Data/PSP2011/csvs/")
files = sub("_.*","",files)
files = unique(files)


#Find species errors
sp_err = data.frame()
remove_line = vector()
r=1
psp_names = vector()
s=1
for (i in files){
  file = read.csv(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/AB/Data/PSP2011/csvs/",i,"_tree.csv"))
  file = file[file$Tree_num<8000,]
  file = file[file$Tree_num!=0,]
  file$Sp[file$Sp=="  "] = NA
  
  for (k in unique(file$subplot_num)){
    temp = file[file$subplot_num==k,]
    psp = paste0(unique(temp$Group_num),".",k)
    psp_names[s] = psp
    s=s+1
    temp = arrange(temp,Meas_num)
    for (n in unique(temp$Tree_num)){
      sp = temp$Sp[temp$Tree_num==n]
      names(sp) = temp$Meas_num[temp$Tree_num==n]+1
      sp = sp[!is.na(sp)]
      if (length(sp)>0){
        for (j in 1:length(sp)){
          if (names(sp)[j] %in% names(sp)[-j] & !any(sp[j]==sp[-j])){
            sp[j] = NA
            remove_line[r] = paste0(psp,"_",temp$X[temp$Tree_num==n][j])
            r=r+1
          }
        }
        sp = sp[!is.na(sp)]
        if (length(unique(sp))>1){
          for (j in 1:length(sp)){
            sp_err[paste0(psp,"_",n),names(sp)[j]] = as.character(sp[j])
          }
          #some of these errors are a result of reusing tree numbers, this filters out those trees
          nans = !is.na(sp_err[paste0(psp,"_",n),])
          if(sum(nans,na.rm=T)!=0){
            if(!identical(which(nans),min(which(nans)):max(which(nans)))){
              sp_err = sp_err[!rownames(sp_err) %in% paste0(psp,"_",n),]
            }
          }
        }
      }
    }
  }
}

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

dbh_err = data.frame()
cond_code = data.frame()
for (i in files){
  file = read.csv(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/AB/Data/PSP2011/csvs/",i,"_tree.csv"))
  file = file[file$Tree_num<8000,]
  file = file[file$Tree_num!=0,]
  file$Sp[file$Sp=="  "] = NA
  
  for (k in unique(file$subplot_num)){
    temp = file[file$subplot_num==k,]
    psp = paste0(unique(temp$Group_num),".",k)
    temp = arrange(temp,Meas_num)
    temp = arrange(temp,Meas_num)
    for (n in unique(temp$Tree_num)){
      dbh = temp$DBH[temp$Tree_num==n]
      cond = temp$Cond_code1[temp$Tree_num==n]
      nans = !is.na(dbh)
      names(dbh) = temp$Meas_num[temp$Tree_num==n]+1
      names(cond) = temp$Meas_num[temp$Tree_num==n]+1
      if(sum(nans,na.rm=T)!=0){
        if(!identical(which(nans),min(which(nans)):max(which(nans)))){
          for (j in 1:length(dbh)){
            dbh_err[paste0(psp,"_",n),names(dbh)[j]] = as.character(dbh[j])
            cond_code[paste0(psp,"_",n),names(cond)[j]] = as.character(cond[j])
            
          }
        }
      }
    }
  }
}
cond_code[is.na(cond_code)] = 0
dbh_err2=dbh_err
for (i in 1:dim(dbh_err)[1]){
  for (j in which(!is.na(dbh_err[i,]))[1]:which(!is.na(dbh_err[i,]))[length(which(!is.na(dbh_err[i,])))]){
    if (is.na(dbh_err[i,j])&any(cond_code[i,j]==c("0","98"))){
      if(!is.na(dbh_err[i,j+1])){
        dbh_err[i,j] = mean(c(as.numeric(dbh_err[i,j-1]),as.numeric(dbh_err[i,j+1])))
      }else{
        dbh_err[i,j] = as.numeric(dbh_err[i,j-1])+(as.numeric(dbh_err[i,j+2])-as.numeric(dbh_err[i,j-1]))/3
      }
    }else if(!is.na(dbh_err[i,j])&any(cond_code[i,j]==c("25","61"))){
      dbh_err[i,j] = NA
    }
  }
}

cut_percent = matrix(ncol = 8,nrow = length(psp_names))
rownames(cut_percent) = psp_names
colnames(cut_percent) = paste0("t",1:8)

damage = matrix(nrow = length(psp_names),ncol = 6*8)
rownames(damage) = psp_names
dams = c("Disease/pathogen","Insect","Fire","Drought","Human","Other")
colnames(damage) = c(paste0(dams,"_t1"),paste0(dams,"_t2"),paste0(dams,"_t3"),paste0(dams,"_t4"),paste0(dams,"_t5"),paste0(dams,"_t6"),paste0(dams,"_t7"),paste0(dams,"_t8"))
codes = c(0,1,2,5,6,7,8,9)
dam_codes = list(
  "Disease/pathogen" = c(2,4,19,20,21,47,51,63,65,68,69,70,71,72,73,74,81,91:96),
  "Insect" = c(1,30,31,32,33,38,62,64,75,76,77,78,78,80,82,83,85),
  "Fire" = c(6),
  "Drought" = c(),
  "Human" = c(11,18,29,34,7),
  "Other" = c(8,9,10,3,5,42,43,44,45,60,84,86,87,88,89,90,37)
)
surveys= data.frame()
#Create consolidated csvs
for (i in files){
  file = read.csv(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/AB/Data/PSP2011/csvs/",i,"_tree.csv"),stringsAsFactors = F)
  #PSP 458, 459 and 625 were burned sometime between 1993 and 1998, and all the trees were killed. Because of this, they didn't record a lot of info,
  #including the plot size which creates an error for me. Therefore, I am filling in that field where it is empty.
  if(i=="PSP458"| i=="PSP459" | i=="PSP625"){
    file$Tree_plot_size=1000
  }
  file = file[file$Tree_num<8000,]
  file = file[file$Tree_num!=0,]
  file$Sp[file$Sp=="  "] = NA
    
  for (k in unique(file$subplot_num)){
    temp = file[file$subplot_num==k,]
    psp = paste0(unique(temp$Group_num),".",k)
    temp = arrange(temp,Meas_num)

    #Resetting a duplicated tree number
    if (psp=="131.2"){
      temp[c("277","554","831","1108","1385"),17] = 277
    }
    #Resetting a misnumbered tree to the correct number
    if (psp=="564.1"){
      temp["349",17] = 163
    }
    samp_vect = unique(temp$Year_meas)
    
    for(j in 1:length(samp_vect)){
      surveys[psp,paste0("t",j)] = samp_vect[j]
    }
    #Correct species on the fly
    if (any(sub("_.*","",rownames(sp_err)) %in% psp)){
      err = sp_err[sub("_.*","",rownames(sp_err)) %in% psp,]
      for (j in 1:dim(err)[1]){
        indices = colnames(err[j,!is.na(err[j,])])
        for (n in indices){
          temp$Sp[temp$Tree_num==as.numeric(strsplit(rownames(err)[j],"_")[[1]][2])&temp$Year_meas==samp_vect[as.numeric(n)]] = err[j,n]
        }
      }
    }
    #Correct dbh on the fly
    if (any(sub("_.*","",rownames(dbh_err)) %in% psp)){
      err = dbh_err[sub("_.*","",rownames(dbh_err)) %in% psp,]
      for (j in 1:dim(err)[1]){
        indices = colnames(err[j,!is.na(err[j,])])
        for (n in indices){
          temp$DBH[temp$Tree_num==as.numeric(strsplit(rownames(err)[j],"_")[[1]][2])&temp$Year_meas==samp_vect[as.numeric(n)]] = as.numeric(err[j,n])
        }
      }
    }
    tree_range = unique(temp$Tree_num)
    tree_range = sort(tree_range)
    df = matrix(nrow = length(tree_range),ncol = length(samp_vect)*3)
    rownames(df) = tree_range
    colnames(df) = c(paste0("sp_t",1:length(samp_vect)),paste0("dbh_t",1:length(samp_vect)),paste0("status_t",1:length(samp_vect)))
    for (n in 1:length(samp_vect)){
      temp_samp = temp[temp$Year_meas==samp_vect[n],]
      temp_samp = arrange(temp_samp,Tree_num)
      #Removes some doubled trees
      if (psp=="53.3"&n==6){
        temp_samp = temp_samp[-18,]
      }
      
      if(!identical(temp_samp$Tree_num,tree_range)){
        add = tree_range[!tree_range %in% temp_samp$Tree_num]
        add_mat = data.frame(matrix(ncol = dim(temp_samp)[2],nrow = length(add)))
        colnames(add_mat) = colnames(temp_samp)
        add_mat$Tree_num = add
        temp_samp = rbind(temp_samp,add_mat)
        temp_samp = arrange(temp_samp,Tree_num)
      }
      for (j in 1:dim(df)[1]){
        df[j,paste0("sp_t",n)] = as.character(temp_samp$Sp[temp_samp$Tree_num==rownames(df)[j]])
        df[j,paste0("dbh_t",n)] = temp_samp$DBH[temp_samp$Tree_num==rownames(df)[j]]/10
        if(!is.na(temp_samp$DBH[temp_samp$Tree_num==rownames(df)[j]])){
          df[j,paste0("status_t",n)] = "L"
        }
      }
      cut_percent[psp,n] = sum(temp_samp$Cond_code1==29|temp_samp$Cond_code2==29|temp_samp$Cond_code3==29,na.rm=T)/sum(!is.na(temp_samp$Cond_code1))
      flags = as.matrix(temp_samp[,c("Cond_code1","Cond_code2","Cond_code3")])
      
      for (s in 1:length(dam_codes)){
        flags[flags %in% dam_codes[[s]]] = names(dam_codes)[s]
      }
      for (r in 1:dim(flags)[1]){
        flags[r,duplicated(flags[r,],incomparables = NA)] = NA
      }
      for(j in 1:length(dam_codes)){
        damage[psp,paste0(names(dam_codes)[j],"_t",n)] = sum(flags==names(dam_codes)[j],na.rm=T)/sum(!is.na(flags[,1]))
      }
      
    }
    #Normalize sp codes using LUT
    for (j in 1:dim(LUT)[1]){
      df[df==LUT[j,"AB"]] = LUT[j,1]
    }
    plot_size = unique(temp$Tree_plot_size)[1]/10000
    df = data.frame(df,plot_size)
    write.csv(df,paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/AB/checks/",psp,"_check.csv"))
  }
}
write.csv(cut_percent,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/AB/AB_cut_percent.csv")
write.csv(damage,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/AB/AB_flags_percent.csv")
write.csv(surveys,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/AB/AB_surveys.csv")