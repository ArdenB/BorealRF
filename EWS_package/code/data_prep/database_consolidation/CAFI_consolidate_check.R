library(readxl)
library(plyr)
library(dplyr)

LUT = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SP_LUT.csv",stringsAsFactors = F)
CAFI1 <- read_excel("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CAFI/CAFI TREE INVENTORY 1 2015.xlsx", 
                    col_types = c("numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "text", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric"))
CAFI2 <- read_excel("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CAFI/CAFI TREE INVENTORY 2 2015.xlsx", 
                    col_types = c("numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "text", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric"))
CAFI3 <- read_excel("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CAFI/CAFI TREE INVENTORY 3 2015.xlsx", 
                    col_types = c("numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "text", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric"))
CAFI4 <- read_excel("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CAFI/CAFI TREE INVENTORY 4 2015.xlsx", 
                    col_types = c("numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "text", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric","numeric","numeric"))
CAFI4[26539,29] = 22064
CAFI4[26539,30] = 22081
CAFI5 <- read_excel("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CAFI/CAFI TREE INVENTORY 5 2015.xlsx", 
                    col_types = c("numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "text", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric", "numeric", 
                                  "numeric", "numeric","numeric","numeric"))
CAFI_sites = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CAFI/CAFI_Site_Description_2015_GE2_wFires_msin_final.csv", stringsAsFactors = F)

#Make disease codes match for all inventories
CAFI1$Disease_Code=paste0(CAFI1$Disease_Code,",",CAFI1$Disease_Codes)
CAFI1$Disease_Code = gsub(",NA","",CAFI1$Disease_Code)
CAFI1 = CAFI1[,-30]

CAFI2$Disease_Code=paste0(CAFI2$Disease_Code,",",CAFI2$Disease_Codes)
CAFI2$Disease_Code = gsub(",NA","",CAFI2$Disease_Code)
CAFI2 = CAFI2[,-30]

CAFI3$Disease_Code=paste0(CAFI3$Disease_Code,",",CAFI3$Disease_Codes)
CAFI3$Disease_Code = gsub(",NA","",CAFI3$Disease_Code)
CAFI3 = CAFI3[,-30]

colnames(CAFI4)[29] = "Disease_Code"
CAFI4$Disease_Code = paste0(CAFI4$Disease_Code,",",CAFI4$Disease_Code2,",",CAFI4$Disease_Code3,",",CAFI4$Disease_Code3)
CAFI4$Disease_Code = gsub(",NA","",CAFI4$Disease_Code)
CAFI4 = CAFI4[,c(-30,-31,-32)]

colnames(CAFI5)[29] = "Disease_Code"
CAFI5$Disease_Code = paste0(CAFI5$Disease_Code,",",CAFI5$Disease_Codes2,",",CAFI5$Disease_Codes3,",",CAFI5$Disease_Codes3)
CAFI5$Disease_Code = gsub(",NA","",CAFI5$Disease_Code)
CAFI5 = CAFI5[,c(-30,-31,-32)]

#Had to remove inventory 2 for PSP10151, adjusting the observations to match the description file
CAFI2[CAFI2$PSP==10151,] = NA
CAFI2 = CAFI2[rowSums(!is.na(CAFI2))!=0,]
CAFI2 = rbind(CAFI2,CAFI3[CAFI3$PSP==10151,])
CAFI3[CAFI3$PSP==10151,] = NA
CAFI3 = CAFI3[rowSums(!is.na(CAFI3))!=0,]
CAFI3 = rbind(CAFI3,CAFI4[CAFI4$PSP==10151,])
CAFI4[CAFI4$PSP==10151,] = NA
CAFI4 = CAFI4[rowSums(!is.na(CAFI4))!=0,]

#Combine
MEAS_NO = 1
CAFI1 = data.frame(CAFI1,MEAS_NO)
MEAS_NO = 2
CAFI2 = data.frame(CAFI2,MEAS_NO)
MEAS_NO = 3
CAFI3 = data.frame(CAFI3,MEAS_NO)
MEAS_NO = 4
CAFI4 = data.frame(CAFI4,MEAS_NO)
MEAS_NO = 5
CAFI5 = data.frame(CAFI5,MEAS_NO)
CAFI_trees = rbind(CAFI1,CAFI2,CAFI3,CAFI4,CAFI5)

CAFI_trees = CAFI_trees[CAFI_trees$PSP %in% CAFI_sites$PSP,]

#Remove measurements 3 and 4 from PSP 10138 due to harvest
remove = which(CAFI_trees$PSP==10138&(CAFI_trees$MEAS_NO==3|CAFI_trees$MEAS_NO==4))
CAFI_trees = CAFI_trees[-remove,]

#Check for status issues
st_err = data.frame()
for (i in unique(CAFI_trees$TREE_ID)){
  status = CAFI_trees$STAT[CAFI_trees$TREE_ID==i]
  if (any(status=="NOT IN PLOT") | any(is.na(status))){
    names(status) = as.character(CAFI_trees$MEAS_NO[CAFI_trees$TREE_ID==i])
    status[is.na(status)] = NaN
    for (j in 1:length(status)){
      st_err[as.character(i),names(status)[j]] = status[j]
    }
  }
}
st_err2=st_err
for (i in 1:dim(st_err)[1]){
  if (any(st_err[i,]=="NOT IN PLOT",na.rm=T)){
    st_err[i,] = NA
  }else if (any(st_err[i,]=="NaN")){
    if (any(!is.na(st_err[i,c((which(st_err[i,]=="NaN")+1):5)]))){
      st_err[i,which(st_err[i,]=="NaN")] = st_err[i,which(!is.na(st_err[i,c((which(st_err[i,]=="NaN")+1):5)]))+which(st_err[i,]=="NaN")]
    }else{
      st_err[i,which(st_err[i,]=="NaN")] = "2"
    }
  }
}
for(i in 1:dim(st_err)[1]){
  for (j in which(!is.na(st_err2[i,]))){
    CAFI_trees$STAT[CAFI_trees$TREE_ID==rownames(st_err)[i]&CAFI_trees$MEAS_NO==colnames(st_err)[j]] = st_err[i,j]
  }
}
CAFI_trees = CAFI_trees[!is.na(CAFI_trees$STAT),]

#Adjust status indicators
CAFI_trees[CAFI_trees$STAT=="13","STAT"] = "0" #Adjusting those indicated with just an incorrect height to "live" since I don't use height in this
CAFI_trees[CAFI_trees$STAT=="MISSING","STAT"] = "2" #Change missing to dead
CAFI_trees[CAFI_trees$STAT=="12+13"|CAFI_trees$STAT=="12 + 13"|CAFI_trees$STAT=="12 A 13"|CAFI_trees$STAT=="12, 13"|CAFI_trees$STAT=="12 13","STAT"] = "12" #Change anything with height and dbh issues to just dbh issues
CAFI_trees[CAFI_trees$STAT=="2 10"|CAFI_trees$STAT=="2   10","STAT"] = "2" #Change "2 10" to dead
CAFI_trees[CAFI_trees$STAT=="3"|CAFI_trees$STAT=="4"|CAFI_trees$STAT=="10"|CAFI_trees$STAT=="11"|CAFI_trees$STAT=="8","STAT"] = "0" #stats 3,4 and 8 are just referring to ingrowth, which I don't need to differentiate at the moment. stats 10 and 1 appear to have had some postprocessing that fixed errors

#check dbh errors
dbh_err = data.frame()
for (i in unique(CAFI_trees$TREE_ID)){
  if (any(CAFI_trees$STAT[CAFI_trees$TREE_ID==i]=="12")){
    dbh = CAFI_trees$DBH[CAFI_trees$TREE_ID==i]
    names(dbh) = CAFI_trees$MEAS_NO[CAFI_trees$TREE_ID==i]
    for (j in 1:length(dbh)){
      dbh_err[as.character(i),names(dbh)[j]] = dbh[j]
    }
    dbh_err[as.character(i),"6"] = paste0(which(CAFI_trees$STAT[CAFI_trees$TREE_ID==i]=="12"),collapse = ",")
  }
}
dbh_err_value = matrix(nrow = dim(dbh_err)[1],ncol = dim(dbh_err)[2]-2)
for (i in 1:dim(dbh_err)[1]){
  for (j in 2:5){
    dbh_err_value[i,j-1] = dbh_err[i,j]-dbh_err[i,j-1]
  }
}
under1 = vector(length=dim(dbh_err)[1])
for (i in 1:length(under1)){
  if(any(dbh_err_value[i,]<(-0.5),na.rm = T)){
    under1[i]=1
  }
}
dbh_err = dbh_err[under1==1,]
for (i in 1:dim(dbh_err)[1]){
  if(nchar(dbh_err[i,6])==1){
    if(dbh_err[i,6]!=colnames(dbh_err[i,!is.na(dbh_err[i,])])[1]){
      dbh_err[i,dbh_err[i,6]] = dbh_err[i,as.numeric(dbh_err[i,6])-1]
    }else if (dbh_err[i,6]==colnames(dbh_err[i,!is.na(dbh_err[i,])])[1]){
      dbh_err[i,dbh_err[i,6]] = dbh_err[i,as.numeric(dbh_err[i,6])+1]
    }
  }else if(strsplit(dbh_err[i,6],",")[[1]][1]==colnames(dbh_err[i,!is.na(dbh_err[i,])])[1]){
    dbh_err[i,strsplit(dbh_err[i,6],",")[[1]][1]] = dbh_err[i,as.numeric(strsplit(dbh_err[i,6],",")[[1]][1])+1]
    dbh_err[i,strsplit(dbh_err[i,6],",")[[1]][2]] = dbh_err[i,as.numeric(strsplit(dbh_err[i,6],",")[[1]][2])-1]
  }else{
    for (j in 1:length(strsplit(dbh_err[i,6],",")[[1]])){
      dbh_err[i,strsplit(dbh_err[i,6],",")[[1]][j]] = dbh_err[i,as.numeric(strsplit(dbh_err[i,6],",")[[1]][j])-1]
    }
  }
}
dbh_err = dbh_err[,1:5]
#Fix dbh errors
for (i in 1:dim(dbh_err)[1]){
  indices = colnames(dbh_err[i,!is.na(dbh_err[i,])])
  for (j in indices){
    CAFI_trees$DBH[CAFI_trees$TREE_ID==rownames(dbh_err)[i]&CAFI_trees$MEAS_NO==j] = dbh_err[i,j]
  }
}
#Assuming this fixes the stat 12 issue, so changing to 0
CAFI_trees[CAFI_trees$STAT=="12","STAT"] = "0"

#Find species errors
sp_err = data.frame()
for (i in unique(CAFI_trees$TREE_ID)){
  sp = CAFI_trees$SPCD[CAFI_trees$TREE_ID==i]
  names(sp) = CAFI_trees$MEAS_NO[CAFI_trees$TREE_ID==i]
  if(length(unique(sp))>1){
    for (j in 1:length(sp)){
      sp_err[as.character(i),names(sp)[j]] = sp[j]
    }
  }
}


sp_err2=sp_err
#changing the sp IDs of the errors. This makes a few assumptions:
#1) Later identifications are more accurate
#2) If a tree was identified as one species more often than another, it is likely that species
for (i in 1:dim(sp_err)[1]){
  indices = colnames(sp_err[i,!is.na(sp_err[i,])])
  sp_vect = vector()
  for (n in indices){
    sp_vect[n] = sp_err[i,n]
  }
  if(length(unique(sp_vect[!is.na(sp_vect)]))>1){
    if (sum(sp_vect==unique(sp_vect)[1])==sum(sp_vect==unique(sp_vect)[2])){
      sp = unique(sp_vect)[2]
      sp_err[i,names(sp_vect)] = sp
    }else if(any(duplicated(sp_vect))){
      sp = sp_vect[duplicated(sp_vect)]
      if (length(unique(sp))==1){
        sp = unique(sp)
        sp_err[i,names(sp_vect)] = sp
      }else{
        sp = sp[duplicated(sp)]
        if (length(unique(sp))==1){
          sp = unique(sp)
          sp_err[i,names(sp_vect)] = sp
        }
      }
    }
  }
}

err_check = vector()
for (n in 1:dim(sp_err)[1]){
  for (i in 1:5){
    err_check[i] = sp_err[n,i]
  }
  if (length(unique(err_check[is.na(err_check)]))>1){
    print(rownames(sp_err)[n])
  }
}

#correct species in CAFI_tree
for (i in 1:dim(sp_err)[1]){
  indices = colnames(sp_err[i,!is.na(sp_err[i,])])
  for (j in indices){
    CAFI_trees$SPCD[CAFI_trees$TREE_ID==rownames(sp_err)[i]&CAFI_trees$MEAS_NO==j] = sp_err[i,j]
  }
}

plot_vect = unique(CAFI_trees$PSP)

cut_percent = matrix(ncol = length(unique(CAFI_trees$MEAS_NO)),nrow = length(plot_vect))
rownames(cut_percent) = plot_vect
colnames(cut_percent) = paste0("t",unique(CAFI_trees$MEAS_NO))


damage = matrix(nrow = length(plot_vect),ncol = 6*5)
rownames(damage) = plot_vect
dams = c("Disease/pathogen","Insect","Fire","Drought","Human","Other")
colnames(damage) = c(paste0(dams,"_t1"),paste0(dams,"_t2"),paste0(dams,"_t3"),paste0(dams,"_t4"),paste0(dams,"_t5"))
dam_codes = list(
  "Disease/pathogen" = c(20:49),
  "Insect" = c(50:59),
  "Fire" = c(70:79),
  "Drought" = c(84),
  "Human" = c(10:19),
  "Other" = c(0,60:69,80:83,85:99)
)

#Create consolidated csvs
for (i in plot_vect){
  temp = CAFI_trees[CAFI_trees$PSP==i,]
  temp = arrange(temp,MEAS_NO)
  if(i==10200){
    temp[683,"TREE_ID"] = 400490
  }else if(i==10369){
    temp[c(106,202),"TREE_ID"] = 123038
  }
  samp_vect = unique(temp$MEAS_NO)
  
  tree_range = unique(temp$TREE_ID)
  tree_range = sort(tree_range)
  df = matrix(nrow = length(tree_range),ncol = length(samp_vect)*2)
  sp_df = matrix(nrow = length(tree_range),ncol = length(samp_vect))
  rownames(df) = tree_range
  colnames(df) = c(paste0("dbh_t",1:length(samp_vect)),paste0("status_t",1:length(samp_vect)))
  colnames(sp_df) = paste0("sp_t",1:length(samp_vect))
  rownames(sp_df) = tree_range
  for (n in 1:length(samp_vect)){
    temp_samp = temp[temp$MEAS_NO==samp_vect[n],]
    if(!identical(temp_samp$TREE_ID,tree_range)){
      add = tree_range[!tree_range %in% temp_samp$TREE_ID]
      add_mat = data.frame(matrix(ncol = dim(temp_samp)[2],nrow = length(add)))
      colnames(add_mat) = colnames(temp_samp)
      add_mat$TREE_ID = add
      temp_samp = rbind(temp_samp,add_mat)
      temp_samp = arrange(temp_samp,TREE_ID)
    }
    for (j in 1:dim(df)[1]){
      sp_df[j,paste0("sp_t",n)] = as.numeric(temp_samp$SPCD[temp_samp$TREE_ID==rownames(df)[j]])
      df[j,paste0("dbh_t",n)] = temp_samp$DBH[temp_samp$TREE_ID==rownames(df)[j]]*2.54
      if(!is.na(temp_samp$STAT[temp_samp$TREE_ID==rownames(df)[j]]) & (temp_samp$STAT[temp_samp$TREE_ID==rownames(df)[j]]=="0" | temp_samp$STAT[temp_samp$TREE_ID==rownames(df)[j]]=="1")){
        df[j,paste0("status_t",n)] = "L"
      }
    }
    cut_percent[as.character(i),paste0("t",samp_vect[n])] = sum(temp_samp$STAT=="1"|temp_samp$STAT=="5",na.rm=T)/sum(!is.na(temp_samp$STAT))
    flags = as.matrix(temp_samp[,c("DTY1","DTY2","DTY3","DTY4","DSP1","DSP2","DSP3","DSP4")])
    flags[is.na(flags)] = ""
    for(s in 1:dim(flags)[1]){
      flags[s,1] = paste0(flags[s,1],flags[s,5],collapse = "")
      flags[s,2] = paste0(flags[s,2],flags[s,6],collapse = "")
      flags[s,3] = paste0(flags[s,3],flags[s,7],collapse = "")
      flags[s,4] = paste0(flags[s,4],flags[s,8],collapse = "")
    }
    flags = flags[,c(1:4)]
    if(!empty(flags)){
      for (s in 1:length(dam_codes)){
        flags[flags %in% dam_codes[[s]]] = names(dam_codes)[s]
      }
      for (r in 1:dim(flags)[1]){
        flags[r,duplicated(flags[r,],incomparables = NA)] = NA
      }
      for(j in 1:length(dam_codes)){
        damage[as.character(i),paste0(names(dam_codes)[j],"_t",n)] = sum(flags==names(dam_codes)[j],na.rm=T)/sum(!is.na(temp_samp$STAT))
      }
    }
  }
  sp_df2 = sp_df
  for (j in 1:dim(LUT)[1]){
    sp_df2[sp_df==as.numeric(LUT[j,"CAFI"])] = LUT[j,1]
  }
  plot_size = 0.0404686
  df = data.frame(sp_df2,df,plot_size)
  write.csv(df,paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CAFI/checks/",i,"_check.csv"))
}
write.csv(cut_percent,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CAFI/CAFI_cut_percent.csv")
write.csv(damage,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CAFI/CAFI_flags_percent.csv")
