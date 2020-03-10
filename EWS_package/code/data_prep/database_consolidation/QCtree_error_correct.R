library(readxl)
library(dplyr)

LUT = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SP_LUT.csv",stringsAsFactors = F)
QC_tree <- read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/QC/DENDRO_ARBRES.csv",stringsAsFactors = F)
QC_meas = read_excel("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/QC/PLACETTE_MES.xlsx")

#Remove trees that are indicated as outside each plot
outside_plot = unique(as.numeric(QC_tree[QC_tree$IN_1410=="O",6]))
QC_tree = QC_tree[!QC_tree$ID_ARBRE %in% outside_plot,]

plot_vect = unique(QC_tree$ID_PE)
QC_tree = arrange(QC_tree,NO_MES)

timestamp()
#Find species errors
sp_err = data.frame()
for (i in unique(QC_tree$ID_ARBRE)){
  if(length(QC_tree$ESSENCE[QC_tree$ID_ARBRE==i])>1){
    sp = QC_tree$ESSENCE[QC_tree$ID_ARBRE==i]
    if (length(unique(sp[sp!=""]))>1){
      names(sp) = QC_tree$NO_MES[QC_tree$ID_ARBRE==i]
      if(length(unique(sp))>1){
        for (j in 1:length(sp)){
          sp_err[as.character(i),names(sp)[j]] = sp[j]
        }
      }
    }
  }
  #print(paste0("sp1",i))
}
sp_err = sp_err[,paste0(1:dim(sp_err)[2])]
print("species errors found")
timestamp()
#changing the sp IDs of the errors. This makes a few assumptions:
#1) Later identifications are more accurate
#2) If a tree was identified as one species more often than another, it is likey that species
for (i in 1:dim(sp_err)[1]){
  indices = colnames(sp_err[i,!is.na(sp_err[i,])])
  sp_vect = vector()
  for (n in indices){
    sp_vect[n] = sp_err[i,n]
  }
  if (any(sp_vect=="")){
    if (length(unique(sp_vect[sp_vect!=""]))==1){
      sp_err[i,sp_vect==""] = unique(sp_vect[sp_vect!=""])
      sp_vect[sp_vect==""] = unique(sp_vect[sp_vect!=""])
    }
  }
  if (length(unique(sp_vect))>1){
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
      }else if(sum(sp==unique(sp)[1])==sum(sp==unique(sp)[2])){
        sp = unique(sp)[2]
        sp_err[i,names(sp_vect)] = sp
      }
    }
  }
  #print(paste0("sp2",i))
}
print("species errors changed")
timestamp()
err_check = vector()
for (n in 1:dim(sp_err)[1]){
  for (i in 1:6){
    err_check[i] = sp_err[n,i]
  }
  if (length(unique(err_check[is.na(err_check)]))>1){
    print(rownames(sp_err)[n])
  }
}

#correct species in QC_tree
for (i in 1:dim(sp_err)[1]){
  indices = paste0(rownames(sp_err)[i],0,colnames(sp_err)[which(!is.na(sp_err[i,]))])
  for (n in 1:length(indices)){
    QC_tree$ESSENCE[QC_tree$ID_ARB_MES==as.numeric(indices[n])] = sp_err[i,n]
    #QC_tree$ESSENCE[QC_tree$ID_ARBRE==rownames(sp_err)[i]&QC_tree$NO_MES==n] = sp_err[i,n]
  }
  #print(paste0("sp3",i))
}
print("species done")
timestamp()
#Check for "" status issues
st_err = data.frame()
dbh = data.frame()
for (i in unique(QC_tree$ID_ARBRE)){
  status = QC_tree$ETAT[QC_tree$ID_ARBRE==i]
  db = QC_tree$DHP[QC_tree$ID_ARBRE==i]
  if (any(status=="")){
    names(status) = as.character(QC_tree$NO_MES[QC_tree$ID_ARBRE==i])
    status[is.na(status)] = NaN
    for (j in 1:length(status)){
      st_err[as.character(i),names(status)[j]] = status[j]
      dbh[as.character(i),names(status)[j]] = db[j]
    }
  }
  #print(paste0("st1",i))
}
print("status errors found")
timestamp()
st_err = st_err[,paste0(1:dim(st_err)[2])]
dbh = dbh[,paste0(1:dim(dbh)[2])]
st_err2=st_err
dbh2=dbh
gaul = c("GA","GM","GV")
#Change status ommissions, this basically is changing everything with a missing dbh to dead, a dbh<90 to "5" which I made up and a dbh>9 to "10"-alive
for (i in 1:dim(st_err)[1]){
  if (any(st_err[i,]=="")){
    if (sum(st_err[i,]=="",na.rm = T)==1){
      if (any(!is.na(st_err[i,c((which(st_err[i,]=="")+1):dim(st_err)[2])]))){
        if(st_err[i,which(st_err[i,]=="")+1]=="40"|any(st_err[i,which(st_err[i,]=="")+1]==gaul)){
          st_err[i,which(st_err[i,]=="")] = "5"
        }else{
          for(j in which(st_err[i,]=="")){
            if (is.na(dbh[i,j])){
              st_err[i,j] = "14"
            }else if(dbh[i,j]<91){
              st_err[i,j] = "5"
            }else if (dbh[i,j]>90){
              st_err[i,j] = "10"
            }
          }
        }
      }else if(!any(st_err[i,!is.na(st_err[i,])]!="")){
        for(j in which(st_err[i,]=="")){
          if (!is.na(dbh[i,j])&dbh[i,j]<91){
            st_err[i,j] = "5"
          }else if(is.na(dbh[i,j])){
            st_err[i,j] = "14"
          }
        }
      }else{
        for(j in which(st_err[i,]=="")){
          if (is.na(dbh[i,j])){
            st_err[i,j] = "14"
          }else if(dbh[i,j]<91){
            st_err[i,j] = "5"
          }else if (dbh[i,j]>90){
            st_err[i,j] = "10"
          }
        }
      }
    }else if(sum(st_err[i,]=="",na.rm = T)>1){
      if(which(st_err[i,]=="")[length(which(st_err[i,]==""))]!=dim(st_err)[2]){
        if (any(!is.na(st_err[i,c((which(st_err[i,]=="")[length(which(st_err[i,]==""))]+1):dim(st_err)[2])]))){
          if(st_err[i,which(st_err[i,]=="")[length(which(st_err[i,]==""))]+1]=="40"|any(st_err[i,which(st_err[i,]=="")[length(which(st_err[i,]==""))]+1]==gaul)){
            st_err[i,which(st_err[i,]=="")[1:length(which(st_err[i,]==""))]] = "5"
          }else{
            for(j in which(st_err[i,]=="")){
              if (is.na(dbh[i,j])){
                st_err[i,j] = "14"
              }else if(dbh[i,j]<91){
                st_err[i,j] = "5"
              }else if (dbh[i,j]>90){
                st_err[i,j] = "10"
              }
            }
          }
        }else if (!any(st_err[i,!is.na(st_err[i,])]!="")){
          for(j in which(st_err[i,]=="")){
            if (!is.na(dbh[i,j])&dbh[i,j]<91){
              st_err[i,j] = "5"
            }else if(is.na(dbh[i,j])){
              st_err[i,j] = "14"
            }
          }
        }else{
          for(j in which(st_err[i,]=="")){
            if (is.na(dbh[i,j])){
              st_err[i,j] = "14"
            }else if(dbh[i,j]<91){
              st_err[i,j] = "5"
            }else if (dbh[i,j]>90){
              st_err[i,j] = "10"
            }
          }
        }
      }else{
        for (j in which(st_err[i,]=="")){
          if (is.na(dbh[i,j])){
            st_err[i,j] = "14"
          }else if(dbh[i,j]<91){
            st_err[i,j] = "5"
          }else if (dbh[i,j]>90){
            st_err[i,j] = "10"
          }
        }
      }
    }
  }
  #print(paste0("st2",i))
}
print("status errors changed")
timestamp()
#Fix status errors
for (i in 1:dim(st_err)[1]){
  indices = paste0(rownames(st_err)[i],0,colnames(st_err)[which(!is.na(st_err[i,]))])
  for (n in 1:length(indices)){
    QC_tree$ETAT[QC_tree$ID_ARB_MES==as.numeric(indices[n])] = st_err[i,n]
  # indices = colnames(st_err)[which(!is.na(st_err[i,]))]
  # for (n in indices){
  #   QC_tree$ETAT[QC_tree$ID_ARBRE==rownames(st_err)[i]&QC_tree$NO_MES==n] = st_err[i,n]
  }
  #print(paste0("st3",i))
}
print("status done")


timestamp()
#check dbh errors
dbh_err = data.frame()
for (i in unique(QC_tree$ID_ARBRE)){
  if (any(QC_tree$DHP_NC[QC_tree$ID_ARBRE==i]!="")){
    dbh = QC_tree$DHP[QC_tree$ID_ARBRE==i]
    names(dbh) = QC_tree$NO_MES[QC_tree$ID_ARBRE==i]
    for (j in 1:length(dbh)){
      dbh_err[as.character(i),names(dbh)[j]] = dbh[j]
    }
    dbh_err[as.character(i),"8"] = paste0(which(QC_tree$DHP_NC[QC_tree$ID_ARBRE==i]!=""),collapse = ",")
  }
  #print(paste0("dbh1",i))
}
print("dbh errors found")
dbh_err = dbh_err[,paste0(1:dim(dbh_err)[2])]
dbh_err2 = dbh_err

#Determine how much a dbh has changed between measurements
dbh_err_value = matrix(nrow = dim(dbh_err)[1],ncol = dim(dbh_err)[2]-2)
for (i in 1:dim(dbh_err)[1]){
  for (j in 2:7){
    dbh_err_value[i,j-1] = dbh_err[i,j]-dbh_err[i,j-1]
  }
}
#Choosing an arbitrary limit of 1 cm decrease between measurements to not allow. Anything other than that is ok. 
under1 = vector(length=dim(dbh_err)[1])
for (i in 1:length(under1)){
  if(any(dbh_err_value[i,]<(-10),na.rm = T)){
    under1[i]=1
  }
}
#change those that drop more than 1 cm between measurements. This uses a few assumptions:
#1)If a decrease by a significant amount is not followed by a subsequent increase, then the value before the decrease is wrong
#2)If a large decrease is followed by a subsequent increase, then the decreased value is wrong
#3)That increase needs to be at least the amount decreased for this to apply
#4)The opposite of the above, ie. if a large increase is followed by a large decrease, then the increase is wrong
#5)If only two measurements, I rely on the notes in the data for what is an error.
dbh_err_value=dbh_err_value[under1==1,]
dbh_err = dbh_err[under1==1,]
err_index = vector(length = dim(dbh_err)[1])
for (i in 1:dim(dbh_err_value)[1]){
  if(sum(!is.na(dbh_err_value[i,]))>1){
    err_index[i] = which(dbh_err_value[i,]<(-10))+1
    if(err_index[i]==which(!is.na(dbh_err_value[i,]))[1]+1&dbh_err_value[i,which(!is.na(dbh_err_value[i,]))[1]+1]<abs(dbh_err_value[i,which(!is.na(dbh_err_value[i,]))[1]])){
      err_index[i] = which(!is.na(dbh_err_value[i,]))[1]
    }else if(err_index[i]>2){
      if(!is.na(dbh_err_value[i,as.numeric(err_index[i])-2])&dbh_err_value[i,as.numeric(err_index[i])-2]>50){
        err_index[i] = which(dbh_err_value[i,]<(-10))
      }
    }
  }else{
    err_index[i] = as.character(as.numeric(dbh_err[i,8])+which(!is.na(dbh_err[i,]))[1]-1)
  }
}
for (i in 1:dim(dbh_err)[1]){
  if(err_index[i]!=colnames(dbh_err[i,!is.na(dbh_err[i,])])[1]){
    dbh_err[i,err_index[i]] = dbh_err[i,as.numeric(err_index[i])-1]
  }else if (err_index[i]==colnames(dbh_err[i,!is.na(dbh_err[i,])])[1]){
    dbh_err[i,err_index[i]] = dbh_err[i,as.numeric(err_index[i])+1]
  }
  #print(paste0("dbh2",i))
}
print("dbh errors changed")
timestamp()
dbh_err = dbh_err[,c(1:7)]
#correct dbh in QC_tree
for (i in 1:dim(dbh_err)[1]){
  indices = paste0(rownames(dbh_err)[i],0,colnames(dbh_err)[which(!is.na(dbh_err[i,]))])
  for (n in 1:length(indices)){
    QC_tree$DHP[QC_tree$ID_ARB_MES==as.numeric(indices[n])] = dbh_err[i,n]
  # indices = colnames(dbh_err)[which(!is.na(dbh_err[i,]))]
  # for (n in indices){
  #   QC_tree$DHP[QC_tree$ID_ARBRE==rownames(dbh_err)[i]&QC_tree$NO_MES==n] = dbh_err[i,n]
  }
  #print(paste0("dbh3",i))
}
print("dbh done")
timestamp()

#Setting empty species to NA, otherwise it sets them all to the first empty species code on the LUT, with is bigleaf maple
QC_tree$ESSENCE[QC_tree$ESSENCE==""] = NA

write.csv(QC_tree,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/QC/QC_tree_corrected.csv")