library(plyr)
library(readxl)
library(dplyr)

LUT = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SP_LUT.csv",stringsAsFactors = F)
#QC_tree <- read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/QC/DENDRO_ARBRES.csv",stringsAsFactors = F)
QC_meas = read_excel("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/QC/PLACETTE_MES.xlsx")
QC_meas = as.data.frame(QC_meas)
QC_tree = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/QC/QC_tree_corrected.csv",stringsAsFactors = F)
# #Remove trees that are indicated as outside each plot
# outside_plot = unique(as.numeric(QC_tree[QC_tree$IN_1410=="O",7]))
# for (i in 1:length(outside_plot)){
#   QC_tree = QC_tree[QC_tree$ID_ARBRE!=outside_plot[i],]
# }
#There are several codes that refer to something funky going on with the plots. Here I've removed measurements labeled as "abandoned", "reimplanted", and "not found"
QC_meas$STATUT_MES[is.na(QC_meas$STATUT_MES)] = "0"
QC_meas = QC_meas[QC_meas$STATUT_MES!="AB",]
#QC_meas = QC_meas[QC_meas$STATUT_MES!="RE",]
QC_meas = QC_meas[QC_meas$STATUT_MES!="NT",]

plot_vect = unique(QC_tree$ID_PE)
QC_tree = arrange(QC_tree,NO_MES)

# #Find species errors
# sp_err = data.frame()
# for (i in unique(QC_tree$ID_ARBRE)){
#   if(length(QC_tree$ESSENCE[QC_tree$ID_ARBRE==i])>1){
#     sp = QC_tree$ESSENCE[QC_tree$ID_ARBRE==i]
#     names(sp) = QC_tree$NO_MES[QC_tree$ID_ARBRE==i]
#     if(length(unique(sp))>1){
#       for (j in 1:length(sp)){
#         sp_err[as.character(i),names(sp)[j]] = sp[j]
#       }
#     }
#   }
# }
# sp_err = sp_err[,paste0(1:dim(sp_err)[2])]
# 
# #changing the sp IDs of the errors. This makes a few assumptions:
# #1) Later identifications are more accurate
# #2) If a tree was identified as one species more often than another, it is likey that species
# for (i in 1:dim(sp_err)[1]){
#   indices = colnames(sp_err[i,!is.na(sp_err[i,])])
#   sp_vect = vector()
#   for (n in indices){
#     sp_vect[n] = sp_err[i,n]
#   }
#   if (sum(sp_vect==unique(sp_vect)[1])==sum(sp_vect==unique(sp_vect)[2])){
#     sp = unique(sp_vect)[2]
#     sp_err[i,names(sp_vect)] = sp
#   }else if(any(duplicated(sp_vect))){
#     sp = sp_vect[duplicated(sp_vect)]
#     if (length(unique(sp))==1){
#       sp = unique(sp)
#       sp_err[i,names(sp_vect)] = sp
#     }else if(any(duplicated(sp))){
#       sp = sp[duplicated(sp)]
#       if (length(unique(sp))==1){
#         sp = unique(sp)
#         sp_err[i,names(sp_vect)] = sp
#       }
#     }else if(sum(sp==unique(sp)[1])==sum(sp==unique(sp)[2])){
#       sp = unique(sp)[2]
#       sp_err[i,names(sp_vect)] = sp
#     }
#   }
# }
# 
# err_check = vector()
# for (n in 1:dim(sp_err)[1]){
#   for (i in 1:6){
#     err_check[i] = sp_err[n,i]
#   }
#   if (length(unique(err_check[is.na(err_check)]))>1){
#     print(rownames(sp_err)[n])
#   }
# }
# 
# #correct species in QC_tree
# for (i in 1:dim(sp_err)[1]){
#   indices = colnames(sp_err)[which(!is.na(sp_err[i,]))]
#   for (n in indices){
#     QC_tree$ESSENCE[QC_tree$ID_ARBRE==rownames(sp_err)[i]&QC_tree$NO_MES==n] = sp_err[i,n]
#   }
# }
# print("species done")
# #Check for status issues
# st_err = data.frame()
# dbh = data.frame()
# for (i in unique(QC_tree$ID_ARBRE)){
#   status = QC_tree$ETAT[QC_tree$ID_ARBRE==i]
#   db = QC_tree$DHP[QC_tree$ID_ARBRE==i]
#   if (any(status=="")){
#     names(status) = as.character(QC_tree$NO_MES[QC_tree$ID_ARBRE==i])
#     status[is.na(status)] = NaN
#     for (j in 1:length(status)){
#       st_err[as.character(i),names(status)[j]] = status[j]
#       dbh[as.character(i),names(status)[j]] = db[j]
#     }
#   }
# }
# st_err = st_err[,paste0(1:dim(st_err)[2])]
# dbh = dbh[,paste0(1:dim(dbh)[2])]
# st_err2=st_err
# dbh2=dbh
# gaul = c("GA","GM","GV")
# #Change status ommissions, this basically is changing everything with a missing dbh to dead, a dbh<90 to "5" which I made up and a dbh>9 to "10"-alive
# for (i in 1:dim(st_err)[1]){
#   if (any(st_err[i,]=="")){
#     if (sum(st_err[i,]=="",na.rm = T)==1){
#       if (any(!is.na(st_err[i,c((which(st_err[i,]=="")+1):dim(st_err)[2])]))){
#         if(st_err[i,which(st_err[i,]=="")+1]=="40"|any(st_err[i,which(st_err[i,]=="")+1]==gaul)){
#           st_err[i,which(st_err[i,]=="")] = "5"
#         }else{
#           for(j in which(st_err[i,]=="")){
#             if (is.na(dbh[i,j])){
#               st_err[i,j] = "14"
#             }else if(dbh[i,j]<91){
#               st_err[i,j] = "5"
#             }else if (dbh[i,j]>90){
#               st_err[i,j] = "10"
#             }
#           }
#         }
#       }else if(!any(st_err[i,!is.na(st_err[i,])]!="")){
#         for(j in which(st_err[i,]=="")){
#           if (!is.na(dbh[i,j])&dbh[i,j]<91){
#             st_err[i,j] = "5"
#           }else if(is.na(dbh[i,j])){
#             st_err[i,j] = "14"
#           }
#         }
#       }else{
#         for(j in which(st_err[i,]=="")){
#           if (is.na(dbh[i,j])){
#             st_err[i,j] = "14"
#           }else if(dbh[i,j]<91){
#             st_err[i,j] = "5"
#           }else if (dbh[i,j]>90){
#             st_err[i,j] = "10"
#           }
#         }
#       }
#     }else if(sum(st_err[i,]=="",na.rm = T)>1){
#       if(which(st_err[i,]=="")[length(which(st_err[i,]==""))]!=dim(st_err)[2]){
#         if (any(!is.na(st_err[i,c((which(st_err[i,]=="")[length(which(st_err[i,]==""))]+1):dim(st_err)[2])]))){
#           if(st_err[i,which(st_err[i,]=="")[length(which(st_err[i,]==""))]+1]=="40"|any(st_err[i,which(st_err[i,]=="")[length(which(st_err[i,]==""))]+1]==gaul)){
#             st_err[i,which(st_err[i,]=="")[1:length(which(st_err[i,]==""))]] = "5"
#           }else{
#             for(j in which(st_err[i,]=="")){
#               if (is.na(dbh[i,j])){
#                 st_err[i,j] = "14"
#               }else if(dbh[i,j]<91){
#                 st_err[i,j] = "5"
#               }else if (dbh[i,j]>90){
#                 st_err[i,j] = "10"
#               }
#             }
#           }
#         }else if (!any(st_err[i,!is.na(st_err[i,])]!="")){
#           for(j in which(st_err[i,]=="")){
#             if (!is.na(dbh[i,j])&dbh[i,j]<91){
#               st_err[i,j] = "5"
#             }else if(is.na(dbh[i,j])){
#               st_err[i,j] = "14"
#             }
#           }
#         }else{
#           for(j in which(st_err[i,]=="")){
#             if (is.na(dbh[i,j])){
#               st_err[i,j] = "14"
#             }else if(dbh[i,j]<91){
#               st_err[i,j] = "5"
#             }else if (dbh[i,j]>90){
#               st_err[i,j] = "10"
#             }
#           }
#         }
#       }else{
#         for (j in which(st_err[i,]=="")){
#           if (is.na(dbh[i,j])){
#             st_err[i,j] = "14"
#           }else if(dbh[i,j]<91){
#             st_err[i,j] = "5"
#           }else if (dbh[i,j]>90){
#             st_err[i,j] = "10"
#           }
#         }
#       }
#     }
#   }
# }
# 
# #Fix status errors
# for (i in 1:dim(st_err)[1]){
#   indices = colnames(st_err)[which(!is.na(st_err[i,]))]
#   for (n in indices){
#     QC_tree$ETAT[QC_tree$ID_ARBRE==rownames(st_err)[i]&QC_tree$NO_MES==n] = st_err[i,n]
#   }
# }
# print("status done")
# #check dbh errors
# dbh_err = data.frame()
# for (i in unique(QC_tree$ID_ARBRE)){
#   if (any(QC_tree$DHP_NC[QC_tree$ID_ARBRE==i]!="")){
#     dbh = QC_tree$DHP[QC_tree$ID_ARBRE==i]
#     names(dbh) = QC_tree$NO_MES[QC_tree$ID_ARBRE==i]
#     for (j in 1:length(dbh)){
#       dbh_err[as.character(i),names(dbh)[j]] = dbh[j]
#     }
#     dbh_err[as.character(i),"8"] = paste0(which(QC_tree$DHP_NC[QC_tree$ID_ARBRE==i]!=""),collapse = ",")
#   }
# }
# dbh_err = dbh_err[,paste0(1:dim(dbh_err)[2])]
# dbh_err2 = dbh_err
# 
# #Determine how much a dbh has changed between measurements
# dbh_err_value = matrix(nrow = dim(dbh_err)[1],ncol = dim(dbh_err)[2]-2)
# for (i in 1:dim(dbh_err)[1]){
#   for (j in 2:7){
#     dbh_err_value[i,j-1] = dbh_err[i,j]-dbh_err[i,j-1]
#   }
# }
# #Choosing an arbitrary limit of 1 cm decrease between measurements to not allow. Anything other than that is ok. 
# under1 = vector(length=dim(dbh_err)[1])
# for (i in 1:length(under1)){
#   if(any(dbh_err_value[i,]<(-10),na.rm = T)){
#     under1[i]=1
#   }
# }
# #change those that drop more than 1 cm between measurements. This uses a few assumptions:
# #1)If a decrease by a significant amount is not followed by a subsequent increase, then the value before the decrease is wrong
# #2)If a large decrease is followed by a subsequent increase, then the decreased value is wrong
# #3)That increase needs to be at least the amount decreased for this to apply
# #4)The opposite of the above, ie. if a large increase is followed by a large decrease, then the increase is wrong
# #5)If only two measurements, I rely on the notes in the data for what is an error.
# dbh_err_value=dbh_err_value[under1==1,]
# dbh_err = dbh_err[under1==1,]
# err_index = vector(length = dim(dbh_err)[1])
# for (i in 1:dim(dbh_err_value)[1]){
#   if(sum(!is.na(dbh_err_value[i,]))>1){
#     err_index[i] = which(dbh_err_value[i,]<(-10))+1
#     if(err_index[i]==which(!is.na(dbh_err_value[i,]))[1]+1&dbh_err_value[i,which(!is.na(dbh_err_value[i,]))[1]+1]<abs(dbh_err_value[i,which(!is.na(dbh_err_value[i,]))[1]])){
#       err_index[i] = which(!is.na(dbh_err_value[i,]))[1]
#     }else if(err_index[i]>2){
#       if(!is.na(dbh_err_value[i,as.numeric(err_index[i])-2])&dbh_err_value[i,as.numeric(err_index[i])-2]>50){
#         err_index[i] = which(dbh_err_value[i,]<(-10))
#       }
#     }
#   }else{
#     err_index[i] = as.character(as.numeric(dbh_err[i,8])+which(!is.na(dbh_err[i,]))[1]-1)
#   }
# }
# for (i in 1:dim(dbh_err)[1]){
#   if(err_index[i]!=colnames(dbh_err[i,!is.na(dbh_err[i,])])[1]){
#     dbh_err[i,err_index[i]] = dbh_err[i,as.numeric(err_index[i])-1]
#   }else if (err_index[i]==colnames(dbh_err[i,!is.na(dbh_err[i,])])[1]){
#     dbh_err[i,err_index[i]] = dbh_err[i,as.numeric(err_index[i])+1]
#   }
# }
# dbh_err = dbh_err[,c(1:7)]
# #correct dbh in QC_tree
# for (i in 1:dim(dbh_err)[1]){
#   indices = colnames(dbh_err)[which(!is.na(dbh_err[i,]))]
#   for (n in indices){
#     QC_tree$DHP[QC_tree$ID_ARBRE==rownames(dbh_err)[i]&QC_tree$NO_MES==n] = dbh_err[i,n]
#   }
# }
# print("dbh done")

live = c("5","10","12","30","32","40","42","50","52","GV")

damage = matrix(nrow = length(plot_vect),ncol = 6*7)
rownames(damage) = plot_vect
dams = c("Disease/pathogen","Insect","Fire","Drought","Human","Other")
colnames(damage) = c(paste0(dams,"_t1"),paste0(dams,"_t2"),paste0(dams,"_t3"),paste0(dams,"_t4"),paste0(dams,"_t5"),paste0(dams,"_t6"),paste0(dams,"_t7"))
dam_codes = list(
  "Disease/pathogen" = c("P","D"),
  "Insect" = "I",
  "Fire" = c(),
  "Drought" = c(),
  "Human" = "H",
  "Other" = "C"
)

# #Setting empty species to NA, otherwise it sets them all to the first empty species code on the LUT, with is bigleaf maple
# QC_tree$ESSENCE[QC_tree$ESSENCE==""] = NA
surveys = data.frame()
#Create consolidated csvs
for (i in plot_vect){
  print(i)
  temp = QC_tree[QC_tree$ID_PE==i,]
  temp = arrange(temp,NO_MES)
  #Deal with some NA states that have dbhs and are thus likely alive
  NAs = temp[is.na(temp$ETAT),]
  if (!empty(NAs)){
    for (j in 1:dim(NAs)[1]){
      if (!is.na(NAs$DHP[j])){
        NAs$ETAT[j] = "10"
        temp$ETAT[temp$ID_ARB_MES==NAs$ID_ARB_MES[j]] = "10"
      }
    }
  }
  #Removing any subsequent measurements after a site has been replanted
  if(any(QC_meas$STATUT_MES[QC_meas$ID_PE==sprintf("%010.0f",i)]=="RE",na.rm=T)){
    temp = temp[temp$NO_MES<which(QC_meas$STATUT_MES[QC_meas$ID_PE==sprintf("%010.0f",i)]=="RE")+1,]
  }
  samp_vect = unique(temp$NO_MES)
  samp_vect = samp_vect[samp_vect %in% QC_meas$NO_MES[QC_meas$ID_PE==sprintf("%010.0f",i)]]
  if (length(samp_vect)>0){
    for (j in 1:length(samp_vect)){
      surveys[as.character(i),paste0("t_",j)] = strsplit(as.character(QC_meas$DATE_SOND[QC_meas$ID_PE_MES==paste0(sprintf("%010.0f",i),0,samp_vect[j])]),"-")[[1]][1]
    }
    
    tree_range = sort(unique(temp$NO_ARBRE))
    df = matrix(nrow = length(tree_range),ncol = length(samp_vect)*3)
    rownames(df) = tree_range
    colnames(df) = c(paste0("sp_t",1:length(samp_vect)),paste0("dbh_t",1:length(samp_vect)),paste0("status_t",1:length(samp_vect)))
    for (n in 1:length(samp_vect)){
      temp_samp = temp[temp$NO_MES==samp_vect[n],]
      if(!identical(temp_samp$NO_ARBRE,tree_range)){
        add = tree_range[!tree_range %in% temp_samp$NO_ARBRE]
        add_mat = data.frame(matrix(ncol = dim(temp_samp)[2],nrow = length(add)))
        colnames(add_mat) = colnames(temp_samp)
        add_mat$NO_ARBRE = add
        temp_samp = rbind(temp_samp,add_mat)
        temp_samp = arrange(temp_samp,NO_ARBRE)
      }
      for (j in 1:dim(df)[1]){
        df[j,paste0("sp_t",n)] = temp_samp$ESSENCE[temp_samp$NO_ARBRE==rownames(df)[j]]
        df[j,paste0("dbh_t",n)] = temp_samp$DHP[temp_samp$NO_ARBRE==rownames(df)[j]]/10
        if(!is.na(temp_samp$ETAT[temp_samp$NO_ARBRE==rownames(df)[j]]) & any(temp_samp$ETAT[temp_samp$NO_ARBRE==rownames(df)[j]]==live)){
          df[j,paste0("status_t",n)] = "L"
        }
      }
      flags = temp_samp$CAUS_DEFOL
      
      for (s in 1:length(dam_codes)){
        flags[flags %in% dam_codes[[s]]] = names(dam_codes)[s]
      }
      for(j in 1:length(dam_codes)){
        damage[as.character(i),paste0(names(dam_codes)[j],"_t",n)] = (sum(flags==names(dam_codes)[j],na.rm=T))/length(flags)
      }
    }
    for (j in 1:dim(LUT)[1]){
      df[df==LUT[j,"QC"]] = LUT[j,1]
    }
    plot_size = 0.04
    df = data.frame(df,plot_size)
    write.csv(df,paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/QC/checks/",i,"_check.csv"))
  }
}
damage = damage[rowSums(!is.na(damage))>0,]
write.csv(damage,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/QC/QC_flags_percent.csv")
write.csv(surveys,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/QC/QC_surveys.csv")