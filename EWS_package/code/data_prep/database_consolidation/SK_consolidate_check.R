
library(dplyr)

LUT = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SP_LUT.csv",stringsAsFactors = F)
SK_tree <- read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SK/SK_trees.txt",stringsAsFactors = F)
SK_sites = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SK/SK_PSPs_Query.txt", stringsAsFactors = F)

# #Get rid of sites that were only measured before the 1980s
# SK_sites = arrange(SK_sites,MEASUREMENT_YEAR)
# sites = unique(SK_sites$measurement_header_PLOT_ID)
# surveys = matrix(ncol = 9,nrow = length(sites))
# for (i in 1:length(sites)){
#   years = SK_sites$MEASUREMENT_YEAR[SK_sites$measurement_header_PLOT_ID==sites[i]]
#   surveys[i,c(1:length(years))] = years
# }
# surveys = data.frame(sites,surveys)
# for(i in 1:length(sites)){
#   if (max(surveys[i,c(2:10)],na.rm=T)<1980){
#     surveys[i,1] = NA
#   }
# }
# surveys = surveys[!is.na(surveys[,1]),]
# surveys = surveys[!is.na(surveys[,3]),]
# SK_tree = SK_tree[SK_tree$PLOT_ID %in% surveys$sites,]

#Get rid of trees listed as Artifical tree. Not sure what this means, but I don't like it and it only applys to trees before 1961, so I don't need those anyways.
SK_tree = SK_tree[SK_tree$OFFICE_ERROR!="Artificial Tree",]


#Adjust tree names
SK_tree$TREE_NO = as.character(SK_tree$TREE_NO)
SK_tree[nchar(SK_tree$TREE_NO)==5,"TREE_NO"] = substr(SK_tree$TREE_NO[nchar(SK_tree$TREE_NO)==5],2,5)
SK_tree$TREE_NO = as.numeric(SK_tree$TREE_NO)

#A bunch of years weren't recorded, but the year is in the measurement number
no_year = SK_tree[is.na(SK_tree$YEAR),]
for (i in 1:dim(no_year)[1]){
  no_year$YEAR[i] = as.numeric(gsub(no_year$PLOT_ID[i],"",no_year$PLOT_MS[i]))
}
SK_tree[rownames(no_year),"YEAR"] = no_year$YEAR

SK_tree$YEAR[SK_tree$YEAR==2044] = 2011


SK_tree$SPECIES[SK_tree$SPECIES=="UI"] = NA
SK_tree$SPECIES[SK_tree$SPECIES=="XX"] = NA
SK_tree$SPECIES[SK_tree$SPECIES=="DC"] = NA
SK_tree$SPECIES[SK_tree$SPECIES=="DD"] = NA
SK_tree$SPECIES[SK_tree$SPECIES=="DU"] = NA
SK_tree$SPECIES[SK_tree$SPECIES==""] = NA
SK_tree$SPECIES[is.na(SK_tree$DBH)] = NA
SK_tree$SPECIES[is.na(SK_tree$TREE_STATUS)] = NA

#There are some missing statuses and species, but existing condition codes, so I'm using them to fill in the missing status
no_status = SK_tree[is.na(SK_tree$TREE_STATUS),]
dead = c(25,26,27,28,29)
for (i in 1:dim(no_status)[1]){
  if(any(no_status[i,c(15,16,17)] %in% dead)){
    no_status$TREE_STATUS[i] = 3
  }else if(sum(!is.na(no_status[i,c(15,16,17)]))==0){
    no_status$TREE_STATUS[i] = 3
  }else if(any(SK_tree$TREE_STATUS[SK_tree$PLOT_ID==no_status$PLOT_ID[i]&SK_tree$TREE_NO==no_status$TREE_NO[i]]=="3",na.rm=T)){
    no_status$TREE_STATUS[i] = 3
  }else{
    no_status$TREE_STATUS[i] = 1
  }
  if(no_status$TREE_STATUS[i]==1){
    if(length(SK_tree$SPECIES[SK_tree$PLOT_ID==no_status$PLOT_ID[i]&SK_tree$TREE_NO==no_status$TREE_NO[i]][!is.na(SK_tree$SPECIES[SK_tree$PLOT_ID==no_status$PLOT_ID[i]&SK_tree$TREE_NO==no_status$TREE_NO[i]])])>0){
      no_status$SPECIES[i] = unique(SK_tree$SPECIES[SK_tree$PLOT_ID==no_status$PLOT_ID[i]&SK_tree$TREE_NO==no_status$TREE_NO[i]][!is.na(SK_tree$SPECIES[SK_tree$PLOT_ID==no_status$PLOT_ID[i]&SK_tree$TREE_NO==no_status$TREE_NO[i]])])
    }
  }
}
for (i in 1:dim(no_status)[1]){
  SK_tree$TREE_STATUS[SK_tree$PLOT_MS==no_status$PLOT_MS[i]&SK_tree$TREE_NO==no_status$TREE_NO[i]] = no_status[i,"TREE_STATUS"]
  SK_tree$SPECIES[SK_tree$PLOT_MS==no_status$PLOT_MS[i]&SK_tree$TREE_NO==no_status$TREE_NO[i]] = no_status[i,"SPECIES"]
}


plot_vect = unique(SK_tree$PLOT_ID)

#Find species errors
sp_err = data.frame()
for (i in plot_vect){
  temp = SK_tree[SK_tree$PLOT_ID==i,]
  temp = arrange(temp,YEAR)
  samp_vect = unique(temp$YEAR)
  for (n in unique(temp$TREE_NO)){
    sp = temp$SPECIES[temp$TREE_NO==n]
    for (j in 1:length(sp)){
      names(sp)[j] = which(samp_vect==temp$YEAR[temp$TREE_NO==n][j])
    }
    sp = sp[!is.na(sp)]
    if (length(unique(sp))>1){
      for (j in 1:length(sp)){
        sp_err[paste0(i,"_",n),names(sp)[j]] = sp[j]
      }
      
    }
  }
}

#some of these errors are a result of reusing tree numbers, this filters out those trees
sp_err = sp_err[,sort(colnames(sp_err))]
for (i in 1:dim(sp_err)[1]){
  nans = !is.na(sp_err[i,])
  if(!identical(which(nans),min(which(nans)):max(which(nans)))){
    sp_err[i,] = NA
  }
}
sp_err = sp_err[rowSums(!is.na(sp_err))!=0,]

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
  if (sum(sp_vect==unique(sp_vect)[1])==sum(sp_vect==unique(sp_vect)[2])){
    sp = unique(sp_vect)[2]
    sp_err[i,names(sp_vect)] = sp
  }else if(any(duplicated(sp_vect))){
    sp = sp_vect[duplicated(sp_vect)]
    if (length(unique(sp))==1){
      sp = unique(sp)
      sp_err[i,names(sp_vect)] = sp
    }
  }
}

#correct species in SK_tree
for (i in 1:dim(sp_err)[1]){
  indices = colnames(sp_err[i,!is.na(sp_err[i,])])
  samp_vect = unique(SK_tree$YEAR[SK_tree$PLOT_ID==strsplit(rownames(sp_err)[i],"_")[[1]][1]])
  for (n in indices){
    SK_tree$SPECIES[SK_tree$PLOT_ID==strsplit(rownames(sp_err)[i],"_")[[1]][1]&SK_tree$TREE_NO==strsplit(rownames(sp_err)[i],"_")[[1]][2]
                                &SK_tree$YEAR==samp_vect[as.numeric(n)]] = sp_err[i,n]
  }
}

cut_percent = matrix(ncol = 8,nrow = length(plot_vect))
rownames(cut_percent) = plot_vect
colnames(cut_percent) = paste0("t",1:8)

damage = matrix(nrow = length(plot_vect),ncol = 8*6)
rownames(damage) = plot_vect
dams = c("Disease/pathogen","Insect","Fire","Drought","Human","Other")
colnames(damage) = c(paste0(dams,"_t1"),paste0(dams,"_t2"),paste0(dams,"_t3"),paste0(dams,"_t4"),paste0(dams,"_t5"),paste0(dams,"_t6"),paste0(dams,"_t7"),paste0(dams,"_t8"))
dam_codes = list(
  "Disease/pathogen" = c(3,5,8,12,88,1,4),
  "Insect" = c(9,21),
  "Fire" = c(),
  "Drought" = c(),
  "Human" = c(20,29),
  "Other" = c(11,40,51)
)
mort_codes = list(
  "Disease" = 2,
  "Insect" = 3,
  "Human" = 4,
  "Other" = c(5,6,8)
)
surveys = data.frame()

#Create consolidated csvs
for (i in plot_vect){
  temp = SK_tree[SK_tree$PLOT_ID==i,]
  temp = arrange(temp,YEAR)
  
  samp_vect = unique(temp$YEAR)
  
  for (n in 1:length(samp_vect)){
    surveys[paste0("3_",i),paste0("t",n)] = samp_vect[n]
  }
  tree_range = unique(temp$TREE_NO)
  tree_range = sort(tree_range)
  df = matrix(nrow = length(tree_range),ncol = length(samp_vect)*3)
  rownames(df) = tree_range
  colnames(df) = c(paste0("sp_t",1:length(samp_vect)),paste0("dbh_t",1:length(samp_vect)),paste0("status_t",1:length(samp_vect)))
  for (n in 1:length(samp_vect)){
    temp_samp = temp[temp$YEAR==samp_vect[n],]
    if(!identical(temp_samp$TREE_NO,tree_range)){
      add = tree_range[!tree_range %in% temp_samp$TREE_NO]
      add_mat = data.frame(matrix(ncol = dim(temp_samp)[2],nrow = length(add)))
      colnames(add_mat) = colnames(temp_samp)
      add_mat$TREE_NO = add
      temp_samp = rbind(temp_samp,add_mat)
      temp_samp = arrange(temp_samp,TREE_NO)
    }
    for (j in 1:dim(df)[1]){
      df[j,paste0("sp_t",n)] = temp_samp$SPECIES[temp_samp$TREE_NO==rownames(df)[j]]
      df[j,paste0("dbh_t",n)] = temp_samp$DBH[temp_samp$TREE_NO==rownames(df)[j]]
      if(!is.na(temp_samp$TREE_STATUS[temp_samp$TREE_NO==rownames(df)[j]]) & (temp_samp$TREE_STATUS[temp_samp$TREE_NO==rownames(df)[j]]==1 | temp_samp$TREE_STATUS[temp_samp$TREE_NO==rownames(df)[j]]==2)){
        df[j,paste0("status_t",n)] = "L"
      }
    }
    cut_percent[as.character(i),n] = sum(temp_samp$CONDITION_CODE1==29|temp_samp$CONDITION_CODE2==29|temp_samp$CONDITION_CODE3==29,na.rm=T)/sum(!is.na(temp_samp$TREE_STATUS))
    flags = as.matrix(temp_samp[,c("CONDITION_CODE1","CONDITION_CODE2","CONDITION_CODE3")])
    mort = temp_samp[,"MORTALITY"]
    for (s in 1:length(dam_codes)){
      flags[flags %in% dam_codes[[s]]] = names(dam_codes)[s]
    }
    for (s in 1:length(mort_codes)){
      mort[mort %in% mort_codes[[s]]] = names(mort_codes)[s]
    }
    flags = cbind(flags,mort)
    for (r in 1:dim(flags)[1]){
      flags[r,duplicated(flags[r,],incomparables = NA)] = NA
    }
    for(j in 1:6){
      damage[as.character(i),paste0(names(dam_codes)[j],"_t",n)] = (sum(flags[,1]==names(dam_codes)[j],na.rm=T)+sum(flags[,2]==names(dam_codes)[j],na.rm=T)
                                                                    +sum(flags[,3]==names(dam_codes)[j],na.rm=T)+sum(flags[,4]==names(dam_codes)[j],na.rm=T))/dim(flags)[1]
    }
    
  }
  for (j in 1:dim(LUT)[1]){
    df[df==LUT[j,"SK"]] = LUT[j,1]
  }
  plot_size = unique(SK_sites$PLOT_SIZE[SK_sites$measurement_header_PLOT_ID==i])
  plot_size = plot_size[!is.na(plot_size)]
  #Some old plots have 0 as a plot size. I'm just setting them to the most common size for the moment. I don't think I'll ever use these sites.
  if(plot_size == 0){
    plot_size = 0.08
  }
  df = data.frame(df,plot_size)
  write.csv(df,paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SK/checks/",i,"_check.csv"))
}
write.csv(cut_percent,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SK/SK_cut_percent.csv")
write.csv(damage,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SK/SK_flags_percent.csv")
write.csv(surveys,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SK/SK_surveys.csv")
