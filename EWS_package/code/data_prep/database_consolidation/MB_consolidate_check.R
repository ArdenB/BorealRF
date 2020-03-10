
library(dplyr)
library(readxl)

LUT = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SP_LUT.csv",stringsAsFactors = F)

files = list.files("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/MB/PSP_csvs/")

# WPSP = matrix(nrow= length(files),ncol = 2)
# rownames(WPSP) = files
# for (i in files){
#   temp = read.csv(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/MB/PSP_csvs/",i),stringsAsFactors = F)
#   WPSP[i,1] = sum(temp$SPECIES=="WP ",na.rm=T)
#   WPSP[i,2] = sum(temp$SPECIES=="SP ",na.rm=T)
# }
# ELWE = matrix(nrow= length(files),ncol = 2)
# rownames(ELWE) = files
# for (i in files){
#   temp = read.csv(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/MB/PSP_csvs/",i),stringsAsFactors = F)
#   ELWE[i,1] = sum(temp$SPECIES=="EL ",na.rm=T)
#   ELWE[i,2] = sum(temp$SPECIES=="WE ",na.rm=T)
# }

#Find species errors
sp_err = data.frame()
for (i in files){
  temp = read.csv(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/MB/PSP_csvs/",i),stringsAsFactors = F)
  temp = temp[temp$SUBPLOT=="tree plot   ",]
  # if(length(unique(temp$subplot_num))>1){
  #   for (j in 2:length(unique(temp$subplot_num))){
  #     temp$Tree_num[temp$subplot_num==j] = temp$Tree_num[temp$subplot_num==j]+max(temp$Tree_num[temp$subplot_num==j-1])
  #   }
  # }
  temp = arrange(temp,MEASURE_NO)
  #There is no species BA in species list, however, there are no trees identified as LA. 
  #I think it's a colloquial issue between largetooth aspend and bigtooth aspen. Changing all BAs to LAs
  temp[temp$SPECIES=="LA ","SPECIES"] = "BA "
  
  

  samp_vect = unique(temp$YEAR_MEAS)
  for (n in unique(temp$TREE_NO)){
    sp = temp$SPECIES[temp$TREE_NO==n]
    comp = temp$SPECIES_COMP[temp$TREE_NO==n]
    names(sp) = temp$MEASURE_NO[temp$TREE_NO==n]
    names(comp) = temp$MEASURE_NO[temp$TREE_NO==n]
    comp = comp[!is.na(comp)]
    # sp = sp[!is.na(sp)]
    # if (length(sp)>0){
    #   for (j in 1:length(sp)){
    #     if (names(sp)[j] %in% names(sp)[-j] & !any(sp[j]==sp[-j])){
    #       sp[j] = NA
    #       remove_line[r] = paste0(i,"_",temp$X[temp$Tree_num==n][j])
    #       r=r+1
    #     }
    #   }
    sp = sp[!is.na(sp)]
    if (length(unique(sp))>1){
      for (j in 1:length(sp)){
        sp_err[paste0(i,"_",n),names(sp)[j]] = as.character(sp[j])
        sp_err[paste0(i,"_",n),paste0(names(sp)[j],"comp")] = as.character(comp[j])
      }
      #some of these errors are a result of reusing tree numbers, this filters out those trees
      # nans = !is.na(sp_err[paste0(i,"_",n),])
      # if(sum(nans,na.rm=T)!=0){
      #   if(!identical(which(nans),min(which(nans)):max(which(nans)))){
      #     sp_err = sp_err[!rownames(sp_err) %in% paste0(i,"_",n),]
      #   }
      # }
      
    }
  }
}

sp_err2=sp_err
#First using the column that determines species composition to correct species errors for situations where two species are chosen the same number of times
for (i in 1:dim(sp_err)[1]){
  test = sp_err[i,c(1,3,5,7,9,11,13)]
  if(sum(test==unique(test[!is.na(test)])[2],na.rm = T)==sum(test==unique(test[!is.na(test)])[1],na.rm = T)){
    for (n in seq(1,length(sp_err[i,!is.na(sp_err[i,])]),2)){
      if (!is.na(sp_err[i,n])){
        if (!(gsub(" ","",sp_err[i,n]) %in% strsplit(sp_err[i,n+1],"[0-9]")[[1]]) & any(gsub(" ","",sp_err[i,c(1,3,5,7,9,11,13)]) %in% strsplit(sp_err[i,n+1],"[0-9]")[[1]])){
          ind = which(sp_err[i,]==sp_err[i,n])
          sp_err[i,ind] = NA
          sp = sp_err[i,c(1,3,5,7,9,11,13)]
          sp = sp[!is.na(sp)]
          sp = unique(sp)
          sp_err[i,ind] = sp
        }
      }
    }
  }
}

sp_err = sp_err[,c(1,3,5,7,9,11,13)]
#changing the sp IDs of the errors. This makes a few assumptions:
#1) Later identifications are more accurate
#2) If a tree was identified as one species more often than another, it is likey that species
for (i in 1:dim(sp_err)[1]){
  indices = colnames(sp_err[i,!is.na(sp_err[i,])])
  sp_vect = vector()
  for (n in indices){
    sp_vect[n] = sp_err[i,n]
  }
  if(length(unique(sp_vect))!=1){
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

err_check = vector()
for (n in 1:dim(sp_err)[1]){
  for (i in 1:6){
    err_check[i] = sp_err[n,i]
  }
  if (length(unique(err_check[is.na(err_check)]))>1){
    print(rownames(sp_err)[n])
  }
}




cut_percent = matrix(ncol = 7,nrow = length(files))
rownames(cut_percent) = gsub(".csv","",files)
colnames(cut_percent) = paste0("t",1:7)

damage = matrix(nrow = length(files),ncol = 6*7)
rownames(damage) = gsub(".csv","",files)
dams = c("Disease/pathogen","Insect","Fire","Drought","Human","Other")
colnames(damage) = c(paste0(dams,"_t1"),paste0(dams,"_t2"),paste0(dams,"_t3"),paste0(dams,"_t4"),paste0(dams,"_t5"),paste0(dams,"_t6"),paste0(dams,"_t7"))
dam_codes = list(
  "Disease/pathogen" = c(40,42,44,45,46,41,43),
  "Insect" = c(50,51,52,53,54,55,56,57),
  "Fire" = c(13),
  "Drought" = c(11),
  "Human" = c(26),
  "Other" = c(10,12,14,15,28,29,61,62)
)
mort_codes = list(
  "Fire" = 7,
  "Human" = 5
)

#Create consolidated csvs
for (i in files){
  temp = read.csv(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/MB/PSP_csvs/",i),stringsAsFactors = F)
  temp = temp[temp$SUBPLOT=="tree plot   ",]
  temp = arrange(temp,MEASURE_NO)
  temp[temp$TREE_NO=="    .","TREE_NO"] = NA
  temp = temp[!is.na(temp$TREE_NO),]
  #Adjust mistakes in tree labeling
  if (i=="MU12PSP152.csv"){
    temp["1720","TREE_NO"] = "  276"
  }else if (i=="MU14PSP065.csv"){
    temp["269","TREE_NO"] = "  369"
  }else if (i=="MU14PSP071.csv"){
    temp["518","TREE_NO"] = "  169"
  }else if (i=="MU20PSP073.csv"){
    temp = temp[-82,]
  }else if (i=="MU20PSP177.csv"){
    temp = temp[c(-1110,-2053,-2999,-4605,-4892),]
  }else if (i=="MU20PSP455.csv"){
    temp = temp[c(-647,-864,-1081),]
  }else if (i=="MU23PSP114.csv"){
    temp["36","TREE_NO"] = "    2"
  }else if (i=="MU30PSP107P.csv"){
    temp["438","TREE_NO"] = "  226"
  }else if (i=="MU30PSP110.csv"){
    temp = temp[c(-618,-1113,-1618,-2244,-2656),]
    temp[c("769","1264","1769","2351","2807"),"TREE_NO"] = "  171"
    temp[c("765","1260","1765","2348","2803"),"TREE_NO"] = "  523"
    temp[c("857","1352","1857","2426","2895"),"TREE_NO"] = "  524"
    temp[c("1027","1532","2195","2570"),"TREE_NO"] = "   63"
    temp = temp[c(-1274,-1778,-2354,-2814),]
  }else if(i=="MU31PSP634.csv"){
    temp = temp[c(-218,-221),]
  }else if(i=="MU31PSP638.csv"){
    temp["56","TREE_NO"] = "  118"
  }else if(i=="MU31PSP712.csv"){
    temp = temp[c(-738,-926),]
  }else if(i=="MU31PSP720.csv"){
    temp = temp[-1626,]
  }else if(i=="MU41PSP543.csv"){
    temp[c(2160,3854),"TREE_NO"] = " 1691"
  }else if(i=="MU61PSP650.csv"){
    temp["1662","TREE_NO"] = "  146"
  }else if(i=="MU72PSP531.csv"){
    temp = temp[c(-747,-1075,-1397,-1661),]
  }else if(i=="MU83PSP402.csv"){
    temp = temp[c(-3565,-3566,-4841,-4842,-4843,-4844,-4845),]
  }else if(i=="MU83PSP405.csv"){
    temp = temp[-2415,]
  }else if(i=="MU83PSP500.csv"){
    temp = temp[c(-2970,-3519),]
    temp["3651","TREE_NO"] = " 1083"
  }else if(i=="MU87PSP007.csv"){
    temp = temp[-3014,]
  }else if(i=="MU89PSP018.csv"){
    temp["5900","TREE_NO"] = "  507"
  }else if(i=="MU89PSP019.csv"){
    temp = temp[-5012,]
    temp["5932","TREE_NO"] = " 1025"
  }else if(i=="MU20PSP462.csv"){ #Measurement 2 was just repeated for measurement 3, deleting measurement 3
    remove = which(temp$MEASURE_NO==3)
    temp = temp[-remove,]
  }
  #FOr some reason, this site has different plot sizes each measurement but the plot shape for each is for a 500 m2 plot, so I'm hoping it's just that size.
  if(i=="MU14PSP070.csv"){
    temp$PLOT_SIZE = 500.34
  }
  samp_vect = unique(temp$MEASURE_NO)
  #Correct species on the fly
  if (any(sub("_.*","",rownames(sp_err)) %in% i)){
    err = sp_err[sub("_.*","",rownames(sp_err)) %in% i,]
    for (j in 1:dim(err)[1]){
      indices = colnames(err[j,!is.na(err[j,])])
      for (n in indices){
        temp$SPECIES[temp$TREE_NO==sprintf("%05.0d",as.numeric(strsplit(rownames(err)[j],"_")[[1]][2]))&temp$MEASURE_NO==samp_vect[as.numeric(n)]] = err[j,n]
      }
    }
  }
  #There is also no SP in species list, but lots identified as it. There appears to be no overlap between SP and WP so I'm going out on a limb to say that 
  #White pines may be getting identified as soft pines here. The other option is that it's supposed to be jack pine and they're identifying it as scrub pine
  #For now I'm going with white pine though.
  temp[temp$SPECIES=="SP ","SPECIES"] = "WP "
  #Again, very few trees (5 total) IDed as EL, but many IDed as WE (possibly colloquial white elm and this is the ID for white elm in AB), so I'm switching everything 
  #IDed as WE to EL
  temp[temp$SPECIES=="WE ","SPECIES"] = "EL "
  #Inexplicably there are a few red pines IDed as PR instead of RP
  temp[temp$SPECIES=="PR ","SPECIES"] = "RP "
  #Some random "HA"s, I don't know what they are, but lots of aspens in both plots, so I'm chaning to aspens
  temp[temp$SPECIES=="HA ","SPECIES"] = "TA "
  temp[temp$SPECIES=="   ","SPECIES"] = NA
  tree_range = unique(temp$TREE_NO)
  tree_range = sort(tree_range)
  df = matrix(nrow = length(tree_range),ncol = length(samp_vect)*3)
  rownames(df) = tree_range
  colnames(df) = c(paste0("sp_t",1:length(samp_vect)),paste0("dbh_t",1:length(samp_vect)),paste0("status_t",1:length(samp_vect)))
  for (n in 1:length(samp_vect)){
    temp_samp = temp[temp$MEASURE_NO==samp_vect[n],]
    temp_samp = arrange(temp_samp,TREE_NO)
    # if(i=="MU11PSP101.csv" & n==6){
    #   temp_samp = temp_samp[,c(-108,-110)]
    # }
    if(!identical(temp_samp$TREE_NO,tree_range)){
      add = tree_range[!tree_range %in% temp_samp$TREE_NO]
      add_mat = data.frame(matrix(ncol = dim(temp_samp)[2],nrow = length(add)))
      colnames(add_mat) = colnames(temp_samp)
      add_mat$TREE_NO = add
      temp_samp = rbind(temp_samp,add_mat)
      temp_samp = arrange(temp_samp,TREE_NO)
    }
    for (j in 1:dim(df)[1]){
      df[j,paste0("sp_t",n)] = as.character(temp_samp$SPECIES[temp_samp$TREE_NO==rownames(df)[j]])
      df[j,paste0("sp_t",n)] = gsub(" ","",df[j,paste0("sp_t",n)])
      df[j,paste0("dbh_t",n)] = as.numeric(temp_samp$DBH[temp_samp$TREE_NO==rownames(df)[j]])
      temp_samp$STATUS[temp_samp$TREE_NO==rownames(df)[j]] = as.numeric(temp_samp$STATUS[temp_samp$TREE_NO==rownames(df)[j]])
      if(!is.na(temp_samp$STATUS[temp_samp$TREE_NO==rownames(df)[j]]) & (temp_samp$STATUS[temp_samp$TREE_NO==rownames(df)[j]]==0 | temp_samp$STATUS[temp_samp$TREE_NO==rownames(df)[j]]==1 | temp_samp$STATUS[temp_samp$TREE_NO==rownames(df)[j]]==2)){
        df[j,paste0("status_t",n)] = "L"
      }
      #Attempt to correct for trees mistakenly identified as dead or no status indicated
      # if (n!=1){
      #   if(!is.na(df[j,paste0("status_t",n)]) & is.na(df[j,paste0("status_t",n-1)]) & !is.na(df[j,paste0("sp_t",n-1)]) & !is.na(df[j,paste0("dbh_t",n-1)]) & df[j,paste0("sp_t",n)]==df[j,paste0("sp_t",n-1)] & 0<as.numeric(df[j,paste0("dbh_t",n-1)]) & as.numeric(df[j,paste0("dbh_t",n-1)])<as.numeric(df[j,paste0("dbh_t",n)])){
      #     df[j,paste0("status_t",n-1)] = "L"
      #   }
      # }
    }
    cut_percent[as.character(gsub(".csv","",i)),n] = sum(temp_samp$STATUS==5,na.rm=T)/sum(!is.na(temp_samp$STATUS))
    flags = data.frame()
    for (s in 1:dim(temp_samp)[1]){
      fl = strsplit(as.character(temp_samp$COND[s]),",")[[1]]
      names(fl) = 1:length(fl)
      for (r in 1:length(fl)){
        flags[as.character(s),names(fl)[r]] = fl[r]
      }
    }
    mort = temp_samp[,"STATUS"]
    for (s in 1:length(dam_codes)){
      flags[flags %in% dam_codes[[s]]] = names(dam_codes)[s]
    }
    for (s in 1:length(mort_codes)){
      mort[mort %in% mort_codes[[s]]] = names(mort_codes)[s]
    }
    flags = as.matrix(flags)
    flags = cbind(flags,mort)
    for (r in 1:dim(flags)[1]){
      flags[r,duplicated(flags[r,],incomparables = NA)] = NA
    }
    for(j in 1:6){
      damage[as.character(gsub(".csv","",i)),paste0(names(dam_codes)[j],"_t",n)] = (sum(flags==names(dam_codes)[j],na.rm=T)/dim(flags)[2])/sum(!is.na(temp_samp$STATUS))
    }
  }
  #Normalize sp codes using LUT
  for (j in 1:dim(LUT)[1]){
    df[df==LUT[j,"MB"]] = LUT[j,1]
  }
  plot_size = unique(as.numeric(temp$PLOT_SIZE))[1]/10000
  df = data.frame(df,plot_size)
  write.csv(df,paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/MB/checks/",gsub(".csv","",i),"_check.csv"))
}
write.csv(cut_percent,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/MB/MB_cut_percent.csv")
write.csv(damage,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/MB/MB_flags_percent.csv")