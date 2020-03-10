library(readxl)
library(plyr)
library(dplyr)

LUT = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SP_LUT.csv",stringsAsFactors = F)
CIPHA1 <- read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CIPHA/CIPHA_MEN00.txt",stringsAsFactors = F)
CIPHA2 <- read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CIPHA/CIPHA_MEN04.txt",stringsAsFactors = F)
CIPHA3 <- read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CIPHA/CIPHA_MEN08.txt",stringsAsFactors = F)
CIPHA4 <- read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CIPHA/CIPHA_MEN12.txt",stringsAsFactors = F)
CIPHA5 <- read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CIPHA/CIPHA_MEN16.txt",stringsAsFactors = F)
loc = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CIPHA/LOCATION17.txt",stringsAsFactors = F)
loc[,1] = paste0(loc$node,loc$stand)

CIPHA1 = data.frame(CIPHA1,vector(length = dim(CIPHA1)[1]))
CIPHA2 = data.frame(CIPHA2,vector(length = dim(CIPHA2)[1]))
CIPHA2$dom[is.na(CIPHA2$dom)] = 0
CIPHA2[CIPHA2$dom==9,23] = "D"
CIPHA3 = data.frame(CIPHA3,vector(length = dim(CIPHA3)[1]))
for (i in 1:dim(CIPHA3)[1]){
  if (any(CIPHA3$die[i]==c(99,199,299))&CIPHA3$species[i]==311){
    CIPHA3[i,16] = "D"
  }
}
CIPHA4 = data.frame(CIPHA4,vector(length = dim(CIPHA4)[1]))
CIPHA4$dom[is.na(CIPHA4$dom)] = 0
CIPHA4[CIPHA4$dom==9,23] = "D"
CIPHA5 = data.frame(CIPHA5,vector(length = dim(CIPHA5)[1]))
for (i in 1:dim(CIPHA5)[1]){
  if (any(CIPHA5$DIE[i]==c(99,199,299))&CIPHA5$species[i]==311){
    CIPHA5[i,20] = "D"
  }
}
CIPHA1 = CIPHA1[,c(1:7,10,12,13,20)]
CIPHA2 = CIPHA2[,c(1:7,11,13,14,23)]
CIPHA3 = CIPHA3[,c(1:10,16)]
CIPHA4 = CIPHA4[,c(1:7,11,13,14,23)]
CIPHA5 = CIPHA5[,c(1:7,11:13,20)]


colnames(CIPHA1) = c("id1","year","date","nodestand","plot","tree","species","state","dht","dbh","dead")
colnames(CIPHA2) = c("id1","year","date","nodestand","plot","tree","species","state","dht","dbh","dead")
colnames(CIPHA3) = c("id1","year","date","nodestand","plot","tree","species","state","dht","dbh","dead")
colnames(CIPHA4) = c("id1","year","date","nodestand","plot","tree","species","state","dht","dbh","dead")
colnames(CIPHA5) = c("id1","year","date","nodestand","plot","tree","species","state","dht","dbh","dead")


CIPHA_tree = rbind(CIPHA1,CIPHA2,CIPHA3,CIPHA4,CIPHA5)
tree_ID = vector(length=dim(CIPHA_tree)[1])
CIPHA_tree = data.frame(CIPHA_tree,tree_ID)
CIPHA_tree$tree_ID = paste0(CIPHA_tree$nodestand,"_",CIPHA_tree$tree)


plot_vect = unique(CIPHA_tree$nodestand)
states = data.frame()
species = data.frame()
for (i in sprintf("%02.0f",0:17)){
  health = read.csv(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CIPHA/HEALTH",i,".txt"),stringsAsFactors = F)
  for (j in 1:dim(health)[1]){
    species[paste0(health$nodestand[j],"_",health$tree[j]),paste0("20",i)] = health$species[j]
    if(health$species[j]==311){
      if (any(health$die[j]==c(99,199,299))){
        states[paste0(health$nodestand[j],"_",health$tree[j]),paste0("20",i)] = "D"
      }else{
        states[paste0(health$nodestand[j],"_",health$tree[j]),paste0("20",i)] = "L"
      }
    }else{
      non_aspen = CIPHA_tree[CIPHA_tree$tree_ID==paste0(health$nodestand[j],"_",health$tree[j]),]
      for (n in 1:dim(non_aspen)[1]){
        states[non_aspen$tree_ID[n],as.character(non_aspen$year[n])] = non_aspen$dead[n]
        if (non_aspen$dead[n]=="FALSE"&!is.na(non_aspen$dbh[n])){
          states[non_aspen$tree_ID[n],as.character(non_aspen$year[n])] = "L"
        }
      }
    }
  }
}
states = states[,c(paste0("20",sprintf("%02.0f",0:17)))]
dbh = states
dbh[] = NA
for(i in 1:dim(CIPHA_tree)[1]){
  dbh[paste0(CIPHA_tree$nodestand[i],"_",CIPHA_tree$tree[i]),as.character(CIPHA_tree$year[i])] = CIPHA_tree$dbh[i]
}

sp_err= data.frame()
sp = vector(length=dim(states)[1])
for (i in 1:dim(species)[1]){
  s = unique(species[i,])
  sp[i] = s[!is.na(s)]
  if (any(species[i,!is.na(species[i,])]!=as.numeric(species[i,!is.na(species[i,])][1]))){
    for (n in colnames(species)){
      sp_err[rownames(species)[i],n] = species[i,n]
    }
  }
}
#Now figure it out for the non aspens
non=states[sp!=311,]
non_dbhs = data.frame()
for (i in 1:dim(non)[1]){
  for (j in 1:dim(non)[2]){
    if (!is.na(non[i,j])){
      non_dbhs[rownames(non)[i],colnames(non)[j]] = CIPHA_tree$dbh[CIPHA_tree$tree_ID==rownames(non)[i]&CIPHA_tree$year==colnames(non)[j]]
    }
  }
}
non_dbhs = non_dbhs[,c("2000","2001","2004","2005","2008","2012","2016")]
rem = vector(length=dim(non)[1])
for (i in 1:dim(non)[1]){
  if (!any(non[i,!is.na(non[i,])]!="L")){
    rem[i] = 1
  }
}
non = non[!rem,]
non_dbhs = non_dbhs[!rem,]
#Trying to salvage these data.
#1) if a tree is not indicated as alive, but a later dbh measurement is larger than the current, then it must be alive and have always been alive
#2) If any future dbh is smaller than the current one, the tree is dead
#3) If a tree is dead, it will always be dead
for (i in 1:dim(non)[1]){
  years = colnames(non[i,!is.na(non[i,])])
  for (n in 1:(length(years)-1)){
    if(non[i,years[n]]!="L"&!is.na(non_dbhs[i,years[n]])){
      if (any(non_dbhs[i,years[(n+1):length(years)]]>non_dbhs[i,years[n]],na.rm=T)){
        non[i,years[1:n]] = "L"
      }
    }else if(non[i,years[n]]!="D"&is.na(non_dbhs[i,years[n]])){
      non[i,years[n]] = "D"
    }
    if (non[i,years[n]]=="D"){
      non[i,years[(n+1):length(years)]] = "D"
    }
  }
}
#fixing in states
for (i in 1:dim(non)[1]){
  years = colnames(non[i,!is.na(non[i,])])
  states[rownames(non)[i],years] = non[i,!is.na(non[i,])]
}


cut_percent = matrix(ncol = 18,nrow = length(plot_vect))
rownames(cut_percent) = plot_vect
colnames(cut_percent) = paste0("t",1:18)


damage = matrix(nrow = length(plot_vect),ncol = 6*18)
rownames(damage) = plot_vect
dams = c("Disease/pathogen","Insect","Fire","Drought","Human","Other")
col = vector()
for(n in 1:length(dams)){
  col = c(col,paste0(dams[n],"_t",1:18))
}
colnames(damage) = col
dam_codes = list(
  "Disease/pathogen" = c(10,11,12,13,14,19,30,31,32,33,34,35,39),
  "Insect" = c(1,2,3,4,5,6,7,8,9,20,21,22,23,24,25,26,27,28,29),
  "Fire" = c("FIRE"),
  "Drought" = c("DROUGHT"),
  "Human" = c("HERBICI","MECHSCAR","LOGGED"),
  "Other" = c("HAIL","WIND","SNOW","ODOVIR","URSARC","URSAME","CASCAN","EREDOR","DRYPIL","PICSPP","SPHVAR","TAMHUD","FLOOD")
)

nodestand = vector(length = dim(states)[1])
treenum = nodestand
for (i in 1:length(treenum)){
  nodestand[i] = strsplit(rownames(states)[i],"_")[[1]][1]
  treenum[i] = strsplit(rownames(states)[i],"_")[[1]][2]
}
level = 25 #set percent level for including damage codes
CIPHA_surveys = data.frame()
#Create consolidated csvs
for (i in plot_vect){
  temp_state = states[nodestand==i,]
  temp_sp = species[nodestand==i,]
  temp_dbh = dbh[nodestand==i,]
  #Interpolate dbhs
  for (j in 1:dim(temp_dbh)[1]){
    years = which(!is.na(temp_dbh[j,]))
    if (length(years)>1){
      for (n in 1:(length(years)-1)){
        if (temp_dbh[j,years[n+1]]==temp_dbh[j,years[n]]){
          temp_dbh[j,c(years[n]:years[n+1])] = temp_dbh[j,years[n+1]]
        }else if (temp_dbh[j,years[n+1]]>temp_dbh[j,years[n]]){
          temp_dbh[j,c(years[n]:years[n+1])] = approx(temp_dbh[j,c(years[n]:years[n+1])],c(years[n]:years[n+1]),n = length(c(years[n]:years[n+1])))$x
        }else{
          temp_dbh[j,c(years[n]:years[n+1])] = rev(approx(temp_dbh[j,c(years[n]:years[n+1])],c(years[n]:years[n+1]),n = length(c(years[n]:years[n+1])))$x)
        }
      }
    }
    if(is.na(temp_dbh[j,dim(temp_dbh)[2]])){
      temp_dbh[j,dim(temp_dbh)[2]] = temp_dbh[j,dim(temp_dbh)[2]-1]
    }
  }
  if (any(i==c("ALD1","ALD2","ALD3"))){
    for (j in 1:dim(temp_state)[1]){
      if (!is.na(temp_state$`2001`[j])){
        temp_state$`2002`[j] = temp_state$`2001`[j]
        temp_state$`2001`[j] = NA
      }
    }
  }
  #Get rid of years with no measurements
  temp_sp = temp_sp[,colSums(!is.na(temp_state))!=0]
  temp_dbh = temp_dbh[,colSums(!is.na(temp_state))!=0]
  temp_state = temp_state[,colSums(!is.na(temp_state))!=0]
  #fill in state for non aspens
  for (j in 1:dim(temp_state)[1]){
    if(any(is.na(temp_state[j,]))){
      for (n in 2:dim(temp_state)[2]){
        if(is.na(temp_state[j,n])){
          temp_state[j,n] = temp_state[j,n-1]
        }
      }
    }
  }
  df = cbind(temp_dbh,temp_state)
  colnames(df) = c(paste0("dbh_t",1:dim(temp_dbh)[2]),paste0("status_t",1:dim(temp_dbh)[2]))
  colnames(temp_sp) = paste0("sp_t",1:dim(temp_dbh)[2])

  for (j in 1:dim(temp_state)[2]){
    CIPHA_surveys[paste0("14_",i),paste0("t",j)] = colnames(temp_state)[j]
    agent = read.csv(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CIPHA/AGENTS",gsub("20","",colnames(temp_state)[j]),".txt"),stringsAsFactors = F)
    agent = agent[agent$nodestand==i,]
    agent$level[agent$level==99] = 0
    cut_percent[as.character(i),paste0("t",j)] = sum((agent$type==46|agent$name=="LOGGED")&agent$level>level)/dim(df)[1]
    flags = data.frame()
    for(s in unique(agent$tree)){
      fl = as.character(agent$type[agent$tree==s&agent$level>level])
      if (length(fl)>0){
        for (r in 1:length(fl)){
          if(!fl[r] %in% dam_codes["Insect"][[1]]&!fl[r] %in% dam_codes["Disease"][[1]]){
            fl[r] = agent$name[agent$tree==s][r]
          }
          flags[as.character(s),as.character(r)] = fl[r]
        }
      }
    }
    flags = as.matrix(flags)
    if(!empty(flags)){
      for (s in 1:length(dam_codes)){
        flags[flags %in% dam_codes[[s]]] = names(dam_codes)[s]
      }
      for (r in 1:dim(flags)[1]){
        flags[r,duplicated(flags[r,],incomparables = NA)] = NA
      }
      for(r in 1:length(dam_codes)){
        damage[as.character(i),paste0(names(dam_codes)[r],"_t",j)] = sum(flags==names(dam_codes)[r],na.rm=T)/dim(df)[1]
      }
    }
  }
  
  for (j in 1:dim(LUT)[1]){
    temp_sp[temp_sp==as.numeric(LUT[j,"CIPHA"])] = LUT[j,1]
  }
  plot_size = (loc$x1[loc$id1==i]*loc$y1[loc$id1==i] + loc$x2[loc$id1==i]*loc$y2[loc$id1==i])/10000
  df = data.frame(temp_sp,df,plot_size)
  write.csv(df,paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CIPHA/checks/",i,"_check.csv"))
}
write.csv(cut_percent,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CIPHA/CIPHA_cut_percent.csv")
write.csv(damage,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CIPHA/CIPHA_flags_percent.csv")
write.csv(CIPHA_surveys,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CIPHA/CIPHA_surveys.csv")