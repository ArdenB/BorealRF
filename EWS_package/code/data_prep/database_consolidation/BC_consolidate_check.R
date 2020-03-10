
library(dplyr)

LUT = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SP_LUT.csv",stringsAsFactors = F)
BC_tree <- read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/BC/faib_psp_tree_by_meas.csv",stringsAsFactors = F)
sites = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/All_sites_101218.csv",row.names = "X",stringsAsFactors = F)
sites = sites[1:6803,]
sites[,1] = gsub("1_","",sites[,1])
BC_tree = BC_tree[BC_tree$SAMP_ID %in% sites[,1],]

BC_plots = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/BC/faib_data_plot_by_meas.csv",stringsAsFactors = F)
BC_plots = BC_plots[!duplicated(BC_plots$SAMP_ID),]
BC_plots = BC_plots[BC_plots$SAMP_ID %in% sites[,1],]

plot_vect = unique(BC_tree$SAMP_ID)


BC_tree$species[BC_tree$species=="2M"] = "XX"
BC_tree$species[BC_tree$species=="MR"] = "M"
BC_tree$species[BC_tree$species=="ZH"] = "XH"

#apparently no species errors
# #Find species errors
# sp_err = data.frame()
# for (i in plot_vect){
#   temp = BC_tree[BC_tree$SAMP_ID==i,]
#   temp = arrange(temp,meas_no)
#   samp_vect = unique(temp$meas_yr)
#   for (n in unique(temp$tree_no)){
#     sp = temp$species[temp$tree_no==n]
#     for (j in 1:length(sp)){
#       names(sp)[j] = which(samp_vect==temp$meas_yr[temp$tree_no==n][j])
#     }
#     sp = sp[!is.na(sp)]
#     if (length(unique(sp))>1){
#       for (j in 1:length(sp)){
#         sp_err[paste0(i,"_",n),names(sp)[j]] = sp[j]
#       }
#       
#     }
#   }
# }





cut_percent = matrix(ncol = 13,nrow = length(plot_vect))
rownames(cut_percent) = plot_vect
colnames(cut_percent) = paste0("t",1:13)

damage = matrix(nrow = length(plot_vect),ncol = 13*6)
rownames(damage) = plot_vect
dams = c("Disease/pathogen","Insect","Fire","Drought","Human","Other")
colnames(damage) = c(paste0(dams,"_t1"),paste0(dams,"_t2"),paste0(dams,"_t3"),paste0(dams,"_t4"),paste0(dams,"_t5"),paste0(dams,"_t6"),paste0(dams,"_t7"),paste0(dams,"_t8"),paste0(dams,"_t9"),paste0(dams,"_t10"),paste0(dams,"_t11"),paste0(dams,"_t12"),paste0(dams,"_t13"))
dam_codes = list(
  "Disease/pathogen" = c("D","DB","DBF","DBS","DD","DDA","DDB","DDD","DDE",'DDF','DDH','DDO','DDP','DDS','DDT','DF','DFC','DFD','DFE','DFL','DFP','DFR','DFS','DL','DLF','DM','DMF','DMH','DML','DMP','DR','DRA','DRB','DRC','DRL','DRN','DRR','DRT','DS','DSA','DSB','DSC','DSE','DSG','DSP','DSR','DSS','DSY'),
  "Insect" = c("C","CIA","CIX","CSN","I",'IA','IAB','IAC','IAG','IAS','IB','IBB','IBD','IBI','IBM','IBS','IBT','IBW','ID','IDD','IDE','IDF','IDH','IDL','IDM','IDN','IDS','IDT','IDW','IDX','IS','ISP','IW','IWP','IWS','IWW',"M"),
  "Fire" = c('NB'),
  "Drought" = c('ND'),
  "Human" = c('TH','TL','TM','TP','TR','TT'),
  "Other" = c("A","AB","AD","AE","AH","AM","AP","AS","AV","AX","AZ",'N','NF','NG','NGC','NGK','NL','NN','NR','NS','NSG','NW','NWS','NWT','NX','NY','NZ')
)
print("3")
surveys= data.frame()
#Create consolidated csvs
for (i in plot_vect){
  temp = BC_tree[BC_tree$SAMP_ID==i,]
  temp = arrange(temp,meas_no)
  if(i=="29025 G000501"){
    temp = temp[temp$meas_no!=2,]
    temp$meas_no[temp$meas_no==3] = 2
  }else if(i=="62014 G000520"){
    temp = temp[temp$meas_no!=3,]
  }
  samp_vect = unique(temp$meas_yr)
  for (n in 1:length(samp_vect)){
    surveys[paste0("1_",i),paste0("t",n)] = samp_vect[n]
  }
  tree_range = unique(temp$tree_no)
  tree_range = sort(tree_range)
  df = matrix(nrow = length(tree_range),ncol = length(samp_vect)*2)
  sp_df = matrix(nrow = length(tree_range),ncol = length(samp_vect))
  rownames(df) = tree_range
  colnames(df) = c(paste0("dbh_t",1:length(samp_vect)),paste0("status_t",1:length(samp_vect)))
  colnames(sp_df) = paste0("sp_t",1:length(samp_vect))
  rownames(sp_df) = tree_range
  for (n in 1:length(samp_vect)){
    temp_samp = temp[temp$meas_yr==samp_vect[n],]
    temp_samp = arrange(temp_samp,tree_no)
    if(!identical(temp_samp$tree_no,tree_range)){
      add = tree_range[!tree_range %in% temp_samp$tree_no]
      add_mat = data.frame(matrix(ncol = dim(temp_samp)[2],nrow = length(add)))
      colnames(add_mat) = colnames(temp_samp)
      add_mat$tree_no = add
      temp_samp = rbind(temp_samp,add_mat)
      temp_samp = arrange(temp_samp,tree_no)
    }
    for (j in 1:dim(df)[1]){
      sp_df[j,paste0("sp_t",n)] = temp_samp$species[temp_samp$tree_no==rownames(df)[j]]
      df[j,paste0("dbh_t",n)] = temp_samp$dbh[temp_samp$tree_no==rownames(df)[j]]
      if(!is.na(temp_samp$ld[temp_samp$tree_no==rownames(df)[j]]) & any(temp_samp$ld[temp_samp$tree_no==rownames(df)[j]]==c("L","I","V","X"))){
        df[j,paste0("status_t",n)] = "L"
      }
    }
    cut_percent[as.character(i),n] = sum(temp_samp$ld=="C",na.rm=T)/sum(!is.na(temp_samp$ld))

    flags = data.frame(paste0(temp_samp$dam_agent_type1,temp_samp$dam_agent_spc1),paste0(temp_samp$dam_agent_type2,temp_samp$dam_agent_spc2))
    flags = as.matrix(flags)
    for (s in 1:length(dam_codes)){
      flags[flags %in% dam_codes[[s]]] = names(dam_codes)[s]
    }
    for (r in 1:dim(flags)[1]){
      flags[r,duplicated(flags[r,],incomparables = NA)] = NA
    }
    for(j in 1:length(dam_codes)){
      damage[i,paste0(names(dam_codes)[j],"_t",n)] = sum(flags==names(dam_codes)[j],na.rm=T)/sum(!is.na(flags[,1]))
    }
  }

  for (j in 1:dim(LUT)[1]){
    sp_df[sp_df==LUT[j,"BC"]] = LUT[j,1]
  }
  plot_size = BC_plots$area_pm[BC_plots$SAMP_ID==i]
  df = data.frame(sp_df,df,plot_size)
  write.csv(df,paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/BC/checks/",i,"_check.csv"))
}
write.csv(damage,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/BC/BC_flags_percent.csv")
write.csv(surveys,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/BC/BC_surveys.csv")
write.csv(cut_percent,"/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/BC/BC_cut_percent.csv")