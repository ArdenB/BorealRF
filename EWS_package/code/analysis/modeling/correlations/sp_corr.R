library(pspearman)

leadpath = "/att/nobackup/scooperd/"

clean_df = function(in_df) {
  
  # Eliminate null vals
  in_df[in_df == null_val] = NA
  
  add_sites = sites[!(sites %in% rownames(in_df))]
  if(length(add_sites)>0){
    in_df[add_sites,] = NA
  }
  
  # Match up all column names
  add_years = y_range[!(paste0("X",y_range) %in% colnames(in_df))]
  if(length(add_years)>0) {
    in_df[,paste0("X",add_years)] = NA
  }
  
  # Ensure column and row order is set properly
  in_df = in_df[,paste0("X",y_range)]
  in_df = in_df[sites,]
  
  return(in_df)
}

#Read in lengths of time since last survey
raw_shift_df = read.csv(paste0(leadpath,"scooperdock/EWS/data/psp/surv_interval_filled.csv"))
raw_shift_df = arrange(raw_shift_df,X)
rownames(raw_shift_df) = raw_shift_df[,"X"]
sites = rownames(raw_shift_df)


#Read in mortality and damage dataframes
raw_mass_df = read.csv(paste0(leadpath,"scooperdock/EWS/data/psp/biomass_interpolated.csv"),row.names = 'X')
mass_df = clean_df(raw_mass_df)


LUT = read.csv(paste0(leadpath,"scooperdock/EWS/data/raw_psp/SP_LUT.csv"),stringsAsFactors = F)
sp_groups = read.csv(paste0(leadpath,"scooperdock/EWS/data/raw_psp/SP_groups.csv"),stringsAsFactors = F)

sp_out_df = data.frame('site' = rep(sites,37))
rows = vector()
for(i in 1:151){
  if(file.exists(paste0(leadpath,"scooperdock/EWS/data/psp/comp_interp_",i,".csv"))){
    raw_sp_df = read.csv(paste0(leadpath,"scooperdock/EWS/data/psp/comp_interp_",i,".csv"),stringsAsFactors = F,row.names = 'X')
    print(paste0(LUT$scientific[LUT$ID==i]," has ",nrow(raw_sp_df)," rows"))
    rows[LUT$scientific[LUT$ID==i]] = nrow(raw_sp_df)
    # sp_df = clean_df(raw_sp_df)
    # #sp_df[!is.na(mass_df)&is.na(sp_df)] = 0
    # sp = LUT$scientific[LUT$ID==i]
    # sp_out_df[,sp] = unlist(sp_df)
  }
}

sp_out_df = sp_out_df[rowSums(!is.na(sp_out_df[,2:122]))>0,]

data = data.frame('site' = rep(sites,37))
sp_groups = sp_groups[sp_groups$scientific %in% colnames(sp_out_df),]
for(gr in colnames(sp_groups)[2:6]){
  groups = unique(sp_groups[,gr])
  for (g in groups){
    sp = sp_groups$scientific[sp_groups[,gr]==g]
    # for(s in sp){
    #   if(any(colnames(data)==paste0(gr,"_",g))){
    #     data[,paste0(gr,"_",g)] = data[,paste0(gr,"_",g)] + sp_out_df[,s]
    #   }
    # }
    #sp = sp[sp %in% colnames(sp_out_df)]
    if(length(sp)==1){
      data[,paste0(gr,"_",g)] = sp_out_df[,sp]
    }else{
      data[,paste0(gr,"_",g)] = rowSums(sp_out_df[,sp],na.rm=T)
    }
  }
}
data = data[,-1]
#data[data==0] = NA
data = data[!is.na(unlist(mass_df)),]
#data = data[rowSums(!is.na(data))>0,]

corr = cor(data,method='spearman',use = 'pairwise.complete.obs')
corr_vect = as.vector(corr)
x_vect = vector(length = length(corr_vect))
i = 1
for (n in 1:nrow(corr)){
  x_vect[i:(i+nrow(corr)-1)] = rep(rownames(corr)[n],nrow(corr))
  i = i+nrow(corr)
}
y_vect = rep(rownames(corr),nrow(corr))

sig_mat = data.frame(x_vect,y_vect,corr_vect)


colnames(sig_mat) = c("x","y","rho")
sig_mat = sig_mat[sig_mat$x!=sig_mat$y,]

write.csv(sig_mat, "/att/nobackup/scooperd/scooperdock/EWS/data/linear_models/sp_groups_corr_082019.csv")

