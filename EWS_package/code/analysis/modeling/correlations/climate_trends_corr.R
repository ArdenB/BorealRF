library(pspearman)
library(dplyr)

leadpath = "/att/nobackup/scooperd/"

raw_shift_df = read.csv(paste0(leadpath,"scooperdock/EWS/data/psp/surv_interval_filled.csv"))
raw_shift_df = arrange(raw_shift_df,X)
rownames(raw_shift_df) = raw_shift_df[,"X"]
sites = rownames(raw_shift_df)
adj_YT = sites[grep("11_",sites)]
for(i in 1:length(adj_YT)){
  adj_YT[i] = paste0("11_",as.numeric(strsplit(adj_YT[i],"_")[[1]][2]))
}
sites[grep("11_",sites)] = adj_YT


climate_vars = list.files(paste0(leadpath,"scooperdock/EWS/data/psp/Climate/1951-2018/"),pattern = '30years.csv')
climate_vars = climate_vars[!climate_vars %in% 'climate_df_30years.csv']
climate_sites = gsub(" ","_",sites)
all_climate = matrix(ncol = length(climate_vars),nrow = length(sites)*37)
colnames(all_climate) = gsub(".csv","",climate_vars)

for(v in climate_vars){
  var = read.csv(paste0(leadpath,"scooperdock/EWS/data/psp/Climate/1951-2018/",v),row.names = 'X')
  var = var[climate_sites,]
  var = var[,paste0('X',1981:2017)]
  var = unlist(var)
  all_climate[,gsub(".csv","",v)] = var
  #hist(var,breaks=100,main = v)
}

write.csv(all_climate,paste0(leadpath,"scooperdock/EWS/data/psp/Climate/1951-2018/climate_df_30years.csv"))


# sig_mat = c("x","y","rho")
# for (n in 1:ncol(all_climate)){
#   x = all_climate[,n]
#   for (i in c(1:ncol(all_climate))[!1:ncol(all_climate) %in% n]){
#     y = all_climate[,i]
#     test = spearman.test(x,y)
#     sig_mat = rbind(sig_mat,c(colnames(all_climate)[n],colnames(all_climate)[i],test$estimate))
#   }
#   print(colnames(all_climate)[n])
# }
# colnames(sig_mat) = sig_mat[1,]
# sig_mat = sig_mat[-1,]
# write.csv(sig_mat, "/att/nobackup/scooperd/scooperdock/EWS/data/linear_models/climate_corr_082319.csv")
