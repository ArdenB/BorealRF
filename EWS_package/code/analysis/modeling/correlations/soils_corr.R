library(pspearman)

leadpath = "/att/nobackup/scooperd/"
soils = read.csv(paste0(leadpath,"scooperdock/EWS/data/psp/soils/soil_properties_aggregated.csv"))            
soils = soils[,c(-1,-2)]


sig_mat = c("x","y","rho")
for (n in 1:ncol(soils)){
  x = soils[,n]
  for (i in c(1:ncol(soils))[!1:ncol(soils) %in% n]){
    y = soils[,i]
    test = spearman.test(x,y)
    sig_mat = rbind(sig_mat,c(colnames(soils)[n],colnames(soils)[i],test$estimate))
  }
  print(colnames(soils)[n])
}
colnames(sig_mat) = sig_mat[1,]
sig_mat = sig_mat[-1,]
write.csv(sig_mat, "/att/nobackup/scooperd/scooperdock/EWS/data/linear_models/soils_corr_042419.csv")
