library(pspearman)
library(data.table)

leadpath = "/att/nobackup/scooperd/"

vi_df = data.frame('num' = 1:913530)

VIs = c('ndvi','psri','ndii','ndvsi','msi','nirv','ndwi','nbr','satvi','tvfc')

for (VI in VIs){
  df = data.frame(fread(paste0(leadpath,"scooperdock/EWS/data/vi_metrics/metric_dataframe_",VI,"_noshift.csv")))
  rownames(df) = df[,1]
  df = df[,c(-1,-2)]
  print(paste0(VI,' read: ',Sys.time()))
  vi_df = data.frame(vi_df,df)
}

vi_df = vi_df[,-1]

corr = cor(vi_df,method='spearman',use = 'na.or.complete')
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

write.csv(sig_mat, "/att/nobackup/scooperd/scooperdock/EWS/data/linear_models/all_vi_corr_100219.csv")
