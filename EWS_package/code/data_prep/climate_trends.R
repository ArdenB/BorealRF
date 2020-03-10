library(dplyr)

leadpath = "/att/nobackup/scooperd/"


climate = read.csv(paste0(leadpath,"scooperdock/EWS/data/psp/Climate/1951-2018/Climate_NA_data_1951-2018Y.csv"))  
climate_names = read.csv(paste0(leadpath,"scooperdock/EWS/data/psp/Climate/1951-2018/Climate_NA_data.csv"),stringsAsFactors = F)
climate_names = arrange(climate_names,ID2)

vars = colnames(climate)[7:29]
pred = 1:30
for (v in vars){
  mat1 = matrix(ncol = length(1981:2018),nrow = nrow(climate_names))
  colnames(mat1) = 1981:2018
  rownames(mat1) = climate_names$ID1
  mat2 = mat1
  this_var = climate[,c('Year','ID1','ID2',v)]
  this_var[this_var==-9999] = NA
  for(y in 1981:2018){
    out_mat = matrix(nrow = nrow(climate_names),ncol = 30)
    colnames(out_mat) = (y-30):(y-1)
    for(y2 in (y-30):(y-1)){
      data_mat_now = this_var[this_var$Year==y2,]
      data_mat_now = arrange(data_mat_now,ID2)
      out_mat[,as.character(y2)] = data_mat_now[,4]
    }
    mat1[,as.character(y)] = rowMeans(out_mat)
    for (i in 1:nrow(out_mat)){
      resp = as.numeric(out_mat[i,]) 
      if(sum(!is.na(resp))>1){
        mat2[i,as.character(y)] = coefficients(lm(resp~pred))[[2]]
      }
    }

  }
  write.csv(mat1,paste0(leadpath,"scooperdock/EWS/data/psp/Climate/1951-2018/",v,"_mean_30years.csv"))
  write.csv(mat2,paste0(leadpath,"scooperdock/EWS/data/psp/Climate/1951-2018/",v,"_abs_trend_30years.csv"))
}


