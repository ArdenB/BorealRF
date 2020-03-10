library(dplyr)
library(readxl)


#Newfoundland and Labrador
# NL_meas = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NL/tblRemCrewDate.txt",stringsAsFactors = F)
# NL_meas$PlotNumber = paste0(NL_meas$District,NL_meas$PlotNumber)
# sites = unique(NL_meas$PlotNumber)
# NL_surveys = matrix(nrow = length(sites),ncol = max(NL_meas$Remeasurement)+1)
# rownames(NL_surveys) = sites
# colnames(NL_surveys) = paste0("t",1:dim(NL_surveys)[2])
# for (i in 1:dim(NL_meas)[1]){
#   NL_surveys[as.character(NL_meas$PlotNumber[i]),paste0("t",NL_meas$Remeasurement[i]+1)] = NL_meas$Year[i]
# }
NL_surveys = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NL/NL_surveys.csv",stringsAsFactors = F,row.names = "X")
rownames(NL_surveys) = paste0("7_",rownames(NL_surveys))
NL_surveys = data.frame(NL_surveys,matrix(nrow = dim(NL_surveys)[1],ncol = 18-dim(NL_surveys)[2]))
colnames(NL_surveys) = paste0("t",sprintf("%02.0f",1:18))

#Alberta
# files = list.files("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/AB/Data/PSP2011/csvs/")
# files = sub("_.*","",files)
# files = unique(files)
# AB_surveys = matrix(nrow = length(files),ncol = 8)
# rownames(AB_surveys) = files
# colnames(AB_surveys) = paste0("t",1:8)
# AB_surveys = data.frame()
# for (i in files){
#   file = read.csv(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/AB/Data/PSP2011/csvs/",i,"_tree.csv"))
#   for (k in unique(file$subplot_num)){
#     temp = file[file$subplot_num==k,]
#     psp = paste0(unique(temp$Group_num),".",k)
#     temp = arrange(temp,Year_meas)
#     years = unique(temp$Year_meas)
#     for(j in 1:length(years)){
#       AB_surveys[psp,paste0("t",j)] = years[j]
#     }
#   }
# }
AB_surveys <- read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/AB/AB_surveys.csv",stringsAsFactors = F,row.names = "X")
rownames(AB_surveys) = paste0("2_",rownames(AB_surveys))
AB_surveys = data.frame(AB_surveys,matrix(nrow = dim(AB_surveys)[1],ncol = 18-dim(AB_surveys)[2]))
colnames(AB_surveys) = paste0("t",sprintf("%02.0f",1:18))

#Northwest Territories
NWT_sites <- read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NWT/NWT_Sites.csv",stringsAsFactors = F)
sites = NWT_sites$Old_Label
NWT_surveys = matrix(nrow = length(sites),ncol = 3)
NWT_surveys = NWT_sites[,c(9,11,13)]
rownames(NWT_surveys) = paste0("12_",gsub("PSP ","",sites))
colnames(NWT_surveys) = paste0("t",1:3)
NWT_surveys[NWT_surveys==0] = NA
NWT_surveys = data.frame(NWT_surveys,matrix(nrow = dim(NWT_surveys)[1],ncol = 18-dim(NWT_surveys)[2]))
colnames(NWT_surveys) = paste0("t",sprintf("%02.0f",1:18))

#Yukon Territories
YT_surveys = as.data.frame(read_excel("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/YT/PSP_data_Overview.xlsx"))
rownames(YT_surveys) = paste0("11_",sprintf("%03.0f",YT_surveys$'PSP Number'))
YT_surveys = YT_surveys[,18:24]
colnames(YT_surveys) = paste0("t",1:7)
YT_surveys = data.frame(YT_surveys,matrix(nrow = dim(YT_surveys)[1],ncol = 18-dim(YT_surveys)[2]))
colnames(YT_surveys) = paste0("t",sprintf("%02.0f",1:18))

#Nova Scotia
# NS_sample <- read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NS/Sample.txt",stringsAsFactors = F)
# sites = unique(NS_sample$PlotNumber)
# NS_surveys = matrix(nrow = length(sites),ncol = 11)
# rownames(NS_surveys) = sites
# colnames(NS_surveys) = paste0("t",1:dim(NS_surveys)[2])
# for (i in 1:dim(NS_sample)[1]){
#   samp_vect = sort(NS_sample$FieldSeason[NS_sample$PlotNumber==NS_sample$PlotNumber[i]])
#   NS_surveys[as.character(NS_sample$PlotNumber[i]),paste0("t",which(samp_vect==NS_sample$FieldSeason[i]))] = NS_sample$FieldSeason[i]
# }
NS_surveys = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NS/NS_surveys.csv",row.names = "X")
rownames(NS_surveys) = paste0("9_",rownames(NS_surveys))
NS_surveys = data.frame(NS_surveys,matrix(nrow = dim(NS_surveys)[1],ncol = 18-dim(NS_surveys)[2]))
colnames(NS_surveys) = paste0("t",sprintf("%02.0f",1:18))

#Manitoba
files = list.files("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/MB/PSP_csvs/")
MB_surveys = matrix(nrow = length(files),ncol = 7)
rownames(MB_surveys) = files
for (i in files){
  file = read.csv(paste0("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/MB/PSP_csvs/",i),stringsAsFactors = F)
  year = sort(unique(file$YEAR_MEAS))
  MB_surveys[i,1:length(year)] = year
}
rownames(MB_surveys) = paste0("4_",gsub(".csv","",files))
colnames(MB_surveys) = paste0("t",1:7)
MB_surveys = data.frame(MB_surveys,matrix(nrow = dim(MB_surveys)[1],ncol = 18-dim(MB_surveys)[2]))
colnames(MB_surveys) = paste0("t",sprintf("%02.0f",1:18))

#New Brunswick
# NB_plots_yr = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NB/PSP_PLOTS_YR.csv",stringsAsFactors = F)
# NB_plots_yr[3,2] = "10001"
# NB_plots_yr["10498","measNum"]=6
# NB_plots_yr["10858","measNum"]=6
# NB_plots_yr["10864","measNum"]=6
# NB_plots_yr["10870","measNum"]=6
# NB_plots_yr["10940","measNum"]=6
# NB_plots_yr["10957","measNum"]=6
# NB_plots_yr["10974","measNum"]=6
# NB_plots_yr["10985","measNum"]=6
# NB_plots_yr["10991","measNum"]=6
# NB_plots_yr["12192","measNum"]=5
# NB_plots_yr["12198","measNum"]=5
# NB_plots_yr["12258","measNum"]=6
# NB_plots_yr["12264","measNum"]=6
# NB_plots_yr["12270","measNum"]=6
# NB_plots_yr["12308","measNum"]=5
# NB_plots_yr["12314","measNum"]=5
# NB_plots_yr["12344","measNum"]=5
# NB_plots_yr["12350","measNum"]=5
# NB_plots_yr["12390","measNum"]=6
# NB_plots_yr["12432","measNum"]=6
# NB_plots_yr["12372","measNum"]=6
# NB_plots_yr["11805","measNum"]=1
# NB_plots_yr["11812","measNum"]=1
# NB_plots_yr["11821","measNum"]=1
# NB_plots_yr["11822","measNum"]=1
# NB_plots_yr["11837","measNum"]=1
# NB_plots_yr["11848","measNum"]=1
# NB_plots_yr["11853","measNum"]=1
# NB_plots_yr["11878","measNum"]=1
# NB_plots_yr["11897","measNum"]=1
# NB_plots_yr["11898","measNum"]=1
# NB_plots_yr["11917","measNum"]=1
# NB_plots_yr["11962","measNum"]=1
# NB_plots_yr["12921","measNum"]=1
# NB_plots_yr["12924","measNum"]=1
# NB_plots_yr["12925","measNum"]=1
# NB_plots_yr["12926","measNum"]=1
# NB_plots_yr["12933","measNum"]=1
# NB_plots_yr["12948","measNum"]=1
# NB_plots_yr["12951","measNum"]=1
# NB_plots_yr["12953","measNum"]=1
# NB_plots_yr["12955","measNum"]=1
# NB_plots_yr["13032","measNum"]=1
# NB_plots_yr["13033","measNum"]=1
# NB_plots_yr["13046","measNum"]=1
# NB_plots_yr["13047","measNum"]=1
# 
# NB_plots_yr = NB_plots_yr[c(-12191,-12197,-12307,-12313,-12343,-12349),]
# sites = unique(NB_plots_yr$Plot)
# NB_surveys = matrix(nrow = length(sites),ncol = max(NB_plots_yr$measNum))
# rownames(NB_surveys) = sites
# colnames(NB_surveys) = paste0("t",1:dim(NB_surveys)[2])
# for (i in 1:dim(NB_plots_yr)[1]){
#   NB_surveys[as.character(NB_plots_yr$Plot[i]),paste0("t",NB_plots_yr$measNum[i])] = NB_plots_yr$MeasYr[i]
# }
NB_surveys = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/NB/NB_surveys.csv",row.names = "X")
rownames(NB_surveys) = paste0("8_",rownames(NB_surveys))
NB_surveys = data.frame(NB_surveys,matrix(nrow = dim(NB_surveys)[1],ncol = 18-dim(NB_surveys)[2]))
colnames(NB_surveys) = paste0("t",sprintf("%02.0f",1:18))
#There was a massive windstorm between 2006 and 2011 at site 1001, such that there were no trees to record in 2011. I'm removing that visit from the record.
NB_surveys["8_1001","t07"] = NA
#Other missing records, removing inventory from surveys csv
NB_surveys["8_10182","t02"] = NA

#Quebec
# QC_meas = as.data.frame(read_excel("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/QC/PLACETTE_MES.xlsx"))
# sites = unique(QC_meas$ID_PE)
# QC_surveys = matrix(nrow = length(sites),ncol = max(QC_meas$NO_MES))
# rownames(QC_surveys) = sites
# for (i in 1:dim(QC_meas)[1]){
#   QC_surveys[QC_meas$ID_PE[i],QC_meas$NO_MES[i]] = strsplit(as.character(QC_meas$DATE_SOND[i]),"-")[[1]][1]
# }
QC_surveys = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/QC/QC_surveys.csv",row.names = "X")
rownames(QC_surveys) = paste0("6_",rownames(QC_surveys))
colnames(QC_surveys) = paste0("t",1:6)
QC_surveys = data.frame(QC_surveys,matrix(nrow = dim(QC_surveys)[1],ncol = 18-dim(QC_surveys)[2]))
colnames(QC_surveys) = paste0("t",sprintf("%02.0f",1:18))

#Ontario
# ON_plots = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/ON/ON_plots.csv",stringsAsFactors = F)
# ON_package = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/ON/ON_package.csv",stringsAsFactors = F)
# ON_visit = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/ON/ON_visit.csv",stringsAsFactors = F)
# ON_visit = ON_visit[ON_visit$VisitTypeCode==2|ON_visit$VisitTypeCode==4,]
# ON_surveys = matrix(nrow = dim(ON_plots)[1],ncol = 6)
# rownames(ON_surveys) = ON_plots$PlotName
# for (i in 1:dim(ON_plots)[1]){
#   ON_package[ON_plots$ï..PlotKey[i]==ON_package$PlotKey,4] = ON_plots$PlotName[i]
# }
# for (i in 1:dim(ON_visit)[1]){
#   ON_package[ON_visit$PackageKey[i]==ON_package$ï..PackageKey,5] = ON_visit$FieldSeasonYear[i]
# }
# ON_package[ON_package[,5]==0,5] = NA
# for (i in 1:dim(ON_surveys)[1]){
#   samp_vect = sort(ON_package$CoOpMethod[ON_package$ApproachCode==rownames(ON_surveys)[i]])
#   ON_surveys[i,1:length(samp_vect)] = samp_vect
# }
# rownames(ON_surveys) = paste0("5_",rownames(ON_surveys))
# colnames(ON_surveys) = paste0("t",1:6)
ON_surveys = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/ON/ON_surveys.csv",stringsAsFactors = F,row.names = "X")
ON_surveys = data.frame(ON_surveys,matrix(nrow = dim(ON_surveys)[1],ncol = 18-dim(ON_surveys)[2]))
colnames(ON_surveys) = paste0("t",sprintf("%02.0f",1:18))

#CIPHA
CIPHA_surveys = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CIPHA/CIPHA_surveys.csv",stringsAsFactors = F,row.names = "X")
colnames(CIPHA_surveys) = paste0("t",sprintf("%02.0f",1:18))

#CAFI
CAFI_surveys = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/CAFI/CAFI_Site_Description_2015_GE2_wFires_msin_final.csv",stringsAsFactors = F)
rownames(CAFI_surveys) = paste0("13_",CAFI_surveys$PSP)
CAFI_surveys = CAFI_surveys[,c(25:29)]
colnames(CAFI_surveys) = paste0("t",1:5)
#For some reason, I don't have any data from years 3 and 4 for this site, so removing it from survey csv
CAFI_surveys["13_10319",c("t3","t4")] = NA
CAFI_surveys["13_10320",c("t3","t4")] = NA
CAFI_surveys["13_10321",c("t3","t4")] = NA
#Same with this one for year 2
CAFI_surveys["13_10409","t2"] = NA
CAFI_surveys["13_10410","t2"] = NA
CAFI_surveys["13_10411","t2"] = NA
#Same with this one for year 2
CAFI_surveys["13_10415","t2"] = NA
CAFI_surveys["13_10416","t2"] = NA
CAFI_surveys["13_10417","t2"] = NA
#Same with this one for year 2
CAFI_surveys["13_10421","t2"] = NA
CAFI_surveys["13_10422","t2"] = NA
CAFI_surveys["13_10423","t2"] = NA
CAFI_surveys = data.frame(CAFI_surveys,matrix(nrow = dim(CAFI_surveys)[1],ncol = 18-dim(CAFI_surveys)[2]))
colnames(CAFI_surveys) = paste0("t",sprintf("%02.0f",1:18))

#Saskatchewan
SK_surveys = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/SK/SK_surveys.csv",stringsAsFactors = F,row.names = "X")
SK_surveys = data.frame(SK_surveys,matrix(nrow = dim(SK_surveys)[1],ncol = 18-dim(SK_surveys)[2]))
colnames(SK_surveys) = paste0("t",sprintf("%02.0f",1:18))

#British Columbia
# BC_plots = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/BC/faib_data_plot_by_meas.csv",stringsAsFactors = F)
# BC_plots = BC_plots[!is.na(BC_plots$meas_no)]
# BC_surveys = matrix(nrow = length(unique(BC_plots$SAMP_ID)),ncol = max(BC_plots$no_meas))
# rownames(BC_surveys) = unique(BC_plots$SAMP_ID)
# colnames(BC_surveys) = paste0("t",1:max(BC_plots$no_meas))
# for (i in 1:dim(BC_plots)[1]){
#   BC_surveys[BC_plots$SAMP_ID[i],paste0("t",BC_plots$meas_no[i]+1)] = BC_plots$meas_yr[i]
# }
BC_surveys = read.csv("/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/BC/BC_surveys.csv",stringsAsFactors = F,row.names = "X")
BC_surveys = data.frame(BC_surveys,matrix(nrow = dim(BC_surveys)[1],ncol = 18-dim(BC_surveys)[2]))
colnames(BC_surveys) = paste0("t",sprintf("%02.0f",1:18))

surveys = rbind(AB_surveys,BC_surveys,SK_surveys,MB_surveys,ON_surveys,QC_surveys,NL_surveys,NB_surveys,NS_surveys,YT_surveys,NWT_surveys,CAFI_surveys,CIPHA_surveys)

write.csv(surveys,"/att/nobackup/scooperd/scooperdock/EWS/data/psp/survey_dates.csv")
