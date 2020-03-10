#####
# ksolvik@whrc.org
# 1/11/16
# For Landsat, we assumed that if the site was touched by fire, the Landsat pixel fully burned.
# Otherwise, no burn.
# Uses the output of an intersection done in QGIS with the LFDB polygons: "burnCSV"
#####
rm(list=ls())
require(raster)
require(maptools)


### Inputs
yearStart = 1917
yearEnd = 2017
dataType = "LANDSAT"
region = "cipha"
basePath = "/Volumes/"
plotLev = TRUE # TRUE if plot level, FALSE if site level analysis
plotLevText="PSP"
noData = 0

fire_proj = "+proj=sinu +lon_0=0 +x_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
fires = readOGR(dsn="/att/nobackup/scooperd/scooperdock/EWS/data/burn/CAN_AK_burnyear_dissolved.shp")

### Define a couple vars based on inputs
# if(plotLev) {
#   plotLevText = "PSP"
# } else {
#   plotLevText = "Site"
# }

outFile = paste0("/att/nobackup/scooperd/scooperdock/EWS/data/fire/",dataType,"_fire.csv")

plotsCSV = "/att/nobackup/scooperd/scooperdock/EWS/data/raw_psp/All_sites_101218.csv"

yList = yearStart:yearEnd

### Burn Years



### Create output data frame
plotLocs = read.csv(plotsCSV)
plotLocs = plotLocs[c(-31654,-33445),]
rownames(plotLocs) = plotLocs$Plot_ID
plots = SpatialPoints(plotLocs[,c("Longitude","Latitude")],proj4string=CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"))
plots = spTransform(plots,fires@proj4string)
outData = data.frame(plotLocs[,"Plot_ID"],matrix(data = noData, ncol = length(yList),nrow = dim(plotLocs)[1]))
colnames(outData) = c("Plot_ID",yList)

inter2 = intersect(plots,fires)
overlap = matrix(ncol = 2,nrow = dim(inter2@coords)[1])
for (i in 1:dim(inter2@coords)[1]){
  long = inter2@coords[i,1]
  lat = inter2@coords[i,2]
  overlap[which(inter2@coords[,1]==long&inter2@coords[,2]==lat),1] = names(which(plots@coords[,1]==long&plots@coords[,2]==lat))
  overlap[i,2] = as.character(inter2$BurnYear[i])
}
colnames(overlap) = c("Plot_ID","year")

### Loop over years and get the burn % tiff
for(r in 1:dim(overlap)[1]) {
  outData[(as.character(outData$Plot_ID)==as.character(overlap[r,1])),as.character(overlap[r,2])] = 100
}
  
write.csv(outData,file = outFile)

