
library(dplyr)

# ======================================================================
# ======================================================================
# ========== This scipt has been modified by ab to fic a bug ==========
# All modifications are maked like this section
# ========== Check what system i'm running on ==========
rm(list = ls()) #cleans any files in the environment out
#if (startsWith(Sys.info()["nodename"], "BURRELL")){
#  # Preserves compatibility with script
#  setwd("C:/Users/aburrell/Documents/Boreal")
#}else{
#  # setwd("/att/nobackup/scooperd/scooperdock")
#  setwd("/mnt/data1/boreal/scooperdock")
#}
setwd("/media/arden/Alexithymia/Boreal")

# ========== dataframe check ==========
check_dataframe = function(df) {
  # +++++ Check if the dataframe coming in is empty
  if (empty(df)) {
    print("Dataset is empty")
    browser()
    stop("A dataset that should have values is empty. exiting code")
  }else if (all(is.na(df))){
    print("Dataset is all NA")
    browser()
    stop("A dataset that should have values is empty. exiting code")
  }
}

# ======================================================================

surveys = read.csv("./EWS_package/data/psp/survey_datesV2.csv",row.names = "X")


provinces = rbind(c(1,"BC",9),
                  c(2,"AB",9.1),
                  c(3,"SK",7.1),
                  c(4,"MB",7.1),
                  c(5,"ON",9),
                  c(6,"QC",9),
                  c(7,"NL",8),
                  c(8,"NB",8),
                  c(9,"NS",9.1),
                  c(11,"YT",7.5),
                  c(12,"NWT",6.7),
                  c(13,"CAFI",7.5)
                  )


dbh_estimates = list(
  "1" = c(0.0510, 2.4529, 0.0297, 2.1131, 0.0120, 2.4165, 0.0276, 1.6215), #Lambert2005
  "2" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "3" = c(0.0510, 2.4529, 0.0297, 2.1131, 0.0120, 2.4165, 0.0276, 1.6215), #Lambert2005, balsam poplar
  "4" = c(0.0460, 2.4312, 0.0074, 2.4442, 0.0086, 2.7326, 0.0114, 2.0860), #Ung2008, black cottonwood & red alder
  "5" = c(0.0959, 2.3430, 0.0308, 2.2240, 0.0047, 2.6530, 0.0080, 2.0149), #Lambert2005
  "6" = c(0.1014, 2.3448, 0.0291, 2.0893, 0.0175, 2.4846, 0.0515, 1.5198), #Lambert2005
  "7" = c(0.1315, 2.3129, 0.0631, 1.9241, 0.0330, 2.3741, 0.0393, 1.6930), #Lambert2005
  "8" = c(0.0608, 2.4735, 0.0159, 2.4123, 0.0082, 2.5139, 0.0235, 1.6656), #Ung2008
  "9" = c(0.0424, 2.4289, 0.0057, 2.4789, 0.0322, 2.1313, 0.0645, 1.9400), #Ung2008
  "10" = c(0.0534, 2.4030, 0.0115, 2.3484, 0.0070, 2.5406, 0.0840, 1.6695), #Lambert2005
  "11" = c(0.0648, 2.3927, 0.0162, 2.1959, 0.0156, 2.2916, 0.0861, 1.6261), #Lambert2005, softwood
  "12" = c(0.0250, 2.6378, 0.0061, 2.5375, 0.0178, 2.4255, 0.0416, 2.0130), #Ung2008
  "14" = c(0.0648, 2.3927, 0.0162, 2.1959, 0.0156, 2.2916, 0.0861, 1.6261), #Lambert2005, softwood
  "16" = c(0.0654, 2.2121, 0.0114, 2.1432, 0.0335, 1.9367, 0.0499, 1.7278), #Lambert2005
  "17" = c(0.3743, 1.9406, 0.0679, 1.8377, 0.0796, 2.0103, 0.0840, 1.2319), #Lambert2005
  "18" = c(0.0111, 2.8027, 0.0003, 3.2721, 0.1158, 1.7196, 0.1233, 1.5152), #Ung2008 
  "19" = c(0.0648, 2.3927, 0.0162, 2.1959, 0.0156, 2.2916, 0.0861, 1.6261), #Lambert2005, softwood
  "20" = c(0.0460, 2.4312, 0.0074, 2.4442, 0.0086, 2.7326, 0.0114, 2.0860), #Ung2008, black cottonwood & red alder
  "21" = c(0.0604, 2.4959, 0.0140, 2.3923, 0.0147, 2.5227, 0.0591, 1.6036), #Ung2008
  "22" = c(0.0402, 2.5804, 0.0073, 2.4859, 0.0401, 2.1826, 0.0750, 1.3436), #Lambert2005, american elm
  "23" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "24" = c(0.1932, 2.1569, 0.0192, 2.2475, 0.0305, 2.4044, 0.1119, 1.3973), #Lambert2005
  "25" = c(0.0204, 2.6974, 0.0069, 2.5462, 0.0404, 2.1388, 0.1233, 1.6636), #Ung2008
  "26" = c(0.1478, 2.2986, 0.0120, 2.2388, 0.0370, 2.3680, 0.0376, 1.6164), #Lambert2005
  "27" = c(0.1861, 2.1665, 0.0406, 1.9946, 0.0461, 2.2291, 0.1106, 1.2277), #Lambert2005
  "28" = c(0.0941, 2.3491, 0.0323, 2.0761, 0.0448, 1.9771, 0.0538, 1.3584), #Lambert2005
  "29" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "30" = c(0.0619, 2.3821, 0.0139, 2.3282, 0.0217, 2.2653, 0.0776, 1.6995), #Lambert2005
  "31" = c(0.0141, 2.8668, 0.0025, 2.8062, 0.0703, 1.9547, 0.1676, 1.4339), #Ung2008, western hemlock
  "33" = c(0.0141, 2.8668, 0.0025, 2.8062, 0.0703, 1.9547, 0.1676, 1.4339), #Ung2008
  "34" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "38" = c(0.0648, 2.3927, 0.0162, 2.1959, 0.0156, 2.2916, 0.0861, 1.6261), #Lambert2005, softwood
  "40" = c(0.1277, 1.9778, 0.0377, 1.6064, 0.0254, 2.2884, 0.0550, 1.8656), #Lambert2005
  "42" = c(0.0625, 2.4475, 0.0174, 2.1109, 0.0196, 2.2652, 0.0801, 1.4875), #Lambert2005
  "43" = c(0.0625, 2.4475, 0.0174, 2.1109, 0.0196, 2.2652, 0.0801, 1.4875), #Lambert2005, tamarack
  "45" = c(0.0804, 2.4041, 0.0184, 2.0703, 0.0079, 2.4155, 0.0389, 1.7290), #Lambert2005
  "48" = c(0.0323, 2.6825, 0.0144, 2.1768, 0.0209, 2.1772, 0.0584, 1.6432), #Ung2008
  "49" = c(0.0997, 2.2709, 0.0192, 2.2038, 0.0056, 2.6011, 0.0284, 1.9375), #Lambert2005, E white pine
  "50" = c(0.0564, 2.4465, 0.0188, 2.0527, 0.0033, 2.7515, 0.0212, 2.0690), #Lambert2005
  "53" = c(0.0997, 2.2709, 0.0192, 2.2038, 0.0056, 2.6011, 0.0284, 1.9375), #Lambert2005
  "54" = c(0.0648, 2.3927, 0.0162, 2.1959, 0.0156, 2.2916, 0.0861, 1.6261), #Lambert2005, softwood
  "58" = c(0.0762, 2.3335, 0.0338, 1.9845, 0.0113, 2.6211, 0.0188, 1.7881), #Lambert2005, white oak
  "60" = c(0.1754, 2.1616, 0.0381, 2.0991, 0.0085, 2.7790, 0.0373, 1.6740), #Lambert2005
  "61" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "62" = c(0.0762, 2.3335, 0.0338, 1.9845, 0.0113, 2.6211, 0.0188, 1.7881), #Lambert2005
  "63" = c(0.0648, 2.3927, 0.0162, 2.1959, 0.0156, 2.2916, 0.0861, 1.6261), #Lambert2005, softwood
  "64" = c(0.0494, 2.5025, 0.0148, 2.2494, 0.0291, 2.0751, 0.1631, 1.4222), #Ung2008
  "65" = c(0.0223, 2.7169, 0.0118, 2.2733, 0.0336, 2.2123, 0.0683, 1.8022), #Ung2008
  "66" = c(0.0989, 2.2814, 0.0220, 2.0908, 0.0005, 3.2750, 0.0066, 2.4213), #Lambert2005
  "68" = c(0.0302, 2.5776, 0.0066, 2.4433, 0.0739, 1.8342, 0.0157, 2.3113), #Ung2008
  "69" = c(0.0334, 2.5980, 0.0114, 2.3057, 0.0302, 2.0927, 0.1515, 1.5012), #Ung2008
  "70" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "71" = c(0.0562, 2.4102, 0.0302, 2.0976, 0.0230, 2.2382, 0.0288, 1.6378), #Lambert2005
  "73" = c(0.0402, 2.5804, 0.0073, 2.4859, 0.0401, 2.1826, 0.0750, 1.3436), #Lambert2005
  "74" = c(0.0402, 2.5804, 0.0073, 2.4859, 0.0401, 2.1826, 0.0750, 1.3436), #Lambert2005, american elm
  "75" = c(0.0402, 2.5804, 0.0073, 2.4859, 0.0401, 2.1826, 0.0750, 1.3436), #Lambert2005, american elm
  "78" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "80" = c(0.0741, 2.3875, 0.0182, 2.2181, 0.0277, 2.2797, 0.0764, 1.5861), #Ung2008, All species
  "81" = c(0.0648, 2.3927, 0.0162, 2.1959, 0.0156, 2.2916, 0.0861, 1.6261), #Lambert2005, softwood
  "82" = c(0.0648, 2.3927, 0.0162, 2.1959, 0.0156, 2.2916, 0.0861, 1.6261), #Lambert2005, softwood
  "83" = c(0.1929, 1.9672, 0.0671, 1.5911, 0.0278, 2.1336, 0.0293, 1.9502), #Lambert2005
  "84" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "85" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "86" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, amelanchier ash using hardwood equation
  "87" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, mountain ash using hardwood equation
  "88" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, mountain ash using hardwood equation
  "89" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, mountain maple using hardwood equation
  "90" = c(0.0720, 2.3885, 0.0168, 2.2569, 0.0088, 2.5689, 0.0099, 1.8985), #Lambert2005
  "91" = c(0.2324, 2.1000, 0.0278, 2.0433, 0.0028, 3.1020, 0.1430, 1.2580), #Lambert2005
  "92" = c(0.1571, 2.1817, 0.0416, 2.0509, 0.0177, 2.3370, 0.1041, 1.2185), #Lambert2005
  "93" = c(0.2116, 2.2013, 0.0365, 2.1133, 0.0087, 2.8927, 0.0173, 1.9830), #Lambert2005, for Hickory genus
  "94" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "95" = c(0.0762, 2.3335, 0.0338, 1.9845, 0.0113, 2.6211, 0.0188, 1.7881), #Lambert2005, white oak
  "96" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, apple using hardwood equation
  "97" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, mountain ash using hardwood equation
  "98" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "99" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "100" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "101" = c(0.2116, 2.2013, 0.0365, 2.1133, 0.0087, 2.8927, 0.0173, 1.9830), #Lambert2005, for Hickory genus
  "102" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "103" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Buckthorn, Rhamnus using hardwood equation
  "104" = c(0.0625, 2.4475, 0.0174, 2.1109, 0.0196, 2.2652, 0.0801, 1.4875), #Lambert2005, tamarack
  "105" = c(0.0648, 2.3927, 0.0162, 2.1959, 0.0156, 2.2916, 0.0861, 1.6261), #Lambert2005, softwood
  "106" = c(0.0648, 2.3927, 0.0162, 2.1959, 0.0156, 2.2916, 0.0861, 1.6261), #Lambert2005, softwood
  "107" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "108" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "109" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "110" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "111" = c(0.0625, 2.4475, 0.0174, 2.1109, 0.0196, 2.2652, 0.0801, 1.4875), #Lambert2005, tamarack
  "112" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, alder using hardwood equation
  "113" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "114" = c(0.0604, 2.4959, 0.0140, 2.3923, 0.0147, 2.5227, 0.0591, 1.6036), #Ung2008, paper birch
  "115" = c(0.0604, 2.4959, 0.0140, 2.3923, 0.0147, 2.5227, 0.0591, 1.6036), #Ung2008, paper birch
  "116" = c(0.0648, 2.3927, 0.0162, 2.1959, 0.0156, 2.2916, 0.0861, 1.6261), #Lambert2005, softwood
  "117" = c(0.0648, 2.3927, 0.0162, 2.1959, 0.0156, 2.2916, 0.0861, 1.6261), #Lambert2005, softwood
  "118" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "119" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "120" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "121" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "122" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "123" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "124" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "125" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "126" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "127" = c(0.0625, 2.4475, 0.0174, 2.1109, 0.0196, 2.2652, 0.0801, 1.4875), #Lambert2005, tamarack
  "128" = c(0.0648, 2.3927, 0.0162, 2.1959, 0.0156, 2.2916, 0.0861, 1.6261), #Lambert2005, softwood
  "129" = c(0.0625, 2.4475, 0.0174, 2.1109, 0.0196, 2.2652, 0.0801, 1.4875), #Lambert2005, tamarack 
  "130" = c(0.0648, 2.3927, 0.0162, 2.1959, 0.0156, 2.2916, 0.0861, 1.6261), #Lambert2005, softwood 
  "131" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "132" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "133" = c(0.0604, 2.4959, 0.0140, 2.3923, 0.0147, 2.5227, 0.0591, 1.6036), #Ung2008, paper birch
  "134" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "135" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "136" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "137" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "138" = c(0.0204, 2.6974, 0.0069, 2.5462, 0.0404, 2.1388, 0.1233, 1.6636), #Ung2008, Douglas fir
  "139" = c(0.0141, 2.8668, 0.0025, 2.8062, 0.0703, 1.9547, 0.1676, 1.4339), #Ung2008, western hemlock
  "140" = c(0.0625, 2.4475, 0.0174, 2.1109, 0.0196, 2.2652, 0.0801, 1.4875), #Lambert2005, tamarack
  "141" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "142" = c(0.0648, 2.3927, 0.0162, 2.1959, 0.0156, 2.2916, 0.0861, 1.6261), #Lambert2005, softwood
  "143" = c(0.0648, 2.3927, 0.0162, 2.1959, 0.0156, 2.2916, 0.0861, 1.6261), #Lambert2005, softwood
  "144" = c(0.0648, 2.3927, 0.0162, 2.1959, 0.0156, 2.2916, 0.0861, 1.6261), #Lambert2005, softwood
  "145" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "146" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "147" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "148" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "149" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "150" = c(0.0871, 2.3702, 0.0241, 2.1969, 0.0167, 2.4807, 0.0390, 1.6229), #Lambert2005, hardwood
  "151" = c(0.0648, 2.3927, 0.0162, 2.1959, 0.0156, 2.2916, 0.0861, 1.6261) #Lambert2005, softwood
   )

AK_dbh_estimates = list(
  "1" = c(98.26, 2.32, 22.16, 1.62, 17.24, 2.30, 133.71, 2.29), #Alexander2012
  "8" = c(64.01, 2.51, 18.98, 1.53, 41.74, 1.83, 134.10, 2.26), #Alexander2012
  "64" = c(117.91, 1.99, 55.40, 1.47, 83.52, 1.80, 271.46, 1.84), #Alexander2012
  "69" = c(48.44, 2.51, 25.22, 2.04, 29.34, 2.24, 96.77, 2.40), #Alexander2012
  "114" = c(147.96, 2.25, 6.39, 2.10, 15.15, 2.49, 164.18, 2.29), #Alexander2012
  "115" = c(147.96, 2.25, 6.39, 2.10, 15.15, 2.49, 164.18, 2.29), #Alexander2012, Alaska birch
  "116" = c(48.44, 2.51, 25.22, 2.04, 29.34, 2.24, 96.77, 2.40) #Alexander2012, White spruce
)

growth = data.frame()
recruit = data.frame()
recruit_N = data.frame()
mort = data.frame()
mort_N = data.frame()
living_wood = data.frame()
living_wood_N = data.frame()
sp_comp_mass = data.frame()
sp_mort_mass = data.frame()
sp_recruit_mass = data.frame()
sp_growth = data.frame()
sp_live_mass = data.frame()
sp_comp_N = data.frame()
sp_mort_N = data.frame()
sp_recruit_N = data.frame()
sp_ID = data.frame()
sp_live_N = data.frame()


for (i in c(1:12)){
  files = list.files(paste0("./EWS_package/data/raw_psp/",provinces[i,2],"/checks/"))
  for(n in files){
    df = read.csv(paste0("./EWS_package/data/raw_psp/",provinces[i,2],"/checks/",n),row.names = "X",stringsAsFactors = F)
    med_dbh = mean(unlist(df[,grep('dbh',colnames(df))]),na.rm=T)
    too_big_rows = which(apply(as.matrix(df[,grep('dbh',colnames(df))]),1,max,na.rm=T)>20*med_dbh)
    if(length(too_big_rows)>0){
      too_big = df[too_big_rows,] 
      for(j in 1:nrow(too_big)){
        if (length(too_big[j,grep('dbh',colnames(too_big))][!is.na(too_big[j,grep('dbh',colnames(too_big))])])==1|
            any(too_big[j,grep('dbh',colnames(too_big))]<10*med_dbh)){
          change = which(too_big[j,grep('dbh',colnames(too_big))]>20*med_dbh)
          print(paste0(n,": ",too_big[j,grep('dbh',colnames(too_big))][j,change]))
          too_big[j,grep('dbh',colnames(too_big))][j,change] = NA
          
        }
      }
      df[too_big_rows,] = too_big
    }
   
    
    measurements = (dim(df)[2]-1)/3
    biomass = matrix(nrow = dim(df)[1],ncol = measurements*7)
    rownames(biomass) = rownames(df)
    cols = vector(length=dim(biomass)[2])
    for (j in 1:measurements){
      cols[((j-1)*7+1):((j-1)*7+7)] = paste0(c("wood_t","bark_t","stem_t","branches_t","foliage_t","crown_t","total_t"),j)
    }
    colnames(biomass) = cols
    print(n)
    for (j in 1:dim(biomass)[1]){
      for(k in 1:measurements){
        if(!is.na(df[j,paste0("status_t",k)]) & !is.na(df[j,paste0("dbh_t",k)]) & as.numeric(df[j,paste0("dbh_t",k)])>=as.numeric(provinces[i,3])){
          #Setting sp NAs that are alive with dbh to "80" - unknown
          if (is.na(df[j,paste0("sp_t",k)])){
            df[j,paste0("sp_t",k)] = 80
          }
          if(provinces[i,2]!="CAFI"){
            biomass[j,paste0("wood_t",k)]=dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][1]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][2])
            biomass[j,paste0("bark_t",k)]=dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][3]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][4])
            biomass[j,paste0("stem_t",k)]=biomass[j,paste0("wood_t",k)]+biomass[j,paste0("bark_t",k)]
            biomass[j,paste0("branches_t",k)]=dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][5]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][6])
            biomass[j,paste0("foliage_t",k)]=dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][7]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][8])
            biomass[j,paste0("crown_t",k)]=biomass[j,paste0("branches_t",k)]+biomass[j,paste0("foliage_t",k)]
            biomass[j,paste0("total_t",k)]=biomass[j,paste0("wood_t",k)]+biomass[j,paste0("bark_t",k)]+biomass[j,paste0("branches_t",k)]+biomass[j,paste0("foliage_t",k)]
          }else{#Different equations for Alaska
            if(any(df[j,paste0("sp_t",k)]==names(AK_dbh_estimates))){ #For trees I have AK specific equations for, use Alexander equations
              biomass[j,paste0("stem_t",k)]=AK_dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][1]*((as.numeric(df[j,paste0("dbh_t",k)]))^AK_dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][2])/1000
              biomass[j,paste0("foliage_t",k)]=AK_dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][3]*((as.numeric(df[j,paste0("dbh_t",k)]))^AK_dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][4])/1000
              biomass[j,paste0("crown_t",k)]=AK_dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][5]*((as.numeric(df[j,paste0("dbh_t",k)]))^AK_dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][6])/1000
              biomass[j,paste0("total_t",k)]=AK_dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][7]*((as.numeric(df[j,paste0("dbh_t",k)]))^AK_dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][8])/1000
            # }else if(any(df[j,paste0("sp_t",k)]==c(69,116,8,1))){ #For White spruce, Lutz spruce, aspen, and balsam poplar average Alexander and Lambert/Ung
            #   biomass[j,paste0("stem_t",k)]=((AK_dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][1]*((as.numeric(df[j,paste0("dbh_t",k)]))^AK_dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][2])/1000) +
            #     (dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][1]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][2])+
            #        dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][3]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][4])))/2
            #   biomass[j,paste0("foliage_t",k)]=((AK_dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][3]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][4])/1000) +
            #     dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][7]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][8]))/2
            #   biomass[j,paste0("crown_t",k)]=((AK_dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][5]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][6])/1000) +
            #     dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][5]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][6]) +
            #        dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][7]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][8]))/2
            #   biomass[j,paste0("total_t",k)]=((AK_dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][7]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][8])/1000) +
            #     dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][1]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][2]) +
            #       dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][3]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][4]) +
            #       dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][5]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][6]) +
            #       dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][7]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][8]))/2
            }else{ #For trees without AK equations, use Lambert/Ung
              biomass[j,paste0("wood_t",k)]=dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][1]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][2])
              biomass[j,paste0("bark_t",k)]=dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][3]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][4])
              biomass[j,paste0("stem_t",k)]=biomass[j,paste0("wood_t",k)]+biomass[j,paste0("bark_t",k)]
              biomass[j,paste0("branches_t",k)]=dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][5]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][6])
              biomass[j,paste0("foliage_t",k)]=dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][7]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][8])
              biomass[j,paste0("crown_t",k)]=biomass[j,paste0("branches_t",k)]+biomass[j,paste0("foliage_t",k)]
              biomass[j,paste0("total_t",k)]=biomass[j,paste0("wood_t",k)]+biomass[j,paste0("bark_t",k)]+biomass[j,paste0("branches_t",k)]+biomass[j,paste0("foliage_t",k)]
            }
          }
        }
      }
    }
    if(sum(!is.na(biomass))>0){
      write.csv(biomass,paste0("./EWS_package/data/raw_psp/",provinces[i,2],"/PSP/biomass/",gsub("_check.csv","",n),".csv"))
      if(measurements>1){
        growth_mass = matrix(nrow = dim(biomass)[1], ncol = measurements-1)
        rownames(growth_mass) = rownames(biomass)
        colnames(growth_mass) = paste0("t",2:measurements)
        
        mort_mass = growth_mass
        mort_n = mort_mass
        recruit_n = mort_n
        recruit_mass = mort_n
        for (j in 1:dim(biomass)[1]){
          for (k in 2:measurements){
            if (!is.na(biomass[j,paste0("total_t",k)])&!is.na(biomass[j,paste0("total_t",k-1)])){
              growth_mass[j,k-1] = biomass[j,paste0("total_t",k)] - biomass[j,paste0("total_t",k-1)]
            }else if(is.na(biomass[j,paste0("total_t",k)])&!is.na(biomass[j,paste0("total_t",k-1)])){
              mort_mass[j,k-1] = biomass[j,paste0("total_t",k-1)]
              mort_n[j,k-1] = 1
            }else if(!is.na(biomass[j,paste0("total_t",k)])&is.na(biomass[j,paste0("total_t",k-1)])){
              recruit_mass[j,k-1] = biomass[j,paste0("total_t",k)]
              recruit_n[j,k-1] = 1
            }
          }
        }
        for (k in 1:measurements){
          living_wood[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("live_mass_t",k)] = sum(biomass[,paste0("total_t",k)],na.rm = T)/1000/as.numeric(unique(df$plot_size))
          living_wood_N[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("live_N_t",k)] = sum(!is.na(biomass[,paste0("total_t",k)]))/as.numeric(unique(df$plot_size))
          if (k!=1){
            #Calcucate mortality, recruitment and growth in biomass in Mg/ha/yr and in stemcount instems/ha/year
            int = surveys[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),k]-surveys[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),k-1]
            growth[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("growth_t",k)] = sum(growth_mass[,paste0("t",k)],na.rm=T)/1000/int/as.numeric(unique(df$plot_size))
            mort[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("mort_mass_t",k)] = sum(mort_mass[,paste0("t",k)],na.rm=T)/1000/int/as.numeric(unique(df$plot_size))
            mort_N[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("mort_N_t",k)] = sum(!is.na(mort_n[,paste0("t",k)]))/int/as.numeric(unique(df$plot_size))
            recruit[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("recruit_mass_t",k)] = sum(recruit_mass[,paste0("t",k)],na.rm=T)/1000/int/as.numeric(unique(df$plot_size))
            recruit_N[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("recruit_N_t",k)] = sum(!is.na(recruit_n[,paste0("t",k)]))/int/as.numeric(unique(df$plot_size))
          }
        }
      }
      sp = unique(unlist(df[1,paste0("sp_t",1:measurements)]))[!is.na(unique(unlist(df[1,paste0("sp_t",1:measurements)])))]
      for (k in 2:dim(df)[1]){
        sp = c(sp,unique(unlist(df[k,paste0("sp_t",1:measurements)]))[!is.na(unique(unlist(df[k,paste0("sp_t",1:measurements)])))])
      }
      species = unique(sp)
      count = vector(length = length(species))
      for (k in 1:length(species)){
        count[k] = sum(sp==species[k])
      }
      species_count = data.frame(species,count)
      species_count = arrange(species_count,desc(count))
      for (k in sprintf("%02.0f",1:length(species))){
        sp_ID[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("sp",k)] = species_count[as.numeric(k),1]
        # sp_mort[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("sp",k)] = species_count[k,1]
        # sp_recruit[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("sp",k)] = species_count[k,1]
        # sp_growth[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("sp",k)] = species_count[k,1]
        for (s in sprintf("%02.0f",1:measurements)){
          sp_comp_mass[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("sp",k,"_mass_%_t",s)] = sum(biomass[df[,paste0("sp_t",as.numeric(s))]==species_count[as.numeric(k),1],paste0("total_t",as.numeric(s))],na.rm = T)/sum(biomass[,paste0("total_t",as.numeric(s))],na.rm = T)
          sp_comp_N[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("sp",k,"_N_%_t",s)] = sum(df[!is.na(biomass[,paste0("total_t",as.numeric(s))]),paste0("sp_t",as.numeric(s))]==species_count[as.numeric(k),1],na.rm = T)/sum(!is.na(df[!is.na(biomass[,paste0("total_t",as.numeric(s))]),paste0("sp_t",as.numeric(s))]))
          sp_live_mass[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("sp",k,"_mass_t",s)] = sum(biomass[,paste0("total_t",as.numeric(s))],na.rm = T)/1000/as.numeric(unique(df$plot_size))*sp_comp_mass[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("sp",k,"_mass_%_t",s)]
          sp_live_N[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("sp",k,"_N_t",s)] = sum(!is.na(biomass[,paste0("total_t",as.numeric(s))]))/as.numeric(unique(df$plot_size))*sp_comp_N[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("sp",k,"_N_%_t",s)]
          if (s!="01"){
            #Calcucate mortality, recruitment and growth in biomass in Mg/ha/yr and in stemcount in stems/ha/year
            int = surveys[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),as.numeric(s)]-surveys[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),as.numeric(s)-1]
            sp_mort_mass[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("sp",k,"_mort_mass_t",s)] = sum(mort_mass[df[,paste0("sp_t",as.numeric(s))]==species_count[as.numeric(k),1],paste0("t",as.numeric(s))],na.rm = T)/1000/int/as.numeric(unique(df$plot_size))
            sp_mort_N[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("sp",k,"_mort_N_t",s)] = sum(mort_n[df[,paste0("sp_t",as.numeric(s))]==species_count[as.numeric(k),1],paste0("t",as.numeric(s))],na.rm = T)/int/as.numeric(unique(df$plot_size))
            sp_recruit_mass[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("sp",k,"_recruit_mass_t",s)] = sum(recruit_mass[df[,paste0("sp_t",as.numeric(s))]==species_count[as.numeric(k),1],paste0("t",as.numeric(s))],na.rm = T)/1000/int/as.numeric(unique(df$plot_size))
            sp_recruit_N[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("sp",k,"_recruit_N_t",s)] = sum(recruit_n[df[,paste0("sp_t",as.numeric(s))]==species_count[as.numeric(k),1],paste0("t",as.numeric(s))],na.rm = T)/int/as.numeric(unique(df$plot_size))
            sp_growth[paste0(provinces[i,1],"_",gsub("_check.csv","",n)),paste0("sp",k,"_growth_mass_t",s)] = sum(growth_mass[df[,paste0("sp_t",as.numeric(s))]==species_count[as.numeric(k),1],paste0("t",as.numeric(s))],na.rm = T)/1000/int/as.numeric(unique(df$plot_size))
          }
        }
      }
    }
  }
  print(paste0(provinces[i,2]," done"))
}

all = data.frame(living_wood,living_wood_N,growth,recruit,recruit_N,mort,mort_N)

# ========== AB modified the pathways to a new version 2 ==========
check_dataframe(all)
write.csv(all,"./EWS_package/data/psp/PSP_total_changesV2.csv")
check_dataframe(sp_ID)
write.csv(sp_ID,"./EWS_package/data/psp/PSP_sp_IDsV2.csv")
check_dataframe(sp_comp_mass)
write.csv(sp_comp_mass,"./EWS_package/data/psp/PSP_sp_comp_massV2.csv")
check_dataframe(sp_mort_mass)
write.csv(sp_mort_mass,"./EWS_package/data/psp/PSP_sp_mort_massV2.csv")
check_dataframe(sp_recruit_mass)
write.csv(sp_recruit_mass,"./EWS_package/data/psp/PSP_sp_recruit_massV2.csv")
check_dataframe(sp_live_mass)
write.csv(sp_live_mass,"./EWS_package/data/psp/PSP_sp_live_massV2.csv")
check_dataframe(sp_comp_N)
write.csv(sp_comp_N,"./EWS_package/data/psp/PSP_sp_comp_NV2.csv")
check_dataframe(sp_mort_N)
write.csv(sp_mort_N,"./EWS_package/data/psp/PSP_sp_mort_NV2.csv")
check_dataframe(sp_recruit_N)
write.csv(sp_recruit_N,"./EWS_package/data/psp/PSP_sp_recruit_NV2.csv")
check_dataframe(sp_growth)
write.csv(sp_growth,"./EWS_package/data/psp/PSP_sp_growthV2.csv")
check_dataframe(sp_live_N)
write.csv(sp_live_N,"./EWS_package/data/psp/PSP_sp_live_NV2.csv")

# 
# #Do it separately for CIPHA
# CIPHA_growth = data.frame()
# CIPHA_recruit = data.frame()
# CIPHA_recruit_N = data.frame()
# CIPHA_mort = data.frame()
# CIPHA_mort_N = data.frame()
# CIPHA_living_wood = data.frame()
# CIPHA_living_wood_N = data.frame()
# CIPHA_sp_comp = data.frame()
# CIPHA_sp_mort = data.frame()
# CIPHA_sp_recruit = data.frame()
# CIPHA_sp_growth = data.frame()
# CIPHA_sp_ID = data.frame()
# CIPHA_sp_live = data.frame()
# 
# files = list.files(paste0("./EWS_package/data/raw_psp/CIPHA/checks/"))
# 
# for(n in files){
#   df = read.csv(paste0("./EWS_package/data/raw_psp/CIPHA/checks/",n),row.names = "X",stringsAsFactors = F)
#   measurements = (dim(df)[2]-1)/3
#   biomass = matrix(nrow = dim(df)[1],ncol = measurements*7)
#   rownames(biomass) = rownames(df)
#   cols = vector(length=dim(biomass)[2])
#   for (j in 1:measurements){
#     cols[((j-1)*7+1):((j-1)*7+7)] = paste0(c("wood_t","bark_t","stem_t","branches_t","foliage_t","crown_t","total_t"),j)
#   }
#   colnames(biomass) = cols
#   print(n)
#   df[df=='D'] = NA
#   for (j in 1:dim(biomass)[1]){
#     for(k in 1:measurements){
#       if(!is.na(df[j,paste0("status_t",k)]) & !is.na(df[j,paste0("dbh_t",k)]) & as.numeric(df[j,paste0("dbh_t",k)])>=7.5){
#         #Setting sp NAs that are alive with dbh to "80" - unknown
#         if (is.na(df[j,paste0("sp_t",k)])){
#           df[j,paste0("sp_t",k)] = 80
#         }
#         biomass[j,paste0("wood_t",k)]=dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][1]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][2])
#         biomass[j,paste0("bark_t",k)]=dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][3]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][4])
#         biomass[j,paste0("stem_t",k)]=biomass[j,paste0("wood_t",k)]+biomass[j,paste0("bark_t",k)]
#         biomass[j,paste0("branches_t",k)]=dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][5]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][6])
#         biomass[j,paste0("foliage_t",k)]=dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][7]*((as.numeric(df[j,paste0("dbh_t",k)]))^dbh_estimates[[as.character(df[j,paste0("sp_t",k)])]][8])
#         biomass[j,paste0("crown_t",k)]=biomass[j,paste0("branches_t",k)]+biomass[j,paste0("foliage_t",k)]
#         biomass[j,paste0("total_t",k)]=biomass[j,paste0("wood_t",k)]+biomass[j,paste0("bark_t",k)]+biomass[j,paste0("branches_t",k)]+biomass[j,paste0("foliage_t",k)]
#       }
#     }
#   }
#   write.csv(biomass,paste0("./EWS_package/data/raw_psp/CIPHA/PSP/biomass/",gsub("_check.csv","",n),".csv"))
#   if(measurements>1){
#     growth_mass = matrix(nrow = dim(biomass)[1], ncol = measurements-1)
#     rownames(growth_mass) = rownames(biomass)
#     colnames(growth_mass) = paste0("t",2:measurements)
#     
#     mort_mass = growth_mass
#     mort_n = mort_mass
#     recruit_n = mort_n
#     recruit_mass = mort_n
#     for (j in 1:dim(biomass)[1]){
#       for (k in 2:measurements){
#         if (!is.na(biomass[j,paste0("total_t",k)])&!is.na(biomass[j,paste0("total_t",k-1)])){
#           growth_mass[j,k-1] = biomass[j,paste0("total_t",k)] - biomass[j,paste0("total_t",k-1)]
#         }else if(is.na(biomass[j,paste0("total_t",k)])&!is.na(biomass[j,paste0("total_t",k-1)])){
#           mort_mass[j,k-1] = biomass[j,paste0("total_t",k-1)]
#           mort_n[j,k-1] = 1
#         }else if(!is.na(biomass[j,paste0("total_t",k)])&is.na(biomass[j,paste0("total_t",k-1)])){
#           recruit_mass[j,k-1] = biomass[j,paste0("total_t",k)]
#           recruit_n[j,k-1] = 1
#         }
#       }
#     }
#     for (k in 1:measurements){
#       CIPHA_living_wood[paste0("14_",gsub("_check.csv","",n)),paste0("live_mass_t",k)] = sum(biomass[,paste0("total_t",k)],na.rm = T)/1000/as.numeric(unique(df$plot_size))
#       CIPHA_living_wood_N[paste0("14_",gsub("_check.csv","",n)),paste0("live_N_t",k)] = sum(!is.na(biomass[,paste0("total_t",k)]))/as.numeric(unique(df$plot_size))
#       if (k!=1){
#         #Calculate mortality, recruitment and growth in biomass in Mg/ha/yr and in stemcount instems/ha/year
#         int = surveys[paste0("14_",gsub("_check.csv","",n)),k]-surveys[paste0("14_",gsub("_check.csv","",n)),k-1]
#         CIPHA_growth[paste0("14_",gsub("_check.csv","",n)),paste0("growth_t",k)] = sum(growth_mass[,paste0("t",k)],na.rm=T)/1000/int/as.numeric(unique(df$plot_size))
#         CIPHA_mort[paste0("14_",gsub("_check.csv","",n)),paste0("mort_mass_t",k)] = sum(mort_mass[,paste0("t",k)],na.rm=T)/1000/int/as.numeric(unique(df$plot_size))
#         CIPHA_mort_N[paste0("14_",gsub("_check.csv","",n)),paste0("mort_N_t",k)] = sum(!is.na(mort_n[,paste0("t",k)]))/int/as.numeric(unique(df$plot_size))
#         CIPHA_recruit[paste0("14_",gsub("_check.csv","",n)),paste0("recruit_mass_t",k)] = sum(recruit_mass[,paste0("t",k)],na.rm=T)/1000/int/as.numeric(unique(df$plot_size))
#         CIPHA_recruit_N[paste0("14_",gsub("_check.csv","",n)),paste0("recruit_N_t",k)] = sum(!is.na(recruit_n[,paste0("t",k)]))/int/as.numeric(unique(df$plot_size))
#       }
#     }
#   }
#   sp = unique(unlist(df[1,paste0("sp_t",1:measurements)]))[!is.na(unique(unlist(df[1,paste0("sp_t",1:measurements)])))]
#   for (k in 2:dim(df)[1]){
#     sp = c(sp,unique(unlist(df[k,paste0("sp_t",1:measurements)]))[!is.na(unique(unlist(df[k,paste0("sp_t",1:measurements)])))])
#   }
#   species = unique(sp)
#   count = vector(length = length(species))
#   for (k in 1:length(species)){
#     count[k] = sum(sp==species[k])
#   }
#   species_count = data.frame(species,count)
#   species_count = arrange(species_count,desc(count))
#   for (k in 1:length(species)){
#     CIPHA_sp_ID[paste0("14_",gsub("_check.csv","",n)),paste0("sp",k)] = species_count[k,1]
#     for (s in 1:measurements){
#       CIPHA_sp_comp[paste0("14_",gsub("_check.csv","",n)),paste0("sp",k,"_mass_%_t",s)] = sum(biomass[df[,paste0("sp_t",s)]==species_count[k,1],paste0("total_t",s)],na.rm = T)/sum(biomass[,paste0("total_t",s)],na.rm = T)
#       CIPHA_sp_comp[paste0("14_",gsub("_check.csv","",n)),paste0("sp",k,"_N_%_t",s)] = sum(df[,paste0("sp_t",s)]==species_count[k,1],na.rm = T)/sum(!is.na(df[,paste0("sp_t",s)]))
#       CIPHA_sp_live[paste0("14_",gsub("_check.csv","",n)),paste0("sp",k,"_mass_t",s)] = sum(biomass[,paste0("total_t",s)],na.rm = T)/1000/as.numeric(unique(df$plot_size))*CIPHA_sp_comp[paste0("14_",gsub("_check.csv","",n)),paste0("sp",k,"_mass_%_t",s)]
#       CIPHA_sp_live[paste0("14_",gsub("_check.csv","",n)),paste0("sp",k,"_N_t",s)] = sum(!is.na(biomass[,paste0("total_t",s)]))/as.numeric(unique(df$plot_size))*CIPHA_sp_comp[paste0("14_",gsub("_check.csv","",n)),paste0("sp",k,"_N_%_t",s)]
#       if (s!=1){
#         #Calcucate mortality, recruitment and growth in biomass in Mg/ha/yr and in stemcount in stems/ha/year
#         int = surveys[paste0("14_",gsub("_check.csv","",n)),s]-surveys[paste0("14_",gsub("_check.csv","",n)),s-1]
#         CIPHA_sp_mort[paste0("14_",gsub("_check.csv","",n)),paste0("sp",k,"_mort_mass_t",s)] = sum(mort_mass[df[,paste0("sp_t",s)]==species_count[k,1],paste0("t",s)],na.rm = T)/1000/int/as.numeric(unique(df$plot_size))
#         CIPHA_sp_mort[paste0("14_",gsub("_check.csv","",n)),paste0("sp",k,"_mort_N_t",s)] = sum(mort_n[df[,paste0("sp_t",s)]==species_count[k,1],paste0("t",s)],na.rm = T)/int/as.numeric(unique(df$plot_size))
#         CIPHA_sp_recruit[paste0("14_",gsub("_check.csv","",n)),paste0("sp",k,"_recruit_mass_t",s)] = sum(recruit_mass[df[,paste0("sp_t",s)]==species_count[k,1],paste0("t",s)],na.rm = T)/1000/int/as.numeric(unique(df$plot_size))
#         CIPHA_sp_recruit[paste0("14_",gsub("_check.csv","",n)),paste0("sp",k,"_recruit_N_t",s)] = sum(recruit_n[df[,paste0("sp_t",s)]==species_count[k,1],paste0("t",s)],na.rm = T)/int/as.numeric(unique(df$plot_size))
#         CIPHA_sp_growth[paste0("14_",gsub("_check.csv","",n)),paste0("sp",k,"_growth_mass_t",s)] = sum(growth_mass[df[,paste0("sp_t",s)]==species_count[k,1],paste0("t",s)],na.rm = T)/1000/int/as.numeric(unique(df$plot_size))
#       }
#     }
#   }
# }
# print("CIPHA done")
# 
# 
# CIPHA_all = data.frame(CIPHA_living_wood,CIPHA_living_wood_N,CIPHA_growth,CIPHA_recruit,CIPHA_recruit_N,CIPHA_mort,CIPHA_mort_N)
# write.csv(CIPHA_all,"./EWS_package/data/psp/CIPHA_total_changes.csv")
# write.csv(CIPHA_sp_ID,"./EWS_package/data/psp/CIPHA_sp_IDs.csv")
# write.csv(CIPHA_sp_comp,"./EWS_package/data/psp/CIPHA_sp_comp.csv")
# write.csv(CIPHA_sp_mort,"./EWS_package/data/psp/CIPHA_sp_mort.csv")
# write.csv(CIPHA_sp_recruit,"./EWS_package/data/psp/CIPHA_sp_recruit.csv")
# write.csv(CIPHA_sp_growth,"./EWS_package/data/psp/CIPHA_sp_growth.csv")
# write.csv(CIPHA_sp_live,"./EWS_package/data/psp/CIPHA_sp_live.csv")
