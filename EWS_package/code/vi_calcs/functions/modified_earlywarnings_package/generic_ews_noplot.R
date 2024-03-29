# 
generic_ews_noplot = function (timeseries, winsize = 50, detrending = c("no", "gaussian",
"loess", "linear", "first-diff"), bandwidth = NULL, span = NULL,
degree = NULL, logtransform = FALSE, interpolate = FALSE,
AR_n = FALSE, powerspectrum = FALSE)
{
    timeseries <- data.matrix(timeseries)
    if (dim(timeseries)[2] == 1) {
        Y = timeseries
        timeindex = 1:dim(timeseries)[1]
    }
    else if (dim(timeseries)[2] == 2) {
        Y <- timeseries[, 2]
        timeindex <- timeseries[, 1]
    }
    else {
        warning("not right format of timeseries input")
    }
    if (interpolate) {
        YY <- approx(timeindex, Y, n = length(Y), method = "linear")
        Y <- YY$y
    }
    else {
        Y <- Y
    }
    if (logtransform) {
        Y <- log(Y + 1)
    }
    detrending <- match.arg(detrending)
    if (detrending == "gaussian") {
        if (is.null(bandwidth)) {
            bw <- round(bw.nrd0(timeindex))
        }
        else {
            bw <- round(length(Y) * bandwidth/100)
        }
        smYY <- ksmooth(timeindex, Y, kernel = "normal", bandwidth = bw,
        range.x = range(timeindex), x.points = timeindex)
        nsmY <- Y - smYY$y
        smY <- smYY$y
    }
    else if (detrending == "linear") {
        nsmY <- resid(lm(Y ~ timeindex))
        smY <- fitted(lm(Y ~ timeindex))
    }
    else if (detrending == "loess") {
        if (is.null(span)) {
            span <- 25/100
        }
        else {
            span <- span/100
        }
        if (is.null(degree)) {
            degree <- 2
        }
        else {
            degree <- degree
        }
        smYY <- loess(Y ~ timeindex, span = span, degree = degree,
        normalize = FALSE, family = "gaussian")
        smY <- predict(smYY, data.frame(x = timeindex), se = FALSE)
        nsmY <- Y - smY
    }
    else if (detrending == "first-diff") {
        nsmY <- diff(Y)
        timeindexdiff <- timeindex[1:(length(timeindex) - 1)]
    }
    else if (detrending == "no") {
        smY <- Y
        nsmY <- Y
    }
    mw <- round(length(Y) * winsize/100)
    omw <- length(nsmY) - mw + 1
    low <- 6
    high <- omw
    nMR <- matrix(data = NA, nrow = mw, ncol = omw)
    x1 <- 1:mw
    for (i in 1:omw) {
        Ytw <- nsmY[i:(i + mw - 1)]
        nMR[, i] <- Ytw
    }
    nARR <- numeric()
    nSD <- numeric()
    nSK <- numeric()
    nKURT <- numeric()
    nACF <- numeric()
    nDENSITYRATIO <- numeric()
    nSPECT <- matrix(0, nrow = omw, ncol = ncol(nMR))
    nCV <- numeric()
    smARall <- numeric()
    smARmaxeig <- numeric()
    detB <- numeric()
    ARn <- numeric()
    nSD <- apply(nMR, 2, sd, na.rm = TRUE)
    for (i in 1:ncol(nMR)) {
      #Edit by Sol. The functions ar.ols and spec.ar require variation in a timeseries. For some reason, for short timeseries
      #there is often a singularity in the nMR[,i] data, so here I am changing this to NAs if that is the case. What this does 
      #is create NAs for ar1, returnrate and densratio for any periods where this occurs, but it also means that sk, kurt, and acf1
      #come out as NaNs, because they are using those same data. sd and cv return zero because they are using the standard deviation
      if(length(unique(nMR[,i]))>1){
        nYR <- ar.ols(nMR[, i], aic = FALSE, order.max = 1, dmean = FALSE,
        intercept = FALSE)
        nARR[i] <- nYR$ar
        spectfft <- spec.ar(nMR[, i], n.freq = omw, plot = FALSE,
                            order = 1)
        nSPECT[, i] <- spectfft$spec
        nDENSITYRATIO[i] <- spectfft$spec[low]/spectfft$spec[high]
      }else{
        nARR[i] <- NA
        spectfft <- NA
        nSPECT[, i] <- NA
        nDENSITYRATIO[i] <- NA
      }
        nSK[i] <- abs(moments::skewness(nMR[, i], na.rm = TRUE))
        nKURT[i] <- moments::kurtosis(nMR[, i], na.rm = TRUE)
        nCV[i] <- nSD[i]/mean(nMR[, i])
        ACF <- acf(nMR[, i], lag.max = 1, type = c("correlation"),
        plot = FALSE)
        nACF[i] <- ACF$acf[2]
        
        
        if (AR_n) {
            ARall <- ar.ols(nMR[, i], aic = TRUE, order.max = 6,
            demean = F, intercept = F)
            smARall[i] <- ARall$ar[1]
            ARn[i] <- ARall$order
            roots <- Mod(polyroot(c(rev(-ARall$ar), 1)))
            smARmaxeig[i] <- max(roots)
            detB[i] <- (prod(roots))^(2/ARn[i])
        }
    }
    nRETURNRATE = 1/nARR
    timevec <- seq(1, length(nARR))
    KtAR <- cor.test(timevec, nARR, alternative = c("two.sided"),
    method = c("kendall"), conf.level = 0.95)
    KtACF <- cor.test(timevec, nACF, alternative = c("two.sided"),
    method = c("kendall"), conf.level = 0.95)
    KtSD <- cor.test(timevec, nSD, alternative = c("two.sided"),
    method = c("kendall"), conf.level = 0.95)
    KtSK <- cor.test(timevec, nSK, alternative = c("two.sided"),
    method = c("kendall"), conf.level = 0.95)
    KtKU <- cor.test(timevec, nKURT, alternative = c("two.sided"),
    method = c("kendall"), conf.level = 0.95)
    KtDENSITYRATIO <- cor.test(timevec, nDENSITYRATIO, alternative = c("two.sided"),
    method = c("kendall"), conf.level = 0.95)
    KtRETURNRATE <- cor.test(timevec, nRETURNRATE, alternative = c("two.sided"),
    method = c("kendall"), conf.level = 0.95)
    KtCV <- cor.test(timevec, nCV, alternative = c("two.sided"),
    method = c("kendall"), conf.level = 0.95)
 
    out <- data.frame(timeindex[mw:length(nsmY)], nARR, nSD,
    nSK, nKURT, nCV, nRETURNRATE, nDENSITYRATIO, nACF)
    colnames(out) <- c("timeindex", "ar1", "sd", "sk", "kurt",
    "cv", "returnrate", "densratio", "acf1")
    return(out)
}
