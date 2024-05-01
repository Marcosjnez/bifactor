#' @export
tetrad_fit_index <- function(SampleCov, ImpliedCov, nfac=1, type="rmsd") {
  ## Get all the possible four-tuple of variables
  possibilities <- combn(1:ncol(SampleCov), 4)

  ## Create a matrix indicating what the four-tuple is
  ## and its respective tetrad
  res0 <- matrix(NA, ncol(possibilities), 7) # Empirical tetrads
  res1 <- matrix(NA, ncol(possibilities), 7) # Model tetrads
  for(i in 1:ncol(possibilities)) {
    # Empirical tetrads
    res0[i,] <- c(possibilities[,i],calculate_fourtuple_tetrads(SampleCov[possibilities[,i],possibilities[,i]]))
    # Model tetrads
    res1[i,] <- c(possibilities[,i],calculate_fourtuple_tetrads(ImpliedCov[possibilities[,i],possibilities[,i]]))
  }

  ## Calculate and return the tetrad fit index
  tM <- atanh(c(res1[,5:7]))
  er <- tM - atanh(c(res0[,5:7]))
  n <- length(er) * {{nfac*{nfac+1}}/{2*nfac}}
  if(type == "mad") {
    TFI <- sum(abs(er))/n
  } else if(type == "rmsd") {
    TFI <- sqrt(sum(er^2)/n)
  } else if(type == "madW"){
    i <- exp(-abs(tM))
    weight <- i/sum(i)
    TFI <- sum(weight * abs(er))
  } else if(type == "rmsdW"){
    i <- exp(-abs(tM))
    weight <- i/sum(i)
    TFI <- sqrt(sum(weight * (er^2)))
  } else {
    stop("Unkown tetrad type!")
  }
  return(TFI)
}

