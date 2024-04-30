#' @export
tetrad_fit_index <- function(SampleCov, ImpliedCov, type="rmsd") {
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
  tM <- c(res1[,5:7])
  er <- tM - c(res0[,5:7])
  n <- length(er)
  if(type == "mad") {
    TFI <- 1 - mean(abs(er))
  } else if(type == "rmsd") {
    TFI <- 1 - sqrt(mean(er^2))
  } else if(type == "exp"){
    i <- exp(-abs(tM))
    weight <- i/sum(i)
    TFI <- 1 - sum(weight * abs(er))
  } else if(type == "sqr"){
    i <- exp(-abs(tM))
    weight <- i/sum(i)
    TFI <- 1 - sqrt(sum(weight * (er^2)))
  } else {
    stop("Unkown tetrad type!")
  }
  return(TFI)
}
