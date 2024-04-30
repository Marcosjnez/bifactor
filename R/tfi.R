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
  er <- c(res1[,5:7] - res0[,5:7])
  if(type == "mad") {
    TFI <- 1 - mean(abs(er))
  } else if(type == "rmsd") {
    TFI <- 1 - sqrt(mean(er^2))
  } else if(type == "fvu"){
    TFI <- 1 - {var(er)/var(c(res0[,5:7]))}
  } else if(type == "test"){
    TFI <- 1 - mean(abs(er/2))
  } else if(type == "wellfare"){
    n <- length(er)
    i <- rank(abs(er))
    weight <- (2*n + 1 - 2*i)/(n^2)
    TFI <- 1 - sum(weight * er)
  } else if(type == "dich"){
    diff0 <- sqrt(mean(er[c(res1[,5:7]) == 0]^2))
    diff1 <- mean(abs(er[c(res1[,5:7]) != 0]))
    if(length(diff0) == 0) {
      TFI <- 1 - diff1
    } else {
      TFI <- 1 - ((diff0 + diff1)/2)
    }
  } else {
    stop("Unkown tetrad type!")
  }
  return(TFI)
}
