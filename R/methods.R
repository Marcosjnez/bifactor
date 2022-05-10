summary.efa <- function(efa, nobs=NULL, suppress=0, order=FALSE, print.max=101) {
  # Check if nobs was provided
  if(is.null(nobs)) {
    if(is.null(efa$modelInfo$n_obs)) {
      warning("Sample size was not provided. Chi-squared-based statistics will not be computed.")
    } else {
      nobs <- efa$modelInfo$n_obs
    }
  }
  
  ### Pattern matrix with communalities, uniqueness, and complexity
  lambda <- efa$rotation$loadings[,order(colSums(efa$rotation$loadings^2), decreasing=T)]
  colnames(lambda) <- paste("F",sprintf(paste("%0",nchar(ncol(lambda)),"d",sep=""),1:ncol(lambda)),sep="")
  if(is.null(colnames(efa$modelInfo$R))) {
    rownames(lambda) <- paste("item_",sprintf(paste("%0",nchar(nrow(lambda)),"d",sep=""),1:nrow(lambda)),sep="")
  } else {
    rownames(lambda) <- colnames(efa$modelInfo$R)
  }
  lambda <- lambda * {abs(lambda) > abs(suppress)}
  if(order) {
    lambda  <- lambda[order(apply(abs(lambda), 1, which.max), decreasing=FALSE),]
  }
  h2     <- c(1 - efa$rotation$uniquenesses)
  u2     <- c(efa$rotation$uniquenesses)
  com    <- {rowSums(efa$rotation$loadings ^ 2)^2}/rowSums(efa$rotation$loadings ^ 4)
  names(h2) <- names(u2) <- names(com) <- rownames(lambda)
  loadsM <- round(data.frame(lambda, h2, u2, com), 3)
  
  ### Variance accounted for
  SSloads <- colSums(lambda ^ 2)
  propVar <- SSloads/nrow(lambda)
  cumsVar <- cumsum(propVar)
  propExp <- propVar/sum(propVar)
  cumsExp <- cumsum(propExp)
  VAF <- round(t(data.frame(SSloads, propVar, cumsVar, propExp, cumsExp)),2)
  rownames(VAF) <- c("SS loadings", "Proportion Var", "Cumulative Var",
                     "Proportion Explained", "Cumulative Proportion")
  
  ### Factor correlations
  Phi    <- efa$rotation$Phi[order(colSums(efa$rotation$loadings^2), decreasing=T),
                             order(colSums(efa$rotation$loadings^2), decreasing=T)] 
  rownames(Phi) <- colnames(Phi) <- colnames(lambda)
  
  ### Fit statistics
  fit  <- suppressWarnings(fitMeasures(efa, nobs))
  
  ### Print
  # Basic info
  cat("Factor Analysis using method = ", efa$modelInfo$method, "\n", sep="")
  cat("Elapsed time of ", round(efa$elapsed * 1e-9, 3), " seconds", "\n", sep="")
  # Loadings
  cat("Standardized loadings (pattern matrix) based upon correlation matrix\n", sep=""); print(loadsM)
  # Variance accounted for
  cat("\n","Variance accounted for after rotation\n",sep=""); print(VAF)
  # Latent correlations
  cat("\n","Factor correlations after rotation\n",sep=""); print(Phi)
  # Fit
  cat("\n","Goodness-of-fit and model misfit indices", sep="")
  cat("\n","Mean item complexity = ", round(mean(com),1), sep="")
  cat("\n","The standardized root mean square residual (SRMR) is ", round(fit["srmr"],3), sep="")
  cat("\n","The largest absolute value of standardized residual correlation is ", round(fit["lavsrc"],3), sep="")
  cat("\n","The degrees of freedom for the null model are ", efa$modelInfo$df_null,
      " and the objective function was ", round(efa$modelInfo$f_null,2), sep="")
  cat("\n","The degrees of freedom for the model are ", efa$modelInfo$df,
      " and the objective function was ", round(efa$rotation$f,2), "\n", sep="")
  if(!is.null(nobs)) {
    cat("The total number of observations was ", nobs, " with Unbiased Chi-squared = ",
        round(fit["chisq.unbiased"],1), " with prob < ", fit["pvalue.unbiased"] , sep="")
    cat("\n","Unbiased Tucker Lewis Index of factoring reliability = ", round(fit["tli.unbiased"],3), sep="")
    cat("\n","Unbiased RMSEA index = ", round(fit["rmsea.unbiased"],3), sep="")
    cat("\n","Unbiased AIC = ", round(fit["aic.unbiased"],1), sep="")
    cat("\n","Unbiased BIC = ", round(fit["bic.unbiased"],1), sep="")
    cat("\n","Unbiased HQ = ", round(fit["hq.unbiased"],1), "\n", sep="")
  }
  
  Results <- list("loadings"=lambda, "communalities"=h2, "uniqueness"=u2, "complexity"=com,
                  "VAF"=VAF, "Phi"=Phi, "fit"=fit)
  invisible(Results)
}

print.efa <- function(efa, nobs=NULL) {
  # Check if nobs was provided
  if(is.null(nobs)) {
    if(is.null(efa$modelInfo$n_obs)) {
      warning("Sample size was not provided. Chi-squared-based statistics will not be computed.")
    } else {
      nobs <- efa$modelInfo$n_obs
    }
  }
  fit  <- suppressWarnings(fitMeasures(efa, nobs))
  Phi    <- efa$rotation$Phi[order(colSums(efa$rotation$loadings^2), decreasing=T),
                             order(colSums(efa$rotation$loadings^2), decreasing=T)] 
  rownames(Phi) <- colnames(Phi) <- paste("F",sprintf(paste("%0",nchar(efa$modelInfo$n_factors),"d",sep=""),
                                                      1:efa$modelInfo$n_factors),sep="")
  # Print
  cat("Factor Analysis using method = ", efa$modelInfo$method, "\n", sep="")
  cat("Elapsed time of ", round(efa$elapsed * 1e-9, 3), " seconds", "\n", sep="")
  cat("\n","Goodness-of-fit and model misfit indices", sep="")
  cat("\n","The standardized root mean square residual (SRMR) is ", round(fit["srmr"],3), sep="")
  cat("\n","The largest absolute value of standardized residual correlation is ", round(fit["lavsrc"],3), sep="")
  cat("\n","The degrees of freedom for the model are ", efa$modelInfo$df,
      " and the objective function was ", round(efa$rotation$f,2), "\n", sep="")
  if(!is.null(nobs)) {
    cat("The total number of observations was ", nobs, " with Unbiased Chi-squared = ",
        round(fit["chisq.unbiased"],1), " with prob < ", fit["pvalue.unbiased"], sep="")
    cat("\n","Unbiased Tucker Lewis Index of factoring reliability = ", round(fit["tli.unbiased"],3), sep="")
    cat("\n","Unbiased RMSEA index = ", round(fit["rmsea.unbiased"],3), sep="")
    cat("\n","Unbiased AIC = ", round(fit["aic.unbiased"],1), sep="")
    cat("\n","Unbiased BIC = ", round(fit["bic.unbiased"],1), sep="")
    cat("\n","Unbiased HQ = ", round(fit["hq.unbiased"],1), "\n", sep="")
  }
  cat("\n","Factor correlations after rotation\n",sep=""); print(Phi)
}
