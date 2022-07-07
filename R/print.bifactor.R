#' @title
#' Print summary information from exploratory factor models.
#' @description
#'
#' Print summary information from exploratory factor models.
#'
#' @usage
#'
#' ## S3 method for class 'bifactor'
#' print(bifactor, nobs=NULL, ...)
#' print.bifactor(bifactor, nobs=NULL, ...)
#'
#' @param efa Object of class bifactor
#' @param nobs Sample size. Defaults to NULL.
#' @param ... Arguments to be passed to or from other methods.
#'
#' @details to be explained
#'
#' @return Matrix of variance accounted for the factors.
#'
#' @author
#'
#' Vithor R. Franco & Marcos Jiménez
#'
#' @export
print.bifactor <- function(bifactor, nobs=NULL, ...) {
  efa <- bifactor
  # Check if nobs was provided
  if(is.null(nobs)) {
    if(is.null(efa$modelInfo$nobs)) {
      warning("Sample size was not provided. Chi-squared-based statistics will not be computed.")
    } else {
      nobs <- efa$modelInfo$nobs
    }
  }
  ### Pattern matrix with communalities, uniqueness, and complexity
  lambda <- efa$bifactor$loadings
  Phi <- efa$bifactor$Phi

  ObjFn <- efa$efa$f
  ordering <- order(diag(Phi %*% t(lambda) %*% lambda), decreasing=T)
  fit <- suppressWarnings(fitMeasures(efa, nobs))
  Phi <- Phi[ordering, ordering]
  rownames(Phi) <- colnames(Phi) <- paste("F",sprintf(paste("%0",nchar(efa$modelInfo$nfactors),"d",sep=""),
                                                      ordering),sep="")
  colnames(lambda) <- rownames(Phi) <-
    colnames(Phi) <- paste("F",sprintf(paste("%0",nchar(ncol(lambda)),"d",sep=""),1:ncol(lambda)),sep="")
  if(is.null(colnames(efa$modelInfo$R))) {
    rownames(lambda) <- paste("item_",sprintf(paste("%0",nchar(nrow(lambda)),"d",sep=""),1:nrow(lambda)),sep="")
  } else {
    rownames(lambda) <- colnames(efa$modelInfo$R)
  }
  # Print
  cat("Factor Analysis using method = ", efa$modelInfo$method, "\n", sep="")
  cat("Elapsed time of ", round(efa$elapsed * 1e-9, 3), " seconds", "\n", sep="")
  cat("\n","Goodness-of-fit and model misfit indices", sep="")
  cat("\n","The standardized root mean square residual (SRMR) is ", round(fit["srmr"],3), sep="")
  cat("\n","The largest absolute value of standardized residual correlation is ", round(fit["lavsrc"],3), sep="")
  cat("\n","The degrees of freedom for the model are ", efa$modelInfo$df,
      " and the objective function was ", round(ObjFn,2), "\n", sep="")
  if(!is.null(nobs)) {
    cat("The total number of observations was ", nobs, " with Unbiased Chi-squared = ",
        round(fit["chisq.unbiased"],1), " with prob < ", fit["pvalue.unbiased"], sep="")
    cat("\n","Unbiased Tucker Lewis Index of factoring reliability = ", round(fit["tli.unbiased"],3), sep="")
    cat("\n","Unbiased RMSEA index = ", round(fit["rmsea.unbiased"],3), sep="")
    cat("\n","Unbiased AIC = ", round(fit["aic.unbiased"],1), sep="")
    cat("\n","Unbiased BIC = ", round(fit["bic.unbiased"],1), sep="")
    cat("\n","Unbiased HQ = ", round(fit["hq.unbiased"],1), "\n", sep="")
  }
  # Loadings
  cat("Standardized loadings (pattern matrix)\n", sep=""); print(round(lambda, 2))
  if(efa$modelInfo$rotation != "none" & efa$modelInfo$projection != "orth") {
    cat("\n","Factor correlations after bifactor rotation\n",sep=""); print(round(Phi, 2))
  }
  invisible(NULL)
}
