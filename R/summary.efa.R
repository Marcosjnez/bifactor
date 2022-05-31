#' @title
#' Compute summaries, fit indices, and information criteria from exploratory factor models.
#' @description
#'
#' Compute summaries, fit indices, and information criteria from exploratory factor models.
#'
#' @usage
#'
#' ## S3 method for class 'efa'
#' summary(efa, nobs=NULL, suppress=0, order=FALSE, ...)
#'
#' @param efa Object of class efa.
#' @param nobs Sample size. Defaults to NULL.
#' @param suppress Hide the loadings whose absolute magnitudes are smaller than this cutoff. Defaults to 0.
#' @param order Order the columns of the pattern matrix according to the variance they account for. Defaults to FALSE.
#' @param ... Arguments to be passed to or from other methods.
#' @details \code{summary.efa}... to be explained
#'
#' @return List with the following objects:
#' \item{loadings}{Loading matrix.}
#' \item{Phi}{Factor correlation matrix.}
#' \item{communalities}{Communalities.}
#' \item{uniquenesses}{Uniquenesses.}
#' \item{complexity}{Item complexities.}
#' \item{VAR}{Variance accounted for the factors.}
#' \item{fit}{fit indices.}
#'
#'@author
#'
#' Vithor R. Franco & Marcos Jiménez
#'
#' @export
summary.efa <- function(efa, nobs=NULL, suppress=0, order=FALSE, ...) {
  # Check if nobs was provided
  if(is.null(nobs)) {
    if(is.null(efa$modelInfo$nobs)) {
      warning("Sample size was not provided. Chi-squared-based statistics will not be computed.")
    } else {
      nobs <- efa$modelInfo$nobs
    }
  }

  ### Pattern matrix with communalities, uniqueness, and complexity
  if(efa$modelInfo$rotation == "none") { # For efa without rotation
    lambda <- efa$efa$loadings
    Phi <- diag(ncol(lambda))
  } else { # For efa with rotation
    lambda <- efa$rotation$loadings
    Phi <- efa$rotation$Phi
  }
  uniquenesses <- c(efa$efa$uniquenesses)
  ObjFn <- efa$efa$f
  SSloads <- diag(Phi %*% t(lambda) %*% lambda) # Generalizing to oblique rotation
  ordering <- order(SSloads, decreasing=T)
  colnames(lambda) <- paste("F",sprintf(paste("%0",nchar(ncol(lambda)),"d",sep=""),1:ncol(lambda)),sep="")
  lambda <- lambda[,ordering]
  if(is.null(colnames(efa$modelInfo$R))) {
    rownames(lambda) <- paste("item_",sprintf(paste("%0",nchar(nrow(lambda)),"d",sep=""),1:nrow(lambda)),sep="")
  } else {
    rownames(lambda) <- colnames(efa$modelInfo$R)
  }
  lambda <- lambda * {abs(lambda) > abs(suppress)}
  if(order) {
    orderItems <- apply(abs(lambda), 1, which.max)
    lambda  <- lambda[order(orderItems, decreasing=FALSE),]
    uniquenesses <- uniquenesses[order(orderItems, decreasing=FALSE)]
  }
  h2     <- 1 - uniquenesses
  u2     <- uniquenesses
  com    <- {rowSums(lambda ^ 2)^2}/rowSums(lambda ^ 4)
  names(h2) <- names(u2) <- names(com) <- rownames(lambda)
  loadsM <- round(data.frame(lambda, h2, u2, com), 3)

  ### Variance accounted for
  SSloads <- sort(SSloads, decreasing=T)
  propVar <- SSloads/nrow(lambda)
  cumsVar <- cumsum(propVar)
  propExp <- propVar/sum(propVar)
  cumsExp <- cumsum(propExp)
  VAF <- round(t(data.frame(SSloads, propVar, cumsVar, propExp, cumsExp)),2)
  rownames(VAF) <- c("SS loadings", "Proportion Var", "Cumulative Var",
                     "Proportion Explained", "Cumulative Proportion")

  ### Factor correlations
  Phi <- Phi[ordering, ordering]
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
      " and the objective function was ", round(ObjFn,2), "\n", sep="")
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
