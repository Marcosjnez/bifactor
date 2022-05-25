#' @title
#' Compute fit measures for exploratory factor models.
#' @description
#'
#' Compute fit measures for exploratory factor models.
#'
#' @usage
#'
#' fitMeasures(efa, nobs=NULL)
#'
#' @param efa Object of class efa.
#' @param nobs Sample size. Defaults to NULL.
#'
#' @details \code{fitMeasures}... to be explained
#'
#' @return Vector of fit measures.
#'
#' @author
#'
#' Vithor R. Franco & Marcos Jiménez
#'
#' @export
fitMeasures <- function(efa, nobs=NULL) {
  # Check if nobs was provided
  if(is.null(nobs)) {
    if(is.null(efa$modelInfo$nobs)) {
      warning("Sample size was not provided. Most fit indices will not be computed.")
    } else {
      nobs <- efa$modelInfo$nobs
    }
  }

  # Basic measures
  # if(efa$modelInfo$rotation == "none") { # For efa without rotation
  #   ObjFn <- efa$efa$f
  #   residuals <- efa$efa$residuals
  # } else { # For efa with rotation
  #   ObjFn <- efa$rotation$f
  #   residuals <- efa$rotation$residuals
  # }
  ObjFn <- efa$efa$f
  residuals <- efa$efa$residuals
  p <- efa$modelInfo$n_vars
  q <- efa$modelInfo$nfactors
  correction <- if(is.null(nobs)) NULL else nobs-1-1/6*(2*p+5)-2/3*q
  df_null <- efa$modelInfo$df_null
  df <- efa$modelInfo$df
  t <- df_null - df
  chisq_null <- if(is.null(nobs)) NULL else {nobs - 1} * efa$modelInfo$f_null
  chisq_null.unbiased <- if(is.null(nobs)) NULL else correction * efa$modelInfo$f_null
  chisq <- if(is.null(nobs)) NULL else {nobs - 1} * ObjFn
  pvalue <- if(is.null(nobs)) NULL else 1 - stats::pchisq(chisq, df)
  chisq.unbiased <- if(is.null(nobs)) NULL else correction * ObjFn
  pvalue.unbiased <- if(is.null(nobs)) NULL else 1 - stats::pchisq(chisq.unbiased, df)

  # Incremental fit indices
  cfi <- if(is.null(nobs)) NULL else {
    {max(chisq_null-df_null,0)-max(chisq-df,0)}/max(chisq_null-df_null,0)
  }
  cfi.unbiased <- if(is.null(nobs)) NULL else {
    {max(chisq_null.unbiased-df_null,0)-max(chisq.unbiased-df,0)}/
      max(chisq_null.unbiased-df_null,0)
  }
  tli <- if(is.null(nobs)) NULL else {
    {{chisq_null/df_null} - {chisq/df}}/{chisq_null/{df_null-1}}
  }
  tli.unbiased <- if(is.null(nobs)) NULL else {
    {{chisq_null.unbiased/df_null} - {chisq.unbiased/df}}/
      {chisq_null.unbiased/{df_null-1}}
  }
  nfi <- {efa$modelInfo$f_null - ObjFn}/efa$modelInfo$f_null

  # Absolute fit indices
  rmsea <- if(is.null(nobs)) NULL else {
    sqrt(max(chisq-df,0)/{df*{nobs-1}})
  }
  rmsea.unbiased <- if(is.null(nobs)) NULL else {
    sqrt(max(chisq.unbiased-df,0)/{df*{nobs-1}})
  }
  srmr <- sqrt(sum(residuals[lower.tri(residuals,diag=T)]^2)/
                 {{efa$modelInfo$n_vars*{efa$modelInfo$n_vars+1}}/2})
  lavsrc <- max(abs(residuals))

  # Comparative fit indices
  aic           <- if(is.null(nobs)) NULL else chisq + {2 * t}
  aic.unbiased  <- if(is.null(nobs)) NULL else chisq.unbiased + {2 * t}
  bic           <- if(is.null(nobs)) NULL else chisq + {log(nobs) * t}
  bic.unbiased  <- if(is.null(nobs)) NULL else chisq.unbiased + {log(nobs) * t}
  hq            <- if(is.null(nobs)) NULL else chisq + {2 * log(log(nobs)) * t}
  hq.unbiased   <- if(is.null(nobs)) NULL else chisq.unbiased + {2 * log(log(nobs)) * t}
  ecvi          <- if(is.null(nobs)) NULL else {chisq/{nobs-1}} + {2*{t/{nobs-1}}}
  ecvi.unbiased <- if(is.null(nobs)) NULL else {chisq.unbiased/{nobs-1}} + {2*{t/{nobs-1}}}

  # Results
  Results <- c("chisq_null"=chisq_null, "chisq_null.unbiased"=chisq_null.unbiased,
               "df_null"=df_null, "chisq"=chisq, "pvalue"=pvalue,
               "chisq.unbiased"=chisq.unbiased, "pvalue.unbiased"=pvalue.unbiased,
               "df"=df, "cfi"=cfi, "cfi.unbiased"=cfi.unbiased, "tli"=tli,
               "tli.unbiased"=tli.unbiased, "nfi"=nfi, "rmsea"=rmsea,
               "rmsea.unbiased"=rmsea.unbiased, "srmr"=srmr, "lavsrc"=lavsrc,
               "aic"=aic, "aic.unbiased"=aic.unbiased, "bic"=bic,
               "bic.unbiased"=bic.unbiased, "hq"=hq, "hq.unbiased"=hq.unbiased,
               "ecvi"=ecvi, "ecvi.unbiased"=ecvi.unbiased)
  return(Results)
}
