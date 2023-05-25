#' @title
#' Compute fit measures for exploratory factor models.
#' @description
#'
#' Compute fit measures for exploratory factor models.
#'
#' @usage
#'
#' fitMeasures(efa, nobs = NULL)
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
fitMeasures <- function(efa, nobs = NULL) {
  # Check if nobs was provided
  if(is.null(nobs)) {
    if(is.null(efa$modelInfo$nobs)) {
      warning("Sample size was not provided. Most fit indices will not be computed.")
    } else {
      nobs <- efa$modelInfo$nobs
    }
  }

  # Basic measures
  ObjFn <- efa$efa$f
  ObjFn_null <- efa$modelInfo$f_null
  residuals <- efa$efa$residuals
  p <- efa$modelInfo$n_vars
  q <- efa$modelInfo$nfactors
  correction <- if(is.null(nobs)) NULL else nobs-1-1/6*(2*p+5)-2/3*q
  df_null <- efa$modelInfo$df_null
  df <- efa$modelInfo$df
  t <- df_null - df
  chisq_null <- if(is.null(nobs)) NULL else {nobs - 1} * ObjFn_null
  chisq_null.corrected <- if(is.null(nobs)) NULL else correction * ObjFn_null
  chisq <- if(is.null(nobs)) NULL else {nobs - 1} * ObjFn
  pvalue <- if(is.null(nobs)) NULL else 1 - stats::pchisq(chisq, df)
  chisq.corrected <- if(is.null(nobs)) NULL else correction * ObjFn
  pvalue.corrected <- if(is.null(nobs)) NULL else 1 - stats::pchisq(chisq.corrected, df)

  # Incremental fit indices
  cfi <- if(is.null(nobs)) NULL else {
    {max(chisq_null-df_null,0)-max(chisq-df,0)}/max(chisq_null-df_null,0)
  }
  cfi.corrected <- if(is.null(nobs)) NULL else {
    {max(chisq_null.corrected-df_null,0)-max(chisq.corrected-df,0)}/
      max(chisq_null.corrected-df_null,0)
  }
  tli <- if(is.null(nobs)) NULL else {
    {{chisq_null/df_null} - {chisq/df}}/{chisq_null/{df_null-1}}
  }
  tli.corrected <- if(is.null(nobs)) NULL else {
    {{chisq_null.corrected/df_null} - {chisq.corrected/df}}/
      {chisq_null.corrected/{df_null-1}}
  }
  nfi <- {ObjFn_null - ObjFn}/ObjFn_null

  # Absolute fit indices
  rmsea <- if(is.null(nobs)) NULL else {
    sqrt(max(chisq-df,0)/{df*{nobs-1}})
  }
  rmsea.corrected <- if(is.null(nobs)) NULL else {
    sqrt(max(chisq.corrected-df,0)/{df*{nobs-1}})
  }
  srmr <- sqrt(sum(residuals[lower.tri(residuals,diag=T)]^2)/
                 {{efa$modelInfo$n_vars*{efa$modelInfo$n_vars+1}}/2})
  lavsrc <- max(abs(residuals))

  # Comparative fit indices
  aic           <- if(is.null(nobs)) NULL else chisq + {2 * t}
  aic.corrected  <- if(is.null(nobs)) NULL else chisq.corrected + {2 * t}
  bic           <- if(is.null(nobs)) NULL else chisq + {log(nobs) * t}
  bic.corrected  <- if(is.null(nobs)) NULL else chisq.corrected + {log(nobs) * t}
  hq            <- if(is.null(nobs)) NULL else chisq + {2 * log(log(nobs)) * t}
  hq.corrected   <- if(is.null(nobs)) NULL else chisq.corrected + {2 * log(log(nobs)) * t}
  ecvi          <- if(is.null(nobs)) NULL else {chisq/{nobs-1}} + {2*{t/{nobs-1}}}
  ecvi.corrected <- if(is.null(nobs)) NULL else {chisq.corrected/{nobs-1}} + {2*{t/{nobs-1}}}

  # Results
  Results <- c("chisq_null"=chisq_null, "chisq_null.corrected"=chisq_null.corrected,
               "df_null"=df_null, "chisq"=chisq, "pvalue"=pvalue,
               "chisq.corrected"=chisq.corrected, "pvalue.corrected"=pvalue.corrected,
               "df"=df, "cfi"=cfi, "cfi.corrected"=cfi.corrected, "tli"=tli,
               "tli.corrected"=tli.corrected, "nfi"=nfi, "rmsea"=rmsea,
               "rmsea.corrected"=rmsea.corrected, "srmr"=srmr, "lavsrc"=lavsrc,
               "aic"=aic, "aic.corrected"=aic.corrected, "bic"=bic,
               "bic.corrected"=bic.corrected, "hq"=hq, "hq.corrected"=hq.corrected,
               "ecvi"=ecvi, "ecvi.corrected"=ecvi.corrected)
  return(Results)
}
