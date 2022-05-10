fitMeasures <- function(efa, nobs=NULL) {
  # Check if nobs was provided
  if(is.null(nobs)) {
    if(is.null(efa$modelInfo$n_obs)) {
      warning("Sample size was not provided. Most fit indices will not be computed.")
    } else {
      n_obs <- efa$modelInfo$n_obs
    }
  } else {
    n_obs <- nobs
  }
  
  # Basic measures
  p <- efa$modelInfo$n_vars
  q <- efa$modelInfo$n_factors
  correction <- if(is.null(nobs)) NULL else n_obs-1-1/6*(2*p+5)-2/3*q
  df_null <- efa$modelInfo$df_null
  df <- efa$modelInfo$df
  t <- df_null - df
  chisq_null <- if(is.null(nobs)) NULL else {n_obs - 1} * efa$modelInfo$f_null
  chisq_null.unbiased <- if(is.null(nobs)) NULL else correction * efa$modelInfo$f_null
  chisq <- if(is.null(nobs)) NULL else {n_obs - 1} * efa$rotation$f
  pvalue <- if(is.null(nobs)) NULL else 1 - pchisq(chisq, df)
  #chisq_df <- chisq/df
  #chisq_df.unbiased <- chisq/df
  chisq.unbiased <- if(is.null(nobs)) NULL else correction * efa$rotation$f
  pvalue.unbiased <- if(is.null(nobs)) NULL else 1 - pchisq(chisq.unbiased, df)
  
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
  nfi <- {efa$modelInfo$f_null - efa$rotation$f}/efa$modelInfo$f_null
  
  # Absolute fit indices
  rmsea <- if(is.null(nobs)) NULL else {
    sqrt(max(chisq-df,0)/{df*{n_obs-1}})
  }
  rmsea.unbiased <- if(is.null(nobs)) NULL else {
    sqrt(max(chisq.unbiased-df,0)/{df*{n_obs-1}})
  }
  srmr <- sqrt(sum(efa$rotation$residuals[lower.tri(efa$rotation$residuals,diag=T)]^2)/
                 {{efa$modelInfo$n_vars*{efa$modelInfo$n_vars+1}}/2})
  lavsrc <- max(abs(efa$rotation$residuals))
  
  # Comparative fit indices
  aic           <- if(is.null(nobs)) NULL else chisq + {2 * t}
  aic.unbiased  <- if(is.null(nobs)) NULL else chisq.unbiased + {2 * t}
  bic           <- if(is.null(nobs)) NULL else chisq + {log(n_obs) * t}
  bic.unbiased  <- if(is.null(nobs)) NULL else chisq.unbiased + {log(n_obs) * t}
  hq            <- if(is.null(nobs)) NULL else chisq + {2 * log(log(n_obs)) * t}
  hq.unbiased   <- if(is.null(nobs)) NULL else chisq.unbiased + {2 * log(log(n_obs)) * t}
  ecvi          <- if(is.null(nobs)) NULL else {chisq/{n_obs-1}} + {2*{t/{n_obs-1}}}
  ecvi.unbiased <- if(is.null(nobs)) NULL else {chisq.unbiased/{n_obs-1}} + {2*{t/{n_obs-1}}}
  
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