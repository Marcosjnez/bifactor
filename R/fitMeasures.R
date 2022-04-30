fitMeasures <- function(efa) {
  # Basic measures
  df_null <- efa$modelInfo$n_vars * {{efa$modelInfo$n_vars-1}}/2
  chisq_null <- {efa$modelInfo$n_obs - 1} * efa$modelInfo$f_null
  df <- df_null - efa$modelInfo$t
  chisq <- {efa$modelInfo$n_obs - 1} * efa$rotation$f
  chisq_df <- chisq/df
  # Incremental fit indices
  cfi <- {max(chisq_null-df_null,0)-max(chisq-df,0)}/max(chisq_null-df_null,0)
  tli <- {{chisq_null/df_null} - {chisq/df}}/{chisq_null/{df_null-1}}
  nfi <- {efa$modelInfo$f_null - efa$rotation$f}/efa$modelInfo$f_null
  # Absolute fit indices
  rmsea <- sqrt(max(chisq-df,0)/{df*{efa$modelInfo$n_obs-1}})
  obs_cor <- cov2cor(efa$modelInfo$R)[lower.tri(efa$modelInfo$R)]
  imp_cor <- cov2cor(efa$rotation$Rhat)[lower.tri(efa$modelInfo$R)]
  srmr <- sqrt(sum({obs_cor - imp_cor}^2)/{{efa$modelInfo$n_vars*{efa$modelInfo$n_vars+1}}/2})
  # Comparative fit indices
  aic <- chisq + {2 * efa$modelInfo$t}
  bic <- chisq + {log(efa$modelInfo$n_obs) * efa$modelInfo$t}
  ecvi <- {chisq/{efa$modelInfo$n_obs-1}} + {2*{efa$modelInfo$t/{efa$modelInfo$n_obs-1}}}
  
  # Results
  Results <- c("df_null"=df_null, "chisq_null"=chisq_null, "df"=df, "chisq"=chisq,
               "cfi"=cfi, "tli"=tli, "nfi"=nfi, "rmsea"=rmsea, "srmr"=srmr,
               "aic"=aic, "bic"=bic, "ecvi"=ecvi)
  return(Results)
}