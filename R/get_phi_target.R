#' Get a target for the correlation matrix
#' @export
get_phi_target <- function(n_generals, specifics_per_general, g_rho = TRUE, s_rho = FALSE) {

  n_specifics <- n_generals * specifics_per_general
  n_factors <- n_specifics + n_generals
  Phi_Target <- matrix(0, n_factors, n_factors)

  if(g_rho) {
    Phi_Target[1:n_generals, 1:n_generals] <- 1
  } else {
    Phi_Target[1:n_generals, 1:n_generals] <- 0
  }

  if(s_rho) {
    Phi_Target[-(1:n_generals), -(1:n_generals)] <- 1
  } else {
    Phi_Target[-(1:n_generals), -(1:n_generals)] <- 0
  }

  Phi_Weight <- 1 - Phi_Target
  diag(Phi_Weight) <- 0
  diag(Phi_Target) <- 0

  return( list(Phi_Target = Phi_Target, Phi_Weight = Phi_Weight) )

}
