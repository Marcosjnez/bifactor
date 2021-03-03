#' Simulate a bifactor structure
#' @export
sim_bifactor <- function(n_generals, specifics_per_general, items_per_specific,
                         loadings_g = "medium", loadings_s = "medium",
                         crossloadings = 0, pure = FALSE,
                         generals_rho = 0, specifics_rho = 0) {

  if(crossloadings > 0.4) {

    stop("Crossloadings are too large")

  }

  # Rangos de pesos:

  if(loadings_g == "low") {
    loadings_g. = c(.3, .5)
  } else if(loadings_g == "medium") {
    loadings_g. = c(.4, .6)
  } else if(loadings_g == "high") {
    loadings_g. = c(.5, .7)
  }

  if(loadings_s == "low") {
    loadings_s. = c(.3, .5)
  } else if(loadings_s == "medium") {
    loadings_s. = c(.4, .6)
  } else if(loadings_s == "high") {
    loadings_s. = c(.5, .7)
  }

  # Numero factores especificos:
  n_specifics <- n_generals * specifics_per_general

  # Numero items:
  n_items <- n_specifics * items_per_specific

  # Numero items por general:
  items_per_general <- n_items / n_generals

  # Numero total factores:
  n_factors <- n_generals + n_specifics

  # Inicializar matriz de pesos:
  A <- matrix(NA, nrow = n_items, ncol = n_factors)

  # Pesos en factores específicos:

  sequen <- seq(loadings_s.[2], loadings_s.[1], length.out = items_per_specific)

  for(i in 0:(n_specifics-1)) {

    start_row <- 1 + i*items_per_specific
    end_row <- start_row + items_per_specific - 1
    A[start_row:end_row , 1+i+n_generals] <- sequen

  }

  # Pesos en factores generales:

  for(i in 0:(n_generals-1)) {

    start_row <- 1 + i*items_per_general
    end_row <- start_row + items_per_general - 1
    A[start_row:end_row , i+1] <- runif(items_per_general, loadings_g.[1], loadings_g.[2])

  }

  colnames(A) <- c(paste("G", 1:n_generals, sep = ""), paste("S", 1:n_specifics, sep = ""))

  # Items puros:

  if(pure) {

    value <- sequen[floor(items_per_specific/2 + 1)]
    row_indexes <- unlist(apply(A, 2, FUN = function(x) which(x == value)))
    column_indexes <- apply(A[row_indexes, ], 1, FUN = function(x) which(x > 0))
    n <- n_specifics * n_generals
    indexes <- which(!is.na(A[row_indexes, 1:n_generals]))
    m <- sqrt(A[row_indexes, 1:n_generals][indexes]^2 +
                A[row_indexes, ][which(A[row_indexes, ] == value)]^2)
    A[row_indexes, 1:n_generals][indexes] <- m
    A[row_indexes, ][which(A[row_indexes, ] == value)] <- 0.01

  }

  # Pesos cruzados

  if(crossloadings != 0) {

    ratio <- specifics_per_general
    row_index <- seq(items_per_specific+1, n_items, by = items_per_specific)
    col_index <- seq(n_generals+1, n_factors-1, by = 1)

    if(ratio < length((row_index))) {
      delete <- seq(ratio, length(row_index), by = ratio)
      row_index <- row_index[-delete]
      col_index <- col_index[-delete]
    }

    # Introducimos los crossloadings y recalibramos los correspondientes al general y especifico:

    for(i in 1:length(row_index)) {

      row_indexes <- row_index[i]:(row_index[i])
      col_index_2 <- which(A[row_indexes[1], ] > 0)
      A[row_indexes, col_index[i]] <- crossloadings
      A[row_indexes, col_index_2] <- sqrt(A[row_indexes, col_index_2]^2 - crossloadings^2/2)

    }

    for(i in 1:n_generals) {

      row_index <- items_per_general*(i-1)+1
      row_indexes <- row_index:(row_index)
      col_index_2 <- which(A[row_indexes[1], ] > 0)
      A[row_indexes, n_generals+i*ratio] <- crossloadings
      A[row_indexes, col_index_2] <- sqrt(A[row_indexes, col_index_2]^2 - crossloadings^2/2)

    }

  }

  A[is.na(A)] <- 0
  rownames(A) <- paste("item", 1:nrow(A), sep = "")

  # Correlaciones entre factores:

  Phi <- matrix(0, n_factors, n_factors)
  Phi[1:n_generals, 1:n_generals] <- generals_rho
  Phi[-(1:n_generals), -(1:n_generals)] <- specifics_rho
  diag(Phi) <- 1

  # Matriz de correlaciones entre items poblacional:

  R <- A %*% Phi %*% t(A)
  uniquenesses <- 1 - diag(R)
  diag(R) <- 1

  # Ejecutamos 'sim_bifactor' recursivamente hasta que ninguna comunalidad sea superior a 1:

  if( any(uniquenesses < 0) ) {

    warning("At least a communality greater than 1 found \n Resampling...")

    sim <- sim_bifactor(n_generals, n_specifics, items_per_specific,
                        loadings, generals_rho, specifics_rho,
                        crossloadings, pure)

    A = sim$A; R = sim$R; Phi = sim$Phi; uniquenesses = sim$uniquenesses

  }

  return( list(A = A, R = R, Phi = Phi, uniquenesses = uniquenesses) )

}
