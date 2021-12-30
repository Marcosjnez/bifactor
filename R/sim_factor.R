#' @title
#'
#' Simulate a bi-factor or generalized bifactor population structure.
#'
#' @description
#'
#' Simulate a bi-factor or generalized bifactor population structure with cross-loading, pure items and correlated factors.
#'
#' @usage
#'
#' sim_factor(n_generals, groups_per_general, items_per_group,
#' loadings_g = "medium", loadings_s = "medium",
#' crossloadings = 0, pure = FALSE,
#' generals_rho = 0, groups_rho = 0, confirmatory = TRUE,
#' method = "minres", fit = "rmsr", misfit = 0)
#'
#' @param n_generals Number of general factors.
#' @param groups_per_general Number of group factors per general factor.
#' @param items_per_group Number of items per group factor.
#' @param loadings_g Loadings' magnitude on the general factors: "low", "medium" or "high". Defaults to "medium".
#' @param loadings_s Loadings' magnitude on the group factors: "low", "medium" or "high". Defaults to "medium".
#' @param crossloadings Magnitude of the cross-loadings among the group factors. Defaults to 0.
#' @param pure Fix a pure item on each general factor. Defaults to FALSE.
#' @param generals_rho Correlation among the general factors. Defaults to 0.
#' @param groups_rho Correlation among the group factors. Defaults to 0.
#' @param confirmatory Logical. Should the misfit value be computed according to a confirmatory model (TRUE) or an exploratory model (FALSE). Defaults to TRUE.
#' @param method Method used to generate population error: "minres" or "ml".
#' @param fit Fit index to control the population error.
#' @param misfit Misfit value to generate population error.
#'
#' @details \code{sim_factor} generates bi-factor and generalized bifactor patterns with cross-loadings, pure items and
#' correlations among the general and group factors. When \code{crossloading} is different than 0, one cross-loading
#' is introduced for an item pertaining to each group factor. When \code{pure} is TRUE, one item loading of each group
#' factor is removed so that the item loads entirely on the general factor. To maintain the item communalities
#' constant upon these modifications, the item loading on the other factors may shrunk (if adding cross-loadings)
#' or increase (if setting pure items).
#'
#' Loading magnitudes may range between 0.3-0.5 ("low"), 0.4-0.6 ("medium") and 0.5-0.7 ("high").
#'
#' @return List with the following objects:
#' \item{lambda}{Population loading matrix.}
#' \item{Phi}{Population factor correlation matrix.}
#' \item{R}{Population correlation matrix.}
#' \item{R_error}{Population correlation matrix with error.}
#' \item{uniquenesses}{Vector of population uniquenesses.}
#' \item{delta}{Minimum of the objective function that correspond to the misfit value.}
#'
#' @references
#'
#' Jiménez, M., Abad, F.J., Garcia-Garzon, E., Garrido, L.E. (2021, June 24). Exploratory generalized bifactor Modeling. Under review. Retrieved from https://osf.io/7aszj/?view_only=8f7bd98025104347a96f60a6736f5a64
#'
#' @export
sim_factor <- function(n_generals, groups_per_general, items_per_group,
                       loadings_g = "medium", loadings_s = "medium",
                       crossloadings = 0, pure = FALSE,
                       generals_rho = 0, groups_rho = 0,
                       confirmatory = TRUE, method = "minres",
                       fit = "rmsr", misfit = 0) {

  root_ml <- function(x, delta, G) {

    I <- diag(nrow(G))
    f <- x*sum(diag(G)) - log(det(I + x*G)) - delta

    return(f^2)

  }

  groot_ml <- function(x, delta, G) {

    I <- diag(nrow(G))
    f <- x*sum(diag(G)) - log(det(I + x*G)) - delta
    g <- sum(diag(G)) - sum(diag(solve(I + x*G) %*% G))
    g <- 2*g*f

    return(g)

  }

  dxt <- function(X) {

    # derivative wrt transpose (just a permutation matrix)

    p <- nrow(X)
    q <- ncol(X)
    pq <- p*q

    res <- array(0, dim = c(pq, pq))
    null <- matrix(0, p, q)

    for(i in 1:pq) {
      temp <- null
      temp[i] <- 1
      res[, i] <- c(t(temp))
    }

    return(res)

  }

  gLRhat <- function(Lambda, Phi) {

    # derivative of Lambda wrt Rhat

    p <- nrow(Lambda)
    g1 <- (Lambda %*% Phi) %x% diag(p)
    g21 <- diag(p) %x% (Lambda %*% Phi)
    g2 <- g21 %*% dxt(Lambda)
    g <- g1 + g2

    return(g)

  }

  gPRhat <- function(Lambda, Phi) {

    g1 <- Lambda %x% Lambda
    g2 <- g1 %*% dxt(Phi)
    g <- g1 + g2
    g <- g[, which(lower.tri(Phi))]

    return(g)

  }

  guRhat <- function(p) {

    gu <- matrix(0, p*p, p)

    for(i in 1:p) {

      index <- (i-1)*p + i
      gu[index, i] <- 1

    }

    return(gu)

  }

  ng <- n_generals
  condition <- n_generals == 0
  if(condition) n_generals <- 1

  if(crossloadings > 0.4) {

    stop("Crossloadings are too large")

  }

  # loadings' range:

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

  # Total number of group factors:
  n_groups <- n_generals * groups_per_general

  # Total number of items:
  n_items <- n_groups * items_per_group

  # Number of items per general:
  items_per_general <- n_items / n_generals

  # Total number of factors:
  n_factors <- n_generals + n_groups

  # Initialize the loading matrix:
  lambda <- matrix(NA, nrow = n_items, ncol = n_factors)

  # Item loadings on the group factors:

  sequen <- seq(loadings_s.[2], loadings_s.[1], length.out = items_per_group)

  for(i in 0:(n_groups-1)) {

    start_row <- 1 + i*items_per_group
    end_row <- start_row + items_per_group - 1
    lambda[start_row:end_row , 1+i+n_generals] <- sequen
    # lambda[start_row:end_row , 1+i+n_generals] <- mean(loadings_s.)

  }

  # Simulate item loadings on the general factors:

  for(i in 0:(n_generals-1)) {

    start_row <- 1 + i*items_per_general
    end_row <- start_row + items_per_general - 1
    lambda[start_row:end_row , i+1] <- stats::runif(items_per_general, loadings_g.[1], loadings_g.[2])
    # lambda[start_row:end_row , i+1] <- mean(loadings_g.)

  }

  colnames(lambda) <- c(paste("G", 1:n_generals, sep = ""), paste("S", 1:n_groups, sep = ""))

  # Pure items:

  if(pure) {

    value <- sequen[floor(items_per_group/2 + 1)]
    row_indexes <- unlist(apply(lambda, 2, FUN = function(x) which(x == value)))
    column_indexes <- apply(lambda[row_indexes, ], 1, FUN = function(x) which(x > 0))
    n <- n_groups * n_generals
    indexes <- which(!is.na(lambda[row_indexes, 1:n_generals]))
    m <- sqrt(lambda[row_indexes, 1:n_generals][indexes]^2 +
                lambda[row_indexes, ][which(lambda[row_indexes, ] == value)]^2)
    lambda[row_indexes, 1:n_generals][indexes] <- m
    lambda[row_indexes, ][which(lambda[row_indexes, ] == value)] <- 0.01

  }

  # Cross-loadings:

  if(crossloadings != 0) {

    ratio <- groups_per_general
    row_index <- seq(items_per_group+1, n_items, by = items_per_group)
    col_index <- seq(n_generals+1, n_factors-1, by = 1)

    if(ratio < length((row_index))) {
      delete <- seq(ratio, length(row_index), by = ratio)
      row_index <- row_index[-delete]
      col_index <- col_index[-delete]
    }

    # Insert cross-loadings and the recalibrate the general and group factors to maintain the previous communality:

    for(i in 1:length(row_index)) {

      row_indexes <- row_index[i]:(row_index[i])
      col_index_2 <- which(lambda[row_indexes[1], ] > 0)
      lambda[row_indexes, col_index[i]] <- crossloadings
      lambda[row_indexes, col_index_2] <- sqrt(lambda[row_indexes, col_index_2]^2 - crossloadings^2/2)

    }

    for(i in 1:n_generals) {

      row_index <- items_per_general*(i-1)+1
      row_indexes <- row_index:(row_index)
      col_index_2 <- which(lambda[row_indexes[1], ] > 0)
      lambda[row_indexes, n_generals+i*ratio] <- crossloadings
      lambda[row_indexes, col_index_2] <- sqrt(lambda[row_indexes, col_index_2]^2 - crossloadings^2/2)

    }

  }

  lambda[is.na(lambda)] <- 0
  rownames(lambda) <- paste("item", 1:nrow(lambda), sep = "")

  # Factor correlations:

  Phi <- matrix(0, n_factors, n_factors)
  Phi[1:n_generals, 1:n_generals] <- generals_rho
  Phi[-(1:n_generals), -(1:n_generals)] <- groups_rho
  diag(Phi) <- 1

  if(condition) { # if n_generals == 0, remove the general factor

    lambda <- lambda[, -1, drop = FALSE]
    Phi <- Phi[-1, , drop = FALSE][, -1, drop = FALSE]

  }

  # Population correlation matrix:

  R <- lambda %*% Phi %*% t(lambda)
  uniquenesses <- 1 - diag(R)
  diag(R) <- 1
  R_error <- R
  delta <- 0

  # Execute sim_factor recursively until no communality is greater than 1:

  if( any(uniquenesses < 0) ) {

    warning("At least a communality greater than 1 found \n Resampling...")

    sim <- sim_factor(ng, groups_per_general, items_per_group,
                        loadings_g, loadings_s, crossloadings, pure,
                        generals_rho, groups_rho)

    lambda = sim$lambda; R = sim$R; Phi = sim$Phi; uniquenesses = sim$uniquenesses

  } else if(misfit != 0 & misfit != "zero") { # Population error?

    p <- nrow(R)
    q <- ncol(lambda)

    tdiag <- TRUE
    dS_du <- guRhat(p)

    if(confirmatory){

      # Select the columns corresponding to estimated loadings (only works when
      # not estimating correlations)
      # if(!correlation) dS_dL <- dS_dL[, which(lambda > 0)]
      pars <- sum(lambda > 0) + p + sum(abs(Phi[lower.tri(Phi)]) > 0)
      dS_dL <- gLRhat(lambda, Phi)[, which(lambda != 0)]
      dS_dP <- gPRhat(lambda, Phi)[, which(Phi[lower.tri(Phi)] != 0)]
      gS <- cbind(dS_dL, dS_dP, dS_du) # matrix of derivatives wrt the correlation model

    } else {

      # dS_dP <- gPRhat(lambda, Phi)
      # gS <- cbind(dS_dL, dS_dP, dS_du)
      pars <- p*q + p - 0.5*q*(q-1) # Bartholomew's book (Chapter 3.12.1; 3rd edition)
      dS_dL <- gLRhat(lambda, Phi)
      # dS_dP <- gPRhat(lambda, Phi)
      # gS <- cbind(dS_dL, dS_dP, dS_du) # matrix of derivatives wrt the correlation model
      gS <- cbind(dS_dL, dS_du) # matrix of derivatives wrt the correlation model

    }

    df <- p*(p+1)/2 - pars

    # Cudeck and Browne (1992):

    if(method == "minres" || method == "ols") {

      B <- -2*gS[lower.tri(R, diag = tdiag), ]
      # B <- -2*lambda %*%

    } else if(method == "ml") {

      # K <- transition(p)
      # MP_inv <- solve(t(K) %*% K) %*% t(K)
      # D <- MP_inv %*% t(MP_inv)
      indexes <- vector(length = p)
      indexes[1] <- 1
      for(i in 2:p) {
        increment <- i
        indexes[i] <- indexes[i-1]+increment
      }
      D <- matrix(0, p*(p+1)/2, p*(p+1)/2)
      diag(D) <- 2
      diag(D)[indexes] <- 1
      R_inv <- solve(R)
      vecs <- apply(gS, 2, FUN = function(x) -t(R_inv %*% matrix(x, p, p) %*% R_inv))
      B <- t(vecs[which(upper.tri(R, diag = tdiag)), ]) %*% D
      B <- t(B)

    }

    # BtB <- t(B) %*% B
    m <- p+1
    U <- replicate(p, stats::runif(m, 0, 1))
    A1 <- t(U) %*% U
    sq <- diag(1/sqrt(diag(A1)))
    A2 <- sq %*% A1 %*% sq
    diag_u <- diag(sqrt(uniquenesses))
    y <- diag_u %*% A2 %*% diag_u
    y <- y[lower.tri(y, diag = tdiag)]
    # y <- A2[lower.tri(A2, diag = tdiag)]
    # y <- stats::runif(p*(p+1)/2, 0, 1)
    # e <- qr.Q(qr(cbind(B, y)))[, ncol(B)+1]
    # v <- MASS::ginv(BtB) %*% t(B) %*% y
    # e <- y - B %*% v # equation 7
    # B.qr <- qr(B)
    # e <- qr.resid(B.qr, y)
    e <- unname(stats::lm(y ~ B, model = FALSE, qr = TRUE)$residuals)

    if(fit == "rmsr") {
      if(misfit == "close") {
        r2 <- mean(1-uniquenesses)
        misfit <- 0.05*r2
      } else if(misfit == "acceptable") {
        r2 <- mean(1-uniquenesses)
        misfit <- 0.10*r2
      }
      delta <- misfit^2*0.5*p*(p-1)
      # delta <- (1-misfit2)*(0.5*(sum(R_error^2) - p))
    } else if(fit == "cfi") {
      null_f <- 0.5*(sum(R^2) - p)
      delta <- (1-misfit)*null_f
    } else if(fit == "rmsea") {
      delta <- misfit^2 * df
    } else if(fit == "raw") {
      delta <- misfit
    }

    if(method == "minres" || method == "ols") {

      E <- matrix(0, p, p)
      E[lower.tri(E, diag = tdiag)] <- e
      E <- t(E) + E
      diag(E) <- 0
      k <- sqrt(2*delta/sum(E*E))
      E <- k*E

    } else if(method == "ml") {

      E <- matrix(0, p, p)
      E[upper.tri(R, diag = tdiag)] <- e
      E <- t(E) + E
      diag(E) <- 0
      E <- 1e-04*E # Fix this to avoid NAs
      G <- R_inv %*% E
      x <- stats::runif(1, 0, 1)
      root <- stats::optim(x, fn = root_ml, gr = groot_ml, method = "L-BFGS-B",
                    lower = -Inf, upper = Inf, G = G, delta = delta)
      k <- root$par
      E <- k*E

    }

    R_error <- R + E

  }

  return( list(lambda = lambda, Phi = Phi, R = R, R_error = R_error,
               uniquenesses = uniquenesses, delta = delta) )

}
