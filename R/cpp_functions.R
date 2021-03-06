#' @title
#'
#' Generate random orthogonal matrices.
#'
#' @description
#'
#' Generate random orthogonal matrices from a standard normal distribution. First, a matrix of random standard normal variables is simulated and then, the Q factor from the QR decomposition is returned.
#'
#' @usage
#'
#' random_orth(p, q)
#'
#' @param p Number of rows.
#' @param q Number of columns. Should not be greater than p.
#'
#' @return An orthogonal matrix with normally distributed data.
#'
#' @references
#'
#' Jiménez, M., Abad, F.J., Garcia-Garzon, E., Garrido, L.E. (2021, June 24). Exploratory Two-tier Modeling. Under review. Retrieved from https://osf.io/7aszj/?view_only=8f7bd98025104347a96f60a6736f5a64
#'
#' @export
random_orth <- function(p, q) {
  .Call(`_bifactor_random_orth`, p, q)
}

#' @title
#'
#' Generate random oblique matrices.
#'
#' @description
#'
#' Generate random oblique matrices from a standard normal distribution.
#'
#' @usage
#'
#' random_oblq(p, q)
#'
#' @param p Number of rows.
#' @param q Number of columns. Should not be greater than p.
#'
#' @return An oblique matrix with normally distributed data.
#'
#' @references
#'
#' Jiménez, M., Abad, F.J., Garcia-Garzon, E., Garrido, L.E. (2021, June 24). Exploratory Two-tier Modeling. Under review. Retrieved from https://osf.io/7aszj/?view_only=8f7bd98025104347a96f60a6736f5a64
#'
#' @export
random_oblq <- function(p, q) {
  .Call(`_bifactor_random_oblq`, p, q)
}

#' @title
#'
#' Generate a random partially oblique matrix.
#'
#' @description
#'
#' First, a matrix is simulated from a standard normal distribution. Second,the matrix is normalized and the Gram-Schmidt process is performed between the oblique blocks. Finally, the orthogonal blocks correspond to those columns of the Q matrix from the QR decomposition.
#'
#' @usage
#'
#' random_poblq(p, q, oblq_blocks)
#'
#' @param p Number of rows.
#' @param q Number of columns. Should not be greater than p.
#' @param oblq_blocks A vector with the number of factors for each oblique block. E.g.: c(2, 4) means that there are two blocks of oblique factors: one with 2 factors and another with 4 factors. Everything else is orthogonal.
#'
#' @return A partially oblique matrix.
#'
#' @examples
#'
#' random_poblq(p = 7, q = 7, oblq_blocks = c(3, 2))
#'
#' @references
#'
#' Jiménez, M., Abad, F.J., Garcia-Garzon, E., Garrido, L.E. (2021, June 24). Exploratory Two-tier Modeling. Under review. Retrieved from https://osf.io/7aszj/?view_only=8f7bd98025104347a96f60a6736f5a64
#'
#' @export
random_poblq <- function(p, q, oblq_blocks) {
  .Call(`_bifactor_random_poblq`, p, q, oblq_blocks)
}

#' @title
#'
#' Retraction of a matrix onto the orthogonal manifold.
#'
#' @description
#'
#' Transform a matrix into an orthogonal matrix.
#'
#' @usage
#'
#' retr_orth(X)
#'
#' @param X A matrix.
#'
#' @return An orthogonal matrix.
#'
#' @references
#'
#' Jiménez, M., Abad, F.J., Garcia-Garzon, E., Garrido, L.E. (2021, June 24). Exploratory Two-tier Modeling. Under review. Retrieved from https://osf.io/7aszj/?view_only=8f7bd98025104347a96f60a6736f5a64
#'
#' @export
retr_orth <- function(X) {
  .Call(`_bifactor_retr_orth`, X)
}

#' @title
#'
#' Retraction of a matrix onto the oblique manifold.
#'
#' @description
#'
#' Transform a matrix into an oblique matrix.
#'
#' @usage
#'
#' retr_oblq(X)
#'
#' @param X A matrix.
#'
#' @return An oblique matrix.
#'
#' @references
#'
#' Jiménez, M., Abad, F.J., Garcia-Garzon, E., Garrido, L.E. (2021, June 24). Exploratory Two-tier Modeling. Under review. Retrieved from https://osf.io/7aszj/?view_only=8f7bd98025104347a96f60a6736f5a64
#'
#' @export
retr_oblq <- function(X) {
  .Call(`_bifactor_retr_oblq`, X)
}

#' @title
#'
#' Retraction of a matrix onto the partially oblique manifold.
#'
#'  @description
#'
#' Transform a matrix into a partially oblique matrix.
#'
#' @usage
#'
#' retr_poblq(X, oblq_blocks)
#'
#' @param X A matrix.
#' @param oblq_blocks A vector with the number of factors for each oblique block. E.g.: c(2, 4) means that there are two blocks of oblique factors: one with 2 factors and another with 4 factors. Everything else is orthogonal.
#'
#' @return A partially oblique matrix.
#'
#' @references
#'
#' Jiménez, M., Abad, F.J., Garcia-Garzon, E., Garrido, L.E. (2021, June 24). Exploratory Two-tier Modeling. Under review. Retrieved from https://osf.io/7aszj/?view_only=8f7bd98025104347a96f60a6736f5a64
#'
#' @examples
#'
#' X <- replicate(8, rnorm(8))
#' retr_poblq(X, c(2, 3, 3))
#'
#' @export
retr_poblq <- function(X, oblq_blocks) {
  .Call(`_bifactor_retr_poblq`, X, oblq_blocks)
}

#' @title
#'
#' Schmid-Leiman Transformation.
#'
#' @description
#'
#' Schmid-Leiman transformation into a bi-factor or two-tier pattern.
#'
#' @usage
#'
#' sl(R, n_generals, n_groups, first_efa = NULL, second_efa = NULL)
#'
#' @param R Correlation matrix.
#' @param n_generals Number of general factors.
#' @param n_groups Number of group factors.
#' @param first_efa Arguments to pass to \code{efast} in the first-order factor extraction. See \code{efast} for the default arguments.
#' @param second_efa Arguments to pass to \code{efast} in the second-order factor extraction. See \code{efast} for the default arguments.
#'
#' @details First, a hierarchical factor model is fitted using a second-order factor analysis on the factor correlation obtained from a first-order factor analysis. Then, the item loadings on the general factors are assumed to be the direct effects of the general factors according to such hierarchical model.
#' On the other hand, the item loadings on the group factors become the originally first-order loadings post-multiplied by the diagonal matrix containing the root of the item uniquenesses.
#'
#' Obviously, the first-order factor analysis should be oblique to perform a second exploratory factor analysis.
#'
#' If the second-order solution does not use an orthogonal projection, then the correlation matrix among the general factors for the Schmid-Leiman solution is simply that obtained from such second-order solution.
#'
#' @return
#'
#' \item{loadings}{Loading matrix of the Schmid-Leiman solution.}
#' \item{first_order_solution}{Object of class \code{efast} with the first-order solution.}
#' \item{second_order_solution}{Object of class \code{efast} with the second-order solution.}
#' \item{uniquenesses}{Vector of uniquenesses.}
#' \item{Rhat}{Correlation matrix predicted by the (hierarchical) model.}
#'
#' @references
#'
#' Jiménez, M., Abad, F.J., Garcia-Garzon, E., Garrido, L.E. (2021, June 24). Exploratory Two-tier Modeling. Under review. Retrieved from https://osf.io/7aszj/?view_only=8f7bd98025104347a96f60a6736f5a64
#'
#' @examples
#'
#' \dontrun{
#' # Simulate data:
#' sim <- sim_twoTier(n_generals = 2, groups_per_general = 3, items_per_group = 5)
#' lambda <- sim$lambda
#' Target <- ifelse(lambda > 0, 1, 0)
#'
#' # Target rotation for the first-order efa and oblimin for the second-order efa:
#' first <- list(rotation = "target", projection = "oblq", Target = Target)
#' second <- list(rotation = "oblimin", projection = "oblq", gamma = 0)
#'
#' SL <- sl(sim$R, n_generals = 2, n_groups = 6, first, second)
#'}
#'
#' @export
sl <- function(R, n_generals, n_groups, first_efa = NULL, second_efa = NULL) {
  .Call(`_bifactor_sl`, R, n_generals, n_groups, first_efa, second_efa)
}

#' @title
#'
#' Fast rotation algorithm for factor analysis.
#'
#' @description
#'
#' Riemannian Newton Trust-Region algorithm to quickly perform (parallel) rotations with different random starting values. The main purpose is finding a global minimum.
#'
#' @usage
#'
#' rotate(loadings, rotation = "oblimin", projection = "oblq",
#' Target = NULL, Weight = NULL, PhiTarget = NULL, PhiWeight = NULL,
#' oblq_blocks = NULL, gamma = 0, epsilon = 0.01, k = 0, w = 1,
#' random_starts = 1L, cores = 1L, rot_control = NULL)
#'
#' @param loadings Unrotated loading matrix.
#' @param rotation Rotation criterion. Available rotations: "varimax", "cf" (Crawford-Ferguson), "oblimin", "geomin", "target", "xtarget" (extended target) and "none". Defaults to "oblimin".
#' @param projection Projection method. Available projections: "orth" (orthogonal), "oblq" (oblique), "poblq" (partially oblique). Defaults to "oblq".
#' @param Target Target matrix for the loadings. Defaults to NULL.
#' @param Weight Weight matrix for the loadings. Defaults to NULL.
#' @param PhiTarget Target matrix for the factor correlations. Defaults to NULL.
#' @param PhiWeight Weight matrix for the factor correlations. Defaults to NULL.
#' @param oblq_blocks Vector with the number of factors for each oblique block. E.g.: c(2, 4) means that there are two blocks of oblique factors: one block with 2 factors and another block with 4 factors. Everything else is orthogonal. Defaults to NULL.
#' @param gamma \eqn{\gamma} parameter for the oblimin criterion. Defaults to 0 (quartimin).
#' @param epsilon \eqn{\epsilon} parameter for the geomin criterion. Defaults to 0.01.
#' @param k \eqn{k} parameter for the Crawford-Ferguson family of rotation criteria. Defaults to 0.
#' @param w \eqn{w} parameter for the extended target criterion ("xtarget"). Defaults to 1.
#' @param random_starts Number of rotations with different random starting values. The rotation with the smallest cost function value is returned. Defaults to 1L.
#' @param cores Number of cores for parallel execution of random starts. Defaults to 1L.
#' @param rot_control List of control parameters for the rotation algorithm. Defaults to NULL.
#'
#' @details
#'
#' If \code{rot_control = NULL}, then \code{list(maxit = 1000, eps = 1e-05)} is passed to \code{rot_control}, where \code{eps} is the absolute tolerance. When the objective function does not make a larger improvement than \code{eps}, the algorithm is assumed to converge.
#' If \code{Target} is provided but not \code{Weight}, then \code{Weight = 1 - Target} by default, which means a partially specified target rotation is performed. The same applies for \code{PhiTarget} and \code{PhiWeight}.
#'
#' @return List of class \code{rotation} with the following components:
#' \item{loadings}{Rotated loading matrix.}
#' \item{Phi}{Correlation matrix among the factors.}
#' \item{T}{Rotation matrix.}
#' \item{f}{Objective value at the minimum.}
#' \item{iterations}{Number of iterations for the rotation algorithm to converge.}
#' \item{convergence}{TRUE if the algorithm converged and FALSE otherwise.}
#' \item{elapsed}{Total amount of time spent for execution (in nanoseconds).}
#'
#' @references
#'
#' Jiménez, M., Abad, F.J., Garcia-Garzon, E., Garrido, L.E. (2021, June 24). Exploratory Two-tier Modeling. Under review. Retrieved from https://osf.io/7aszj/?view_only=8f7bd98025104347a96f60a6736f5a64
#'
#' Zhang, G., Hattori, M., Trichtinger, L. A., & Wang, X. (2019). Target rotation with both factor loadings and factor correlations. Psychological Methods, 24(3), 390–402. https://doi.org/10.1037/met0000198
#'
#' @export
rotate <- function(loadings, rotation = "oblimin", projection = "oblq", Target = NULL, Weight = NULL, PhiTarget = NULL, PhiWeight = NULL, oblq_blocks = NULL, gamma = 0, epsilon = 0.01, k = 0, w = 1, random_starts = 1L, cores = 1L, rot_control = NULL) {
  .Call(`_bifactor_rotate`, loadings, rotation, projection, Target, Weight, PhiTarget, PhiWeight, oblq_blocks, gamma, epsilon, k, w, random_starts, cores, rot_control)
}

#' @title
#'
#' Fast exploratory factor analysis.
#'
#' @description
#'
#' Fast exploratory factor analysis.
#'
#' @usage
#'
#' efast(R, n_factors, method = "minres", rotation = "oblimin", projection = "oblq",
#' Target = NULL, Weight = NULL, PhiTarget = NULL, PhiWeight = NULL,
#' oblq_blocks = NULL, normalize = FALSE, gamma = 0, epsilon = 1e-02, k = 0, w = 1,
#' random_starts = 1L, cores = 1L, init = NULL, efa_control = NULL, rot_control = NULL)
#'
#' @param R Correlation matrix.
#' @param n_factors Number of common factors to extract.
#' @param method EFA fitting method: "ml" (maximum likelihood for multivariate normal items), "minres" (minimum residuals), "pa" (principal axis) and "minrank" (minimum rank). Defaults to "minres".
#' @param rotation Rotation criterion. Available rotations: "varimax", "cf" (Crawford-Ferguson), "oblimin", "geomin", "target", "xtarget" (extended target) and "none". Defaults to "oblimin".
#' @param projection Projection method. Available projections: "orth" (orthogonal), "oblq" (oblique), "poblq" (partially oblique). Defaults to "oblq".
#' @param Target Target matrix for the loadings. Defaults to NULL.
#' @param Weight Weight matrix for the loadings. Defaults to NULL.
#' @param PhiTarget Target matrix for the factor correlations. Defaults to NULL.
#' @param PhiWeight Weight matrix for the factor correlations. Defaults to NULL.
#' @param oblq_blocks Vector with the number of factors for each oblique block. E.g.: c(2, 4) means that there are two blocks of oblique factors: one block with 2 factors and another block with 4 factors. Everything else is orthogonal. Defaults to NULL.
#' @param normalize Kaiser normalization. Defaults to FALSE.
#' @param gamma \eqn{\gamma} parameter for the oblimin criterion. Defaults to 0 (quartimin).
#' @param epsilon \eqn{\epsilon} parameter for the geomin criterion. Defaults to 0.01.
#' @param k \eqn{k} parameter for the Crawford-Ferguson family of rotation criteria. Defaults to 0.
#' @param w \eqn{w} parameter for the extended target criterion ("xtarget"). Defaults to 1L.
#' @param random_starts Number of rotations with different random starting values. The rotation with the smallest cost function value is returned. Defaults to 1L.
#' @param cores Number of cores for parallel execution of random starts. Defaults to 1L.
#' @param init Initial uniquenesses values for exploratory factor analsyis estimation. Defaults to NULL.
#' @param efa_control List of control parameters for efa fitting. Defaults to NULL.
#' @param rot_control List of control parameters for the rotation algorithm. Defaults to NULL.
#'
#' @details
#'
#' If \code{efa.control = NULL}, then \code{list(maxit = 1e4)} is passed to \code{efa.control}. If \code{rot_control = NULL}, then \code{list(maxit = 1000, eps = 1e-05)} is passed to \code{rot_control}, where \code{eps} is the absolute tolerance. When the objective function does not make a larger improvement than \code{eps}, the algorithm is assumed to converge.
#'
#' If \code{Target} is provided but not \code{Weight}, then \code{Weight = 1 - Target} by default, which means a partially specified target rotation is performed. The same applies for \code{PhiTarget} and \code{PhiWeight}.
#'
#' If \code{init = NULL}, then the squared multiple correlations of each item with the remaining ones are used as initial values (These are known to be upper bounds).
#'
#' If a Heywood case is encountered, then \code{method =} "minrank" is automatically applied to ensure positive uniquenesses.
#'
#' @return List of class \code{efast} with the following components:
#' \item{efa}{List containing the following objects:}
#'
#' \itemize{
#' \item loadings - Unrotated loadings.
#' \item uniquenesses - Vector of uniquenesses.
#' \item Rhat - Correlation matrix predicted by the model.
#' \item residuals - Residual correlation matrix.
#' \item f - Objective value at the minimum.
#' \item Heywood - TRUE if any Heywood case is encountered and FALSE otherwise.
#' \item iterations - Number of iterations for the L-BFGS-B algorithm to converge.
#' \item convergence - TRUE if the L-BFGS-B algorithm converged and FALSE otherwise.
#' \item method - Method used to fit the exploratory factor analysis.
#' }
#'
#' \item{rotation}{List of class \code{rotation}. Only if the argument \code{rotation} is not "none". See \code{rotate} for the components.}
#' \item{elapsed}{Total amount spent for execution (in nanoseconds).}
#'
#' @references
#'
#' Jiménez, M., Abad, F.J., Garcia-Garzon, E., Garrido, L.E. (2021, June 24). Exploratory Two-tier Modeling. Under review. Retrieved from https://osf.io/7aszj/?view_only=8f7bd98025104347a96f60a6736f5a64
#'
#' @examples
#'
#' \dontrun{
#' # Simulate data:
#' sim <- sim_twoTier(n_generals = 2, groups_per_general = 5, items_per_group = 6)
#' scores <- MASS::mvrnorm(1e3, rep(0, nrow(sim$R)), Sigma = sim$R)
#' s <- cor(scores)
#'
#' # Fit efa:
#' efa <- efast(s, n_factors = 12, method = "minres", rotation = "oblimin",
#' projection = "oblq", gamma = 0, random_starts = 10L, cores = 1L)
#'}
#'
#' @export
efast <- function(R, n_factors, method = "minres", rotation = "oblimin", projection = "oblq", Target = NULL, Weight = NULL, PhiTarget = NULL, PhiWeight = NULL, oblq_blocks = NULL, normalize = FALSE, gamma = 0, epsilon = 1e-02, k = 0, w = 1, random_starts = 1L, cores = 1L, init = NULL, efa_control = NULL, rot_control = NULL) {
  .Call(`_bifactor_efast`, R, n_factors, method, rotation, projection, Target, Weight, PhiTarget, PhiWeight, oblq_blocks, normalize, gamma, epsilon, k, w, random_starts, cores, init, efa_control, rot_control)
}

#' @title
#'
#' Get a target from a loading matrix.
#'
#' @description
#'
#' Get a target for the loading matrix using a custom or empirical cut-off.
#'
#' @param loadings A matrix of loadings.
#' @param Phi A correlation matrix among the factors. Defaults to NULL.
#' @param cutoff The cut-off used to create the target matrix. Defaults to 0.
#'
#' @details
#'
#' If \code{cutoff} is not 0, loadings smaller than such a cut-off are fixed to 0. When \code{cutoff = 0}, an empirical cut-off is used for each column of the loading matrix. They are the mean of the one-lagged differences of the sorted squared normalized loadings. Then, the target is determined by fixing to 0 the squared normalized loadings smaller than such cut-offs.
#'
#' @return A target matrix.
#'
#' @references
#'
#' Garcia-Garzon, E., Abad, F. J., & Garrido, L. E. (2019). Improving bi-factor exploratory modeling: Empirical target rotation based on loading differences. Methodology: European Journal of Research Methods for the Behavioral and Social Sciences, 15(2), 45–55. https://doi.org/10.1027/1614-2241/a000163
#'
#' Jiménez, M., Abad, F.J., Garcia-Garzon, E., Garrido, L.E. (2021, June 24). Exploratory Two-tier Modeling. Under review. Retrieved from https://osf.io/7aszj/?view_only=8f7bd98025104347a96f60a6736f5a64
#'
#' @export
get_target <- function(loadings, Phi = NULL, cutoff = 0) {
  .Call(`_bifactor_get_target`, loadings, Phi, cutoff)
}

#' @title
#'
#' Fit an exploratory bi-factor, two-tier or multiple-tier model.
#'
#' @usage
#'
#' twoTier(R, n_generals, n_groups, twoTier_method = "GSLiD", projection = "oblq",
#' PhiTarget = NULL, PhiWeight = NULL, oblq_blocks = NULL,
#' init_Target = NULL, method = "minres", maxit = 20L,
#' cutoff = 0, w = 1, random_starts = 1L, cores = 1L, init = NULL,
#' efa_control = NULL, rot_control = NULL,
#' SL_first_efa = NULL, SL_second_efa = NULL, verbose = TRUE)
#'
#' @description
#'
#' Fit an exploratory bi-factor, two-tier or multiple-tier model with correlated factors.
#'
#' @param R Correlation matrix.
#' @param n_generals Number of general factors to extract.
#' @param n_groups Number of group factors to extract.
#' @param twoTier_method "GSLiD" and "SL" (Schmid-Leiman) Defaults to "GSLiD".
#' @param projection Projection method. Available projections: "orth" (orthogonal), "oblq" (oblique) and "poblq" (partially oblique). Defaults to "oblq".
#' @param PhiTarget Target matrix for the factor correlations. Defaults to NULL.
#' @param PhiWeight Weight matrix for the factor correlations. Defaults to NULL.
#' @param init_Target Target matrix for the loadings. Defaults to NULL.
#' @param method EFA fitting method: "ml" (maximum likelihood for multivariate normal items), "minres" (minimum residuals), "pa" (principal axis) or "minrank" (minimum rank). Defaults to "minres".
#' @param maxit Maximum number of iterations for the GSLiD algorithm. Defaults to 20L.
#' @param cutoff Cut-off used to update the target matrix upon each iteration. Defaults to 0.
#' @param oblq_blocks Vector with the number of factors for each oblique block. E.g.: c(2, 4) means that there are two blocks of oblique factors: one block with 2 factors and another block with 4 factors. Everything else is orthogonal. Defaults to NULL.
#' @param w \eqn{w} parameter for the extended target criterion ("xtarget"). Defaults to 1L.
#' @param random_starts Number of rotations with different random starting values. The rotation with the smallest cost function value is returned. Defaults to 1L.
#' @param cores Number of cores for parallel execution of multiple rotations. Defaults to 1L.
#' @param init Initial uniquenesses values for exploratory factor analsyis estimation. Defaults to NULL.
#' @param efa_control List of control parameters for efa fitting. Defaults to NULL.
#' @param rot_control List of control parameters for the rotation algorithm. Defaults to NULL.
#' @param SL_first_efa List of arguments to pass to \code{efast} to perform the first-order solution for the Schmid-Leiman method. Defaults to NULL.
#' @param SL_second_efa List of arguments to pass to \code{efast} to perform the second-order solution for the Schmid-Leiman method. Defaults to NULL.
#' @param verbose Print the convergence progress information. Defaults to TRUE.
#'
#' @details
#'
#' If \code{efa.control = NULL}, then \code{list(maxit = 1e4)} is passed to \code{efa.control}. If \code{rot_control = NULL}, then \code{list(maxit = 1000, eps = 1e-05)} is passed to \code{rot_control}, where \code{eps} is the absolute tolerance. When the objective function does not make a larger improvement than \code{eps}, the algorithm is assumed to converge.
#'
#' If \code{Target} is provided but not \code{Weight}, then \code{Weight = 1 - Target} by default, which means a partially specified target rotation is performed. The same applies for \code{PhiTarget} and \code{PhiWeight}.
#'
#' If \code{init = NULL}, then the squared multiple correlations of each item with the remaining ones are used as initial values (These are known to be upper bounds).
#'
#' If \code{init_Target} is provided, then an initial target by means of the Schmid-Leiman transformation is not necessary.
#'
#' If \code{cutoff} is not 0, loadings smaller than such a cut-off are fixed to 0. When \code{cutoff} = 0, an empirical cut-off is used for each column of the loading matrix. They are the mean of the one-lagged differences of the sorted squared normalized loadings. Then, the target is determined by fixing to 0 the squared normalized loadings smaller than such cut-offs.
#'
#' @return List of class \code{twoTier}.
#' \item{efa}{List containing objects related to the exploratory factor analysis estimation. See \code{efast}.}
#' \item{twoTier}{List with the following components:}
#' \itemize{
#' \item loadings - Rotated loading matrix.
#' \item Phi - Factor correlation matrix.
#' \item T - Transformation matrix.
#' \item f - Objective value at the minimum.
#' \item iterations - Number of iterations performed by the rotation algorithm.
#' \item convergence - Convergence of the rotation algorithm.
#' \item uniquenesses - Vector of uniquenesses.
#' \item Rhat - Correlation matrix predicted by the model.
#' \item Target - Updated target matrix.
#' \item Weight - Weight matrix. It is the complement of the updated target.
#' \item GSLiD_iterations - Number of iterations performed by the GSLiD algorithm.
#' \item GSLiD_convergence - Convergence of the GSLiD algorithm.
#' \item min_congruences - Vector containing, for each iteration, the minimum Tucker's congruence between
#'  the current loading matrix and the previous loading matrix.
#' \item max_abs_diffs - Vector containing, for each iteration, the maximum absolute difference between the
#' current loading matrix and the previous loading matrix.
#' }
#'
#' \item{elapsed}{Total amount of time spent for execution (in nanoseconds).}
#'
#' @references
#'
#' Jiménez, M., Abad, F.J., Garcia-Garzon, E., Garrido, L.E. (2021, June 24). Exploratory Two-tier Modeling. Under review. Retrieved from https://osf.io/7aszj/?view_only=8f7bd98025104347a96f60a6736f5a64
#'
#' @examples
#'
#' \dontrun{# Simulate data:
#' sim <- sim_twoTier(n_generals = 3, groups_per_general = 5, items_per_group = 6,
#' generals_rho = 0.3)
#' scores <- MASS::mvrnorm(1e4, rep(0, nrow(sim$R)), Sigma = sim$R)
#' s <- cor(scores)
#'
#' # Fit an exploratory two-tier model with GSLiD:
#' GSLiD <- twoTier(s, n_generals = 3, n_groups = 15, method = "minres",
#' projection = "poblq", twoTier_method = "GSLiD", oblq_blocks = 3,
#' random_starts = 10, cores = 8, w = 1, maxit = 20, verbose = TRUE)
#'}
#'
#' @export
twoTier <- function(R, n_generals, n_groups, twoTier_method = "GSLiD", projection = "oblq", PhiTarget = NULL, PhiWeight = NULL, oblq_blocks = NULL, init_Target = NULL, method = "minres", maxit = 20L, cutoff = 0, w = 1, random_starts = 1L, cores = 1L, init = NULL, efa_control = NULL, rot_control = NULL, SL_first_efa = NULL, SL_second_efa = NULL, verbose = TRUE) {
  .Call(`_bifactor_twoTier`, R, n_generals, n_groups, twoTier_method, projection, PhiTarget, PhiWeight, oblq_blocks, init_Target, method, maxit, cutoff, w, random_starts, cores, init, efa_control, rot_control, SL_first_efa, SL_second_efa, verbose)
}


