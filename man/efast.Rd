% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cpp_functions.R
\name{efast}
\alias{efast}
\title{Fast exploratory factor analysis.}
\usage{
efast(R, n_factors, method = "minres", rotation = "oblimin", projection = "oblq",
Target = NULL, Weight = NULL, PhiTarget = NULL, PhiWeight = NULL,
oblq_blocks = NULL, normalize = FALSE, gamma = 0, epsilon = 1e-02, k = 0, w = 1,
random_starts = 1L, cores = 1L, init = NULL, efa_control = NULL, rot_control = NULL)
}
\arguments{
\item{R}{Correlation matrix.}

\item{n_factors}{Number of common factors to extract.}

\item{method}{EFA fitting method: "ml" (maximum likelihood for multivariate normal items), "minres" (minimum residuals), "pa" (principal axis) and "minrank" (minimum rank). Defaults to "minres".}

\item{rotation}{Rotation criterion. Available rotations: "varimax", "cf" (Crawford-Ferguson), "oblimin", "geomin", "target", "xtarget" (extended target) and "none". Defaults to "oblimin".}

\item{projection}{Projection method. Available projections: "orth" (orthogonal), "oblq" (oblique), "poblq" (partially oblique). Defaults to "oblq".}

\item{Target}{Target matrix for the loadings. Defaults to NULL.}

\item{Weight}{Weight matrix for the loadings. Defaults to NULL.}

\item{PhiTarget}{Target matrix for the factor correlations. Defaults to NULL.}

\item{PhiWeight}{Weight matrix for the factor correlations. Defaults to NULL.}

\item{oblq_blocks}{Vector with the number of factors for each oblique block. E.g.: c(2, 4) means that there are two blocks of oblique factors: one block with 2 factors and another block with 4 factors. Everything else is orthogonal. Defaults to NULL.}

\item{normalize}{Kaiser normalization. Defaults to FALSE.}

\item{gamma}{\eqn{\gamma} parameter for the oblimin criterion. Defaults to 0 (quartimin).}

\item{epsilon}{\eqn{\epsilon} parameter for the geomin criterion. Defaults to 0.01.}

\item{k}{\eqn{k} parameter for the Crawford-Ferguson family of rotation criteria. Defaults to 0.}

\item{w}{\eqn{w} parameter for the extended target criterion ("xtarget"). Defaults to 1L.}

\item{random_starts}{Number of rotations with different random starting values. The rotation with the smallest cost function value is returned. Defaults to 1L.}

\item{cores}{Number of cores for parallel execution of random starts. Defaults to 1L.}

\item{init}{Initial uniquenesses values for exploratory factor analsyis estimation. Defaults to NULL.}

\item{efa_control}{List of control parameters for efa fitting. Defaults to NULL.}

\item{rot_control}{List of control parameters for the rotation algorithm. Defaults to NULL.}
}
\value{
List of class \code{efast} with the following components:
\item{efa}{List containing the following objects:}

\itemize{
\item loadings - Unrotated loadings.
\item uniquenesses - Vector of uniquenesses.
\item Rhat - Correlation matrix predicted by the model.
\item residuals - Residual correlation matrix.
\item f - Objective value at the minimum.
\item Heywood - TRUE if any Heywood case is encountered and FALSE otherwise.
\item iterations - Number of iterations for the L-BFGS-B algorithm to converge.
\item convergence - TRUE if the L-BFGS-B algorithm converged and FALSE otherwise.
\item method - Method used to fit the exploratory factor analysis.
}

\item{rotation}{List of class \code{rotation}. Only if the argument \code{rotation} is not "none". See \code{rotate} for the components.}
\item{elapsed}{Total amount spent for execution (in nanoseconds).}
}
\description{
Fast exploratory factor analysis.
}
\details{
If \code{efa.control = NULL}, then \code{list(maxit = 1e4)} is passed to \code{efa.control}. If \code{rot_control = NULL}, then \code{list(maxit = 1000, eps = 1e-05)} is passed to \code{rot_control}, where \code{eps} is the absolute tolerance. When the objective function does not make a larger improvement than \code{eps}, the algorithm is assumed to converge.

If \code{Target} is provided but not \code{Weight}, then \code{Weight = 1 - Target} by default, which means a partially specified target rotation is performed. The same applies for \code{PhiTarget} and \code{PhiWeight}.

If \code{init = NULL}, then the squared multiple correlations of each item with the remaining ones are used as initial values (These are known to be upper bounds).

If a Heywood case is encountered, then \code{method =} "minrank" is automatically applied to ensure positive uniquenesses.
}
\examples{

\dontrun{
# Simulate data:
sim <- sim_twoTier(n_generals = 2, groups_per_general = 5, items_per_group = 6)
scores <- MASS::mvrnorm(1e3, rep(0, nrow(sim$R)), Sigma = sim$R)
s <- cor(scores)

# Fit efa:
efa <- efast(s, n_factors = 12, method = "minres", rotation = "oblimin",
projection = "oblq", gamma = 0, random_starts = 10L, cores = 1L)
}

}
\references{
Jiménez, M., Abad, F.J., Garcia-Garzon, E., Garrido, L.E. (2021, June 24). Exploratory Two-tier Modeling. Under review. Retrieved from https://osf.io/7aszj/?view_only=8f7bd98025104347a96f60a6736f5a64
}
