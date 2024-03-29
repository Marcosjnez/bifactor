% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cpp_functions.R
\name{poly}
\alias{poly}
\title{Fast polychoric correlations.}
\usage{
poly(X, cores = 1L, acov = "none")
}
\arguments{
\item{X}{Matrix of categorical scores. The lowest score must start at 0.}

\item{cores}{Number of parallel cores to compute the polychoric correlations.}

\item{acov}{Use acov = 'cov' to obtain the asymptotic covariance matrix and acov = 'var' to simply obtain the asymptotic variances. Defaults to "none".}
}
\value{
A list with the polychoric correlations, the thresholds, and the elapsed time in nanoseconds.
}
\description{
Compute huge polychoric correlation matrices very fast.
}
\details{
None yet.
}
