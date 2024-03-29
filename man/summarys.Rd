% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/summarys.R
\name{summarys}
\alias{summarys}
\alias{summary.efa}
\alias{summary.bifactor}
\title{S3Methods for summarying}
\usage{
\method{summary}{efa}(object, nobs=NULL, suppress=0, order=FALSE, digits=2, ...)

\method{summary}{bifactor}(object, nobs=NULL, suppress=0, order=FALSE, digits=2, ...)
}
\arguments{
\item{object}{Object from \code{bifactor} package.}

\item{nobs}{Optional number of observations. If not provided, Chi-squared-based statistics will not be computed.}

\item{suppress}{Hide the loadings which absolute magnitudes are smaller than this cutoff. Defaults to 0.}

\item{order}{Order the columns of the pattern matrix according to the variance they account for. Defaults to FALSE.}

\item{digits}{Number of digits to display in the loading and factor correlation matrices.}

\item{...}{Arguments to be passed to or from other methods.}
}
\value{
summarys \code{bifactor} object
}
\description{
summarys for \code{bifactor} objects
}
\author{
Marcos Jimenez <marcosjnezhquez@gmail.com> and Vithor R. Franco <vithorfranco@gmail.com>
}
