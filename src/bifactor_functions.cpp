#define ARMA_DONT_USE_OPENMP
#define ARMA_NO_DEBUG
// #define ARMA_DONT_OPTIMISE_BAND
// #define ARMA_OPENMP_THREADS 10
// #define ARMA_DONT_OPTIMISE_SYMPD
// #define ARMA_DONT_USE_SUPERLU
// #define ARMA_DONT_USE_BLAS
// #define ARMA_OPENMP_THRESHOLD 240
// #define ARMA_64BIT_WORD
// #define ARMA_MAT_PREALLOC 4

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

#include <RcppArmadillo.h>
#include <omp.h>
#include "fit_indices.h"

// [[Rcpp::export]]
arma::mat random_orth(int p, int q);

// [[Rcpp::export]]
arma::mat random_oblq(int p, int q);

// [[Rcpp::export]]
arma::mat random_poblq(int p, int q, arma::uvec oblq_blocks);

// [[Rcpp::export]]
arma::mat retr_orth(arma::mat X);

// [[Rcpp::export]]
arma::mat retr_oblq(arma::mat X);

// [[Rcpp::export]]
arma::mat retr_poblq(arma::mat X, arma::uvec oblq_blocks);

// [[Rcpp::export]]
Rcpp::List sl(arma::mat R, int n_generals, int n_groups,
              Rcpp::Nullable<Rcpp::List> first_efa = R_NilValue,
              Rcpp::Nullable<Rcpp::List> second_efa = R_NilValue);

// [[Rcpp::export]]
Rcpp::List rotate(arma::mat loadings, std::string rotation = "oblimin",
                  std::string projection = "oblq",
                  Rcpp::Nullable<arma::mat> Target = R_NilValue,
                  Rcpp::Nullable<arma::mat> Weight = R_NilValue,
                  Rcpp::Nullable<arma::mat> PhiTarget = R_NilValue,
                  Rcpp::Nullable<arma::mat> PhiWeight = R_NilValue,
                  Rcpp::Nullable<arma::uvec> oblq_blocks = R_NilValue,
                  double gamma = 0, double epsilon = 0.01, double k = 0,
                  double w = 1, int random_starts = 1, int cores = 1,
                  Rcpp::Nullable<Rcpp::List> rot_control = R_NilValue);

// [[Rcpp::export]]
Rcpp::List efast(arma::mat R, int n_factors, std::string method = "minres",
                 std::string rotation = "oblimin", std::string projection = "oblq",
                 Rcpp::Nullable<arma::mat> Target = R_NilValue,
                 Rcpp::Nullable<arma::mat> Weight = R_NilValue,
                 Rcpp::Nullable<arma::mat> PhiTarget = R_NilValue,
                 Rcpp::Nullable<arma::mat> PhiWeight = R_NilValue,
                 Rcpp::Nullable<arma::uvec> oblq_blocks = R_NilValue,
                 bool normalize = false, double gamma = 0, double epsilon = 0.01,
                 double k = 0, double w = 1,
                 int random_starts = 1, int cores = 1,
                 Rcpp::Nullable<arma::vec> init = R_NilValue,
                 Rcpp::Nullable<Rcpp::List> efa_control = R_NilValue,
                 Rcpp::Nullable<Rcpp::List> rot_control = R_NilValue);

// [[Rcpp::export]]
arma::mat get_target(arma::mat loadings, Rcpp::Nullable<arma::mat> Phi, double cutoff = 0);

// [[Rcpp::export]]
Rcpp::List twoTier(arma::mat R, int n_generals, int n_groups,
                   std::string twoTier_method = "GSLiD",
                   std::string projection = "oblq",
                   Rcpp::Nullable<arma::mat> PhiTarget = R_NilValue,
                   Rcpp::Nullable<arma::mat> PhiWeight = R_NilValue,
                   Rcpp::Nullable<arma::uvec> oblq_blocks = R_NilValue,
                   Rcpp::Nullable<arma::mat> init_Target = R_NilValue,
                   std::string method = "minres", int maxit = 20, double cutoff = 0,
                   double w = 1, int random_starts = 1, int cores = 1,
                   Rcpp::Nullable<arma::vec> init = R_NilValue,
                   Rcpp::Nullable<Rcpp::List> efa_control = R_NilValue,
                   Rcpp::Nullable<Rcpp::List> rot_control = R_NilValue,
                   Rcpp::Nullable<Rcpp::List> SL_first_efa = R_NilValue,
                   Rcpp::Nullable<Rcpp::List> SL_second_efa = R_NilValue,
                   bool verbose = true);
