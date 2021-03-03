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
arma::mat random_orthogonal(int p, int q);

// [[Rcpp::export]]
Rcpp::List multiple_rotations(arma::mat loadings, std::string rotation, arma::mat Target, arma::mat Weight, arma::mat Phi_Target, arma::mat Phi_Weight,
                              double gamma = 0, double epsilon = 1e-02, double k = 0, double w = 1, int random_starts = 10,
                              int cores = 1, double eps = 1e-05, int max_iter = 1e4);

// [[Rcpp::export]]
Rcpp::List efast(arma::mat R, int n_factors, std::string method = "minres",
                 std::string rotation = "oblimin", Rcpp::Nullable<Rcpp::NumericVector> init = R_NilValue,
                 Rcpp::Nullable<Rcpp::NumericMatrix> Target = R_NilValue,
                 Rcpp::Nullable<Rcpp::NumericMatrix> Weight = R_NilValue,
                 Rcpp::Nullable<Rcpp::NumericMatrix> PhiTarget = R_NilValue,
                 Rcpp::Nullable<Rcpp::NumericMatrix> PhiWeight = R_NilValue,
                 bool normalize = false, double gamma = 0, double epsilon = 1e-02, double k = 0, double w = 1,
                 int random_starts = 1, int cores = 1,
                 int efa_max_iter = 1e4, double efa_factr = 1e7, int m = 5,
                 int rot_max_iter = 1e4, double rot_eps = 1e-05);

// [[Rcpp::export]]
arma::mat get_target(arma::mat L, arma::mat Phi);

// [[Rcpp::export]]
arma::mat get_target_with_cutoff(arma::mat L, double cutoff = 0.20);

// [[Rcpp::export]]
Rcpp::List bifactor(arma::mat R, int n_generals, int n_specifics, std::string method,
                    std::string rotation, Rcpp::Nullable<Rcpp::NumericVector> init = R_NilValue,
                    bool normalize = false, double gamma = 0, double epsilon = 1e-02,
                    double k = 0, double w = 1, std::string bifactor_method = "SL",
                    int SLiD_max_iter = 10, double cutoff = 0.20,
                    Rcpp::Nullable<Rcpp::NumericMatrix> LTarget = R_NilValue,
                    Rcpp::Nullable<Rcpp::NumericMatrix> PhiTarget = R_NilValue,
                    Rcpp::Nullable<Rcpp::NumericMatrix> PhiWeight = R_NilValue,
                    int random_starts = 1, int cores = 1,
                    int efa_max_iter = 1e4, double efa_factr = 1e7, int m = 5,
                    int rot_max_iter = 1e4, double rot_eps = 1e-05, bool verbose = true);
