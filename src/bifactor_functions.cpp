#define ARMA_NO_DEBUG
// #define ARMA_DONT_USE_OPENMP
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

#ifdef _OPENMP
  #include <omp.h>
#endif

#include <RcppArmadillo.h>
#include "dimensionality.h"

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
Rcpp::List rotate(arma::mat loadings,
                  Rcpp::CharacterVector rotation = Rcpp::CharacterVector::create("oblimin"),
                  std::string projection = "oblq",
                  double gamma = 0, arma::vec epsilon = Rcpp::NumericVector::create(0.01),
                  arma::vec k = 0, double w = 1, double alpha = 1,
                  Rcpp::Nullable<arma::mat> Target = R_NilValue,
                  Rcpp::Nullable<arma::mat> Weight = R_NilValue,
                  Rcpp::Nullable<arma::mat> PhiTarget = R_NilValue,
                  Rcpp::Nullable<arma::mat> PhiWeight = R_NilValue,
                  Rcpp::Nullable<arma::uvec> blocks = R_NilValue,
                  Rcpp::Nullable<std::vector<arma::uvec>> blocks_list = R_NilValue,
                  Rcpp::Nullable<arma::vec> block_weights = R_NilValue,
                  Rcpp::Nullable<arma::uvec> oblq_blocks = R_NilValue,
                  std::string penalization = "none",
                  Rcpp::Nullable<Rcpp::List> rot_control = R_NilValue,
                  int random_starts = 1, int cores = 1);

// [[Rcpp::export]]
Rcpp::List efast(arma::mat R, int n_factors, std::string method = "minres",
                 Rcpp::CharacterVector rotation = Rcpp::CharacterVector::create("oblimin"),
                 std::string projection = "oblq",
                 Rcpp::Nullable<arma::mat> Target = R_NilValue,
                 Rcpp::Nullable<arma::mat> Weight = R_NilValue,
                 Rcpp::Nullable<arma::mat> PhiTarget = R_NilValue,
                 Rcpp::Nullable<arma::mat> PhiWeight = R_NilValue,
                 Rcpp::Nullable<arma::uvec> blocks = R_NilValue,
                 Rcpp::Nullable<std::vector<arma::uvec>> blocks_list = R_NilValue,
                 Rcpp::Nullable<arma::vec> block_weights = R_NilValue,
                 Rcpp::Nullable<arma::uvec> oblq_blocks = R_NilValue,
                 bool normalize = false, std::string penalization = "none",
                 double gamma = 0, arma::vec epsilon = Rcpp::NumericVector::create(0.01),
                 arma::vec k = 0, double w = 1, double alpha = 1,
                 int random_starts = 1, int cores = 1,
                 Rcpp::Nullable<arma::vec> init = R_NilValue,
                 Rcpp::Nullable<Rcpp::List> efa_control = R_NilValue,
                 Rcpp::Nullable<Rcpp::List> rot_control = R_NilValue);

// [[Rcpp::export]]
arma::mat get_target(arma::mat loadings, Rcpp::Nullable<arma::mat> Phi, double cutoff = 0);

// [[Rcpp::export]]
Rcpp::List bifactor(arma::mat R, int n_generals, int n_groups,
                   std::string bifactor_method = "GSLiD",
                   std::string projection = "oblq",
                   Rcpp::Nullable<arma::mat> PhiTarget = R_NilValue,
                   Rcpp::Nullable<arma::mat> PhiWeight = R_NilValue,
                   Rcpp::Nullable<arma::uvec> blocks = R_NilValue,
                   Rcpp::Nullable<std::vector<arma::uvec>> blocks_list = R_NilValue,
                   Rcpp::Nullable<arma::vec> block_weights = R_NilValue,
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

// [[Rcpp::export]]
arma::mat asymp_cov(arma::mat R,
                    Rcpp::Nullable<arma::mat> X = R_NilValue,
                    double eta = 1, std::string type = "normal");

// [[Rcpp::export]]
Rcpp::List se(Rcpp::Nullable<Rcpp::List> fit = R_NilValue,
              Rcpp::Nullable<int> n = R_NilValue,
              Rcpp::Nullable<arma::mat> X = R_NilValue,
              std::string type = "normal", double eta = 1);

// [[Rcpp::export]]
Rcpp::List parallel(arma::mat X, int n_boot = 100, Rcpp::Nullable<arma::vec> quant = R_NilValue,
                    bool mean = false, bool replace = false,
                    Rcpp::Nullable<std::vector<std::string>> PA = R_NilValue,
                    bool hierarchical = false, Rcpp::Nullable<Rcpp::List> efa = R_NilValue,
                    int cores = 1);

// [[Rcpp::export]]
Rcpp::List cv_eigen(arma::mat X, int N = 100, bool hierarchical = false,
                    Rcpp::Nullable<Rcpp::List> efa = R_NilValue, int cores = 1);


