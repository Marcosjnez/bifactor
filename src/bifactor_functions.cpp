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

#include <omp.h>
#include <RcppArmadillo.h>
#include <Rcpp/Benchmark/Timer.h>
#include "structures.h"
#include "auxiliary_manifolds.h"
#include "rotation_manifolds.h"
#include "auxiliary_criteria.h"
#include "rotation_criteria.h"
#include "auxiliary_rotation_optim.h"
#include "auxiliary_checks.h"
#include "efa_checks.h"
#include "rotation_checks.h"
#include "rotation_optim.h"
#include "rotate.h"
#include "efa_fit.h"
#include "efa.h"
#include "efast.h"
#include "bifactor.h"
#include "asymptotic_cov.h"
#include "method_derivatives.h"
#include "se.h"
#include "dimensionality.h"
#include "check_deriv.h"

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
              Rcpp::Nullable<int> nobs = R_NilValue,
              Rcpp::Nullable<Rcpp::List> first_efa = R_NilValue,
              Rcpp::Nullable<Rcpp::List> second_efa = R_NilValue);

// [[Rcpp::export]]
Rcpp::List rotate(arma::mat loadings,
                  Rcpp::CharacterVector rotation = Rcpp::CharacterVector::create("oblimin"),
                  std::string projection = "oblq",
                  arma::vec gamma = 0, arma::vec epsilon = Rcpp::NumericVector::create(0.01),
                  arma::vec k = 0, double w = 1, double alpha = 1,
                  double a = 30, double b = 0.36,
                  Rcpp::Nullable<arma::mat> Target = R_NilValue,
                  Rcpp::Nullable<arma::mat> Weight = R_NilValue,
                  Rcpp::Nullable<arma::mat> PhiTarget = R_NilValue,
                  Rcpp::Nullable<arma::mat> PhiWeight = R_NilValue,
                  Rcpp::Nullable<arma::uvec> blocks = R_NilValue,
                  Rcpp::Nullable<std::vector<arma::uvec>> blocks_list = R_NilValue,
                  Rcpp::Nullable<arma::vec> block_weights = R_NilValue,
                  Rcpp::Nullable<arma::uvec> oblq_blocks = R_NilValue,
                  std::string between_blocks = "none",
                  std::string normalization = "none",
                  Rcpp::Nullable<Rcpp::List> rot_control = R_NilValue,
                  int random_starts = 1, int cores = 1);

// [[Rcpp::export]]
Rcpp::List efast(arma::mat R, int nfactors, std::string method = "minres",
                 Rcpp::CharacterVector rotation = Rcpp::CharacterVector::create("oblimin"),
                 std::string projection = "oblq",
                 Rcpp::Nullable<int> nobs = R_NilValue,
                 Rcpp::Nullable<arma::mat> Target = R_NilValue,
                 Rcpp::Nullable<arma::mat> Weight = R_NilValue,
                 Rcpp::Nullable<arma::mat> PhiTarget = R_NilValue,
                 Rcpp::Nullable<arma::mat> PhiWeight = R_NilValue,
                 Rcpp::Nullable<arma::uvec> blocks = R_NilValue,
                 Rcpp::Nullable<std::vector<arma::uvec>> blocks_list = R_NilValue,
                 Rcpp::Nullable<arma::vec> block_weights = R_NilValue,
                 Rcpp::Nullable<arma::uvec> oblq_blocks = R_NilValue,
                 std::string normalization = "none", std::string between_blocks = "none",
                 arma::vec gamma = 0, arma::vec epsilon = Rcpp::NumericVector::create(0.01),
                 arma::vec k = 0, double w = 1, double alpha = 1,
                 double a = 30, double b = 0.36,
                 int random_starts = 1, int cores = 1,
                 Rcpp::Nullable<arma::vec> init = R_NilValue,
                 Rcpp::Nullable<Rcpp::List> efa_control = R_NilValue,
                 Rcpp::Nullable<Rcpp::List> rot_control = R_NilValue);

// [[Rcpp::export]]
arma::mat get_target(arma::mat loadings, Rcpp::Nullable<arma::mat> Phi, double cutoff = 0);

// [[Rcpp::export]]
Rcpp::List bifad(arma::mat R, int n_generals, int n_groups,
                 std::string projection = "orth",
                 Rcpp::Nullable<arma::uvec> oblq_blocks = R_NilValue,
                 double cutoff = 0,
                 std::string normalization = "none",
                 Rcpp::Nullable<int> nobs = R_NilValue,
                 Rcpp::Nullable<Rcpp::List> first_efa = R_NilValue,
                 Rcpp::Nullable<Rcpp::List> second_efa = R_NilValue,
                 Rcpp::Nullable<Rcpp::List> rot_control = R_NilValue,
                 int random_starts = 1, int cores = 1);

// [[Rcpp::export]]
Rcpp::List bifactor(arma::mat R, int n_generals, int n_groups,
                   std::string bifactor_method = "GSLiD",
                   std::string method = "minres",
                   std::string projection = "oblq",
                   Rcpp::Nullable<int> nobs = R_NilValue,
                   Rcpp::Nullable<arma::mat> PhiTarget = R_NilValue,
                   Rcpp::Nullable<arma::mat> PhiWeight = R_NilValue,
                   Rcpp::Nullable<arma::uvec> blocks = R_NilValue,
                   Rcpp::Nullable<std::vector<arma::uvec>> blocks_list = R_NilValue,
                   Rcpp::Nullable<arma::vec> block_weights = R_NilValue,
                   Rcpp::Nullable<arma::uvec> oblq_blocks = R_NilValue,
                   Rcpp::Nullable<arma::mat> init_Target = R_NilValue,
                   int maxit = 20, double cutoff = 0, std::string normalization = "none",
                   double w = 1, int random_starts = 1, int cores = 1,
                   Rcpp::Nullable<arma::vec> init = R_NilValue,
                   Rcpp::Nullable<Rcpp::List> efa_control = R_NilValue,
                   Rcpp::Nullable<Rcpp::List> rot_control = R_NilValue,
                   Rcpp::Nullable<Rcpp::List> first_efa = R_NilValue,
                   Rcpp::Nullable<Rcpp::List> second_efa = R_NilValue,
                   bool verbose = true);

// [[Rcpp::export]]
arma::mat asymp_cov(arma::mat R,
                    Rcpp::Nullable<arma::mat> X = R_NilValue,
                    double eta = 1, std::string type = "normal");

// [[Rcpp::export]]
Rcpp::List se(Rcpp::List fit = R_NilValue,
              Rcpp::Nullable<int> nobs = R_NilValue,
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

// [[Rcpp::export]]
Rcpp::List check_deriv(arma::mat L, arma::mat Phi,
                       arma::mat dL, arma::mat dP,
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
                       std::string between_blocks = "none",
                       arma::vec gamma = 0, arma::vec epsilon = Rcpp::NumericVector::create(0.01),
                       arma::vec k = 0, double w = 1, double alpha = 1,
                       double a = 30, double b = 0.36);
