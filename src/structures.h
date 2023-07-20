/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 18/03/2022
 *
 */

typedef struct arguments_rotate{

  /*
   * Structure of arguments for rotation
   */

  int p, q, iteration = 0;
  std::vector<int> pi, qi;
  double w = 1, f, q2;
  arma::vec k = {0}, gamma = {0}, epsilon = {0.01}, clf_epsilon = {0.01};
  double ss = 1, inprod = 1, ng = 1;
  bool convergence = false;
  int maxit = 1e04;
  double eps = 1e-05;
  std::string optim = "newtonTR";
  std::string normalization = "none";

  Rcpp::Nullable<arma::mat> nullable_Target, nullable_Weight,
  nullable_PhiTarget, nullable_PhiWeight;
  Rcpp::Nullable<arma::uvec> nullable_oblq_factors, nullable_block_weights;
  Rcpp::Nullable<std::vector<std::vector<arma::uvec>>> nullable_blocks; // NEW
  Rcpp::Nullable<Rcpp::List> nullable_rot_control;

  arma::mat lambda, T, L, Phi, Inv_T, dL, dP, Inv_T_dt, dT, g,
  gL, gP, dg, dgL, dgP, hL, hP, d_constr, d_constr_temp, rg, A,
  dH, f1, f2, L2, L2N, LoL2, ML2, IgCL2N, I_gamma_C, N, M, H, HL2, h, L2h,
  Target, Weight, Phi_Target, Phi_Weight, Weight2, Phi_Weight2, S,
  Ls, Lg, L2s, L2g, exp_aL2g, g_exp_aL2g, gL2g, gL2s, C, logC, logCN,
  gC1, gC, glogC, glogCN, gexplogCN, exp_lCN, gL1, gL2, I, I1, I2, Ng,
  dxtL, dxt_L2s, dmudL, dc2dL, dmudP, dc2dP, LtLxI, dxtP, expmu, expmmu, dir;

  arma::vec Lvec;
  arma::uvec lower, larger;
  std::vector<arma::uvec> loweri, largeri;

  arma::rowvec var, varq;
  std::vector<arma::rowvec> vari, varqi;
  arma::colvec mu, c2, dc2dmu;
  std::vector<arma::colvec> mui, c2i, expmui, expmmui, dc2dmui;
  arma::mat dvarqdL;
  std::vector<arma::mat> dvarqdLi, dmudLi, dc2dLi, dmudPi, dc2dPi, LtLxIi,
  dxtLi, dxtPi;
  double prodvar, prodvarq, prodc2;
  std::vector<double> prodvari, prodvarqi, prodc2i;

  arma::vec term;
  arma::uvec oblq_indexes, loblq_indexes, orth_indexes;
  std::vector<arma::uvec> list_oblq_indexes;
  std::vector<std::vector<arma::uvec>> blocks;
  std::vector<arma::uvec> rows_list, cols_list;
  arma::vec block_weights;
  int n_blocks = 1, n_rotations = 1, i, n_loads;

  std::vector<std::string> rotations = {"oblimin"};
  std::string projection = "oblq";
  std::vector<arma::mat> Li, Phii, Li2, Mi, Ni, HLi2, LoLi2, IgCL2Ni, V,
  f1i, f2i, Hi, Ii, HL2i, I_gamma_Ci;
  std::vector<arma::vec> termi;

  Rcpp::Nullable<arma::mat> nullable_indexes1, nullable_indexes2;
  arma::mat indexes1, indexes2;

} args;

typedef struct arguments_efast{

  int p, q;
  std::string cor = "pearson", estimator = "uls", projection = "oblq";
  std::vector<std::string> rotation = {"oblimin"};
  Rcpp::Nullable<int> nullable_nobs = R_NilValue;
  Rcpp::Nullable<arma::mat> nullable_Target = R_NilValue, nullable_Weight = R_NilValue,
    nullable_PhiTarget = R_NilValue, nullable_PhiWeight = R_NilValue;
  Rcpp::Nullable<std::vector<std::vector<arma::uvec>>> nullable_blocks = R_NilValue;
  Rcpp::Nullable<arma::vec> nullable_block_weights = R_NilValue;
  Rcpp::Nullable<arma::uvec> nullable_oblq_factors = R_NilValue;
  std::string normalization = "none";
  double w = 1;
  arma::vec k = {0}, gamma = {0}, epsilon = {0.01}, clf_epsilon = {0.1};
  int random_starts = 10L, cores = 1L;
  Rcpp::Nullable<arma::vec> nullable_init = R_NilValue;
  Rcpp::Nullable<Rcpp::List> nullable_efa_control = R_NilValue,
    nullable_rot_control = R_NilValue;

} args_efast;

typedef struct arguments_efa{

  arma::mat init;
  arma::mat R, loadings, Rhat, residuals, Phi, Inv_W, X, gL, gU, g, smoothed;
  int q, p;
  double f = 0;
  arma::mat lambda, phi, reduced_R, eigvec;
  arma::vec u, uniquenesses, eigval, psi, sqrt_psi, psi2, g_psi2;
  std::string estimator = "uls";
  double efa_factr = 1e07;
  std::string optim = "gradient";
  arma::vec lower = {0.005}, upper = {0.995};
  arma::mat rg, dir, dpsi, dH;
  int iteration, iterations = 0L, maxit = 1000L;
  bool convergence = false, heywood = false;
  std::string manifold = "identity";
  int random_starts = 1L, cores = 1L;
  double ss = 1, inprod = 1, ng = 1, eps = 1e-05,
    c1 = 10e-04, c2 = 0.5, rho = 0.5;
  int M = 5L, armijo_maxit = 10L;
  std::string search = "back";
  std::string normalization = "none";
  int lambda_parameters;
  arma::uvec lower_tri_ind;
  Rcpp::Nullable<Rcpp::List> nullable_efa_control, nullable_init;

} args_efa;

typedef struct arguments_cor{

  int nobs, q, iteration = 0L, maxit = 1e04;
  double f = 0.00, eps = 1e-05, ng = 1, ss = 1, inprod = 1, n_pairs;
  bool convergence = false;
  std::vector<size_t> s;
  std::vector<std::vector<double>> taus, mvphi;
  std::vector<std::vector<std::vector<int>>> n;
  arma::mat T, cor, dT, dir;
  arma::mat g, rg;
  arma::mat dg, dH;
  arma::mat dcor, gcor, dgcor;

} args_cor;

typedef std::tuple<arma::mat, arma::vec, arma::mat, double, int, bool> efa_NTR;
typedef std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> NTR;
typedef std::tuple<arma::mat, arma::mat, double, int, bool> cor_NTR;
