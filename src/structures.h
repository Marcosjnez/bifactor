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
  std::vector<int> qi;
  double w = 1, alpha = 10, f, q2;
  arma::vec k = {0}, gamma = {0}, epsilon = {0.01}, clf_epsilon = {0.01};
  double a = 30, b = 0.36, ss = 1, inprod = 1, ng = 1;
  bool convergence = false;
  int maxit = 1e04;
  double eps = 1e-05;
  std::string optim = "newtonTR";
  std::string normalization = "none";

  Rcpp::Nullable<arma::mat> nullable_Target, nullable_Weight,
  nullable_PhiTarget, nullable_PhiWeight;
  Rcpp::Nullable<arma::uvec> nullable_blocks, nullable_oblq_blocks,
  nullable_block_weights;
  Rcpp::Nullable<std::vector<arma::uvec>> nullable_blocks_list;
  Rcpp::Nullable<std::vector<arma::uvec>> nullable_between_blocks_list;
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
  arma::uvec oblq_indexes, loblq_indexes, orth_indexes, blocks_vector; // REMOVE blocks_vector?
  std::vector<arma::uvec> list_oblq_indexes, blocks_list;
  arma::vec block_weights;
  int n_blocks = 1, n_rotations = 1, i, n_loads;

  std::string between_blocks = "none";
  bool between = false;
  std::vector<arma::uvec> between_blocks_list;

  std::vector<std::string> rotations = {"oblimin"};
  std::string projection = "oblq";
  std::vector<arma::mat> Li, Phii, Li2, Ni, HLi2, LoLi2, IgCL2Ni, V,
  f1i, f2i, Hi, Ii, HL2i, I_gamma_Ci;
  std::vector<arma::vec> termi;

} args;

typedef struct arguments_efast{

  int p, q;
  std::string method = "minres", projection = "oblq";
  std::vector<std::string> rotation = {"oblimin"};
  Rcpp::Nullable<int> nullable_nobs = R_NilValue;
  Rcpp::Nullable<arma::mat> nullable_Target = R_NilValue, nullable_Weight = R_NilValue,
    nullable_PhiTarget = R_NilValue, nullable_PhiWeight = R_NilValue;
  Rcpp::Nullable<arma::uvec> nullable_blocks = R_NilValue;
  Rcpp::Nullable<std::vector<arma::uvec>> nullable_blocks_list = R_NilValue;
  Rcpp::Nullable<arma::vec> nullable_block_weights = R_NilValue;
  Rcpp::Nullable<arma::uvec> nullable_oblq_blocks = R_NilValue;
  std::string normalization = "none";
  std::string between_blocks = "none";
  double w = 1, alpha = 1;
  arma::vec k = {0}, gamma = {0}, epsilon = {0.01}, clf_epsilon = {0.1};
  double a = 30, b = 0.36;
  int random_starts = 10L, cores = 1L;
  Rcpp::Nullable<arma::vec> nullable_init = R_NilValue;
  Rcpp::Nullable<Rcpp::List> nullable_efa_control = R_NilValue,
    nullable_rot_control = R_NilValue;

} args_efast;

typedef struct arguments_efa{

  arma::mat R;
  arma::vec psi, sqrt_psi;
  int q, p;
  double f = 0;
  arma::mat lambda, phi, reduced_R, eigvec;
  arma::vec u, eigval, sc;

} args_efa;

typedef std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> NTR;
