/*
 * Author: Marcos Jiménez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 31/08/2023
 *
 */

// Criteria for factor extraction

class cfa_criterion {

public:

  virtual void F(arguments_cfa& x) = 0;

  virtual void G(arguments_cfa& x) = 0;

  virtual void dG(arguments_cfa& x) = 0;

  virtual void H(arguments_cfa& x) = 0;

  virtual void H2(arguments_cfa& x) = 0;

  virtual void outcomes(arguments_cfa& x) = 0;

};

/*
 * GLS / DWLS / ULS
 */

class gls: public cfa_criterion {

public:

  void F(arguments_cfa& x) {

    x.Rhat = x.lambda * x.phi * x.lambda.t() + x.psi;
    x.residuals = x.R - x.Rhat;
    x.f = 0.5*arma::accu(x.residuals % x.residuals % x.W);

  }

  void G(arguments_cfa& x) {

    x.lambda_phi = x.lambda * x.phi;
    x.W_residuals = x.W % x.residuals;
    x.glambda = -2* x.W_residuals * x.lambda_phi;
    x.W_residuals_lambda = x.W_residuals * x.lambda;
    x.gphi = -2*x.lambda.t() * x.W_residuals_lambda;
    x.gphi.diag() *= 0.5;
    x.gpsi = -2*x.W_residuals;
    x.gpsi.diag() *= 0.5;

    x.gradient = arma::join_cols(x.glambda.elem(x.lambda_indexes),
                                 x.gphi.elem(x.phi_indexes),
                                 x.gpsi.elem(x.psi_indexes));

  }

  void dG(arguments_cfa& x) {

    x.dlambda.elem(x.lambda_indexes) = x.dparameters.head(x.n_lambda);
    x.dphi.elem(x.phi_indexes) = x.dparameters.subvec(x.n_lambda, x.n_lambda + x.n_phi - 1);
    x.dphi = arma::symmatl(x.dphi);
    x.dpsi.elem(x.psi_indexes) = x.dparameters.tail(x.n_psi);
    x.dpsi = arma::symmatl(x.dpsi);

    // dglambda:
    arma::mat dg1 = -2*x.W_residuals * x.dlambda * x.phi;
    arma::mat W_dresiduals = -x.W % (x.dlambda * x.lambda_phi.t() + x.lambda_phi * x.dlambda.t());
    arma::mat dg2 = -2*W_dresiduals * x.lambda_phi;
    x.dglambda = dg1 + dg2;

    // dgphi:
    W_dresiduals = -x.W % (x.lambda * x.dphi * x.lambda.t());
    x.dgphi = -2*x.lambda.t() * W_dresiduals * x.lambda;
    x.dgphi.diag() *= 0.5;

    // dgpsi:
    x.dgpsi = 2*x.W % x.dpsi;
    x.dgpsi.diag() *= 0.5;

    x.dgradient = arma::join_cols(x.dglambda.elem(x.lambda_indexes),
                                  x.dgphi.elem(x.phi_indexes),
                                  x.dgpsi.elem(x.psi_indexes));

  }

  void H(arguments_cfa& x) {

    x.w = arma::vectorise(x.W);
    // Rcpp::Rcout << "hlambda" << std::endl;
    // Lambda
    x.dlambda_dRhat_W = gLRhat(x.lambda, x.phi);
    x.dlambda_dRhat_W.each_col() %= x.w;
    x.lambda_phit_kron_Ip = arma::kron(x.lambda_phi.t(), x.Ip);
    arma::mat g1 = 2*x.lambda_phit_kron_Ip * x.dlambda_dRhat_W;
    arma::mat g2 = -2*arma::kron(x.phi, x.W_residuals);
    arma::mat hlambda = g1 + g2;
    x.hlambda = hlambda; //(x.lambda_indexes, x.lambda_indexes);

    // Rcpp::Rcout << "hphi" << std::endl;
    // Phi
    arma::mat LL = 2*arma::kron(x.lambda, x.lambda).t();
    x.dphi_dRhat_W = gPRhat(x.lambda, x.phi, x.indexes_q);
    x.dphi_dRhat_W.each_col() %= x.w;
    arma::mat hphi_temp = LL * x.dphi_dRhat_W;
    hphi_temp.rows(x.indexes_diag_q) *= 0.5;
    x.hphi = hphi_temp; //(x.phi_indexes, x.phi_indexes);

    // Rcpp::Rcout << "hpsi" << std::endl;
    // Psi
    x.dpsi_dRhat_W = gURhat(x.psi);
    x.dpsi_dRhat_W.each_col() %= x.w;
    arma::mat W2 = 2*x.W;
    W2.diag() *= 0.5;
    arma::vec w2 = arma::vectorise(W2);
    arma::mat hpsi = arma::diagmat(w2);
    x.hpsi = hpsi; //(x.psi_indexes, x.psi_indexes);

    // Rcpp::Rcout << "dlambda_dphi" << std::endl;
    // Lambda & Phi
    g1 = 2*x.lambda_phit_kron_Ip * x.dphi_dRhat_W;
    arma::mat g21 = -2*arma::kron(x.Iq, x.W_residuals_lambda);
    arma::mat dtg21 = g21 * dxt(x.q, x.q);
    g2 = g21 + dtg21;
    g2.cols(x.indexes_diag_q) -= dtg21.cols(x.indexes_diag_q);
    arma::mat dlambda_dphi_temp = g1 + g2;
    x.dlambda_dphi = dlambda_dphi_temp; //(x.lambda_indexes, x.phi_indexes);

    // Rcpp::Rcout << "dlambda_dpsi" << std::endl;
    // Lambda & Psi
    arma::mat dlambda_dpsi_temp = 2*x.lambda_phit_kron_Ip * x.dpsi_dRhat_W;
    x.dlambda_dpsi = dlambda_dpsi_temp; //(x.lambda_indexes, x.psi_indexes);

    // Rcpp::Rcout << "dpsi_dphi" << std::endl;
    // Phi & Psi
    arma::mat dpsi_dphi_temp = x.dphi_dRhat_W;
    dpsi_dphi_temp.rows(x.indexes_diag_p) *= 0.5;
    dpsi_dphi_temp *= 2;
    x.dpsi_dphi = dpsi_dphi_temp; //(x.psi_indexes, x.phi_indexes);

    /*
     * Join all the derivatives such that
     * hLambda           dlambda_dphi   dlambda_dpsi
     * dlambda_dphi.t()  hphi           dpsi_dphi.t()
     * dambda_dpsi.t()   dpsi_dphi      hpsi
     */

    arma::mat col1 = arma::join_cols(x.hlambda(x.lambda_indexes, x.lambda_indexes),
                                     x.dlambda_dphi(x.lambda_indexes, x.phi_indexes).t(),
                                     x.dlambda_dpsi(x.lambda_indexes, x.psi_indexes).t());
    arma::mat col2 = arma::join_cols(x.dlambda_dphi(x.lambda_indexes, x.phi_indexes),
                                     x.hphi(x.phi_indexes, x.phi_indexes),
                                     x.dpsi_dphi(x.psi_indexes, x.phi_indexes));
    arma::mat col3 = arma::join_cols(x.dlambda_dpsi(x.lambda_indexes, x.psi_indexes),
                                     x.dpsi_dphi(x.psi_indexes, x.phi_indexes).t(),
                                     x.hpsi(x.psi_indexes, x.psi_indexes));
    x.hessian = arma::join_rows(col1, col2, col3);

  }

  void H2(arguments_cfa& x) {

    // Rcpp::Rcout << "dlambda_dS" << std::endl;
    arma::mat g1 = -2*x.lambda_phit_kron_Ip;
    g1.each_row() %= x.w.t();
    arma::mat g2 = g1 * dxt(x.p, x.p);
    arma::mat g = g1 + g2;
    g.cols(x.indexes_diag_p) *= 0.5;
    x.dlambda_dS = g; //(x.lambda_indexes, x.S_indexes);

    // Rcpp::Rcout << "dphi_dS" << std::endl;
    g1 = -2*arma::kron(x.lambda.t(), x.lambda.t());
    g1.each_row() %= x.w.t();
    g2 = g1 * dxt(x.p, x.p);
    g = g1 + g2;
    g.cols(x.indexes_diag_p) *= 0.5;
    g.rows(x.indexes_diag_q) *= 0.5;
    x.dphi_dS = g; //(x.phi_indexes, x.S_indexes);

    // Rcpp::Rcout << "dpsi_dS" << std::endl;
    g = -2*arma::diagmat(x.w);
    g.cols(x.indexes_diag_p) *= 0.5;
    x.dpsi_dS = g; //(x.psi_indexes, x.S_indexes);

    x.df2_dLPUdS = arma::join_cols(x.dlambda_dS(x.lambda_indexes, x.S_indexes),
                                   x.dphi_dS(x.phi_indexes, x.S_indexes),
                                   x.dpsi_dS(x.psi_indexes, x.S_indexes));

  }

  void outcomes(arguments_cfa& x) {

    x.uniquenesses = 1 - arma::diagvec(x.Rhat);// x.R.diag() - arma::diagvec(x.Rhat) FIX
    x.Rhat.diag() = x.R.diag();

  };

};

// Choose the cor criteria:

cfa_criterion* choose_cfa_criterion(std::string estimator) {

  cfa_criterion *criterion;

  if (estimator == "gls" | estimator == "uls" | estimator == "dwls") {

    criterion = new gls();

  } else {

    Rcpp::stop("Available estimators: uls");

  }

  return criterion;

}
