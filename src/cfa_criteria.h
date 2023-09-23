/*
 * Author: Marcos Jiménez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 14/09/2023
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

class cfa_dwls: public cfa_criterion {

public:

  void F(arguments_cfa& x) {

    x.Rhat = x.lambda * x.phi * x.lambda.t() + x.psi;
    x.residuals = x.R - x.Rhat;
    x.f = 0.5*arma::accu(x.residuals % x.residuals % x.W);

  }

  void G(arguments_cfa& x) {

    x.gradient.set_size(x.parameters.n_elem); x.gradient.zeros();

    x.lambda_phi = x.lambda * x.phi;
    x.W_residuals = x.W % x.residuals;
    x.glambda = -2* x.W_residuals * x.lambda_phi;
    x.W_residuals_lambda = x.W_residuals * x.lambda;
    x.gphi = -2*x.lambda.t() * x.W_residuals_lambda;
    x.gphi.diag() *= 0.5;
    x.gpsi = -2*x.W_residuals;
    x.gpsi.diag() *= 0.5;
    // x.gphi = -x.lambda.t() * x.W_residuals_lambda;
    // x.gpsi = -x.W_residuals;

    x.gradient(x.lambda_indexes) += x.glambda.elem(x.target_indexes);
    x.gradient(x.phi_indexes) += x.gphi.elem(x.targetphi_indexes);
    x.gradient(x.psi_indexes) += x.gpsi.elem(x.targetpsi_indexes);

  }

  void dG(arguments_cfa& x) {

    x.dgradient.set_size(x.parameters.n_elem); x.dgradient.zeros();

    // dglambda:
    arma::mat dg1 = -2*x.W_residuals * x.dlambda * x.phi;
    arma::mat W_dresiduals = -x.W % (x.dlambda * x.lambda_phi.t() +
      x.lambda_phi * x.dlambda.t());
    arma::mat dg2 = -2*W_dresiduals * x.lambda_phi;
    x.dglambda = dg1 + dg2;

    // dgphi:
    W_dresiduals = -x.W % (x.lambda * x.dphi * x.lambda.t());
    x.dgphi = -2*x.lambda.t() * W_dresiduals * x.lambda;
    x.dgphi.diag() *= 0.5;
    // x.dgphi = -x.lambda.t() * W_dresiduals * x.lambda;

    // dgpsi:
    x.dgpsi = 2*x.W % x.dpsi;
    x.dgpsi.diag() *= 0.5;
    // x.dgpsi = x.W % x.dpsi;

    x.dgradient(x.lambda_indexes) += x.dglambda.elem(x.target_indexes);
    x.dgradient(x.phi_indexes) += x.dgphi.elem(x.targetphi_indexes);
    x.dgradient(x.psi_indexes) += x.dgpsi.elem(x.targetpsi_indexes);

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

    arma::mat col1 = arma::join_cols(x.hlambda, x.dlambda_dphi.t(), x.dlambda_dpsi.t());
    arma::mat col2 = arma::join_cols(x.dlambda_dphi, x.hphi, x.dpsi_dphi);
    arma::mat col3 = arma::join_cols(x.dlambda_dpsi, x.dpsi_dphi.t(), x.hpsi);
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

    x.df2_dLPUdS = arma::join_cols(x.dlambda_dS, x.dphi_dS, x.dpsi_dS);

  }

  void outcomes(arguments_cfa& x) {

    // x.uniquenesses = x.R.diag() - arma::diagvec(x.Rhat);
    // x.Rhat.diag() = x.R.diag();

  };

};

/*
 * GLS / DWLS / ULS
 */

class cfa_ml: public cfa_criterion {

public:

  void F(arguments_cfa& x) {

    x.Rhat = x.lambda * x.phi * x.lambda.t() + x.psi;
    if(!x.Rhat.is_sympd()) {
      arma::vec eigval;
      arma::mat eigvec;
      eig_sym(eigval, eigvec, x.Rhat);
      arma::vec d = arma::clamp(eigval, 0.1, eigval.max());
      x.Rhat = eigvec * arma::diagmat(d) * eigvec.t();
    }
    x.Rhat_inv = arma::inv_sympd(x.Rhat);
    x.f = arma::log_det_sympd(x.Rhat) - x.logdetR +
      arma::accu(x.R % x.Rhat_inv) - x.p;

  }

  void G(arguments_cfa& x) {

    x.gradient.set_size(x.parameters.n_elem); x.gradient.zeros();

    x.residuals = x.R - x.Rhat;
    x.Ri_res_Ri = 2*x.Rhat_inv * -x.residuals * x.Rhat_inv;
    x.lambda_phi = x.lambda * x.phi;
    x.glambda = x.Ri_res_Ri * x.lambda_phi;

    x.gphi = x.lambda.t() * x.Ri_res_Ri * x.lambda;
    x.gphi.diag() *= 0.5;

    arma::mat dlogdetRhatdU = 2*x.Rhat_inv;
    dlogdetRhatdU.diag() *= 0.5;
    arma::mat dRhat_invdU = 2*(-x.Rhat_inv * x.R * x.Rhat_inv);
    dRhat_invdU.diag() *= 0.5;
    x.gpsi = dRhat_invdU + dlogdetRhatdU;

    x.gradient(x.lambda_indexes) += x.glambda.elem(x.target_indexes);
    x.gradient(x.phi_indexes) += x.gphi.elem(x.targetphi_indexes);
    x.gradient(x.psi_indexes) += x.gpsi.elem(x.targetpsi_indexes);

  }

  void dG(arguments_cfa& x) {

    x.dgradient.set_size(x.parameters.n_elem); x.dgradient.zeros();

    // dglambda:
    arma::mat dRhat = x.dlambda * x.lambda_phi.t() + x.lambda_phi * x.dlambda.t();
    arma::mat dresiduals = -dRhat;
    arma::mat dRhat_inv = -x.Rhat_inv * -dresiduals * x.Rhat_inv;
    arma::mat dRi_res_Ri = 2*(dRhat_inv * -x.residuals * x.Rhat_inv -
      x.Rhat_inv * -x.residuals * dRhat_inv + x.Rhat_inv * -dresiduals * x.Rhat_inv);
    x.dglambda = x.lambda.t() * dRi_res_Ri * x.lambda;

    // dgphi:
    dRhat = x.lambda * x.phi * x.lambda;
    dresiduals = -dRhat;
    dRhat_inv = -x.Rhat_inv * -dRhat * x.Rhat_inv;
    dRi_res_Ri = 2*(dRhat_inv * -x.residuals * x.Rhat_inv + x.Rhat_inv * -x.residuals * dRhat_inv +
      x.Rhat_inv * -dresiduals * x.Rhat_inv);
    x.dgphi = x.lambda.t() * dRi_res_Ri * x.lambda;
    x.dgphi.diag() *= 0.5;

    // dgpsi:
    dRhat = x.dpsi;
    dRhat_inv = -x.Rhat_inv * dRhat * x.Rhat_inv;
    arma::mat ddlogdetRhat = 2*dRhat_inv;
    ddlogdetRhat.diag() *= 0.5;
    arma::mat ddRhat_inv = 2*(-dRhat_inv * x.R * x.Rhat_inv + -x.Rhat_inv * x.R * dRhat_inv);
    x.dgpsi = ddlogdetRhat + ddRhat_inv;
    x.dgpsi.diag() *= 0.5;

    x.dgradient(x.lambda_indexes) += x.dglambda.elem(x.target_indexes);
    x.dgradient(x.phi_indexes) += x.dgphi.elem(x.targetphi_indexes);
    x.dgradient(x.psi_indexes) += x.dgpsi.elem(x.targetpsi_indexes);

  }

  void H(arguments_cfa& x) {

    // Rcpp::Rcout << "hlambda" << std::endl;
    // Lambda
    arma::mat h1 = arma::kron(x.phi, x.Ri_res_Ri);

    arma::mat dRhat_dL = gLRhat(x.lambda, x.phi);
    arma::mat dRi_res_Ri_dRhat = 2*arma::kron(x.Rhat_inv, x.Rhat_inv) -
      arma::kron(x.Ri_res_Ri, x.Rhat_inv) - arma::kron(x.Rhat_inv, x.Ri_res_Ri);
    arma::mat dRi_res_Ri_dL = dRi_res_Ri_dRhat * dRhat_dL;
    arma::mat h2 = arma::kron(x.lambda_phi.t(), x.Ip) * dRi_res_Ri_dL;

    x.hlambda = h1 + h2;

    // Rcpp::Rcout << "hphi" << std::endl;
    // Phi
    arma::mat dRhat_dP = gPRhat(x.lambda, x.phi, x.indexes_diag_q);
    arma::mat dRi_res_Ri_dP = dRi_res_Ri_dRhat * dRhat_dP;

    x.hphi = arma::kron(x.lambda.t(), x.lambda.t()) * dRi_res_Ri_dP;
    x.hphi.rows(x.indexes_diag_q) *= 0.5;

    // Rcpp::Rcout << "hpsi" << std::endl;
    // Psi
    arma::mat dRhat_dU = gURhat(x.psi);
    arma::mat dRi_res_Ri_dU = dRi_res_Ri_dRhat * dRhat_dU;
    x.hpsi = dRi_res_Ri_dU;
    x.hpsi.rows(x.indexes_diag_p) *= 0.5;

    // Rcpp::Rcout << "dlambda_dphi" << std::endl;
    // Lambda & Phi
    arma::mat h21 = arma::kron(x.Iq, x.Ri_res_Ri * x.lambda);
    h2 = h21;
    h2 += h21 * dxt(x.q, x.q);
    h2.cols(x.indexes_diag_q) = h21.cols(x.indexes_diag_q);
    x.dlambda_dphi = h1 + h2; //(x.lambda_indexes, x.phi_indexes);

    // Rcpp::Rcout << "dlambda_dpsi" << std::endl;
    // Lambda & Psi
    x.dlambda_dpsi = arma::kron(x.lambda_phi.t(), x.Ip) * dRi_res_Ri_dU;

    // Rcpp::Rcout << "dpsi_dphi" << std::endl;
    // Phi & Psi
    x.dpsi_dphi = dRi_res_Ri_dP;
    x.dpsi_dphi *= 0.5;

    /*
     * Join all the derivatives such that
     * hLambda           dlambda_dphi   dlambda_dpsi
     * dlambda_dphi.t()  hphi           dpsi_dphi.t()
     * dambda_dpsi.t()   dpsi_dphi      hpsi
     */

    arma::mat col1 = arma::join_cols(x.hlambda, x.dlambda_dphi.t(), x.dlambda_dpsi.t());
    arma::mat col2 = arma::join_cols(x.dlambda_dphi, x.hphi, x.dpsi_dphi);
    arma::mat col3 = arma::join_cols(x.dlambda_dpsi, x.dpsi_dphi.t(), x.hpsi);
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

    x.df2_dLPUdS = arma::join_cols(x.dlambda_dS, x.dphi_dS, x.dpsi_dS);

  }

  void outcomes(arguments_cfa& x) {

    // x.uniquenesses = x.R.diag() - arma::diagvec(x.Rhat);
    // x.Rhat.diag() = x.R.diag();

  };

};

/*
 * ULS
 */

class efa_uls: public cfa_criterion {

public:

  void F(arguments_cfa& x) {

    // x.psi2 = 0.5*(x.lower + x.upper) + 0.5*abs(x.upper - x.lower) % sin(x.psi);
    x.reduced_R = x.R - arma::diagmat(x.psi2);
    // x.reduced_R = x.R - arma::diagmat(x.psi);
    eig_sym(x.eigval, x.eigvec, x.reduced_R);
    arma::vec e = x.eigval(arma::span(0, x.p - x.q - 1));

    x.f = 0.5*arma::accu(e % e);

  }

  void G(arguments_cfa& x) {

    arma::vec e_values = x.eigval(arma::span(0, x.p - x.q - 1));
    arma::mat e_vectors = x.eigvec(arma::span::all, arma::span(0, x.p - x.q - 1));
    x.g_psi2 = -arma::diagvec(e_vectors * arma::diagmat(e_values) * e_vectors.t());

  }

  void dG(arguments_cfa& x) {
    Rcpp::stop("uls estimator not available with this optimization algorithm");
  }

  void H(arguments_cfa& x) {}

  void H2(arguments_cfa& x) {}

  void outcomes(arguments_cfa& x) {

    arma::vec eigval;
    arma::mat eigvec;
    eig_sym(eigval, eigvec, x.reduced_R);

    arma::vec eigval2 = reverse(eigval);
    arma::mat eigvec2 = reverse(eigvec, 1);

    arma::mat A = eigvec2(arma::span::all, arma::span(0, x.q-1));
    arma::vec eigenvalues = eigval2(arma::span(0, x.q-1));
    for(int i=0; i < x.q; ++i) {
      if(eigenvalues(i) < 0) eigenvalues(i) = 0;
    }
    arma::mat D = arma::diagmat(sqrt(eigenvalues));

    x.lambda = A * D;
    x.Rhat = x.lambda * x.lambda.t();
    x.uniquenesses = 1 - arma::diagvec(x.Rhat);// x.R.diag() - arma::diagvec(x.Rhat) FIX
    x.Rhat.diag() = x.R.diag();

  };

};

/*
 * ml
 */

class efa_ml: public cfa_criterion {

public:

  void F(arguments_cfa& x) {

    // x.psi2 = 0.5*(x.lower + x.upper) + 0.5*abs(x.upper - x.lower) % sin(x.psi);
    x.sqrt_psi = sqrt(x.psi2);
    // x.sqrt_psi = sqrt(x.psi);
    arma::mat sc = arma::diagmat(1/x.sqrt_psi);
    x.reduced_R = sc * x.R * sc;
    eig_sym(x.eigval, x.eigvec, x.reduced_R);
    arma::vec e = x.eigval(arma::span(0, x.p - x.q - 1));

    // double objective = -arma::accu(log(e) + 1/e - 1);
    x.f = -(arma::accu(log(e) - e) + x.p - x.q);

  }

  void G(arguments_cfa& x) {

    arma::mat A = x.eigvec(arma::span::all, arma::span(x.p-x.q, x.p-1));
    arma::vec eigenvalues = x.eigval(arma::span(x.p-x.q, x.p-1));
    // x.g = ((A % A) * (eigenvalues - 1) + 1 - arma::diagvec(x.R)/x.psi)/x.psi;
    x.g_psi2 = ((A % A) * (eigenvalues - 1) + 1 - arma::diagvec(x.R)/x.psi2)/x.psi2;

  }

  void dG(arguments_cfa& x) {
    Rcpp::stop("ml estimator not available with this optimization algorithm");
  }

  void H(arguments_cfa& x) {
  }

  void H2(arguments_cfa& x) {
  }

  void outcomes(arguments_cfa& x) {

    arma::vec eigval;
    arma::mat eigvec;
    eig_sym(eigval, eigvec, x.reduced_R);

    arma::vec eigval2 = reverse(eigval);
    arma::mat eigvec2 = reverse(eigvec, 1);

    arma::mat A = eigvec2(arma::span::all, arma::span(0, x.q-1));
    arma::vec eigenvalues = eigval2(arma::span(0, x.q-1)) - 1;
    for(int i=0; i < x.q; ++i) {
      if(eigenvalues[i] < 0) eigenvalues[i] = 0;
    }
    arma::mat D = diagmat(sqrt(eigenvalues));
    arma::mat w = A * D;

    x.lambda = diagmat(x.sqrt_psi) * w;
    x.Rhat = x.lambda * x.lambda.t();
    x.uniquenesses = 1 - diagvec(x.Rhat);// x.R.diag() - arma::diagvec(x.Rhat) FIX
    x.Rhat.diag() = x.R.diag();

  };

};

/*
 * DWLS
 */

class efa_dwls: public cfa_criterion {

public:

  void F(arguments_cfa& x) {

    // W is a matrix with the inverse variance of the polychoric correlations
    // Only the variance, not the covariances, are considered in W
    x.Rhat = x.lambda * x.lambda.t();// + arma::diagmat(x.uniquenesses);
    // x.Rhat.diag().ones();
    x.residuals = x.R - x.Rhat;
    x.f = 0.5*arma::accu(x.residuals % x.residuals % x.W);

  }

  void G(arguments_cfa& x) {

    x.gL = -2*(x.residuals % x.W) * x.lambda; // * x.Phi;
    // arma::mat DW_res = x.residuals % x.DW;
    // x.gU = -arma::diagvec(DW_res);

  }

  void dG(arguments_cfa& x) {
    Rcpp::stop("dwls estimator not available with this optimization algorithm");
  }

  void H(arguments_cfa& x) {}

  void H2(arguments_cfa& x) {}

  void outcomes(arguments_cfa& x) {

    x.Rhat = x.lambda * x.lambda.t();
    x.uniquenesses = 1 - arma::diagvec(x.Rhat); // x.R.diag() - arma::diagvec(x.Rhat) FIX
    x.Rhat.diag() = x.R.diag();

  };

};

// Choose the cor criteria:

cfa_criterion* choose_cfa_criterion(std::string estimator) {

  cfa_criterion *criterion;

  if(estimator == "uls" | estimator == "dwls") {

    criterion = new cfa_dwls();

  } else if(estimator == "gls") {

    Rcpp::stop("estimator gls not available yet");

  } else if (estimator == "efa_uls") {

    criterion = new efa_uls();

  } else if (estimator == "ml") {

    criterion = new cfa_ml();

  } else if(estimator == "efa_ml") {

    criterion = new efa_ml();

  } else if(estimator == "efa_dwls") {

    criterion = new efa_dwls();

  } else if(estimator == "minrank") {

    Rcpp::stop("The 'minrank' estimator is not available yet");

  } else {

    Rcpp::stop("Available estimators: uls, ml, dwls, and gls");

  }

  return criterion;

}

class cfa_criterion2 {

public:

  virtual void F(arguments_optim& x, std::vector<arguments_cfa>& structs) = 0;

  virtual void G(arguments_optim& x, std::vector<arguments_cfa>& structs) = 0;

  virtual void dG(arguments_optim& x, std::vector<arguments_cfa>& structs) = 0;

  virtual void H(arguments_optim& x, std::vector<arguments_cfa>& structs) = 0;

  virtual void H2(arguments_optim& x, std::vector<arguments_cfa>& structs) = 0;

  virtual void outcomes(arguments_optim& x, std::vector<arguments_cfa>& structs) = 0;

};

class ultimate_criterion: public cfa_criterion2 {

public:

  void F(arguments_optim& x, std::vector<arguments_cfa>& structs) {

    cfa_criterion* criterion;
    x.f = 0;

    for(int i=0; i < x.nblocks; ++i) {

      criterion = choose_cfa_criterion(structs[i].estimator);
      criterion->F(structs[i]);
      x.f += structs[i].f;

    }

  }

  void G(arguments_optim& x, std::vector<arguments_cfa>& structs) {

    cfa_criterion* criterion;
    x.gradient.set_size(x.parameters.n_elem); x.gradient.zeros();

    for(int i=0; i < x.nblocks; ++i) {

      criterion = choose_cfa_criterion(structs[i].estimator);
      criterion->G(structs[i]);
      x.gradient += structs[i].gradient;

    }

  }

  void dG(arguments_optim& x, std::vector<arguments_cfa>& structs) {

    cfa_criterion* criterion;
    x.dgradient.set_size(x.parameters.n_elem); x.dgradient.zeros();

    for(int i=0; i < x.nblocks; ++i) {

      criterion = choose_cfa_criterion(structs[i].estimator);
      criterion->dG(structs[i]);
      x.dgradient += structs[i].dgradient;

    }

  }

  void H(arguments_optim& x, std::vector<arguments_cfa>& structs) {

  }

  void H2(arguments_optim& x, std::vector<arguments_cfa>& structs) {

  }

  void outcomes(arguments_optim& x, std::vector<arguments_cfa>& structs) {

    x.lambda = resizeList(x.lambda, x.nblocks), x.phi = resizeList(x.phi, x.nblocks),
      x.psi = resizeList(x.psi, x.nblocks), x.R = resizeList(x.R, x.nblocks),
      x.Rhat = resizeList(x.Rhat, x.nblocks), x.residuals = resizeList(x.residuals, x.nblocks);

    x.fs.resize(x.nblocks), x.nobs.resize(x.nblocks), x.p.resize(x.nblocks),
    x.q.resize(x.nblocks);

    x.cor = resizeChar(x.cor, x.nblocks), x.estimator = resizeChar(x.estimator, x.nblocks),
      x.projection = resizeChar(x.projection, x.nblocks);

    for(int i=0; i < x.nblocks; ++i) {

      x.lambda(i) = structs[i].lambda;
      x.phi(i) = structs[i].phi;
      x.psi(i) = structs[i].psi;
      x.Rhat(i) = structs[i].Rhat;
      x.residuals(i) = structs[i].residuals;
      x.fs[i] = structs[i].f;
      x.R(i) = structs[i].R;
      x.df += structs[i].df;
      x.df_null += structs[i].df_null;
      x.f_null += structs[i].f_null;
      x.cor(i) = structs[i].cor;
      x.estimator(i) = structs[i].estimator;
      x.projection(i) = structs[i].projection;
      x.p[i] = structs[i].p;
      x.q[i] = structs[i].q;
      x.nobs[i] = structs[i].nobs;

    }

    x.df += x.parameters.size() * (x.nblocks-1L);

  }

};
