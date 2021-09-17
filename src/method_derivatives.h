#include "asymptotic_cov.h"

/*
 * Derivatives for minres:
 * d2f/(dtheta dtheta') and d2f/(dtheta ds)
 */

/*
 * Hessian matrix: d2f/(dtheta dtheta')
 */

arma::mat gLRhat(arma::mat Lambda, arma::mat Phi) {

  int p = Lambda.n_rows;
  arma::mat I(p, p, arma::fill::eye);
  arma::mat LP = Lambda * Phi;
  arma::mat g1 = arma::kron(LP, I);
  arma::mat g21 = arma::kron(I, LP);
  arma::mat g2 = g21 * dxt(Lambda);
  arma::mat g = g1 + g2;

  return g;

}

arma::mat gPRhat(arma::mat Lambda, arma::mat Phi) {

  arma::uvec indexes = trimatl_ind(arma::size(Phi), -1);
  arma::mat g1 = arma::kron(Lambda, Lambda);
  arma::mat g2 = g1 * dxt(Phi);
  arma::mat g_temp = g1 + g2;
  arma::mat g = g_temp.cols(indexes);

  return g;
}

arma::mat hessian_Lambda(arma::mat S, arma::mat Lambda,
                         arma::mat Phi, arma::vec psi) {

  int p = Lambda.n_rows;
  int q = Phi.n_rows;
  int pq = p*q;

  arma::mat Rhat = Lambda * Phi * Lambda.t() + arma::diagmat(psi);
  arma::mat residuals = S - Rhat;
  arma::mat I(p, p, arma::fill::eye);
  arma::mat h1 = 2*arma::kron((Lambda * Phi).t(), I) * gLRhat(Lambda, Phi);
  arma::mat h2 = -2*arma::kron(Phi, residuals);

  arma::mat hessian_Lambda = h1 + h2; // second derivatives for Lambda

  return hessian_Lambda;

}

arma::mat hessian_Phi(arma::mat Lambda, arma::mat Phi) {

  arma::uvec indexes = trimatl_ind(arma::size(Phi), -1);
  arma::mat LxL = 2*arma::kron(Lambda, Lambda);
  arma::mat LxLt = LxL.t();
  arma::mat hessian_Phi = LxLt.rows(indexes) * gPRhat(Lambda, Phi);

  return hessian_Phi;

}

arma::mat hessian_psi(arma::vec psi) {

  int p = psi.size();
  arma::mat I(p, p, arma::fill::eye);

  return I;

}

arma::mat hessian_LambdaPhi(arma::mat S, arma::mat Lambda,
                            arma::mat Phi, arma::vec psi) {

  int p = Lambda.n_rows;
  int q = Phi.n_rows;
  int q_cor = q*(q-1)/2;

  arma::mat LambdaPhi = Lambda * Phi;
  arma::mat Rhat = LambdaPhi * Lambda.t() + arma::diagmat(psi);
  arma::mat residuals = S - Rhat;
  arma::mat I1(p, p, arma::fill::eye);
  arma::mat I2(q, q, arma::fill::eye);

  arma::uvec indexes = trimatl_ind(arma::size(Phi), -1);
  arma::mat h1 = 2*arma::kron(LambdaPhi.t(), I1) * gPRhat(Lambda, Phi);
  arma::mat h21 = -2*arma::kron(I2, residuals * Lambda);
  arma::mat h22 = h21 + h21 * dxt(Phi);
  arma::mat h2 = h22.cols(indexes);
  arma::mat h = h1 + h2;

  return h;

}

arma::mat hessian_Lambdapsi(arma::mat Lambda, arma::mat Phi) {

  int p = Lambda.n_rows;
  int q = Phi.n_rows;

  arma::mat LP = 2*Lambda * Phi;
  arma::mat h_Lambdapsi = arma::diagmat(LP.col(0));

  for(int i=1; i < q; ++i) {

    arma::mat aug = arma::diagmat(LP.col(i));
    h_Lambdapsi = arma::join_cols(h_Lambdapsi, aug);

  }

  return h_Lambdapsi;

}

arma::mat hessian_Phipsi(arma::mat Lambda, arma::mat Phi) {

  int p = Lambda.n_rows;

  arma::mat gPR = gPRhat(Lambda, Phi);
  arma::uvec indexes(p);
  for(int i=0; i < p; ++i) indexes[i] = i*p + i;
  arma::mat h_Phipsi = gPR.rows(indexes);

  return h_Phipsi.t();

}

arma::mat hessian_minres(arma::mat S, arma::mat Lambda, arma::mat Phi,
                         std::string projection) {

  /*
   * Compute all the functions above and create the hessian matrix
   */

  /*
   * Second derivatives for Lambda
   */

  int p = Lambda.n_rows;
  int q = Phi.n_rows;
  int pq = p*q;

  arma::mat LambdaPhi = Lambda * Phi;
  arma::mat Rhat = Lambda * Phi * Lambda.t();
  Rhat.diag().ones();
  arma::mat residuals = S - Rhat;
  arma::mat I1(p, p, arma::fill::eye);
  arma::mat gLR = gLRhat(Lambda, Phi);
  arma::mat LambdaPhitxI1 = 2*arma::kron(LambdaPhi.t(), I1);
  arma::mat h1 = LambdaPhitxI1 * gLR;
  arma::mat h2 = -2*arma::kron(Phi, residuals);

  arma::mat h_Lambda = h1 + h2;

  arma::mat gPR;
  arma::uvec indexes;
  arma::mat h_Phi;

  if(projection == "oblq") {

    /*
     * Second derivatives for Phi
     */

    indexes = trimatl_ind(arma::size(Phi), -1);
    arma::mat LxL = 2*arma::kron(Lambda, Lambda);
    arma::mat LxLt = LxL.t();
    gPR = gPRhat(Lambda, Phi);
    h_Phi = LxLt.rows(indexes) * gPR;

  }
  /*
   * Second derivatives for psi
   */

  arma::mat h_psi = I1;
  arma::mat h_LambdaPhi;

  if(projection == "oblq") {

    /*
     * Second derivatives for Lambda and Phi
     */

    arma::mat I2(q, q, arma::fill::eye);

    h1 = LambdaPhitxI1 * gPR;
    arma::mat h21 = -2*arma::kron(I2, residuals * Lambda);
    arma::mat h22 = h21 + h21 * dxt(Phi);
    h2 = h22.cols(indexes);
    h_LambdaPhi = h1 + h2;

  }

    /*
     * Second derivatives for Lambda and Psi
     */

    arma::mat LP = 2*LambdaPhi;
    arma::mat h_Lambdapsi = arma::diagmat(LP.col(0));

    for(int i=1; i < q; ++i) {

      arma::mat aug = arma::diagmat(LP.col(i));
      h_Lambdapsi = arma::join_cols(h_Lambdapsi, aug);

    }

    arma::mat h_Phipsi;

    if(projection == "oblq") {

    /*
     * Second derivatives for Phi and Psi
     */

    arma::uvec indexes2(p);
    for(int i=0; i < p; ++i) indexes2[i] = i*p + i;
    h_Phipsi = gPR.rows(indexes2).t();

  }

    arma::mat hessian;

    if(projection == "oblq") {

      /*
       * Join all the derivatives such that
       * h_Lambda         h_LambdaPhi   h_Lambdapsi
       * h_LambdaPhi.t()  h_Phi         h_Phipsi
       * h_Lambdapsi.t()  h_Phipsi.t()  h_psi
       */

      int q_cor = q*(q-1)/2;

      // insert columnwise:
      arma::mat hessian1 = h_Lambda; // pq x pq
      hessian1.insert_cols(pq, h_LambdaPhi); // pq x q_cor
      hessian1.insert_cols(pq + q_cor, h_Lambdapsi); // pq x p

      // insert columnwise:
      arma::mat hessian2 = h_LambdaPhi.t(); // q_cor x pq
      hessian2.insert_cols(pq, h_Phi); // q_cor x q_cor
      hessian2.insert_cols(pq + q_cor, h_Phipsi); // q_cor x p

      // insert columnwise:
      arma::mat hessian3 = h_Lambdapsi.t(); // p x pq
      hessian3.insert_cols(pq, h_Phipsi.t()); // p x pq
      hessian3.insert_cols(pq + q_cor, h_psi); // p x p

      // stack the blocks:
      hessian = arma::join_cols(hessian1, hessian2, hessian3);

    } else if(projection == "orth") {

      /*
       * Join all the derivatives such that
       * h_Lambda        h_Lambdapsi
       * h_Lambdapsi.t()    h_psi
       */

      // insert columnwise:
      arma::mat hessian1 = h_Lambda; // pq x pq
      hessian1.insert_cols(pq, h_Lambdapsi); // pq x p

      // insert columnwise:
      arma::mat hessian2 = h_Lambdapsi.t(); // p x pq
      hessian2.insert_cols(pq, h_psi); // p x p

      // stack the blocks:
      hessian = arma::join_cols(hessian1, hessian2);

    }

  return hessian;

}

/*
 * d2f/(dtheta ds) for minres
 */

arma::mat gLS_minres(arma::mat S, arma::mat Lambda, arma::mat Phi) {

  int p = Lambda.n_rows;
  arma::uvec indexes = trimatl_ind(arma::size(S), -1);

  arma::mat LambdaPhi = Lambda * Phi;
  arma::mat I(p, p, arma::fill::eye);
  arma::mat g1 = -2*arma::kron(LambdaPhi.t(), I);
  arma::mat g2 = g1 * dxt(S);
  arma::mat g_temp = g1 + g2;
  arma::mat g = g_temp.cols(indexes);

  return g;

}

arma::mat gPS_minres(arma::mat S, arma::mat Lambda, arma::mat Phi) {

  arma::uvec indexes1 = trimatl_ind(arma::size(Phi), -1);
  arma::uvec indexes2 = trimatl_ind(arma::size(S), -1);

  arma::mat g1 = -2*arma::kron(Lambda.t(), Lambda.t());
  arma::mat g2 = g1 * dxt(S);
  arma::mat g_temp = g1 + g2;
  arma::mat g = g_temp(indexes1, indexes2);

  return g;

}

arma::mat gLPS_minres(arma::mat S, arma::mat Lambda, arma::mat Phi,
                      std::string projection) {

  /*
   * Compute d2f/(dtheta ds)
   */

  int p = Lambda.n_rows;
  int q = Lambda.n_cols;

  arma::uvec indexes1 = trimatl_ind(arma::size(Phi), -1);
  arma::uvec indexes2 = trimatl_ind(arma::size(S), -1);

  arma::mat LambdaPhi = Lambda * Phi;
  arma::mat I(p, p, arma::fill::eye);
  arma::mat g1 = -2*arma::kron(LambdaPhi.t(), I);
  arma::mat g2 = g1 * dxt(S);
  arma::mat g_temp = g1 + g2;
  arma::mat g = g_temp.cols(indexes2);

  arma::mat gd;
  int k;

  if(projection == "orth") {

    gd = g;
    k = p*q;

  } else if(projection == "oblq") {

    arma::mat d1 = -2*arma::kron(Lambda.t(), Lambda.t());
    arma::mat d2 = d1 * dxt(S);
    arma::mat d_temp = d1 + d2;
    arma::mat d = d_temp(indexes1, indexes2);
    gd = arma::join_cols(g, d);
    k = p*q + q*(q-1)/2;

  }

  /*
   * fill p rows with zeros
   */

  gd.insert_rows(k, p);

  return gd;

}

arma::mat B(arma::mat S, arma::mat Lambda, arma::mat Phi,
            Rcpp::Nullable<arma::mat> nullable_X = R_NilValue,
            std::string method = "minres", std::string projection = "oblq",
            std::string type = "continuous", double eta = 1) {

  /*
   * Compute the B matrix from the sandwich covariance matrix estimator
   */

  arma::uvec indexes = trimatl_ind(arma::size(S), -1);
  arma::mat gLPS;

  if(method == "minres") {

    gLPS = gLPS_minres(S, Lambda, Phi, projection);

  } else {

    Rcpp::stop("Standard errors not implemented yet for maximum likelihood");

  }

  arma::mat asymp = asymp_cov(S, nullable_X, eta, type);
  arma::mat asymp2 = asymp(indexes, indexes);

  arma::mat B = gLPS * asymp2 * gLPS.t();

  return B;

}


