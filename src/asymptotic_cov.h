/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 18/03/2022
 *
 */

// #include "auxiliary_manifolds.h"

arma::mat asymptotic_general(arma::mat X) {

  /*
   * Browne and Shapiro (Equation 3.2; 1986)
   */

  arma::mat colmeans = arma::mean(X, 0);
  X.each_row() -= colmeans; // Centered matrix
  arma::mat S = arma::cov(X, 1); // Covariance matrix
  arma::vec d = arma::sqrt(arma::diagvec(S));
  arma::mat diag_d_inv = arma::diagmat(1/d);
  arma::mat P = diag_d_inv * S * diag_d_inv; // Correlation matrix
  arma::vec p = arma::vectorise(P);

  int q = P.n_rows;
  int qq = q*q;
  arma::mat Theta(qq, qq); // Fourth-order moments

  int ij = 0;
  int kh;

  for(int j=0; j < q; ++j) {
    for(int i=0; i < q; ++i) {
      kh = 0;
      for(int h=0; h < q; ++h) {
        for(int k=0; k < q; ++k) {
          arma::vec m = X(arma::span::all, i) % X(arma::span::all, j) %
          X(arma::span::all, k) % X(arma::span::all, h);
          Theta(ij, kh) = arma::mean(m) / (d[i]*d[j]*d[k]*d[h]);
          ++kh;
        }
      }
      ++ij;
    }
  }

  arma::mat Gamma = Theta - p * p.t();

  arma::mat Ms = dxt(q, q)*0.5;
  Ms.diag() += 0.5;
  arma::mat I(q, q, arma::fill::eye);
  arma::mat Kd(qq, q, arma::fill::zeros);
  for(int i=0; i < q; ++i) {
    int ii = i * q + i;
    Kd(ii, i) = 1;
  }
  arma::mat A = Ms * arma::kron(I, P) * Kd;
  arma::mat B = Gamma * Kd;
  arma::mat G = Kd.t() * Gamma * Kd;

  arma::mat asymptotic = Gamma - A*B.t() - B*A.t() + A*G*A.t();

  return asymptotic;

}

arma::mat asymptotic_normal(arma::mat P) {

  /*
   * Browne and Shapiro (Equation 4.1; 1986)
   */

  int q = P.n_rows;
  int qq = q*q;

  arma::mat Ms = dxt(q, q)*0.5;
  Ms.diag() += 0.5;
  arma::mat I(q, q, arma::fill::eye);
  arma::mat Kd(qq, q, arma::fill::zeros);
  for(int i=0; i < q; ++i) {
    int ii = i * q + i;
    Kd(ii, i) = 1;
  }
  arma::mat A = Ms * arma::kron(I, P) * Kd;
  arma::mat Gamma = 2*Ms * arma::kron(P, P);
  arma::mat B = Gamma * Kd;
  arma::mat G = 2*P % P; // Cheaper than Kd.t() * Gamma * Kd

  arma::mat asymptotic = Gamma - A*B.t() - B*A.t() + A*G*A.t();

  return asymptotic;

}

arma::mat asymptotic_elliptical(arma::mat P, double eta) {

  /*
   * Browne and Shapiro (Equation 4.2; 1986)
   */

  arma::mat asymptotic = eta * asymptotic_normal(P);

  return asymptotic;

}

arma::mat asymp_cov(arma::mat S,
                    Rcpp::Nullable<arma::mat> nullable_X,
                    double eta, std::string type) {

  arma::mat asymptotic_cov, X;

  if(type == "normal") {
    asymptotic_cov = asymptotic_normal(S);
  } else if(type == "elliptical") {
    asymptotic_cov = asymptotic_elliptical(S, eta);
  } else if(type == "general") {
    if(nullable_X.isNotNull()) {
      X = Rcpp::as<arma::mat>(nullable_X);
    } else {
      Rcpp::stop("The asymptotic covariance matrix of a general correlation matrix requires raw data");
    }
    asymptotic_cov = asymptotic_general(X);
  } else{
    Rcpp::stop("Available asymptotic covariance estimators are 'normal', 'elliptical' and 'general'");
  }

  return asymptotic_cov;

}

