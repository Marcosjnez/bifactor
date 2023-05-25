/*
 * Author: Marcos Jiménez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 05/03/2023
 *
 */

// Criteria for polychoric correlations

class cor_criterion {

public:

  virtual void F(arguments_cor& x) = 0;

  virtual void g_cor(arguments_cor& x) = 0;

  virtual void dgcor(arguments_cor& x) = 0;

};

/*
 * none
 */

class none: public cor_criterion {

public:

  void F(arguments_cor& x) {}

  void gcor(arguments_cor& x) {}

  void dgcor(arguments_cor& x) {}

};

/*
 * polychoric family
 */

class poly_fam: public cor_criterion {

public:

  void F(arguments_cor& x) {

    for(size_t l=0; l < (x.q-1L); ++l) {
      for(int k=(l+1L); k < x.q; ++k) {
        for (size_t i = 0; i < x.s[l]; ++i) {
          for (size_t j = 0; j < x.s[k]; ++j) {
            // CDF of the bivariate normal:
            double ppi = pbinorm(x.taus[l][i], x.taus[k][j], x.taus[l][i + 1], x.taus[k][j + 1], x.cor(l, k),
                                 x.mvphi1[l][i], x.mvphi2[k][j], x.mvphi1[l][i+1], x.mvphi2[k][j+1]);
            x.f -= x.n[l][k][i][j] * std::log(ppi) / x.nobs; // No need to compute the objective value
          }
        }
      }
    }
  }

  void gcor(arguments_cor& x) {

    int kl = 0;
    for(size_t l=0; l < (x.q-1L); ++l) {
      for(int k=(l+1L); k < x.q; ++k) {
        for (size_t i = 0; i < x.s[l]; ++i) {
          for (size_t j = 0; j < x.s[k]; ++j) {
            // CDF of the bivariate normal:
            double ppi = pbinorm(x.taus[l][i], x.taus[k][j], x.taus[l][i + 1], x.taus[k][j + 1], x.cor(l, k),
                  x.mvphi1[l][i], x.mvphi2[k][j], x.mvphi1[l][i+1], x.mvphi2[k][j+1]);
            // PDF of the Bivariate normal:
            double gij = dbinorm(x.cor(l, k), x.taus[l][i+1], x.taus[k][j+1]) -
              dbinorm(x.cor(l, k), x.taus[l][i], x.taus[k][j+1]) -
              dbinorm(x.cor(l, k), x.taus[l][i+1], x.taus[k][j]) +
              dbinorm(x.cor(l, k), x.taus[l][i], x.taus[k][j]);
            if(ppi < 1e-09) ppi = 1e-09; // Avoid division by zero
            x.gcor(kl) -= x.n[l][k][i][j] / ppi * gij / x.nobs; // Update Gradient
            ++kl;
          }
        }
      }
    }
  }

  void dgcor(arguments_cor& x) {

    int kl = 0;
    for(size_t l=0; l < (x.q-1L); ++l) {
      for(int k=(l+1L); k < x.q; ++k) {
        for (size_t i = 0; i < x.s[l]; ++i) {
          for (size_t j = 0; j < x.s[k]; ++j) {
            // CDF of the bivariate normal:
            double ppi = pbinorm(x.taus[l][i], x.taus[k][j], x.taus[l][i + 1], x.taus[k][j + 1], x.cor(l, k),
                                 x.mvphi1[l][i], x.mvphi2[k][j], x.mvphi1[l][i+1], x.mvphi2[k][j+1]);
            // PDF of the Bivariate normal:
            double gij = dbinorm(x.cor(l, k), x.taus[l][i+1], x.taus[k][j+1]) -
              dbinorm(x.cor(l, k), x.taus[l][i], x.taus[k][j+1]) -
              dbinorm(x.cor(l, k), x.taus[l][i+1], x.taus[k][j]) +
              dbinorm(x.cor(l, k), x.taus[l][i], x.taus[k][j]);
            // Derivative of the PDF of the Bivariate normal:
            double hij = ddbinorm(x.cor(l, k), x.taus[l][i+1], x.taus[k][j+1]) -
              ddbinorm(x.cor(l, k), x.taus[l][i], x.taus[k][j+1]) -
              ddbinorm(x.cor(l, k), x.taus[l][i+1], x.taus[k][j]) +
              ddbinorm(x.cor(l, k), x.taus[l][i], x.taus[k][j]);
            if(ppi < 1e-09) ppi = 1e-09; // Avoid division by zero
            double term = hij - gij*x.cor(l, k);
            x.dgcor(kl) += x.n[l][k][i][j]*(gij*gij - ppi*term)/(ppi*ppi) / x.nobs * x.dcor(l, k);
            ++kl;
          }
        }
      }
    }

  }

};

// Choose the cor criteria:

cor_criterion* choose_criterion(std::string cor_fam) {

  cor_criterion *criterion;

  if (cor_fam == "target") {

    criterion = new poly_fam();

  } else if(cor_fam == "none") {

  } else {

    Rcpp::stop("Available models: poly");

  }

  return criterion;

}
