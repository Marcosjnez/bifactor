/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 28/05/2022
 *
 */

// Manifolds

class efa_manifold {

public:

  virtual void param(arguments_efa& x) = 0;

  virtual void dLPU(arguments_efa& x) = 0;

  virtual void grad(arguments_efa& x) = 0;

  virtual void dgrad(arguments_efa& x) = 0;

  virtual void proj(arguments_efa& x) = 0;

  virtual void hess(arguments_efa& x) = 0;

  virtual void retr(arguments_efa& x) = 0;

};

// Identity manifold:

class identity:public efa_manifold {

public:

  void param(arguments_efa& x) {

    x.psi2 = 0.5*(x.lower + x.upper) + 0.5*abs(x.upper - x.lower) % sin(x.psi);

  }

  void dLPU(arguments_efa& x) {

  }

  void grad(arguments_efa& x) {

    x.g = x.g_psi2 % (0.5*abs(x.upper - x.lower) % cos(x.psi));

  }

  void dgrad(arguments_efa& x) {

    Rcpp::stop("The differential of this estimator is not available yet.");

  }

  void proj(arguments_efa& x) {

    x.rg = x.g;

  }

  void hess(arguments_efa& x) {

  }

  void retr(arguments_efa& x) {

  }

};

// Box-constraint manifold:

class box:public efa_manifold {

public:

  void param(arguments_efa& x) {

    x.psi2 = 0.5*(x.lower + x.upper) + 0.5*abs(x.upper - x.lower) % sin(x.psi);

  }

  void dLPU(arguments_efa& x) {

  }

  void grad(arguments_efa& x) {

    x.g = x.g_psi2 % (0.5*abs(x.upper - x.lower) % cos(x.psi));

  }

  void dgrad(arguments_efa& x) {

    Rcpp::stop("The differential of this estimator is not available yet.");

  }

  void proj(arguments_efa& x) {

    x.rg = x.g;

  }

  void hess(arguments_efa& x) {

  }

  void retr(arguments_efa& x) {

  }

};

// Orthogonal manifold:

class orth_efa:public efa_manifold {

public:

  void param(arguments_efa& x) {

    x.lambda = x.psi.cols(0, x.q-1);
    x.lambda(0, 1) = 0;
    x.lambda(0, 2) = 0;
    x.lambda(1, 2) = 0;
    x.uniquenesses = x.psi.col(x.q);

  }

  void dLPU(arguments_efa& x) {

  }

  void grad(arguments_efa& x) {

    x.g = arma::join_rows(x.gL, x.gU);

  }

  void dgrad(arguments_efa& x) {

  }

  void proj(arguments_efa& x) {

    // arma::mat rgL = x.lambda * skew(x.lambda.t() * x.gL);
    // x.rg = arma::join_rows(rgL, x.gU);
    x.rg = x.g;

  }

  void hess(arguments_efa& x) {

    // arma::mat drg = x.dg - x.dT * symm(x.T.t() * x.g);
    // x.dH = x.T * skew(x.T.t() * drg);

  }

  void retr(arguments_efa& x) {

    // arma::mat Q, R;
    // arma::qr_econ(Q, R, x.lambda);
    // x.psi.cols(0, x.q-1) = Q;

  }

};

// Choose the manifold:

efa_manifold* choose_efa_manifold(std::string mani) {

  efa_manifold* manifold;
  if(mani == "identity") {
    manifold = new identity();
  } else if(mani == "orth") {
    manifold = new orth_efa();
  } else if(mani == "box") {
    manifold = new box();
  } else {

    Rcpp::stop("Available manifolds for factor extraction: \n identity, orth, box");

  }

  return manifold;

}
