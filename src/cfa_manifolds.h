/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 31/08/2023
 *
 */

// Manifolds

class cfa_manifold {

public:

  virtual void param(arguments_cfa& x) = 0;

  virtual void dparam(arguments_cfa& x) = 0;

  virtual void grad(arguments_cfa& x) = 0;

  virtual void dgrad(arguments_cfa& x) = 0;

  virtual void proj(arguments_cfa& x) = 0;

  virtual void hess(arguments_cfa& x) = 0;

  virtual void retr(arguments_cfa& x) = 0;

};

// Identity manifold for CFA:

class cfa_id:public cfa_manifold {

public:

  void param(arguments_cfa& x) {

    x.lambda.elem(x.lambda_indexes) = x.parameters.head(x.n_lambda);
    x.phi.elem(x.phi_indexes) = x.parameters.subvec(x.n_lambda, x.n_lambda + x.n_phi - 1);
    x.phi = arma::symmatl(x.phi);
    x.psi.elem(x.psi_indexes) = x.parameters.tail(x.n_psi);
    x.psi = arma::symmatl(x.psi);

  }

  void dparam(arguments_cfa& x) {

  }

  void grad(arguments_cfa& x) {

    x.g = x.gradient;

  }

  void dgrad(arguments_cfa& x) {

    x.dg = x.dgradient;

  }

  void proj(arguments_cfa& x) {

    x.rg = x.g;

  }

  void hess(arguments_cfa& x) {

    x.dH = x.dg;

  }

  void retr(arguments_cfa& x) {

  }

};

// Identity manifold for EFA:

class cfa_identity:public cfa_manifold {

public:

  void param(arguments_cfa& x) {

    x.psi2 = x.parameters;

  }

  void dparam(arguments_cfa& x) {

  }

  void grad(arguments_cfa& x) {

    x.g = x.g_psi2;

  }

  void dgrad(arguments_cfa& x) {

    Rcpp::stop("The differential of this estimator is not available yet.");

  }

  void proj(arguments_cfa& x) {

    x.rg = x.g;

  }

  void hess(arguments_cfa& x) {

  }

  void retr(arguments_cfa& x) {

  }

};

// Box-constraint manifold:

class cfa_box:public cfa_manifold {

public:

  void param(arguments_cfa& x) {

    x.psi2 = 0.5*(x.lower + x.upper) + 0.5*abs(x.upper - x.lower) % sin(x.parameters);

  }

  void dparam(arguments_cfa& x) {

  }

  void grad(arguments_cfa& x) {

    x.g = x.g_psi2 % (0.5*abs(x.upper - x.lower) % cos(x.parameters));

  }

  void dgrad(arguments_cfa& x) {

    Rcpp::stop("The differential of this estimator is not available yet.");

  }

  void proj(arguments_cfa& x) {

    x.rg = x.g;

  }

  void hess(arguments_cfa& x) {

  }

  void retr(arguments_cfa& x) {

  }

};

// DWLS manifold:

class efa_dwls_manifold:public cfa_manifold {

public:

  void param(arguments_cfa& x) {

    x.lambda.elem(x.lower_tri_ind) = x.parameters;

  }

  void dparam(arguments_cfa& x) {

  }

  void grad(arguments_cfa& x) {

    x.g = arma::vectorise(x.gL.elem(x.lower_tri_ind));

  }

  void dgrad(arguments_cfa& x) {

    // x.dg = x.G;

  }

  void proj(arguments_cfa& x) {

    x.rg = x.g;

  }

  void hess(arguments_cfa& x) {

    // x.dH = x.dg;

  }

  void retr(arguments_cfa& x) {

  }

};

// Choose the manifold:

cfa_manifold* choose_cfa_manifold(std::string projection) {

  cfa_manifold* manifold;
  if(projection == "id") {
    manifold = new cfa_id();
  } else if(projection == "identity") {
    manifold = new cfa_identity();
  } else if(projection == "dwls") {
    manifold = new efa_dwls_manifold();
  } else if(projection == "box") {
    manifold = new cfa_box();
  } else if(projection == "none") {

  } else {

    Rcpp::stop("Available projections: \n id, identity, box, and dwls_efa");

  }

  return manifold;

}
