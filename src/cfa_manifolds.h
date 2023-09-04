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

  virtual void g_constraints(arguments_cfa& x) = 0;

  virtual void proj(arguments_cfa& x) = 0;

  virtual void hess(arguments_cfa& x) = 0;

  virtual void retr(arguments_cfa& x) = 0;

};

// Identity manifold:

class id:public cfa_manifold {

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

  void g_constraints(arguments_cfa& x) {

    Rcpp::Rcout << "Constraints are not available for the Identity manifold" << std::endl;

  };

  void proj(arguments_cfa& x) {

    x.rg = x.g;

  }

  void hess(arguments_cfa& x) {

    x.dH = x.dg;

  }

  void retr(arguments_cfa& x) {

  }

};

// Choose the manifold:

cfa_manifold* choose_cfa_manifold(std::string projection) {

  cfa_manifold* manifold;
  if(projection == "id") {
    manifold = new id();
  } else if(projection == "none") {

  } else {

    Rcpp::stop("Available projections: \n id");

  }

  return manifold;

}
