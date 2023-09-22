/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 15/09/2023
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

    // x.transformed = 0.5*(x.lower + x.upper) + 0.5*abs(x.upper - x.lower) % sin(x.parameters);
    // x.transformed = x.parameters;
    // x.transformed(x.target_positive) = x.parameters(x.target_positive) % x.parameters(x.target_positive);
    // x.lambda.elem(x.target_indexes) = x.transformed(x.lambda_indexes);
    // x.phi.elem(x.targetphi_indexes) = x.transformed(x.phi_indexes);
    // x.psi.elem(x.targetpsi_indexes) = x.transformed(x.psi_indexes);
    x.lambda.elem(x.target_indexes) = x.parameters(x.lambda_indexes);
    x.phi.elem(x.targetphi_indexes) = x.parameters(x.phi_indexes);
    x.psi.elem(x.targetpsi_indexes) = x.parameters(x.psi_indexes);
    x.phi = arma::symmatl(x.phi);
    x.psi = arma::symmatl(x.psi);

  }

  void dparam(arguments_cfa& x) {

    x.dlambda.elem(x.target_indexes) = x.dparameters(x.lambda_indexes);
    x.dphi.elem(x.targetphi_indexes) = x.dparameters(x.phi_indexes);
    x.dpsi.elem(x.targetpsi_indexes) = x.dparameters(x.psi_indexes);
    x.dphi = arma::symmatl(x.dphi);
    x.dpsi = arma::symmatl(x.dpsi);

  }

  void grad(arguments_cfa& x) {

    // x.g = x.gradient % (0.5*abs(x.upper - x.lower) % cos(x.parameters));
    x.g = x.gradient;
    // x.g(x.target_positive) %= 2*x.parameters(x.target_positive);

  }

  void dgrad(arguments_cfa& x) {

    // x.dg = x.dgradient % (0.5*abs(x.upper - x.lower) % cos(x.parameters)) +
    //   x.gradient % (0.5*abs(x.upper - x.lower) % -sin(x.dparameters));
    x.dg = x.dgradient;
    // x.dg(x.target_positive) *= 2;

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

class cfa_manifold2 {

public:

  virtual void param(arguments_optim& x, std::vector<arguments_cfa>& structs) = 0;

  virtual void dparam(arguments_optim& x, std::vector<arguments_cfa>& structs) = 0;

  virtual void grad(arguments_optim& x, std::vector<arguments_cfa>& structs) = 0;

  virtual void dgrad(arguments_optim& x, std::vector<arguments_cfa>& structs) = 0;

  virtual void proj(arguments_optim& x, std::vector<arguments_cfa>& structs) = 0;

  virtual void hess(arguments_optim& x, std::vector<arguments_cfa>& structs) = 0;

  virtual void retr(arguments_optim& x, std::vector<arguments_cfa>& structs) = 0;

};

class ultimate_manifold: public cfa_manifold2 {

public:

  void param(arguments_optim& x, std::vector<arguments_cfa>& structs) {

    cfa_manifold* manifold;

    for(int i=0; i < x.nblocks; ++i) {

      structs[i].parameters = x.parameters;
      manifold = choose_cfa_manifold(structs[i].projection);
      manifold->param(structs[i]);

    }

  }

  void dparam(arguments_optim& x, std::vector<arguments_cfa>& structs) {

    cfa_manifold* manifold;

    for(int i=0; i < x.nblocks; ++i) {

      structs[i].dparameters = x.dparameters;
      manifold = choose_cfa_manifold(structs[i].projection);
      manifold->dparam(structs[i]);

    }

  }

  void grad(arguments_optim& x, std::vector<arguments_cfa>& structs) {

    cfa_manifold* manifold;
    x.g.set_size(x.parameters.n_elem); x.g.zeros();

    for(int i=0; i < x.nblocks; ++i) {

      manifold = choose_cfa_manifold(structs[i].projection);
      manifold->grad(structs[i]);
      x.g += structs[i].g;

    }

  }

  void dgrad(arguments_optim& x, std::vector<arguments_cfa>& structs) {

    cfa_manifold* manifold;
    x.dg.set_size(x.parameters.n_elem); x.dg.zeros();

    for(int i=0; i < x.nblocks; ++i) {

      manifold = choose_cfa_manifold(structs[i].projection);
      manifold->dgrad(structs[i]);
      x.dg += structs[i].dg;

    }

  }

  void proj(arguments_optim& x, std::vector<arguments_cfa>& structs) {

    cfa_manifold* manifold;
    x.rg.set_size(x.parameters.n_elem); x.rg.zeros();

    for(int i=0; i < x.nblocks; ++i) {

      structs[i].g = x.g;
      manifold = choose_cfa_manifold(structs[i].projection);
      manifold->proj(structs[i]);

    }

    x.rg = structs[0].rg;

  }

  void hess(arguments_optim& x, std::vector<arguments_cfa>& structs) {

    cfa_manifold* manifold;
    x.dH.set_size(x.parameters.n_elem); x.dH.zeros();

    for(int i=0; i < x.nblocks; ++i) {

      structs[i].dg = x.dg;
      manifold = choose_cfa_manifold(structs[i].projection);
      manifold->hess(structs[i]);

    }

    x.dH = structs[0].dH;

  }

  void retr(arguments_optim& x, std::vector<arguments_cfa>& structs) {

    cfa_manifold* manifold;

    for(int i=0; i < x.nblocks; ++i) {

      structs[i].parameters = x.parameters;
      manifold = choose_cfa_manifold(structs[i].projection);
      manifold->retr(structs[i]);

    }

    x.parameters = structs[0].parameters;

  }

};
