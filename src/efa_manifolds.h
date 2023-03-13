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

  virtual void g_constraints(arguments_efa& x) = 0;

  virtual void proj(arguments_efa& x) = 0;

  virtual void hess(arguments_efa& x) = 0;

  virtual void retr(arguments_efa& x) = 0;

};

// Identity manifold:

class identity:public efa_manifold {

public:

  void param(arguments_efa& x) {

  }

  void dLPU(arguments_efa& x) {

  }

  void grad(arguments_efa& x) {

  }

  void dgrad(arguments_efa& x) {

  }

  void g_constraints(arguments_efa& x) {

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

  }

  void dLPU(arguments_efa& x) {

  }

  void grad(arguments_efa& x) {

  }

  void dgrad(arguments_efa& x) {

  }

  void g_constraints(arguments_efa& x) {

  }

  void proj(arguments_efa& x) {

    x.rg = x.g;

  }

  void hess(arguments_efa& x) {

  }

  void retr(arguments_efa& x) {

  }

};

// Choose the manifold:

efa_manifold* choose_efa_manifold(std::string mani) {

  efa_manifold* manifold;
  if(mani == "identity") {
    manifold = new identity();
  } else if(mani == "box") {
    manifold = new box();
  } else {

    Rcpp::stop("Available manifolds for factor extraction: \n identity, box");

  }

  return manifold;

}
