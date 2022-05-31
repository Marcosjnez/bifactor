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

    // arma::vec eigval;
    // arma::mat eigvec;
    // eig_sym(eigval, eigvec, Sstar);
    //
    // arma::vec eigval2 = reverse(eigval);
    // arma::mat eigvec2 = reverse(eigvec, 1);
    //
    // arma::mat A = eigvec2(arma::span::all, arma::span(0, nfactors-1));
    // arma::vec eigenvalues = eigval2(arma::span(0, nfactors-1)) - 1;
    // for(int i=0; i < nfactors; ++i) {
    //   if(eigenvalues[i] < 0) eigenvalues[i] = 0;
    // }
    // arma::mat D = diagmat(sqrt(eigenvalues));
    //
    // w = A * D;
    // w = diagmat(sqrt_psi) * w;
    // arma::mat ww = w * w.t();
    // uniquenesses = 1 - diagvec(ww);

  }

  void grad(arguments_efa& x) {


    // Rhat = ww;
    // Rhat.diag() = R.diag();

  }

  void dgrad(arguments_efa& x) {

  }

  void g_constraints(arguments_efa& x) {

  }

  void proj(arguments_efa& x) {

  }

  void hess(arguments_efa& x) {

  }

  void retr(arguments_efa& x) {

  }

};
