/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 28/05/2022
 *
 */

// Criteria for factor extraction

class efa_criterion {

public:

  virtual void F(arguments_efa& x) = 0;

  virtual void G(arguments_efa& x) = 0;

  virtual void dG(arguments_efa& x) = 0;

  virtual void gLPU(arguments_efa& x) = 0;

  virtual void hLPU(arguments_efa& x) = 0;

  virtual void dgLPU(arguments_efa& x) = 0;

};

/*
 * minres
 */

class minres: public efa_criterion {

public:

  void F(arguments_efa& x) {

    x.reduced_R = x.R - diagmat(x.psi);
    eig_sym(x.eigval, x.eigvec, x.reduced_R);
    arma::vec e = eigval(arma::span(0, x.p - x.q - 1));

    x.f = arma::accu(e % e);

  }

  void G(arguments_efa& x) {

    arma::vec e_values = x.eigval(arma::span(0, x.p - x.q - 1));
    arma::mat e_vectors = x.eigvec(arma::span::all, arma::span(0, x.p - x.q - 1));
    x.g = -2*arma::diagvec(e_vectors * arma::diagmat(e_values) * e_vectors.t());

  }

  void dG(arguments_efa& x) {

  }

  void gLPU(arguments_efa& x) {

  }

  void hLPU(arguments_efa& x) {}

  void dgLPU(arguments_efa& x) {}

};

/*
 * ml
 */

class ml: public efa_criterion {

public:

  void F(arguments_efa& x) {

    x.sqrt_psi = sqrt(x.psi);
    x.sc = arma::diagmat(1/x.sqrt_psi);
    x.reduced_R = x.sc * x.R * x.sc;
    eig_sym(x.eigval, x.reduced_R);
    arma::vec e = eigval(arma::span(0, x.p - x.q - 1));

    // double objective = -arma::accu(log(e) + 1/e - 1);
    x.f = arma::accu(log(e) - e) + x.p - x.q;
    x.f *= -1;

  }

  void G(arguments_efa& x) {

    arma::mat A = x.eigvec(arma::span::all, arma::span(0, x.q-1));
    arma::vec eigenvalues = x.eigval(arma::span(0, x.q-1));

    x.g = ((A % A) * (eigenvalues - 1) + 1 - arma::diagvec(x.R)/x.psi)/x.psi;

  }

  void dG(arguments_efa& x) {

  }

  void gLPU(arguments_efa& x) {

  }

  void hLPU(arguments_efa& x) {}

  void dgLPU(arguments_efa& x) {}

};

