/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 28/05/2022
 *
 */

// Criteria for factor analysis

class efa_criterion {

public:

  virtual void F(arguments_efa& x) = 0;

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

    arma::mat reduced_R = x.R - diagmat(x.psi);

    arma::vec eigval;
    eig_sym(eigval, reduced_R);
    arma::vec e = eigval(arma::span(0, x.p - x.q - 1));
    x.f = arma::accu(e % e);

  }

  void gLPU(arguments_efa& x) {}

  void hLPU(arguments_efa& x) {}

  void dgLPU(arguments_efa& x) {}

};

/*
 * minres
 */

class ml: public efa_criterion {

public:

  void F(arguments_efa& x) {

    x.sqrt_psi = sqrt(x.psi);
    arma::mat sc = diagmat(1/x.sqrt_psi);
    x.Sstar = sc * x.R * sc;

    arma::vec eigval;
    eig_sym(eigval, x.Sstar);

    arma::vec e = eigval(arma::span(0, x.p - x.q - 1));

    // double objective = -arma::accu(log(e) + 1/e - 1);
    x.f = arma::accu(log(e) - e) + x.p - x.q;
    x.f *= -1;

  }

  void gLP(arguments_efa& x) {}

  void hLP(arguments_efa& x) {}

  void dgLP(arguments_efa& x) {}

};

