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

  virtual void outcomes(arguments_efa& x) = 0;

};

/*
 * minres
 */

class minres: public efa_criterion {

public:

  void F(arguments_efa& x) {

    x.psi2 = 0.5*(x.lower + x.upper) + 0.5*abs(x.upper - x.lower) % sin(x.psi);
    x.reduced_R = x.R - arma::diagmat(x.psi2);
    // x.reduced_R = x.R - arma::diagmat(x.psi);
    eig_sym(x.eigval, x.eigvec, x.reduced_R);
    arma::vec e = x.eigval(arma::span(0, x.p - x.q - 1));

    x.f = 0.5*arma::accu(e % e);

  }

  void G(arguments_efa& x) {

    arma::vec e_values = x.eigval(arma::span(0, x.p - x.q - 1));
    arma::mat e_vectors = x.eigvec(arma::span::all, arma::span(0, x.p - x.q - 1));
    x.g = -arma::diagvec(e_vectors * arma::diagmat(e_values) * e_vectors.t());
    x.g %= 0.5*abs(x.upper - x.lower) % cos(x.psi);

  }

  void dG(arguments_efa& x) {

    Rcpp::stop("The differential of this factor extraction method is not available yet.");

  }

  void gLPU(arguments_efa& x) {

  }

  void hLPU(arguments_efa& x) {}

  void dgLPU(arguments_efa& x) {}

  void outcomes(arguments_efa& x) {

    arma::vec eigval;
    arma::mat eigvec;
    eig_sym(eigval, eigvec, x.reduced_R);

    arma::vec eigval2 = reverse(eigval);
    arma::mat eigvec2 = reverse(eigvec, 1);

    arma::mat A = eigvec2(arma::span::all, arma::span(0, x.q-1));
    arma::vec eigenvalues = eigval2(arma::span(0, x.q-1));
    for(int i=0; i < x.q; ++i) {
      if(eigenvalues(i) < 0) eigenvalues(i) = 0;
    }
    arma::mat D = diagmat(sqrt(eigenvalues));

    x.lambda = A * D;
    x.Rhat = x.lambda * x.lambda.t();
    x.uniquenesses = 1 - diagvec(x.Rhat);
    x.Rhat.diag() = x.R.diag();

  };

};

/*
 * ml
 */

class ml: public efa_criterion {

public:

  void F(arguments_efa& x) {

    x.psi2 = 0.5*(x.lower + x.upper) + 0.5*abs(x.upper - x.lower) % sin(x.psi);
    x.sqrt_psi = sqrt(x.psi2);
    // x.sqrt_psi = sqrt(x.psi);
    arma::mat sc = arma::diagmat(1/x.sqrt_psi);
    x.reduced_R = sc * x.R * sc;
    eig_sym(x.eigval, x.eigvec, x.reduced_R);
    arma::vec e = x.eigval(arma::span(0, x.p - x.q - 1));

    // double objective = -arma::accu(log(e) + 1/e - 1);
    x.f = -(arma::accu(log(e) - e) + x.p - x.q);

  }

  void G(arguments_efa& x) {

    arma::mat A = x.eigvec(arma::span::all, arma::span(x.p-x.q, x.p-1));
    arma::vec eigenvalues = x.eigval(arma::span(x.p-x.q, x.p-1));

    // x.g = ((A % A) * (eigenvalues - 1) + 1 - arma::diagvec(x.R)/x.psi)/x.psi;
    x.g = ((A % A) * (eigenvalues - 1) + 1 - arma::diagvec(x.R)/x.psi2)/x.psi2;
    x.g %= 0.5*abs(x.upper - x.lower) % cos(x.psi);

  }

  void dG(arguments_efa& x) {

    Rcpp::stop("The differential of this factor extraction method is not available yet.");

  }

  void gLPU(arguments_efa& x) {}

  void hLPU(arguments_efa& x) {}

  void dgLPU(arguments_efa& x) {}

  void outcomes(arguments_efa& x) {

    arma::vec eigval;
    arma::mat eigvec;
    eig_sym(eigval, eigvec, x.reduced_R);

    arma::vec eigval2 = reverse(eigval);
    arma::mat eigvec2 = reverse(eigvec, 1);

    arma::mat A = eigvec2(arma::span::all, arma::span(0, x.q-1));
    arma::vec eigenvalues = eigval2(arma::span(0, x.q-1)) - 1;
    for(int i=0; i < x.q; ++i) {
      if(eigenvalues[i] < 0) eigenvalues[i] = 0;
    }
    arma::mat D = diagmat(sqrt(eigenvalues));
    arma::mat w = A * D;

    x.lambda = diagmat(x.sqrt_psi) * w;
    x.Rhat = x.lambda * x.lambda.t();
    x.uniquenesses = 1 - diagvec(x.Rhat);
    x.Rhat.diag() = x.R.diag();

  };

};

// Choose the factor extraction method:

efa_criterion* choose_efa_criterion(std::string method) {

  efa_criterion *criterion;

  if (method == "minres") {

    criterion = new minres();

  } else if(method == "ml") {

    criterion = new ml();

  } else if(method == "minrank" | method == "pa") {

  } else {

    Rcpp::stop("Available factor extraction methods: \n minres, ml, minrank, pa");

  }

  return criterion;

}
