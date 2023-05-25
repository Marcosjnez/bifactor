/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 28/05/2022
 *
 */

// #include "structures.h"
// #include "auxiliary_manifolds.h"

// Manifolds

class cor_manifold {

public:

  virtual void param(arguments_cor& x) = 0;

  virtual void dcor(arguments_cor& x) = 0;

  virtual void grad(arguments_cor& x) = 0;

  virtual void dgrad(arguments_cor& x) = 0;

  virtual void proj(arguments_cor& x) = 0;

  virtual void hess(arguments_cor& x) = 0;

  virtual void retr(arguments_cor& x) = 0;

};

// Oblique manifold:

class oblq:public cor_manifold {

public:

  void param(arguments_cor& x) {

    x.cor = x.T.t() * x.T;

  }

  void dcor(arguments_cor& x) {

    x.dcor = x.T.t() * x.dT;
    x.dcor += x.dcor.t();
    // x.dcor = x.T.t() * x.dT + x.dT.t() * x.T;

  }

  void grad(arguments_cor& x) {

    x.g = x.T * x.gcor;

  }

  void dgrad(arguments_cor& x) {

    x.dg = x.dgcor % (2*x.T) + x.gcor % (2*x.dT);

  }

  void proj(arguments_cor& x) {

    x.rg = x.g - x.T * arma::diagmat( x.T.t() * x.g );

  }

  void hess(arguments_cor& x) {

    x.dH = x.dg - x.dT * arma::diagmat( x.T.t() * x.g) - x.T * arma::diagmat( x.T.t() * x.dg );
    // arma::mat drg = x.dg - x.dT * arma::diagmat( x.T.t() * x.g) - x.T * arma::diagmat( x.dT.t() * x.g ) -
      // x.T * arma::diagmat( x.T.t() * x.dg );
    // x.dH = drg - x.T * arma::diagmat( x.T.t() * drg );

  }

  void retr(arguments_cor& x) {

    x.T *= arma::diagmat( 1 / sqrt(arma::sum(x.T % x.T, 0)) );

  }

};

// Partially oblique manifold:

class poblq:public cor_manifold {

public:

  void param(arguments_cor& x) {

    x.cor = x.T.t() * x.T;

  }

  void dcor(arguments_cor& x) {

    x.dcor = x.T.t() * x.dT;
    x.dcor += x.dcor.t();

  }

  void grad(arguments_cor& x) {

    x.g = x.gcor % (2*x.T);

  }

  void dgrad(arguments_cor& x) {

    x.dg = x.dgcor % (2*x.T) + x.gcor % (2*x.dT);

  }

  void proj(arguments_cor& x) {

    arma::mat c1 = x.T.t() * x.g;
    arma::mat X0 = c1 + c1.t();
    x.A = lyap_sym(x.Phi, X0);
    x.A(x.oblq_indexes).zeros();
    // x.A.diag() = 0.5*arma::diagvec(X0);

    // x.A = syl(x.Phi, X0);
    // X0(x.oblq_indexes).zeros(); x.A = lyapunov(x.Phi, X0, x.oblq_indexes); x.A(x.oblq_indexes).zeros();
    // x.A = lyapunov_2(x.Phi, X0, x.oblq_indexes);
    arma::mat N = x.T * x.A;
    x.rg = x.g - N;

  }

  void hess(arguments_cor& x) {

    // Implicit differentiation of APhi + PhiA = X0
    arma::mat dc1 = x.dT.t() * x.g + x.T.t() * x.dg; // Differential of c1
    arma::mat dX0 = dc1 + dc1.t(); // Differential of X0
    arma::mat c2 = x.A * x.dP + x.dP * x.A; // Differential of APhi + PhiA wrt Phi
    arma::mat Q = dX0 - c2;
    // dAPhi + PhidA = Q
    arma::mat dA = lyap_sym(x.Phi, Q);
    dA(x.oblq_indexes).zeros();
    // dA.diag() = 0.5*arma::diagvec(Q);

    // arma::mat dA = syl(Phi, Q);
    // Q(x.oblq_indexes).zeros(); arma::mat dA = lyapunov(x.Phi, Q, x.oblq_indexes); dA(x.oblq_indexes).zeros();
    // arma::mat dA = lyapunov_2(x.Phi, Q, x.oblq_indexes);
    arma::mat drg = x.dg - (x.dT * x.A + x.T * dA);

    // projection
    arma::mat c = x.T.t() * drg;
    arma::mat X0 = c + c.t();
    arma::mat A = lyap_sym(x.Phi, X0);
    A(x.oblq_indexes).zeros();
    // A.diag() = 0.5*arma::diagvec(X0);

    // X0(x.oblq_indexes).zeros(); arma::mat A = lyapunov(x.Phi, X0, x.oblq_indexes); A(x.oblq_indexes).zeros();
    // arma::mat A = lyapunov_2(x.Phi, X0, x.oblq_indexes);
    arma::mat N = x.T * A;
    x.dH = drg - N;

  }

  void retr(arguments_cor& x) {

    x.T *= arma::diagmat( 1 / sqrt(arma::diagvec( x.T.t() * x.T )) );

    arma::mat Q;
    arma::mat R;
    qr_econ(Q, R, x.T);

    arma::mat X2 = x.T;
    x.T = Q;
    x.T.cols(x.list_oblq_indexes[0]) = X2.cols(x.list_oblq_indexes[0]);

    int J = x.T.n_cols;
    int I = x.list_oblq_indexes.size()-1;

    // In the following loop, the Gram-Schmidt process is performed between blocks:

    for(int i=0; i < I; ++i) {

      // Select the cumulative indexes of blocks:
      std::vector<arma::uvec> indexes(&x.list_oblq_indexes[0], &x.list_oblq_indexes[i+1]);
      arma::uvec cum_indexes = list_to_vector(indexes);

      arma::mat orthogonals = Q.cols(cum_indexes);
      int n = orthogonals.n_cols;

      for(int j=0; j < x.list_oblq_indexes[i+1].size(); ++j) {

        // orthogonalize every column of the following block:
        int index = x.list_oblq_indexes[i+1][j];
        x.T.col(index) = orthogonalize(orthogonals, X2.col(index), n);

      }

    }

  }

};

// Choose the manifold:

cor_manifold* choose_manifold(std::string projection) {

  cor_manifold* manifold;
  if(projection == "oblq") {
    manifold = new oblq();
  } else if(projection == "poblq") {
    manifold = new poblq();
  } else if(projection == "none") {

  } else {

    Rcpp::stop("Available projections: \n oblq, poblq");

  }

  return manifold;

}
