/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 18/03/2022
 *
 */

// #include "structures.h"
// #include "auxiliary_manifolds.h"

// Manifolds

class base_manifold {

public:

  virtual void param(arguments_rotate& x) = 0;

  virtual void dLP(arguments_rotate& x) = 0;

  virtual void grad(arguments_rotate& x) = 0;

  virtual void dgrad(arguments_rotate& x) = 0;

  virtual void g_constraints(arguments_rotate& x) = 0;

  virtual void proj(arguments_rotate& x) = 0;

  virtual void hess(arguments_rotate& x) = 0;

  virtual void retr(arguments_rotate& x) = 0;

};

// Orthogonal manifold:

class orth:public base_manifold {

public:

  void param(arguments_rotate& x) {

    x.L = x.lambda*x.T;

  }

  void dLP(arguments_rotate& x) {

    x.dL = x.lambda * x.dT;

  }

  void grad(arguments_rotate& x) {

    x.g = x.lambda.t() * x.gL;

  }

  void dgrad(arguments_rotate& x) {

    x.dg = x.lambda.t() * x.dgL;

  }

  void g_constraints(arguments_rotate& x) {

    arma::mat I1(x.p, x.p, arma::fill::eye);
    arma::mat I2(x.q, x.q, arma::fill::eye);
    int pq = x.p*x.q;

    x.d_constr_temp = arma::kron(I2, x.L.t()) * x.hL +
      arma::kron(x.gL.t(), I2) * dxt(x.p, x.q);

    arma::uvec indexes_1 = arma::trimatl_ind(arma::size(I2), -1);
    arma::uvec indexes_2 = arma::trimatu_ind(arma::size(I2), 1);
    arma::mat d_constr2 = dxt(x.q, x.q) * x.d_constr_temp;
    x.d_constr_temp -= d_constr2;
    x.d_constr = x.d_constr_temp.rows(indexes_1);
    x.d_constr.insert_cols(pq, x.p);

  }

  void proj(arguments_rotate& x) {

    x.rg = x.T * skew(x.T.t() * x.g);

  }

  void hess(arguments_rotate& x) {

    arma::mat drg = x.dg - x.dT * symm(x.T.t() * x.g); // update this?
    // Rcpp::Rcout << drg << std::endl;
    // arma::mat drg = x.dT * skew(x.T.t() * x.g) + x.T * skew(x.dT.t() * x.g) + x.T*skew(x.T.t() * x.dg);
    // Rcpp::Rcout << drg << std::endl;
    x.dH = x.T * skew(x.T.t() * drg);

  }

  void retr(arguments_rotate& x) {

    arma::mat Q, R;
    arma::qr_econ(Q, R, x.T);

    x.T = Q;

  }

};

// Oblique manifold:

class oblq:public base_manifold {

public:

  void param(arguments_rotate& x) {

    x.Phi = x.T.t() * x.T;
    x.Inv_T = arma::inv(x.T);
    x.L = x.lambda * x.Inv_T.t();

  }

  void dLP(arguments_rotate& x) {

    x.Inv_T_dt = x.Inv_T * x.dT;
    x.dL = - x.L * x.Inv_T_dt.t();

    x.dP = x.T.t() * x.dT;
    x.dP += x.dP.t();

  }

  void grad(arguments_rotate& x) {

    arma::mat g1 = - x.Inv_T.t() * x.gL.t() * x.L;

    if(x.gP.is_empty()) {

      x.g = g1;

    } else {

      arma::mat g2 = x.T * x.gP;
      x.g = g1 + g2;

    }

  }

  void dgrad(arguments_rotate& x) {

    arma::mat dg1 = - x.g * x.Inv_T_dt.t() - (x.dT * x.Inv_T).t() * x.g - (x.dgL * x.Inv_T).t() * x.L;

    if(x.gP.is_empty()) {

      x.dg = dg1;

    } else {

      arma::mat dg2 = x.dT * x.gP + x.T * x.dgP;
      x.dg = dg1 + dg2;

    }

  }

  void g_constraints(arguments_rotate& x) {

    int pq = x.p*x.q;
    int qq = x.q*x.q;
    int q_cor = x.q*(x.q-1)/2;
    arma::uvec indexes_1(x.q);

    for(int i=0; i < x.q; ++i) indexes_1[i] = ((i+1)*x.q) - (x.q-i);
    arma::uvec indexes_2 = arma::trimatl_ind(arma::size(x.Phi), -1);
    arma::mat I2(x.q, x.q, arma::fill::eye);

    x.d_constr_temp = arma::kron(I2, x.L.t()) * x.hL +
      arma::kron(x.gL.t(), I2) * dxt(x.p, x.q);

    arma::mat inv_Phi = arma::inv_sympd(x.Phi);
    arma::cube B(qq, pq, 1);
    B.slice(0) = x.d_constr_temp;
    B.reshape(x.q, x.q, pq);
    B.each_slice() *= inv_Phi;
    B.reshape(x.q*x.q, pq, 1);
    x.d_constr_temp = B.slice(0);

    arma::mat c1p = -arma::kron(inv_Phi.t(), (x.L.t() * x.gL * inv_Phi));
    arma::mat Phi_t = dxt(x.q, x.q);
    arma::mat HP_temp = c1p + c1p * Phi_t;
    if(!x.hP.is_empty()) HP_temp -= x.hP;
    arma::mat HP = HP_temp.cols(indexes_2);

    x.d_constr = arma::join_rows(x.d_constr_temp, HP);
    x.d_constr.shed_rows(indexes_1);
    x.d_constr.insert_cols(pq + q_cor, x.p);

  };

  void proj(arguments_rotate& x) {

    x.rg = x.g - x.T * arma::diagmat( x.T.t() * x.g );

  }

  void hess(arguments_rotate& x) {

    x.dH = x.dg - x.dT * arma::diagmat( x.T.t() * x.g) - x.T * arma::diagmat( x.T.t() * x.dg );
    // arma::mat drg = x.dg - x.dT * arma::diagmat( x.T.t() * x.g) - x.T * arma::diagmat( x.dT.t() * x.g ) -
      // x.T * arma::diagmat( x.T.t() * x.dg );
    // x.dH = drg - x.T * arma::diagmat( x.T.t() * drg );

  }

  void retr(arguments_rotate& x) {

    x.T *= arma::diagmat( 1 / sqrt(arma::sum(x.T % x.T, 0)) );

  }

};

// Partially oblique manifold:

class poblq:public base_manifold {

public:

  void param(arguments_rotate& x) {

    x.Phi = x.T.t() * x.T;
    x.Inv_T = arma::inv(x.T);
    x.L = x.lambda * x.Inv_T.t();

  }

  void dLP(arguments_rotate& x) {

    x.Inv_T_dt = x.Inv_T * x.dT;
    x.dL = - x.L * x.Inv_T_dt.t();

    x.dP = x.T.t() * x.dT;
    x.dP += x.dP.t();

  }

  void grad(arguments_rotate& x) {

    arma::mat g1 = - x.Inv_T.t() * x.gL.t() * x.L;

    if(x.gP.is_empty()) {

      x.g = g1;

    } else {

      arma::mat g2 = x.T * x.gP;
      x.g = g1 + g2;

    }

  }

  void dgrad(arguments_rotate& x) {

    arma::mat dg1 = - x.g * x.Inv_T_dt.t() - (x.dT * x.Inv_T).t() * x.g - (x.dgL * x.Inv_T).t() * x.L;

    if(x.gP.is_empty()) {

      x.dg = dg1;

    } else {

      arma::mat dg2 = x.dT * x.gP + x.T * x.dgP;
      x.dg = dg1 + dg2;

    }

  }

  void g_constraints(arguments_rotate& x) {

    int pq = x.p*x.q;
    int qq = x.q*x.q;
    int q_cor = x.q*(x.q-1)/2;
    arma::uvec indexes_1(x.q);

    for(int i=0; i < x.q; ++i) indexes_1[i] = ((i+1)*x.q) - (x.q-i);
    arma::uvec indexes_2 = arma::trimatl_ind(arma::size(x.Phi), -1);
    arma::mat I2(x.q, x.q, arma::fill::eye);

    x.d_constr_temp = arma::kron(I2, x.L.t()) * x.hL +
      arma::kron(x.gL.t(), I2) * dxt(x.p, x.q);

    arma::mat inv_Phi = arma::inv_sympd(x.Phi);
    arma::cube B(qq, pq, 1);
    B.slice(0) = x.d_constr_temp;
    B.reshape(x.q, x.q, pq);
    B.each_slice() *= inv_Phi;
    B.reshape(x.q*x.q, pq, 1);
    x.d_constr_temp = B.slice(0);

    arma::mat c1p = -arma::kron(inv_Phi.t(), (x.L.t() * x.gL * inv_Phi));
    arma::mat Phi_t = dxt(x.q, x.q);
    arma::mat HP_temp = c1p + c1p * Phi_t;
    if(!x.hP.is_empty()) HP_temp -= x.hP;
    arma::mat HP = HP_temp.cols(indexes_2);

    x.d_constr = arma::join_rows(x.d_constr_temp, HP);
    x.d_constr.shed_rows(indexes_1);
    x.d_constr.insert_cols(pq + q_cor, x.p);

  };

  void proj(arguments_rotate& x) {

    arma::mat c1 = x.T.t() * x.g;
    arma::mat X0 = c1 + c1.t();
    x.A = lyap_sym(x.Phi, X0);
    x.A(x.oblq_indexes).zeros();

    // x.A = syl(x.Phi, X0);
    // X0(x.oblq_indexes).zeros(); x.A = lyapunov(x.Phi, X0, x.oblq_indexes); x.A(x.oblq_indexes).zeros();
    // x.A = lyapunov_2(x.Phi, X0, x.oblq_indexes);
    arma::mat N = x.T * x.A;
    x.rg = x.g - N;

  }

  void hess(arguments_rotate& x) {

    // Implicit differentiation of APhi + PhiA = X0
    arma::mat dc1 = x.dT.t() * x.g + x.T.t() * x.dg; // Differential of c1
    arma::mat dX0 = dc1 + dc1.t(); // Differential of X0
    arma::mat c2 = x.A * x.dP + x.dP * x.A; // Differential of APhi + PhiA wrt Phi
    arma::mat Q = dX0 - c2;
    // dAPhi + PhidA = Q
    arma::mat dA = lyap_sym(x.Phi, Q);
    dA(x.oblq_indexes).zeros(); // should be set to 0?

    // arma::mat dA = syl(Phi, Q);
    // Q(x.oblq_indexes).zeros(); arma::mat dA = lyapunov(x.Phi, Q, x.oblq_indexes); dA(x.oblq_indexes).zeros();
    // arma::mat dA = lyapunov_2(x.Phi, Q, x.oblq_indexes);
    arma::mat drg = x.dg - (x.dT * x.A + x.T * dA);

    // projection
    arma::mat c = x.T.t() * drg;
    arma::mat X0 = c + c.t();
    arma::mat A = lyap_sym(x.Phi, X0);
    A(x.oblq_indexes).zeros();

    // X0(x.oblq_indexes).zeros(); arma::mat A = lyapunov(x.Phi, X0, x.oblq_indexes); A(x.oblq_indexes).zeros();
    // arma::mat A = lyapunov_2(x.Phi, X0, x.oblq_indexes);
    arma::mat N = x.T * A;
    x.dH = drg - N;

  }

  void retr(arguments_rotate& x) {

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

base_manifold* choose_manifold(std::string projection) {

  base_manifold* manifold;
  if(projection == "orth") {
    manifold = new orth();
  } else if(projection == "oblq") {
    manifold = new oblq();
  } else if(projection == "poblq") {
    manifold = new poblq();
  } else if(projection == "none") {

  } else {

    Rcpp::stop("Available projections: \n orth, oblq, poblq");

  }

  return manifold;

}

// Retractions onto the manifolds:

arma::mat retr_orth(arma::mat X) {

  arma::mat Q, R;
  arma::qr_econ(Q, R, X);

  return Q;

}

arma::mat retr_oblq(arma::mat X) {

  X *= arma::diagmat(1 / sqrt(arma::sum(X % X, 0)));

  return X;

}

arma::mat retr_poblq(arma::mat X, arma::uvec oblq_blocks) {

  int nfactors = arma::accu(oblq_blocks);
  if(nfactors > X.n_rows || nfactors > X.n_cols) Rcpp::stop("Too many factors declared in oblq_blocks");

  std::vector<arma::uvec> list_oblq_blocks = vector_to_list2(oblq_blocks);

  poblq RR;
  arguments_rotate x;
  x.T = X;
  x.list_oblq_indexes = list_oblq_blocks;
  RR.retr(x);

  return x.T;

}

// Random matrix generation:

arma::mat random_orth(int p, int q) {

  arma::mat X(p, q, arma::fill::randn);
  arma::mat Q;
  arma::mat R;
  qr_econ(Q, R, X);

  return Q;

}

arma::mat random_oblq(int p, int q) {

  arma::mat X(p, q, arma::fill::randn);
  X *= arma::diagmat(1 / sqrt(arma::sum(X % X, 0)));

  return X;

}

arma::mat random_poblq(int p, int q, arma::uvec oblq_blocks) {

  int nfactors = arma::accu(oblq_blocks);
  if(nfactors > p || nfactors > q) Rcpp::stop("Too many factors declared in oblq_blocks");

  std::vector<arma::uvec> list_oblq_blocks = vector_to_list2(oblq_blocks);;

  arma::mat X(p, q, arma::fill::randn);
  poblq RR;
  arguments_rotate x;
  x.T = X;
  x.list_oblq_indexes = list_oblq_blocks;
  RR.retr(x);

  return x.T;

}
