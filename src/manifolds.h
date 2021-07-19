arma::mat skew(arma::mat X) {

  return 0.5 * (X - X.t());

}

arma::mat symm(arma::mat X) {

  return 0.5 * (X + X.t());

}

// Solve the lyapunov equation YX + XY = Q for symmetric Q and X:

arma::mat lyap_sym(arma::mat Y, arma::mat Q) {

  int q = Y.n_cols;
  arma::vec I(q, arma::fill::ones);

  arma::vec eigval;
  arma::mat eigvec;
  arma::eig_sym(eigval, eigvec, Y);

  arma::mat M = eigvec.t() * Q * eigvec;
  arma::mat W1 = I * eigval.t();
  arma::mat W = W1 + W1.t();
  arma::mat YY = M / W;
  arma::mat A = eigvec * YY * eigvec.t();

  return A;

}

arma::mat syl(arma::mat Y, arma::mat Q) {

  arma::mat X = arma::syl(Y, Y, -Q);

  return X;

}

arma::mat lyapunov(arma::mat Y, arma::mat Q, arma::uvec indexes) {

  int q = Y.n_rows;
  arma::mat I(q, q, arma::fill::eye);
  arma::mat Y1 = arma::kron(I, Y);
  arma::mat Y2 = arma::kron(Y, I);
  arma::mat X = Y1 + Y2;

  Q = Q.elem(indexes);
  X = X(indexes, indexes);
  arma::vec Lambda = arma::solve(X, Q, arma::solve_opts::fast);
  arma::mat A(q, q, arma::fill::zeros);
  A(indexes) = Lambda;
  A = symmatl(A);

  return A;

}

arma::mat lyapunov_2(arma::mat Y, arma::mat Q, arma::uvec indexes) {

  int q = Y.n_rows;
  arma::mat I(q, q, arma::fill::eye);
  arma::mat Y1 = arma::kron(I, Y);
  arma::mat Y2 = arma::kron(Y, I);
  arma::mat X = Y1 + Y2;

  Q = Q.elem(indexes);
  X = X(indexes, indexes);
  arma::vec Lambda = arma::solve(X, Q, arma::solve_opts::fast);
  arma::mat A(q, q, arma::fill::zeros);
  A(indexes) = Lambda;

  return A;

}

arma::uvec consecutive(int lower, int upper) {

  int size = upper - lower + 1;
  arma::uvec ivec(size);
  std::iota(ivec.begin(), ivec.end(), lower);

  return ivec;
}

std::vector<arma::uvec> vector_to_list(arma::uvec v){

  int n = v.size();
  std::vector<arma::uvec> lista(n);
  v.insert_rows(0, 1);

  for(int i=0; i < n; ++i) {

    lista[i] = v[i] + consecutive(1, v[i+1]);

  }

  return lista;

}

std::vector<arma::uvec> vector_to_list2(arma::uvec v){

  int n = v.size();
  int add = 0;
  std::vector<arma::uvec> lista(n);

  for(int i=0; i < n; ++i) {

    if(i != 0) {
      add = lista[i-1].back() + 1;
    }

    lista[i] = add + consecutive(1, v[i]) - 1;

  }

  return lista;

}

arma::vec orthogonalize(arma::mat orthogonals, arma::vec x, int k) {

  for(int i=0; i < k; ++i) {

    // x -= arma::accu(orthogonals.col(i) % x) / arma::accu(orthogonals.col(i) % orthogonals.col(i)) * orthogonals.col(i);
    x -= arma::accu(orthogonals.col(i) % x) * orthogonals.col(i);

  }

  x /= sqrt(arma::accu(x % x));

  return x;

}

arma::uvec list_to_vector(std::vector<arma::uvec> X) {

  arma::uvec single_vector = std::accumulate(X.begin(), X.end(),
                                             arma::uvec(), [](arma::uvec a, arma::uvec b) {
                                               a = arma::join_cols(a, b);
                                               return a;
                                             });

  return single_vector;

}

// Manifolds

class base_manifold {

public:

  virtual void proj(arma::mat& rg, arma::mat& A, arma::mat g, arma::mat T,
                    arma::mat Phi, arma::uvec indexes_1) = 0;

  virtual void hess(arma::mat& dH, arma::mat delta, arma::mat g, arma::mat dg, arma::mat T,
                    arma::mat Phi, arma::mat A, arma::uvec indexes_1) = 0;

  virtual void retr(arma::mat& T, std::vector<arma::uvec> oblq_indexes) = 0;

};

// Orthogonal manifold:

class orth:public base_manifold {

public:

  void proj(arma::mat& rg, arma::mat& A, arma::mat g, arma::mat T,
            arma::mat Phi, arma::uvec indexes_1) {

    // A = skew(T.t() * g);
    // rg = T * A;
    rg = T * skew(T.t() * g);

  }

  void hess(arma::mat& dH, arma::mat delta, arma::mat g, arma::mat dg, arma::mat T,
            arma::mat Phi, arma::mat A, arma::uvec indexes_1) {

    // arma::mat hessian = delta * A + T * skew(delta.t() * g + T.t() * dg);
    // proj_orth(dH, A, hessian, T, Phi, indexes_1);
    dg -= delta * symm(T.t() * g);
    proj(dH, A, dg, T, Phi, indexes_1);

  }

  void retr(arma::mat& T, std::vector<arma::uvec> oblq_indexes) {

    arma::mat Q, R;
    arma::qr_econ(Q, R, T);

    T = Q;

  }

};

// Oblique manifold:

class oblq:public base_manifold {

public:

  void proj(arma::mat& rg, arma::mat& A, arma::mat g, arma::mat T,
            arma::mat Phi, arma::uvec indexes_1) {

    rg = g - T * diagmat( T.t() * g );

  }

  void hess(arma::mat& dH, arma::mat delta, arma::mat g, arma::mat dg, arma::mat T,
            arma::mat Phi, arma::mat A, arma::uvec indexes_1) {

    dH = dg - T * diagmat( T.t() * dg ) - delta * diagmat( T.t() * g);

  }

  void retr(arma::mat& T, std::vector<arma::uvec> oblq_indexes) {

    T *= arma::diagmat(1 / sqrt(arma::sum(T % T, 0)));

  }

};

// Partially oblique manifold:

class poblq:public base_manifold {

public:

  void proj(arma::mat& rg, arma::mat& A, arma::mat g, arma::mat X,
            arma::mat Phi, arma::uvec indexes_1) {

    arma::mat c1 = X.t() * g;
    arma::mat X0 = c1 + c1.t();
    A = lyap_sym(Phi, X0);
    A(indexes_1).zeros();
    // A = syl(Phi, X0);
    // X0(indexes_1).zeros(); A = lyapunov(Phi, X0, indexes_1); A(indexes_1).zeros();
    // A = lyapunov_2(Phi, X0, indexes_1);
    arma::mat N = X * A;
    rg = g - N;

  }

  void hess(arma::mat& dH, arma::mat Z, arma::mat g, arma::mat dg, arma::mat X,
            arma::mat Phi, arma::mat A, arma::uvec indexes_1) {

    int q = X.n_cols;
    arma::mat c1 = Z.t() * g + X.t() * dg;
    arma::mat c2 = Z.t() * X;
    arma::mat c3 = A * (c2 + c2.t());
    arma::mat Q = c1 + c1.t() - c3 - c3.t();
    arma::mat dA = lyap_sym(Phi, Q);
    dA(indexes_1).zeros();
    // arma::mat dA = syl(Phi, Q);
    // Q(indexes_1).zeros(); arma::mat dA = lyapunov(Phi, Q, indexes_1); dA(indexes_1).zeros();
    // arma::mat dA = lyapunov_2(Phi, Q, indexes_1);
    arma::mat hessian = dg - (Z * A + X * dA);

    proj(dH, A, hessian, X, Phi, indexes_1);

  }

  void retr(arma::mat& X, std::vector<arma::uvec> oblq_indexes) {

    X *= diagmat(1 / sqrt(diagvec(X.t() * X)));

    arma::mat Q;
    arma::mat R;
    qr_econ(Q, R, X);

    arma::mat X2 = X;
    X = Q;
    X.cols(oblq_indexes[0]) = X2.cols(oblq_indexes[0]);

    int J = X.n_cols;
    int I = oblq_indexes.size()-1;

    // In the following loop, the Gram-Schmidt process is performed between blocks:

    for(int i=0; i < I; ++i) {

      // Select the cumulative indexes of blocks:
      std::vector<arma::uvec> indexes(&oblq_indexes[0], &oblq_indexes[i+1]);
      arma::uvec cum_indexes = list_to_vector(indexes);

      arma::mat orthogonals = Q.cols(cum_indexes);
      int n = orthogonals.n_cols;

      for(int j=0; j < oblq_indexes[i+1].size(); ++j) {

        // orthogonalize every column of the following block:
        int index = oblq_indexes[i+1][j];
        X.col(index) = orthogonalize(orthogonals, X2.col(index), n);

      }

    }

  }

};

// Retractions:

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

  int n_factors = arma::accu(oblq_blocks);
  if(n_factors > X.n_rows || n_factors > X.n_cols) Rcpp::stop("Too many factors declared in oblq_blocks");

  std::vector<arma::uvec> list_oblq_blocks = vector_to_list2(oblq_blocks);

  poblq RR;
  RR.retr(X, list_oblq_blocks);

  return X;

}

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

  int n_factors = arma::accu(oblq_blocks);
  if(n_factors > p || n_factors > q) Rcpp::stop("Too many factors declared in oblq_blocks");

  std::vector<arma::uvec> list_oblq_blocks = vector_to_list2(oblq_blocks);;

  arma::mat X(p, q, arma::fill::randn);
  poblq RR;
  RR.retr(X, list_oblq_blocks);

  return X;

}
