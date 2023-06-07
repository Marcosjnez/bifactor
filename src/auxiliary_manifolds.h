/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 18/03/2022
 *
 */

// #include "structures.h"

arma::mat dxt(int p, int q) {

  /*
   * derivative of a matrix wrt its transpose
   */

  int pq = p*q;

  arma::mat res(pq, pq);
  arma::mat temp(p, q);

  for(int i=0; i < pq; ++i) {
    temp.zeros();
    temp(i) = 1;
    res.col(i) = arma::vectorise(temp.t(), 0);
  }

  return res;

}

arma::mat skew(arma::mat X) {

  // Skew-symmetric matrix

  return 0.5 * (X - X.t());

}

arma::mat symm(arma::mat X) {

  // Symmetric matrix

  return 0.5 * (X + X.t());

}

arma::mat lyap_sym(arma::mat Y, arma::mat Q) {

  // Solve the lyapunov equation YX + XY = Q with symmetric Q and X:

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

arma::uvec consecutive(int lower, int upper) {

  // Generate a sequence of integers from lower to upper

  int size = upper - lower + 1;
  arma::uvec ivec(size);
  std::iota(ivec.begin(), ivec.end(), lower);

  return ivec;
}

std::vector<arma::uvec> vector_to_list(arma::uvec v){

  // Pass a vector to a list

  int n = v.size();
  std::vector<arma::uvec> lista(n);
  v.insert_rows(0, 1);

  for(int i=0; i < n; ++i) {

    lista[i] = v[i] + consecutive(1, v[i+1]);

  }

  return lista;

}

std::vector<arma::uvec> vector_to_list2(arma::uvec v){

  // Pass a vector to a list of sequential vectors

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

arma::vec orthogonalize(arma::mat X, arma::vec x, int k) {

  // Make x orthogonal to every column of X

  for(int i=0; i < k; ++i) {

    // x -= arma::accu(X.col(i) % x) / arma::accu(X.col(i) % X.col(i)) * X.col(i);
    x -= arma::accu(X.col(i) % x) * X.col(i);

  }

  x /= sqrt(arma::accu(x % x));

  return x;

}

arma::uvec list_to_vector(std::vector<arma::uvec> X) {

  // Unlist to a vector

  arma::uvec single_vector = std::accumulate(X.begin(), X.end(),
                                             arma::uvec(), [](arma::uvec a, arma::uvec b) {
                                               a = arma::join_cols(a, b);
                                               return a;
                                             });

  return single_vector;

}

std::vector<arma::uvec> increment(arma::uvec oblq_indexes, int p) {

  arma::uvec oblq_indexes_total = oblq_indexes;
  int n_blocks = oblq_indexes.size();
  int total = arma::accu(oblq_indexes);
  if(p != total) {
    oblq_indexes_total.insert_rows(n_blocks, 1);
    oblq_indexes_total[n_blocks] = (p - total + 0.00);
  }
  std::vector<arma::uvec> indexes_list = vector_to_list2(oblq_indexes_total);

  return indexes_list;

}

std::vector<int> subvector(std::vector<int> v, int lower, int upper) {

  std::vector<int> subv(&v[lower], &v[upper]);

  return subv;

}

std::vector<std::vector<int>> subvectors(std::vector<std::vector<int>> v, int lower, int upper) {

  std::vector<std::vector<int>> subv(&v[lower], &v[upper]);

  return subv;

}

// Gram-Schmidt process:

arma::mat gram(arma::mat X) {

  int n = X.n_rows;
  int k = X.n_cols;
  X.col(0) /= sqrt(arma::accu(X.col(0) % X.col(0)));

  for(int i=1; i < k; ++i) {

    for(int j=0; j < i; ++j) {

      X.col(i) -= arma::accu(X.col(j) % X.col(i)) / arma::accu(X.col(j) % X.col(j)) * X.col(j);

    }

    X.col(i) /= sqrt(arma::accu(X.col(i) % X.col(i)));

  }

  return X;

}

