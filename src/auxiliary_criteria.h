/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 18/03/2022
 *
 */

arma::mat kdiag(arma::mat X) {

  /*
   * Transform every column into a diagonal matrix and bind the results
   */

  int pq = X.n_rows;
  int q = X.n_cols;
  int p = pq/q;

  arma::mat res2(pq, 0);

  for(int j=0; j < q; ++j) {

    arma::mat res1(0, p);

    for(int i=0; i < q; ++i) {
      int index_1 = i*p;
      int index_2 = index_1 + (p-1);
      arma::mat temp = arma::diagmat(X(arma::span(index_1, index_2), j));
      res1 = arma::join_cols(res1, temp);
    }

    res2 = arma::join_rows(res2, res1);

  }

  return res2;

}

arma::mat cbind_diag(arma::mat X) {

  /*
   * Transform every column into a diagonal matrix and bind
   */

  int p = X.n_rows;
  int q = X.n_cols;
  arma::mat res(p, 0);

  for(int i=0; i < q; ++i) {
    res = arma::join_rows(res, arma::diagmat(X.col(i)));
  }

  return res;

}

arma::mat bc(int g) {

  // Generate a matrix with all the combination of pairs

  int k = g*(g-1)/2;
  arma::mat Ng(g, k);

  int i = 0;

  for(int k=0; k < (g-1); ++k) {
    for(int j=k+1; j < g; ++j) {

      Ng(k, i) = 1;
      Ng(j, i) = 1;
      ++i;

    }
  }

  return Ng;

}
