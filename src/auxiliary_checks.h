/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 18/03/2022
 *
 */

arma::mat diagg(int q) {

  arma::mat D(q, q, arma::fill::eye);
  arma::mat H(q, q*q, arma::fill::zeros);
  arma::uvec indexes = arma::find(D == 1);
  for(int i=0; i < q; ++i) H(i, indexes[i]) = 1;

  return H;

}

arma::mat zeros(arma::mat X, std::vector<arma::uvec> indexes) {

  int I = indexes.size();
  arma::mat X0 = X;

  for(int i=0; i < I; ++i) {

    X0(indexes[i], indexes[i]).zeros();

  }

  X0.diag() = X.diag();

  return X0;

}

arma::vec new_k(std::vector<std::string> x, std::string y, arma::vec k) {

  // Resize gamma, k and epsilon

  int k_size = k.size();
  int x_size = x.size();

  std::vector<int> id; // Setup storage for found IDs

  for(int i =0; i < x.size(); i++) // Loop through input
    if(x[i] == y) {// check if input matches target
      id.push_back(i);
    }

    arma::uvec indexes = arma::conv_to<arma::uvec>::from(id);
    std::vector<double> ks = arma::conv_to<std::vector<double>>::from(k);

    if(k_size < x_size) {
      ks.resize(x_size, k.back());
      k = arma::conv_to<arma::vec>::from(ks);
      k(indexes) = k(arma::span(0, indexes.size()-1));
    }

    return k; // send locations to R (c++ index shift!)
}

arma::mat tc(int g) {

  // Generate a matrix with all the combination of triads

  if(g == 0) {
    arma::mat Ng; return Ng;
  } else if(g == 1) {
    arma::mat Ng(1, 1, arma::fill::ones); return Ng;
  } else if(g == 2) {
    arma::mat Ng(2, 1, arma::fill::ones); return Ng;
  }

  int k = g*(g-1)*(g-2)/6;
  arma::mat Ng(g, k);

  int i = 0;

  for(int k=0; k < (g-2); ++k) {
    for(int j=k+1; j < (g-1); ++j) {
      for(int l=j+1; l < g; ++l) {

        Ng(k, i) = 1;
        Ng(j, i) = 1;
        Ng(l, i) = 1;
        ++i;

      }
    }
  }

  return Ng;

}
