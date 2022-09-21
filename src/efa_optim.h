// Optimization algorithms for factor analysis

class efa_optim {

public:

  virtual NTR optim(arguments_efa x, efa_manifold *manifold,
                    efa_criterion *criterion) = 0;

};

// Gradient descent:

class RGD:public efa_optim {

public:

  NTR optim(arguments_efa x, efa_manifold *manifold,
            efa_criterion *criterion) {

    return gd(x, manifold, criterion);

  }

};

// Newton Trust-Region:

class RNTR:public efa_optim {

public:

  NTR optim(arguments_efa x, efa_manifold *manifold,
            efa_criterion *criterion) {

    return ntr(x, manifold, criterion);

  }

};

efa_optim* choose_optim(std::string optim) {

  efa_optim* algorithm;
  if(optim == "gradient") {
    algorithm = new RGD();
  } else if(optim == "newtonTR") {
    algorithm = new RNTR();
  } else {

    Rcpp::stop("Available optimization rutines for factor extraction: \n gradient, newtonTR");

  }

  return algorithm;

}
