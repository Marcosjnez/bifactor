// Optimization algorithms for cfa

class cfa_optim {

public:

  virtual void optim(arguments_cfa& x, cfa_manifold *manifold,
                    cfa_criterion *criterion) = 0;

};

// Riemannian gradient descent:

class cfa_RGD:public cfa_optim {

public:

  void optim(arguments_cfa& x, cfa_manifold *manifold,
            cfa_criterion *criterion) {

    gd(x, manifold, criterion);

  }

};

// Riemannian Newton Trust-Region:

class cfa_RNTR:public cfa_optim {

public:

  void optim(arguments_cfa& x, cfa_manifold *manifold,
            cfa_criterion *criterion) {

    ntr(x, manifold, criterion);

  }

};

// BFGS algorithm:

class cfa_BFGS:public cfa_optim {

public:

  void optim(arguments_cfa& x, cfa_manifold *manifold,
            cfa_criterion *criterion) {

    bfgs(x, manifold, criterion);

  }

};

// L-BFGS algorithm:

class cfa_LBFGS:public cfa_optim {

public:

  void optim(arguments_cfa& x, cfa_manifold *manifold,
            cfa_criterion *criterion) {

    lbfgs(x, manifold, criterion);

  }

};

// Select the optimization algorithm:

cfa_optim* choose_cfa_optim(std::string optim) {

  cfa_optim* algorithm;
  if(optim == "gradient") {
    algorithm = new cfa_RGD();
  } else if(optim == "newtonTR") {
    algorithm = new cfa_RNTR();
  } else if(optim == "BFGS") {
    algorithm = new cfa_BFGS();
  } else if(optim == "L-BFGS") {
    algorithm = new cfa_LBFGS();
  } else {

    Rcpp::stop("Available optimization rutines for cfa: \n 'gradient', 'BFGS', 'L-BFGS', 'newtonTR'. The default method is 'newtonTR'.");

  }

  return algorithm;

}
