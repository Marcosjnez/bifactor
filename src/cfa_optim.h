// Optimization algorithms for cfa

class cfa_optim {

public:

  virtual cfa_NTR optim(arguments_cfa x, cfa_manifold *manifold,
                    cfa_criterion *criterion) = 0;

};

// Riemannian gradient descent:

class RGD:public cfa_optim {

public:

  cfa_NTR optim(arguments_cfa x, cfa_manifold *manifold,
            cfa_criterion *criterion) {

    return gd(x, manifold, criterion);

  }

};

// Riemannian Newton Trust-Region:

class RNTR:public cfa_optim {

public:

  cfa_NTR optim(arguments_cfa x, cfa_manifold *manifold,
            cfa_criterion *criterion) {

    return ntr(x, manifold, criterion);

  }

};

// BFGS algorithm:

class BFGS:public cfa_optim {

public:

  cfa_NTR optim(arguments_cfa x, cfa_manifold *manifold,
            cfa_criterion *criterion) {

    return bfgs(x, manifold, criterion);

  }

};

// L-BFGS algorithm:

class LBFGS:public cfa_optim {

public:

  cfa_NTR optim(arguments_cfa x, cfa_manifold *manifold,
            cfa_criterion *criterion) {

    return lbfgs(x, manifold, criterion);

  }

};

// Select the optimization algorithm:

cfa_optim* choose_cfa_optim(std::string optim) {

  cfa_optim* algorithm;
  if(optim == "gradient") {
    algorithm = new RGD();
  } else if(optim == "newtonTR") {
    algorithm = new RNTR();
  } else if(optim == "BFGS") {
    algorithm = new BFGS();
  } else if(optim == "L-BFGS") {
    algorithm = new LBFGS();
  } else {

    Rcpp::stop("Available optimization rutines for cfa: \n 'gradient', 'BFGS', 'L-BFGS', 'newtonTR'. The default method is 'newtonTR'.");

  }

  return algorithm;

}
