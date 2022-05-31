// Optimization algorithms for rotation

class rotation_optim {

public:

  virtual TRN optim(arguments_rotate x, rotation_manifold *manifold,
                     rotation_criterion *criterion) = 0;

};

// Gradient descent:

class RGD:public rotation_optim {

public:

  TRN optim(arguments_rotate x, rotation_manifold *manifold,
             rotation_criterion *criterion) {

    return gd(x, manifold, criterion);

  }

};

// Newton Trust-Region:

class RNTR:public rotation_optim {

public:

  TRN optim(arguments_rotate x, rotation_manifold *manifold,
             rotation_criterion *criterion) {

    return ntr(x, manifold, criterion);

  }

};

rotation_optim* choose_optim(std::string optim) {

  rotation_optim* algorithm;
  if(optim == "gradient") {
    algorithm = new RGD();
  } else if(optim == "newtonTR") {
    algorithm = new RNTR();
  } else {

    Rcpp::stop("Available optimization rutines for rotation: \n gradient, newtonTR");

  }

  return algorithm;

}
