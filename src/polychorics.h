#include <armadillo>
#include <iostream>
#include <array>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <unordered_map>

std::vector<int> count(const std::vector<int>& X, const int n, const int max_X){

  // Allocate memory space for table
  std::vector<int> frequency(max_X + 1);
  // std::vector<int> frequency;
  // frequency.reserve(max_X + 1);

  // Populate table
  for (int i = 0; i < n; i++) {
    frequency[X[i]]++;
  }

  // Return joint frequency table
  return frequency;
}

std::vector<std::vector<int>> joint_frequency_table(const std::vector<int>& X, const int n, const int max_X,
                                                    const std::vector<int>& Y, const int max_Y){

  // Allocate memory space for table
  std::vector<std::vector<int>> joint_frequency(max_X + 1L, std::vector<int>(max_Y + 1L, 0));

  // Populate table
  for (int i = 0; i < n; i++) {
    joint_frequency[X[i]][Y[i]]++;
  }

  // Return joint frequency table
  return joint_frequency;
}

double erf_inverse (double a) {
  double p, r, t;
  t = std::fmaf (a, 0.0f - a, 1.0f);
  t = std::log(t);
  if (fabsf(t) > 6.125f) { // maximum ulp error = 2.35793
    p =              3.03697567e-10f; //  0x1.4deb44p-32
    p = std::fmaf (p, t,  2.93243101e-8f); //  0x1.f7c9aep-26
    p = std::fmaf (p, t,  1.22150334e-6f); //  0x1.47e512p-20
    p = std::fmaf (p, t,  2.84108955e-5f); //  0x1.dca7dep-16
    p = std::fmaf (p, t,  3.93552968e-4f); //  0x1.9cab92p-12
    p = std::fmaf (p, t,  3.02698812e-3f); //  0x1.8cc0dep-9
    p = std::fmaf (p, t,  4.83185798e-3f); //  0x1.3ca920p-8
    p = std::fmaf (p, t, -2.64646143e-1f); // -0x1.0eff66p-2
    p = std::fmaf (p, t,  8.40016484e-1f); //  0x1.ae16a4p-1
  } else { // maximum ulp error = 2.35002
    p =              5.43877832e-9f;  //  0x1.75c000p-28
    p = std::fmaf (p, t,  1.43285448e-7f); //  0x1.33b402p-23
    p = std::fmaf (p, t,  1.22774793e-6f); //  0x1.499232p-20
    p = std::fmaf (p, t,  1.12963626e-7f); //  0x1.e52cd2p-24
    p = std::fmaf (p, t, -5.61530760e-5f); // -0x1.d70bd0p-15
    p = std::fmaf (p, t, -1.47697632e-4f); // -0x1.35be90p-13
    p = std::fmaf (p, t,  2.31468678e-3f); //  0x1.2f6400p-9
    p = std::fmaf (p, t,  1.15392581e-2f); //  0x1.7a1e50p-7
    p = std::fmaf (p, t, -2.32015476e-1f); // -0x1.db2aeep-3
    p = std::fmaf (p, t,  8.86226892e-1f); //  0x1.c5bf88p-1
  }
  r = a * p;
  return r;
}

double Qnorm(double p) {
  return M_SQRT2 * erf_inverse(2 * p - 1);
}

double MVPHI(double z) {
  return 0.5 * std::erfc(-z * M_SQRT1_2);
}

const double TWOPI = 6.283185307179586;
const std::array<std::array<double, 3>, 10> X = {{
  {{-0.9324695142031522, -0.9815606342467191, -0.9931285991850949}},
  {{-0.6612093864662647, -0.9041172563704750, -0.9639719272779138}},
  {{-0.2386191860831970, -0.7699026741943050, -0.9122344282513259}},
  {{ 0.0,              -0.5873179542866171, -0.8391169718222188}},
  {{ 0.0,              -0.3678314989981802, -0.7463319064601508}},
  {{ 0.0,              -0.1252334085114692, -0.6360536807265150}},
  {{ 0.0,               0.0,              -0.5108670019508271}},
  {{ 0.0,               0.0,              -0.3737060887154196}},
  {{ 0.0,               0.0,              -0.2277858511416451}},
  {{ 0.0,               0.0,              -0.07652652113349733}}
  }};

const std::array<std::array<double, 3>, 10> W = {{
  {{ 0.1713244923791705,  0.04717533638651177, 0.01761400713915212}},
  {{ 0.3607615730481384,  0.1069393259953183,  0.04060142980038694}},
  {{ 0.4679139345726904,  0.1600783285433464,  0.06267204833410906}},
  {{ 0.0,                0.2031674267230659,  0.08327674157670475}},
  {{ 0.0,                0.2334925365383547,  0.1019301198172404}},
  {{ 0.0,                0.2491470458134029,  0.1181945319615184}},
  {{ 0.0,                0.0,                0.1316886384491766}},
  {{ 0.0,                0.0,                0.1420961093183821}},
  {{ 0.0,                0.0,                0.1491729864726037}},
  {{ 0.0,                0.0,                0.1527533871307259}}
  }};

double genz(const double sh, const double sk, const double mvphi_h, const double mvphi_k, const double r) {

  int lg, ng;
  if (std::abs(r) < 0.3) {
    ng = 0;
    lg = 3;
  } else if (std::abs(r) < 0.75) {
    ng = 1;
    lg = 6;
  } else {
    ng = 2;
    lg = 10;
  }

  double h = sh;
  double k = sk;
  double hk = h * k;
  double bvn = 0.0;

  if (std::abs(r) < 0.925) {
    double hs = (h * h + k * k) / 2;
    double asr = std::asin(r);
    for (int i = 0; i < lg; ++i) {
      double sn = std::sin(asr * (X[i][ng] + 1) / 2);
      bvn += W[i][ng] * std::exp((sn * hk - hs) / (1 - sn * sn));
      sn = std::sin(asr * (-X[i][ng] + 1) / 2);
      bvn += W[i][ng] * std::exp((sn * hk - hs) / (1 - sn * sn));
    }
    bvn = bvn * asr / (2 * TWOPI) + (1-mvphi_h) * (1-mvphi_k);
  } else {
    if (r < 0) {
      k = -k;
      hk = -hk;
    }
    if (std::abs(r) < 1) {
      double as = (1 - r) * (1 + r);
      double a = std::sqrt(as);
      double bs = (h - k) * (h - k);
      double c = (4 - hk) / 8;
      double d = (12 - hk) / 16;
      bvn = a * std::exp(-(bs / as + hk) / 2)
        * (1 - c * (bs - as) * (1 - d * bs / 5) / 3 + c * d * as * as / 5);
      if (hk > -160) {
        double b = std::sqrt(bs);
        bvn -= std::exp(-hk / 2) * std::sqrt(TWOPI) * MVPHI(-b / a) * b
          * (1 - c * bs * (1 - d * bs / 5) / 3);
      }
      a = a / 2;
      for (int i = 0; i < lg; ++i) {
        double xs = std::pow(a * (X[i][ng] + 1), 2);
        double rs = std::sqrt(1 - xs);
        bvn += a * W[i][ng]
        * (std::exp(-bs / (2 * xs) - hk / (1 + rs)) / rs
             - std::exp(-(bs / xs + hk) / 2) * (1 + c * xs * (1 + d * xs)));
             xs = as * std::pow(-X[i][ng] + 1, 2) / 4;
             rs = std::sqrt(1 - xs);
             bvn += a * W[i][ng] * std::exp(-(bs / xs + hk) / 2)
               * (std::exp(-hk * (1 - rs) / (2 * (1 + rs))) / rs
                    - (1 + c * xs * (1 + d * xs)));
      }
      bvn = -bvn / TWOPI;
    }
    if (r > 0) {
      if(h > k) {
        bvn += 1-mvphi_h;
      } else {
        bvn += 1-mvphi_k;
      }
    }
    if (r < 0) {
      bvn = -bvn + std::max(0.0, (1-mvphi_h) - mvphi_k);
    }
  }

  return bvn;

}

const double neg_inf = -std::numeric_limits<double>::infinity();
const double pos_inf = std::numeric_limits<double>::infinity();

double bivariatenormal_pdf(double p, double x, double y) {

  /*
   * Function for the bivariate normal density
   */

  if(!std::isfinite(x) | !std::isfinite(y)) {
    return 0;
  }
  double z1 = x*x + y*y - 2*p*x*y;
  double p2 = p*p;
  double z2 = std::exp(-z1/2/(1-p2));
  double pd = 0.5*z2/M_PI/sqrt(1-p2);

  return pd;
}

double PBINORM(const double lower0, const double lower1, const double upper0, const double upper1, const double rho,
                const double mvphi0, const double mvphi1, const double mvphi2, const double mvphi3) {

  bool ll1 = lower0 == pos_inf;
  bool ll2 = lower1 == pos_inf;
  bool uu1 = upper0 == neg_inf;
  bool uu2 = upper1 == neg_inf;

  if(lower0 > upper0 || lower1 > upper1 || ll1 || ll2 || uu1 || uu2) {
    return 1.0;
  }

  bool l1 = lower0 == neg_inf;
  bool l2 = lower1 == neg_inf;
  bool u1 = upper0 == pos_inf;
  bool u2 = upper1 == pos_inf;

  if(l1) {
    if(l2) {
      if(u1) {
        // return std::normal_distribution<>{0.0, 1.0}(upper1);
        return mvphi3;
      }
      if(u2) {
        // return std::normal_distribution<>{0.0, 1.0}(upper0);
        return mvphi2;
      }
      return genz(-upper0, -upper1, 1.00-mvphi2, 1.00-mvphi3, rho);
    }
    if(u1) {
      if(u2) {
        // return std::normal_distribution<>{0.0, 1.0}(-lower1);
        return 1.00 - mvphi1;
      }
      // return std::normal_distribution<>{0.0, 1.0}(upper1) - std::normal_distribution<>{0.0, 1.0}(lower1);
      return mvphi3 - mvphi1;
    }
    if(u2) {
      return genz(-upper0, lower1, 1.00-mvphi2, mvphi1, -rho);
    }
    return genz(-upper0, -upper1, 1.00-mvphi2, 1-mvphi3, rho) -
      genz(-upper0, -lower1, 1.00-mvphi2, 1.00-mvphi1, rho);
  }

  if(u1) {
    if(u2) {
      if(l2) {
        // return std::normal_distribution<>{0.0, 1.0}(-lower0);
        return 1.00 - mvphi0;
      }
      return genz(lower0, lower1, mvphi0, mvphi1, rho);
    }
    if(l2) {
      return genz(-upper1, lower0, 1.00-mvphi3, mvphi0, -rho);
    }
    return genz(lower0, lower1, mvphi0, mvphi1, rho) -
      genz(lower0, upper1, mvphi0, mvphi3, rho);
  }

  if(l2) {
    if(u2) {
      // return std::normal_distribution<>{0.0, 1.0}(upper0) - std::normal_distribution<>{0.0, 1.0}(lower0);
      return mvphi2 - mvphi0;
    }
    return genz(-upper0, -upper1, 1.00-mvphi2, 1.00-mvphi3, rho) -
      genz(-lower0, -upper1, 1.00-mvphi0, 1.00-mvphi3, rho);
  }

  if(u2) {
    return genz(lower0, lower1, mvphi0, mvphi1, rho) -
      genz(upper0, lower1, mvphi2, mvphi1, rho);
  }

  return genz(upper0, upper1, mvphi2, mvphi3, rho) -
    genz(lower0, upper1, mvphi0, mvphi3, rho) -
    genz(upper0, lower1, mvphi2, mvphi1, rho) +
    genz(lower0, lower1, mvphi0, mvphi1, rho);
}

double fpoly(double p, const std::vector<double>& a, const std::vector<double>& b, const std::vector<std::vector<int>>& n,
              const size_t s1, const size_t s2, const std::vector<double>& mvphi1, const std::vector<double>& mvphi2) {

  double f = 0.0;
  for (size_t i = 0; i < s1; ++i) {
    for (size_t j = 0; j < s2; ++j) {
      f -= n[i][j] * std::log(PBINORM(a[i], b[j], a[i + 1], b[j + 1], p,
                                      mvphi1[i], mvphi2[j], mvphi1[i+1], mvphi2[j+1]));
    }
  }

  return f;

}

const double GOLDEN_RATIO = (3.0 - std::sqrt(5.0)) / 2.0;
constexpr double ZEPS = 1.0e-10;

std::vector<double> optimize(const std::vector<double>& tau1, const std::vector<double>& tau2, const std::vector<std::vector<int>>& n,
                              const size_t s1, const size_t s2, const std::vector<double>& mvphi1, const std::vector<double>& mvphi2,
                              const int nobs, const double cor) {

  // tau1 = Vector of thresholds for the first variable (It must start at -Infinite and end at Infinite)
  // tau2 = Vector of thresholds for the second variable (It must start at -Infinite and end at Infinite)
  // n =  Contingency table for the variables
  // s1 = Length of tau1 - 1L
  // s2 = Length of tau2 - 1L
  // mvphi1 = pnorm of tau1
  // mvphi1 = pnorm of tau2
  // nobs =  Sample size
  // cor = Initial value for the correlation

  double p = cor; // Parameters to be estimated
  double iteration = 1;

  // Start the iterative algorithm
  for(int i=0; i < 20L; ++i) {
    // double f = 0.0;  // Objective value (no needed)
    double g = 0.0;     // Gradient
    double score = 0.0; // Approximated Hessian

    for (size_t i = 0; i < s1; ++i) {
      for (size_t j = 0; j < s2; ++j) {
        // CDF of the bivariate normal:
        double prop = PBINORM(tau1[i], tau2[j], tau1[i + 1], tau2[j + 1], p,
                               mvphi1[i], mvphi2[j], mvphi1[i+1], mvphi2[j+1]);
        // PDF of the Bivariate normal:
        double gij = bivariatenormal_pdf(p, tau1[i+1], tau2[j+1]) -
          bivariatenormal_pdf(p, tau1[i], tau2[j+1]) -
          bivariatenormal_pdf(p, tau1[i+1], tau2[j]) +
          bivariatenormal_pdf(p, tau1[i], tau2[j]);
        // f -= n[i][j] * std::log(prop) / nobs; // No need to compute the objective value
        if(prop < 1e-09) prop = 1e-09; // Avoid division by zero
        g -= n[i][j] / prop * gij / nobs; // Update Gradient
        score += gij*gij / prop;          // Update Hessian
      }
    }
    double dir = g/score; // Approximated Newton's Descent direction
    p -= dir;             // Update parameter (no need for step-size)
    if(std::abs(g*g) < 1e-09) break; // Tolerance criteria
    ++ iteration;
  }

  return {p, iteration};

}

std::vector<double> cumsum(const std::vector<int> input) {
  std::vector<double> output(input.size());
  std::partial_sum(input.begin(), input.end(), output.begin());
  return output;
}

Rcpp::List poly(const arma::mat& X, const int cores) {

  /*
   * Function to estimate the full polychoric correlation matrix
   */

  Rcpp::Timer timer;

  const int n = X.n_rows;
  const int q = X.n_cols;

  arma::mat cor = arma::cor(X);
  std::vector<std::vector<int>> cols(q);
  std::vector<int> maxs(q);
  std::vector<std::vector<double>> taus(q);
  std::vector<size_t> s(q);
  std::vector<std::vector<double>> mvphi(q);

  for(size_t i = 0; i < q; ++i) {
    cols[i] = arma::conv_to<std::vector<int>>::from(X.col(i));
    maxs[i] = *max_element(cols[i].begin(), cols[i].end());
    std::vector<int> frequencies = count(cols[i], n, maxs[i]-1L);
    mvphi[i] = cumsum(frequencies);
    taus[i] = mvphi[i]; // Cumulative frequencies
    for (size_t j = 0; j < maxs[i]; ++j) {
      mvphi[i][j] /= n;
      taus[i][j] = Qnorm(mvphi[i][j]);
    }
    mvphi[i].push_back(1.0);
    mvphi[i].insert(mvphi[i].begin(), 0.0);
    taus[i].push_back(pos_inf);
    taus[i].insert(taus[i].begin(), neg_inf);
    s[i] = taus[i].size() -1L;
  }

  timer.step("precomputations");

  arma::mat polys(q, q, arma::fill::eye);
  arma::mat iters(q, q, arma::fill::zeros);

#ifdef _OPENMP
  omp_set_num_threads(cores);
#pragma omp parallel for
#endif
  for(size_t i=0; i < (q-1L); ++i) {
    for(int j=(i+1L); j < q; ++j) {
      std::vector<std::vector<int>> tab = joint_frequency_table(cols[i], n, maxs[i], cols[j], maxs[j]);
      std::vector<double> rho = optimize(taus[i], taus[j], tab, s[i], s[j], mvphi[i], mvphi[j], n, cor(i, j));
      polys(i, j) = polys(j, i) = rho[0];
      iters(i, j) = iters(j, i) = rho[1];
    }
  }

  timer.step("polychorics");

  Rcpp::List result;
  result["polychorics"] = polys;
  result["thresholds"] = taus;
  result["iters"] = iters;
  result["elapsed"] = timer;

  return result;

}


