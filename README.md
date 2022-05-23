# bifactor: Exploratory Factor Analysis Models and Tools

Provides general purpose tools to fit exploratory factor, bi-factor, and bi-factor models with multiple general factors.

# Installation in Linux and Windows

Using the `remotes` package:

    install.packages("remotes")
    remotes::install_github("marcosjnez/bifactor")
    
Using the `devtools` package:
    
    install.packages("devtools")
    devtools::install_github("marcosjnez/bifactor")

# Installation in macOS

In order to install the `bifactor` package in macOS, you need to configure the toolchain. Try the following installer: https://github.com/rmacoslib/r-macos-rtools/releases/tag/v4.0.0.

If you cannot execute de installer, you may follow this tutorial: https://thecoatlessprofessor.com/programming/cpp/r-compiler-tools-for-rcpp-on-macos/.

You also need to enable OpenMP to allow parallelization: https://mac.r-project.org/openmp/.
