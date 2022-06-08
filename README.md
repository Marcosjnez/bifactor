# bifactor: Exploratory Factor Analysis Models and Tools

Provides general purpose tools to fit exploratory factor, bi-factor, and bi-factor models with multiple general factors.

# Installation in Linux and Windows

Using the `devtools` package:
    
    devtools::install_github("marcosjnez/bifactor")

# Installation in macOS

To install the `bifactor` package in macOS, you need to configure the C++ toolchain so that C++ code can be compiled from R. Try the following installer: https://github.com/rmacoslib/r-macos-rtools/releases/tag/v4.0.0.

If the installer does not succeed, you may follow this tutorial: https://thecoatlessprofessor.com/programming/cpp/r-compiler-tools-for-rcpp-on-macos/.
Briefly, install `Xcode`from App Store. Then, install the package manager Homebrew and GCC by typing the following lines in the Terminal:

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    brew install gcc
    
Next, download and install the appropriate gfortran binary from https://github.com/fxcoudert/gfortran-for-macOS/releases.

You also need to enable OpenMP to allow parallelization: https://mac.r-project.org/openmp/.
Briely, this involves downloading and installing OpenMP. For this, type the following lines in the terminal:

    curl -O https://mac.r-project.org/openmp/openmp-12.0.1-darwin20-Release.tar.gz
    sudo tar fvxz openmp-12.0.1-darwin20-Release.tar.gz -C /
    
Finally, add the following lines to ~/.R/Makevars:

    CPPFLAGS += -Xclang -fopenmp
    LDFLAGS += -lomp
