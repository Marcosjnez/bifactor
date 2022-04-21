# bifactor: Exploratory Factor Analysis Models and Tools

Provides general purpose tools to fit exploratory factor, bi-factor, and generalized bi-factor models.

# Installation in Linux and Windows

Using the `remotes` package:

    install.packages("remotes")
    remotes::install_github("marcosjnez/bifactor")
    
Using the `devtools` package:
    
    install.packages("devtools")
    devtools::install_github("marcosjnez/bifactor")

# Installation in macOS

Install Homebrew from the terminal:

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"'
    eval "$(/opt/homebrew/bin/brew shellenv)"                                       

Install `libomp`, `lgfortran`, `libgomp`, `llvm` and `gettext` from the terminal:

    brew install libomp   
    brew install lgfortran
    brew install libgomp
    brew install llvm
    brew install gettext

Install `xcode`:

    xcode-select --install

Remove the Makevars and Renviron files from the R console:

    unlink("~/.R/Makevars")
    unlink("~/.Renviron")

Download and install `gfortran` from https://github.com/fxcoudert/gfortran-for-macOS/releases for your macOS version.

Create a file named Makevars in the directory ~/.R/:

    sudo mkdir .R
    sudo echo < .R/Makevars

Open the Makevars file and paste the following lines in it:

    LOC = /usr/local/gfortran/
    CC=$(LOC)/bin/gcc -fopenmp
    CXX=$(LOC)/bin/g++ -fopenmp
    # -O3 should be faster than -O2 (default) level optimisation ..
    CFLAGS=-g -O3 -Wall -pedantic -std=gnu99 -mtune=native -pipe
    CXXFLAGS=-g -O3 -Wall -pedantic -std=c++11 -mtune=native -pipe
    LDFLAGS=-L/usr/local/opt/gettext/lib -L$(LOC)/lib -Wl,-rpath,$(LOC)/lib
    CPPFLAGS=-I/usr/local/opt/gettext/include -I$(LOC)/include
    -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include

Install `Rcpp` and `RcppArmadillo` from the R console:

    install.packages(c('Rcpp', 'RcppArmadillo'))

Install `bifactor`:

    devtools::install_github("marcosjnez/bifactor")
