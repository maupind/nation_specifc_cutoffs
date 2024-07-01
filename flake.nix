{
  description = "A flake to reproduce analysis packages";

 # flake.nix
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = {nixpkgs, ...}: let
    system = "x86_64-linux";
    #       ↑ Swap it for your system if needed
    #       "aarch64-linux" / "x86_64-darwin" / "aarch64-darwin"
    pkgs = import nixpkgs { system = "x86_64-linux"; config.allowUnfree = true;};
    git_archive_pkgs = [(pkgs.rPackages.buildRPackage {
    name = "tidyverse";
    src = pkgs.fetchgit {
      url = "https://github.com/tidyverse/tidyverse";
      branchName = "main";
      rev = "8ec2e1ffb739da925952b779925bb806bba8ff99";
      sha256 = "sha256-HTv0f3arq7KagwoKKnhEFT4NMA/LNd/zAmqabASy11g=";
    };
    propagatedBuildInputs = builtins.attrValues {
      inherit (pkgs.rPackages) broom conflicted cli dbplyr dplyr dtplyr forcats ggplot2 googledrive googlesheets4 haven hms httr jsonlite lubridate magrittr modelr pillar purrr ragg readr readxl reprex rlang rstudioapi rvest stringr tibble tidyr xml2;
    };
  })
(pkgs.rPackages.buildRPackage {
    name = "pROC";
    src = pkgs.fetchgit {
      url = "https://github.com/cran/pROC";
      branchName = "master";
      rev = "f47943152318bcc286cd3a0e1147c0fc8f3256a4";
      sha256 = "sha256-oPJg42M7X33HoiF+Qg5HXvZwg2paHIlJ4OQ/aLxXIoU=";
    };
    propagatedBuildInputs = builtins.attrValues {
      inherit (pkgs.rPackages) plyr Rcpp;
    };
  }) (pkgs.rPackages.buildRPackage {
    name = "haven";
    src = pkgs.fetchzip {
      url = "https://cran.r-project.org/src/contrib/Archive/haven/haven_2.5.2.tar.gz";
      sha256 = "sha256-tXf6NY4yjh42YMjiy+WtI11Eo8gR+b3enplMqRx9ucU=";
    };
    propagatedBuildInputs = builtins.attrValues {
      inherit (pkgs.rPackages) cli forcats hms lifecycle readr rlang tibble tidyselect vctrs cpp11;
    };
    nativeBuildInputs = with pkgs; [pkg-config zlib];
  })
(pkgs.rPackages.buildRPackage {
    name = "feather";
    src = pkgs.fetchzip {
      url = "https://cran.r-project.org/src/contrib/feather_0.3.5.tar.gz";
      sha256 = "sha256-lpVVXsKvQZpk34LIhjS/PbIY513O0E9mR9cq3eTNqQ8=";
    };
    propagatedBuildInputs = builtins.attrValues {
      inherit (pkgs.rPackages) Rcpp tibble hms;
    };
  })
(pkgs.rPackages.buildRPackage {
    name = "car";
    src = pkgs.fetchzip {
      url = "https://cran.r-project.org/src/contrib/car_3.1-2.tar.gz";
      sha256 = "sha256-Sh2ykojEQJYpYMleHAPAHPnJEYuiFHO0jlC7dOP5CyU=";
    };
    propagatedBuildInputs = builtins.attrValues {
      inherit (pkgs.rPackages) carData abind MASS mgcv nnet pbkrtest quantreg lme4 nlme scales;
    };
  })
(pkgs.rPackages.buildRPackage {
    name = "yardstick";
    src = pkgs.fetchzip {
      url = "https://cran.r-project.org/src/contrib/yardstick_1.3.1.tar.gz";
      sha256 = "sha256-tGcjmaHQnmzAenLFMRZu7wXbPVQG83j3R2IMzJ6piCQ=";
    };
    propagatedBuildInputs = builtins.attrValues {
      inherit (pkgs.rPackages) dplyr generics hardhat lifecycle rlang tibble tidyselect vctrs withr;
    };
  })
(pkgs.rPackages.buildRPackage {
    name = "psych";
    src = pkgs.fetchzip {
      url = "https://cran.r-project.org/src/contrib/Archive/psych/psych_2.3.3.tar.gz";
      sha256 = "sha256-0yY7iAHYnBh7AXruXN82ZFG/RLwmcLJVDYOvTaAfGDg=";
    };
    propagatedBuildInputs = builtins.attrValues {
      inherit (pkgs.rPackages) mnormt lattice nlme;
    };
  })
(pkgs.rPackages.buildRPackage {
    name = "DT";
    src = pkgs.fetchzip {
      url = "https://cran.r-project.org/src/contrib/Archive/DT/DT_0.28.tar.gz";
      sha256 = "sha256-w7ggoa9gR1vamf7+rkorsZj8hvJ3Svx888eDC6EUGMM=";
    };
    propagatedBuildInputs = builtins.attrValues {
      inherit (pkgs.rPackages) htmltools htmlwidgets jsonlite magrittr crosstalk jquerylib promises;
    };
  })
(pkgs.rPackages.buildRPackage {
    name = "survey";
    src = pkgs.fetchzip {
      url = "https://cran.r-project.org/src/contrib/survey_4.4-2.tar.gz";
      sha256 = "sha256-t3IQagejvGDzWXyO8yaMI4jO5TJ7mooJCF4b96dqXQw=";
    };
    propagatedBuildInputs = builtins.attrValues {
      inherit (pkgs.rPackages) lattice minqa numDeriv mitools Rcpp survival RcppArmadillo;
    };
  })
(pkgs.rPackages.buildRPackage {
    name = "caret";
    src = pkgs.fetchzip {
      url = "https://cran.r-project.org/src/contrib/caret_6.0-94.tar.gz";
      sha256 = "sha256-ZF+SmMH/qEfl0B7dQdMN3RNvtassqeyymVr3qNQF7Ig=";
    };
    propagatedBuildInputs = builtins.attrValues {
      inherit (pkgs.rPackages) ggplot2 lattice e1071 foreach ModelMetrics nlme plyr pROC recipes reshape2 withr;
    };
  })
 (pkgs.rPackages.buildRPackage {
    name = "pandoc";
    src = pkgs.fetchzip {
      url = "https://cran.r-project.org/src/contrib/pandoc_0.2.0.tar.gz";
      sha256 = "sha256-QrpW4hR9uVvw3Xqf69wgjel3wqt3feaPvmf3rMRoNOM=";
    };
    propagatedBuildInputs = builtins.attrValues {
      inherit (pkgs.rPackages) fs rappdirs rlang;
    };
  })
   (pkgs.rPackages.buildRPackage {
    name = "mltools";
    src = pkgs.fetchzip {
      url = "https://cran.r-project.org/src/contrib/mltools_0.3.5.tar.gz";
      sha256 = "sha256-dDvYchT7QB4JgwFYcoyucl4dE8qyV140bFpDzGGvb5o=";
    };
    propagatedBuildInputs = builtins.attrValues {
      inherit (pkgs.rPackages) data_table Matrix;
    };
  })
#(pkgs.rPackages.buildRPackage {
#    name = "gt";
#    src = pkgs.fetchzip {
#      url = "https://cran.r-project.org/src/contrib/gt_0.10.0.tar.gz";
#      sha256 = "sha256-sBOuWiZH9VhARAlzvjgONUR/+/xoEJsAbmdK2AXk6mQ=";
#    };
#    propagatedBuildInputs = builtins.attrValues {
#      inherit (pkgs.rPackages) base64enc bigD bitops cli commonmark dplyr fs glue htmltools htmlwidgets juicyjuice magrittr markdown reactable rlang sass scales tibble tidyselect xml2;
#    };
#   })
 (pkgs.rPackages.buildRPackage {
    name = "gtExtras";
    src = pkgs.fetchzip {
      url = "https://cran.r-project.org/src/contrib/gtExtras_0.5.0.tar.gz";
      sha256 = "sha256-7tHcXw5rYGw6HlsM1Zl/O5imfjBsQvLraGFbA0yJ35s=";
    };
    propagatedBuildInputs = builtins.attrValues {
      inherit (pkgs.rPackages) gt commonmark dplyr fontawesome ggplot2 glue htmltools paletteer rlang scales knitr cli;
    };
   })
(pkgs.rPackages.buildRPackage {
    name = "gtsummary";
    src = pkgs.fetchzip {
      url = "https://cran.r-project.org/src/contrib/gtsummary_1.7.2.tar.gz";
      sha256 = "sha256-O8ZOrvPD7OeKtI1GnCTkmxvVXZpHRPLcCd35L3GihMQ=";
    };
    propagatedBuildInputs = builtins.attrValues {
      inherit (pkgs.rPackages) broom broom_helpers cli dplyr forcats glue gt knitr lifecycle purrr rlang stringr tibble tidyr vctrs;
    };
   })
   (pkgs.rPackages.buildRPackage {
    name = "cutpointr";
    src = pkgs.fetchzip {
      url = "https://cran.r-project.org/src/contrib/cutpointr_1.1.2.tar.gz";
      sha256 = "sha256-Ks0ZWSCYyFrOhZY2QC+1TiVCVtTu4oNWJ4S3RiXL/68=";
    };
    propagatedBuildInputs = builtins.attrValues {
      inherit (pkgs.rPackages) gridExtra foreach dplyr tidyselect tidyr purrr tibble ggplot2 Rcpp rlang;
    };
   })
#   (pkgs.rPackages.buildRPackage {
#    name = "WRS2";
#    src = pkgs.fetchzip {
#      url = "https://cran.r-project.org/src/contrib/WRS2_1.1-5.tar.gz";
#      sha256 = "sha256-vZL1izWHKZSWr4OrR/qXnTFo/GsZEaMV1l4FduY29b8=";
#    };
#    propagatedBuildInputs = builtins.attrValues {
#      inherit (pkgs.rPackages) MASS reshape plyr;
#    };
#   })
#(pkgs.rPackages.buildRPackage {
#    name = "flextable";
#    src = pkgs.fetchzip {
#      url = "https://cran.r-project.org/src/contrib/flextable_0.9.4.tar.gz";
#      sha256 = "sha256-2lVvyMYr33pN6DiSHbdrpr/uYCNZvvj4JYAdmONpBpc=";
#    };
#    propagatedBuildInputs = builtins.attrValues {
#      inherit (pkgs.rPackages) rmarkdown knitr htmltools data_table rlang ragg officer gdtools xml2 uuid;
#    };
#  })
];
 tex = (pkgs.texlive.combine {
  inherit (pkgs.texlive) scheme-basic ;
});
 system_packages = builtins.attrValues {
  inherit (pkgs) R zlib pkg-config;
};
 rstudio_pkgs = pkgs.rstudioWrapper.override {
  packages = [ git_archive_pkgs 
  ];
};
    
  in {
    devShells.${system}.default = pkgs.mkShell {

      packages = [ 
		pkgs.python311
		pkgs.poetry
		pkgs.python311Packages.xgboost
		pkgs.python311Packages.pyarrow
		pkgs.python311Packages.packaging
		pkgs.python311Packages.pip
		pkgs.python311Packages.numba
		pkgs.python311Packages.numpy
		pkgs.python311Packages.shap
		pkgs.python311Packages.ipykernel
		pkgs.python311Packages.jupyter-core
		pkgs.python311Packages.ipywidgets
		pkgs.python311Packages.scikit-learn
		pkgs.python311Packages.notebook
		pkgs.python311Packages.matplotlib
		pkgs.python311Packages.joblib
		pkgs.python311Packages.botorch
		pkgs.python311Packages.torch
		git_archive_pkgs
          	system_packages
          	rstudio_pkgs
	 ];
	 

        # Workaround in linux: python downloads ELF's that can't find glibc
  # You would see errors like: error while loading shared libraries: name.so: cannot open shared object file: No such file or directory
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc
    #pkgs.lib
    # Add any missing library needed
    # You can use the nix-index package to locate them, e.g. nix-locate -w --top-level --at-root /lib/libudev.so.1
  ];
  
  
  # Put the venv on the repo, so direnv can access it
  POETRY_VIRTUALENVS_IN_PROJECT = "true";
  POETRY_VIRTUALENVS_PATH = "{project-dir}/.venv";
  
  # Use python from path, so you can use a different version to the one bundled with poetry
  POETRY_VIRTUALENVS_PREFER_ACTIVE_PYTHON = "true";


    };
  };
}
