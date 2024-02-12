This a repository that contains the data preparation and analysis code for the article: Identification of important factors in dementia ascertainment in India and development of nation-specific cutoffs: A machine learning and diagnostic analysis 

There are two script files for R and Python. The R files contain a script for data preparation (data_preparation.qmd) to clean the data for use with XGBoost model. The second script (cutoff_identification) is the script used to analyse the various assessments and their cutoffs in this article.

The python files contain the code for the XGBoost models and their associated SHAP value calculations

This repo also contains a couple of .nix files. [Nix](https://nixos.org/) functions as a programming language, package manager, and operating system. It has a declarative nature that allows for reproducible programming environments aimed at solving dependency issues and ensuring a working environment across different machines. 

The code can be run independent of these files. To utilise nix it must be [installed](https://nixos.org/download) on either Linux, MacOS, or Windows using WSL2 though I am exploring an option to make it run completey independent of operating system.
