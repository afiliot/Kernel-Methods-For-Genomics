## Kernel Methods for Machine Learning - Data Challenge - Spring 2019

### Classification task
This repo contains our work within the scope of the Data Challenge https://www.kaggle.com/c/kernel-methods-for-machine-learning-2018-2019 
proposed as part of the Kernel Methods for Machine Learning course (MVA MSc). The objective is a classification task: predicting whether a DNA sequence region is binding site
to a specific transcription factor.Transcription factors (TFs) are regulatory proteins that bind specific sequence motifs in the genome
to activate or repress transcription of target genes. Genome-wide protein-DNA binding maps can be profiled using some experimental techniques
and thus all genomics can be classified into two classes for a TF of interest: bound or unbound. In this challenge, we will work with three
datasets corresponding to three different TFs. The **most important rule** is not to make use of external machine learning libraries (libsvm, liblinear, scikit-learn, ...).


### Structure
The program ``run.py`` aims at reproducing our best submission. ``kernels.py`` contains all of the kernel methods we used. ``KLR.py, KRR.py, SVM.py`` are respectively dedicated to implementing Kernel Logistic and Ridge Regression and Support Vector Machine algorithms. The more complex models used like ALIGNF and NLC Kernels are implemented in the corresponding Python files. At last, ``utils.py`` contains useful methods to run the experiments properly. Those experiments can be found in ``main.py``. The folders ``Data`` provides the data sets and results of cross-validations for each model. 

### Ranking

We finally ranked 
