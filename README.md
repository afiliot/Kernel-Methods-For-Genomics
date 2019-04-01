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

We were finally ranked 24th/75 with an accuracy of 0.70733 (best accuracy 0.73066) on the public leaderboard (20% of test set). And 38th/75 with an accuracy of 0.67466 (best accuracy 0.71200) on the private leaderboard (100% of test set).

### References

Cortes, C.,  Mohri, M., and Rostamizadeh, A. Learning non-linear combinations of kernels. In *Advances in Neural Information Processing Systems 22 - Proceedings of the 2009 Conference*, pp. 396–404, 2009. ISBN9781615679119.

Cortes, C., Mohri, M., and Rostamizadeh, A. Algorithms for learning kernels based on centered alignment. *Journal of Machine Learning Research*, 13:795–828, 2012.

Leslie, C., Eskin, E., Weston, J., and Stafford Noble, W. Mismatch string kernels for svm protein classification. volume 20, pp. 1417–1424, 01 2002.

Sonnenburg, S., Rtsch, G., and Rieck, K. Large Scale Learning with String Kernels, pp. 73–104. 01 2007.

Vert, J., Saigo, H., and Akutsu, T. Local alignment kernels for biological sequences. *Kernel Methods in Computational Biology*, pp. 131–154, 01 2004.

