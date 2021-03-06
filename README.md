# Machine learning methods for predictive analysis of team performance in sports
[![MIT License](https://img.shields.io/badge/License-MIT-blueviolet)](https://opensource.org/licenses/mit-license.php)
[![Platform](https://img.shields.io/badge/MacOS-Catalina%2010.15.2-orange)](https://www.apple.com/it/newsroom/2019/10/macos-catalina-is-available-today/)
[![Library](https://img.shields.io/badge/numpy-1.19.4-informational)](https://pypi.org/project/numpy/)
[![Library](https://img.shields.io/badge/pandas-1.1.5-informational)](https://pypi.org/project/pandas/1.1.5/)
[![Library](https://img.shields.io/badge/scikit--learn-0.24.0-informational)](https://pypi.org/project/scikit-learn/)
[![Library](https://img.shields.io/badge/scipy-1.3.1-informational)](https://pypi.org/project/scipy/1.3.1/)

## Introduction
The objective of this study is to use data relating to the past sports performance of individual players in the Volleyball Serie A, to predict at the beginning of the championship which teams will access the final phase of the championship, the Playoffs, following the Regular Season phase. .
For this work, the data relating to the men's volleyball Serie A seasons from the 2001/02 to the 2017/18 season were taken into consideration.
Specifically, the data regarding the performance of each individual athlete season by season were considered. Each team is therefore represented by the set of players who make up the squad at the beginning of the season.
The aim of the work was to identify **supervised learning models** capable of predicting future events using information on past events.

<p align="center">
  <img src="/Readme_Documents/DatasetFragment.png" alt="Dataset Fragment" width="80%"/>
</p>

> Fragment of the dataset used

## Analysis of the Results
The statistical classifications (also called Metrics), obtained through the Confusion Matrix, which were used for this project are the following:
- **Accuracy**;
- **Balanced_Accuracy**;
- **Precision**;
- **Recall**;
- **F1_score**;
- **Error** (an introduced metric created specifically for this project).

## Supervised Learning Models Used
The supervised learning models that were used for this project are as follows:
- **Logistic Regression**;
- **SVC** with **linear kernel** (Support Vector Classification - an extension of SVM);
- **SVC**  with **RBF** (Radial Basis Function) **kenrel** (Support Vector Classification - an extension of SVM).

In particular, for the SVC model with RBF kernel four different implementations have been made (for more information read the `report.pdf`).

### Logistic Regression
For this model, various parameters are available, the ones we have focused on most are:
- **C** is the penalty parameter of the error term. In our case it takes value
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?C&space;\in&space;[2^{-8},&space;2^{-7},&space;...,&space;2^{7}]" title="C \in [2^{-8}, 2^{-7}, ..., 2^{7}]" />
</p>

- **solver** indicates the algorithm to be used in the optimization problem. In our case it is "lbfgs".

- **max_iter** indicates the maximum number of iterations for the solver to converge. In our case it was assigned a value of 200.

- the other parameters have their default value

#### *To Run this Model*
```sh
$ python3 LogisticRegression.py
```

### SVC with Linear Kernel
For this model, various parameters are available, the ones we have focused on most are:
- **C** is the penalty parameter of the error term. In our case it takes value
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?C&space;\in&space;[2^{-8},&space;2^{-7},&space;...,&space;2^{7}]" title="C \in [2^{-8}, 2^{-7}, ..., 2^{7}]" />
</p>
  
- **max_iter** indicates the maximum number of iterations for the solver to converge. In our case it was assigned a value of 20000.

- the other parameters have their default value

#### *To Run this Model*
```sh
$ python3 LinearSVC.py
```

### SVC with RBF Kernel
For this model, various parameters are available, the ones we have focused on most are:
- **C** is the penalty parameter of the error term. In our case it takes value
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?C&space;\in&space;[2^{-8},&space;2^{-7},&space;...,&space;2^{7}]" title="C \in [2^{-8}, 2^{-7}, ..., 2^{7}]" />
</p>

- **kernel** specifies the type of kernel to be used in the algorithm. It can be "linear", "poly", "rbf" and "sigmoid". In our case it has value "rbf" or Gaussian kernel.

- **gamma γ** is a kernel coefficient for "rbf" types. Possible values for this variable are
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\gamma&space;\in&space;\{&space;\gamma_{0}&space;*2^{i}&space;|&space;i&space;=&space;-8,&space;...,&space;7&space;\}&space;\quad&space;with&space;\quad&space;\gamma_{0}&space;=&space;\frac{1}{numeroFeatures}&space;=&space;\frac{1}{20}" title="\gamma \in \{ \gamma_{0} *2^{i} | i = -8, ..., 7 \} \quad with \quad \gamma_{0} = \frac{1}{numeroFeatures} = \frac{1}{20}" />
</p>

- the other parameters have their default value

#### *To Run this Model*
Four different implementations of this model have been created (for more information see the `report.pdf`)

- To run the **third implementation**
```sh
$ python3 NoLinearSVC.py
```
- To run the **fourth implementation**
```sh
$ python3 NoLinearSVC_with_Probability.py
```

#### Example of Output

<p align="center">
  <img src="/Readme_Documents/OutputSVC.png" alt="Output SVC Example" width="80%"/>
</p>

> Example of output for the 2008 test with this implementation

## Comparison of the Results
The comparison of the results obtained with the various models was made in terms of the F1_score metric. Below is the table summarizing the results obtained:

MODEL | F1_score
------------- | -------------
**Logistic Regression**  | 75,9%
**SVC with Linear Kernel**  | 73,3%
**SVC with RBF Kernel first implementation** | 82,5%
**SVC with RBF Kernel second implementation**  | 80,9%
**SVC with RBF Kernel third implementation**  | 81,4%
**SVC with RBF Kernel fourth implementation**  | 81%

## Libraries Needed
To run the code you need the following libraries:

Library | Version
------------- | -------------
**numpy**  | >= 1.19.4
**pandas** | >= 1.1.5
**scikit-learn**  | >= 0.24.0
**scipy**  | >= 1.3.1

The code has been tested with MacOS Catalina (version 10.15.2).

## License
MIT License. See [LICENSE](LICENSE) file for further information.
