# Machine learning methods for predictive analysis of team performance in sports

## Introduction
The objective of this study is to use data relating to the past sports performance of individual players in the Volleyball Serie A, to predict at the beginning of the championship which teams will access the final phase of the championship, the Playoffs, following the Regular Season phase. .
For this work, the data relating to the men's volleyball Serie A seasons from the 2001/02 to the 2017/18 season were taken into consideration.
Specifically, the data regarding the performance of each individual athlete season by season were considered. Each team is therefore represented by the set of players who make up the squad at the beginning of the season.
The aim of the work was to identify **supervised learning models** capable of predicting future events using information on past events.

<p align="center">
  <img src="/Readme_Documents/Use_Case_Example.gif" alt="Use Case Example"/>
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

### SVC with Linear Kernel
For this model, various parameters are available, the ones we have focused on most are:
- **C** is the penalty parameter of the error term. In our case it takes value
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?C&space;\in&space;[2^{-8},&space;2^{-7},&space;...,&space;2^{7}]" title="C \in [2^{-8}, 2^{-7}, ..., 2^{7}]" />
</p>
  
- **max_iter** indicates the maximum number of iterations for the solver to converge. In our case it was assigned a value of 20000.

- the other parameters have their default value

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


