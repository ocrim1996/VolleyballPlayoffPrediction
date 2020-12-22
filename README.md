# Machine learning methods for predictive analysis of team performance in sports

## Introduction
The objective of this study is to use data relating to the past sports performance of individual players in the Volleyball Serie A, to predict at the beginning of the championship which teams will access the final phase of the championship, the Playoffs, following the Regular Season phase. .
For this work, the data relating to the men's volleyball Serie A seasons from the 2001/02 to the 2017/18 season were taken into consideration.
Specifically, the data regarding the performance of each individual athlete season by season were considered. Each team is therefore represented by the set of players who make up the squad at the beginning of the season.
The aim of the work was to identify **supervised learning models** capable of predicting future events using information on past events.

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
- **SVC** (an extension of SVM) with **linear kernel**;
- **SVC** (an extension of SVM) with **RBF** (Radial Basis Function) **kenrel**.

In particular, for the SVC model with RBF kernel four different implementations have been made (for more information read the `report.pdf`).

### Logistic Regression
For this model, various parameters are available, the ones we have focused on most are:
- **C** is the penalty parameter of the error term. In our case it takes value
```math
C \in [2^{-8}, 2^{-7}, ..., 2^{7}]
```
- 



