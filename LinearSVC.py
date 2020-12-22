import pandas as pd
import statistics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn import metrics


if __name__ == '__main__':
    train = pd.read_csv('VolleyballDataframe.csv')
    X_train = train.loc[train['Stagione'] <= 2014, 'C_bat_pos':'S_mur_neg']
    X_test = train.loc[train['Stagione'] > 2014, 'C_bat_pos':'S_mur_neg']
    y_train = train.loc[train['Stagione'] <= 2014, 'Playoff']
    y_test = train.loc[train['Stagione'] > 2014, 'Playoff']

    f1_scores = []
    C_max_scores = []

    # Loop 50 times and average.
    for i in range(50):

        # StratifiedShuffleSplit divides the initial training set into two other parts:
        # training set (80%) and test set (20%)
        cv = StratifiedShuffleSplit(n_splits=10, train_size=0.80, test_size=0.20)
        best_score = 0.0
        C_values = [2 ** i for i in range(-8, 8)]
        C_max = C_values[0]
        for C in C_values:
            print('C=' + str(C))

            # Linear SVC (an Linear SVM extension) to find C_max through Cross-Validation.
            svclinearmodel = LinearSVC(max_iter=20000, C=C)
            score = cross_val_score(svclinearmodel, X_train, y_train, cv=cv, scoring='f1')
            mean_score = statistics.mean(score)
            print('mean_score=' + str(mean_score))
            if mean_score > best_score:
                C_max = C
                best_score = mean_score
                print('Sto aggiornando C_max e Best_score')
            print()

        # Linear SVC with C_max.
        svclinearmodel = LinearSVC(max_iter=20000, C=C_max)
        svclinearmodel.fit(X_train, y_train)
        predictions = svclinearmodel.predict(X_test)
        print('f1 del C_max=' + str(metrics.f1_score(y_test, predictions)))
        print()
        f1_scores.append(metrics.f1_score(y_test, predictions))
        C_max_scores.append(C_max)

    print()
    print('La lista di f1=' + str(f1_scores))
    print('La lista di C_max=' + str(C_max_scores))

    # Calculate the f1 average.
    f1_avg = sum(f1_scores) / float(len(f1_scores))
    # Calculate the C_max average.
    C_max_avg = sum(C_max_scores) / float(len(C_max_scores))

    print('La media di f1=' + str(f1_avg))
    print('La media di C_max=' + str(C_max_avg))