import pandas as pd
import statistics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import metrics

if __name__ == '__main__':
    train = pd.read_csv('VolleyballDataframe.csv')
    f1_scores_total = []
    accuracy_scores_total = []
    balanced_accuracy_scores_total = []
    precision_scores_total = []
    recall_scores_total = []

    # A cycle that at each iteration runs a different year,
    # the year of the i-th iteration makes up the test set and the others are training set.
    for anno in range(2001, 2018):
        print('TEST SU STAGIONE: '+str(anno))
        print()
        X_train = train.loc[train['Stagione'] != anno, 'C_bat_pos':'S_mur_neg']
        X_test = train.loc[train['Stagione'] == anno, 'C_bat_pos':'S_mur_neg']
        y_train = train.loc[train['Stagione'] != anno, 'Playoff']
        y_test = train.loc[train['Stagione'] == anno, 'Playoff']
        values = []

        # StratifiedShuffleSplit divides the initial training set into two other parts:
        # training set (80%) and test set (20%)
        cv = StratifiedShuffleSplit(n_splits=10, train_size=0.80, test_size=0.20)
        best_score = 0.0
        C_values = [2 ** i for i in range(-8, 8)]
        C_max = C_values[0]
        gamma0 = 1/20
        gamma_values = [gamma0 * (2 ** i) for i in range(-8, 8)]
        gamma_max = gamma_values[0]
        for C in C_values:
            for gamma in gamma_values:

                # No Linear SVC (an No Linear SVM extension) to find C_max and gamma_max through Cross-Validation.
                svcnolinearmodel = SVC(C=C, gamma=gamma)
                score = cross_val_score(svcnolinearmodel, X_train, y_train, cv=cv, scoring='f1')
                mean_score = statistics.mean(score)
                if mean_score > best_score:
                    C_max = C
                    gamma_max = gamma
                    best_score = mean_score
        values.append((C_max, gamma_max))

        # No Linear SVC with C_max and gamma_max.
        svcnolinearmodel = SVC(C=C_max, gamma=gamma_max)
        svcnolinearmodel.fit(X_train, y_train)
        predictions = svcnolinearmodel.predict(X_test)
        df = pd.DataFrame(train.loc[train['Stagione'] == anno, 'Squadra'])
        df['Stagione'] = train.loc[train['Stagione'] == anno, 'Stagione']
        df['PredictionPlayoff'] = predictions
        df['RealPlayoff'] = train.loc[train['Stagione'] == anno, 'Playoff']
        print(df.to_string())
        print()

        # Calculation of the various metrics.
        # f1
        f1 = metrics.f1_score(y_test, predictions)
        f1_scores_total.append(f1)

        # accuracy
        accuracy = metrics.accuracy_score(y_test, predictions)
        accuracy_scores_total.append(accuracy)

        # balanced_accuracy
        balanced_accuracy = metrics.balanced_accuracy_score(y_test, predictions)
        balanced_accuracy_scores_total.append(balanced_accuracy)

        # precision
        precision = metrics.precision_score(y_test, predictions)
        precision_scores_total.append(precision)

        # recall
        recall = metrics.recall_score(y_test, predictions)
        recall_scores_total.append(recall)

        print('f1 del C_max e gamma_max=' + str(f1))
        print('accuracy del C_max e gamma_max=' + str(accuracy))
        print('balanced_accuracy del C_max e gamma_max=' + str(balanced_accuracy))
        print('precision del C_max e gamma_max=' + str(precision))
        print('recall del C_max e gamma_max=' + str(recall))
        print()
        print('C_max e gamma_max=' + str(values))
        print()

    f1_avg_total = sum(f1_scores_total) / float(len(f1_scores_total))
    accuracy_avg_total = sum(accuracy_scores_total) / float(len(accuracy_scores_total))
    balanced_accuracy_avg_total = sum(balanced_accuracy_scores_total) / float(len(balanced_accuracy_scores_total))
    precision_avg_total = sum(precision_scores_total) / float(len(precision_scores_total))
    recall_avg_total = sum(recall_scores_total) / float(len(recall_scores_total))

    print('La media totale di f1=' + str(f1_avg_total))
    print('La media totale di accuracy=' + str(accuracy_avg_total))
    print('La media totale di balanced_accuracy=' + str(balanced_accuracy_avg_total))
    print('La media totale di precision=' + str(precision_avg_total))
    print('La media totale di recall=' + str(recall_avg_total))

