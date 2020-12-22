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
    error_anno = []


    # Takes the n largest elements in the 'elements' array
    def maxN(elements, n):
        return sorted(elements, reverse=True)[:n]


    # A cycle that at each iteration runs a different year,
    # the year of the i-th iteration makes up the test set and the others are training set.
    for anno in range(2001, 2018):
        print('TEST SU STAGIONE: ' + str(anno))
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
        gamma0 = 1 / 20
        gamma_values = [gamma0 * (2 ** i) for i in range(-8, 8)]
        gamma_max = gamma_values[0]
        for C in C_values:
            for gamma in gamma_values:

                # No Linear SVC (an No Linear SVM extension) to find C_max and gamma_max through Cross-Validation.
                # Now it no longer returns 1 or 0 to understand if the team enters the playoffs or not
                # but makes it a probability (probability = True).
                svcnolinearmodel = SVC(C=C, gamma=gamma, probability=True)
                score = cross_val_score(svcnolinearmodel, X_train, y_train, cv=cv, scoring='f1')
                mean_score = statistics.mean(score)
                if mean_score > best_score:
                    C_max = C
                    gamma_max = gamma
                    best_score = mean_score
        values.append((C_max, gamma_max))

        # No Linear SVC with C_max and gamma_max with probability = True.
        svcnolinearmodel = SVC(C=C_max, gamma=gamma_max, probability=True)
        svcnolinearmodel.fit(X_train, y_train)
        predictions = svcnolinearmodel.predict_proba(X_test)

        # The number of teams that go to the playoffs differs between years.
        nSquadrePlayoff = 8
        nSquadreStagione = 14
        if anno == 2011:
            nSquadrePlayoff = 12
        elif anno == 2012:
            nSquadrePlayoff = 10
            nSquadreStagione = 12
        elif anno == 2013:
            nSquadreStagione = 12
        elif anno == 2014:
            nSquadreStagione = 13
        elif anno == 2015:
            nSquadreStagione = 12

        playoffProbabilities = [prediction[1] for prediction in predictions]

        # It puts 1 on the teams that have the highest probability and zero otherwise.
        # (the maximum number of teams that can be assigned 1 is given by the number of teams
        # that go to the playoffs that year).
        finalPredictions = [1 if index in maxN(playoffProbabilities, nSquadrePlayoff) else 0 for index in
                            playoffProbabilities]

        df = pd.DataFrame(train.loc[train['Stagione'] == anno, 'Squadra'])
        df['Stagione'] = train.loc[train['Stagione'] == anno, 'Stagione']
        df['PlayoffProbability'] = playoffProbabilities
        df['PredictionPlayoff'] = finalPredictions
        df['RealPlayoff'] = train.loc[train['Stagione'] == anno, 'Playoff']
        df['Posizione'] = train.loc[train['Stagione'] == anno, 'Posizione']

        # Introduced a further statistical classification called 'Error',
        # in addition to the usual f1_score, accuracy, balanced_accuracy, precision and recall.
        error_team = []

        for row in df.itertuples():
            if row.RealPlayoff == row.PredictionPlayoff:
                error_team.append(0)
            elif row.RealPlayoff == 0 and row.PredictionPlayoff == 1:
                error_team.append(row.Posizione - nSquadrePlayoff)
            else:
                error_team.append((nSquadrePlayoff + 1) - row.Posizione)

        df['Errore'] = error_team
        print(df.to_string())
        print()
        print('Numero Squadre Playoff= ' + str(nSquadrePlayoff))
        print('Numero Squadre Stagione=' + str(nSquadreStagione))
        print()

        # Calculation of the various metrics.
        # errore
        error_team_avg = sum(error_team) / nSquadreStagione
        error_anno.append(error_team_avg)

        # f1
        f1 = metrics.f1_score(y_test, finalPredictions)
        f1_scores_total.append(f1)

        # accuracy
        accuracy = metrics.accuracy_score(y_test, finalPredictions)
        accuracy_scores_total.append(accuracy)

        # balanced_accuracy
        balanced_accuracy = metrics.balanced_accuracy_score(y_test, finalPredictions)
        balanced_accuracy_scores_total.append(balanced_accuracy)

        # precision
        precision = metrics.precision_score(y_test, finalPredictions)
        precision_scores_total.append(precision)

        # recall
        recall = metrics.recall_score(y_test, finalPredictions)
        recall_scores_total.append(recall)

        print('f1 del C_max e gamma_max=' + str(f1))
        print('accuracy del C_max e gamma_max=' + str(accuracy))
        print('balanced_accuracy del C_max e gamma_max=' + str(balanced_accuracy))
        print('precision del C_max e gamma_max=' + str(precision))
        print('recall del C_max e gamma_max=' + str(recall))
        print('errore medio in questo anno=' + str(error_team_avg))
        print()
        print('C_max e gamma_max=' + str(values))
        print()

    f1_avg_total = sum(f1_scores_total) / float(len(f1_scores_total))
    accuracy_avg_total = sum(accuracy_scores_total) / float(len(accuracy_scores_total))
    balanced_accuracy_avg_total = sum(balanced_accuracy_scores_total) / float(len(balanced_accuracy_scores_total))
    precision_avg_total = sum(precision_scores_total) / float(len(precision_scores_total))
    recall_avg_total = sum(recall_scores_total) / float(len(recall_scores_total))
    error_anno_avg = sum(error_anno) / float(len(error_anno))

    print('La media totale di f1=' + str(f1_avg_total))
    print('La media totale di accuracy=' + str(accuracy_avg_total))
    print('La media totale di balanced_accuracy=' + str(balanced_accuracy_avg_total))
    print('La media totale di precision=' + str(precision_avg_total))
    print('La media totale di recall=' + str(recall_avg_total))
    print('La media totale di errore=' + str(error_anno_avg))
