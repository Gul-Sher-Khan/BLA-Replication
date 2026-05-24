# this file is in python2.7

import os
import sys
# train random forest

import pandas as pd

from lace import cliff, morph, lace1, add_to_bin, lace2_simulator
import numpy as np

import logging
import sys

# ignore warnings
import warnings
warnings.filterwarnings("ignore")



if __name__ == "__main__":

    # listing all files in dir
    for filename in os.listdir("../../testing_data_lace"):

        df = pd.read_csv("../../testing_data_lace/" + filename)
        df["bug"] = df["bug"].apply(lambda x: 1 if x > 0 else 0)

        trian_scores = []
        test_scores = []

        # stratified sampling
        # import sklearn.model_selection as model_selection
        # train_df, test_df = model_selection.train_test_split(df, train_size=0.8, stratify=df["bug"])

        # stratified sampling
        from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
        # sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        sss = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)

        for train_index, test_index in sss.split(df, df["bug"]):

            train_df, test_df = df.loc[train_index], df.loc[test_index]
            # train_df = df.sample(frac=0.8, random_state=0)
            # test_df = df.drop(train_df.index)

            duplicate_train = train_df.copy(deep=True)

            cols = list(duplicate_train.columns)
            # cols.remove("bug")
            cols.remove("name")

            remove_bug = list(duplicate_train.columns)
            remove_bug.remove("bug")
            remove_bug.remove("name")


            anon_df =  pd.DataFrame(lace1(
                attribute_names=list(cols),
                data_matrix=duplicate_train[cols].values.tolist(),
                independent_attrs=remove_bug,
                objective_attr="bug",
                objective_as_binary=True,
                cliff_percentage=0.8,
                alpha=0.15,
                beta=0.35,
            ), columns=list(cols))

            # print(df2)

            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import roc_auc_score, f1_score, recall_score, fowlkes_mallows_score
            from sklearn.metrics import accuracy_score

            # from imblearn.metrics import geometric_mean_score

            cols.remove("bug")

            clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
            clf.fit(train_df[cols], train_df["bug"])
            clf_probs = clf.predict(test_df[cols])

            trian_scores.append({
                "pd": recall_score(test_df["bug"], clf_probs),
                "auc" : roc_auc_score(test_df["bug"], clf_probs),
                "gmean": fowlkes_mallows_score(test_df["bug"], clf_probs),
                "fmeasure": f1_score(test_df["bug"], clf_probs),
            })


            # train random forest on LACE data
            clf2 = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
            # anon_df["bug"] = anon_df["bug"].apply(lambda x: 1 if x >= 1 else 0)
            clf2.fit(anon_df[cols], anon_df["bug"])
            clf_probs2 = clf2.predict(test_df[cols])

            test_scores.append({
                "pd": recall_score(test_df["bug"], clf_probs2),
                "auc" : roc_auc_score(test_df["bug"], clf_probs2),
                "gmean": fowlkes_mallows_score(test_df["bug"], clf_probs2),
                "fmeasure": f1_score(test_df["bug"], clf_probs2),
            })

            # print("--------------------------------------------------")


        print("file: ", filename)

        # print average of train scores
        print("train scores")
        print("pd: ", np.mean([x["pd"] for x in trian_scores]))
        print("auc: ", np.mean([x["auc"] for x in trian_scores]))
        print("gmean: ", np.mean([x["gmean"] for x in trian_scores]))
        print("fmeasure: ", np.mean([x["fmeasure"] for x in trian_scores]))

        # print average of test scores
        print("test scores")
        print("pd: ", np.mean([x["pd"] for x in test_scores]))
        print("auc: ", np.mean([x["auc"] for x in test_scores]))
        print("gmean: ", np.mean([x["gmean"] for x in test_scores]))
        print("fmeasure: ", np.mean([x["fmeasure"] for x in test_scores]))
