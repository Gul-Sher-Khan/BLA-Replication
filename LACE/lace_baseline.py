import os
import sys

import pandas as pd

from lace import cliff, morph, lace1, add_to_bin, lace2_simulator
import numpy as np

import logging
import sys

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


# logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("lace_file_log.log"),
        logging.StreamHandler()
    ]
)

# def do_cliff(df, cols, obj, cliff_percentage=0.8):
#     return pd.DataFrame(cliff(
#         attribute_names=list(df.columns),
#         data_matrix=df.values.tolist(),
#         independent_attrs=cols,
#         objective_attr=obj,
#         cliff_percentage=cliff_percentage,
#         objective_as_binary=True,
#     ), columns=list(df.columns))

def do_morph(df, cols, obj, morph_alpha=0.15, morph_beta=0.35):
    return pd.DataFrame(morph(
        attribute_names=list(df.columns),
        data_matrix=df.values.tolist(),
        independent_attrs=cols,
        objective_attr=obj,
        objective_as_binary=True,
        alpha=morph_alpha,
        beta=morph_beta,
    ), columns=list(df.columns))

def do_lace1(df, cols, obj, alpha=0.15, beta=0.35, cliff_percentage=0.8):
    return pd.DataFrame(lace1(
        attribute_names=list(df.columns),
        data_matrix=df.values.tolist(),
        independent_attrs=cols,
        objective_attr=obj,
        objective_as_binary=True,
        cliff_percentage=cliff_percentage,
        alpha=alpha,
        beta=beta,
    ), columns=list(df.columns))

def do_lace2(df, cols, obj, cliff_percentage=0.8, morph_alpha=0.15, morph_beta=0.35, holders=5):
    __x = lace2_simulator(
        attribute_names=list(df.columns),
        data_matrix=df.values.tolist(),
        independent_attrs=cols,
        objective_attr=obj,
        objective_as_binary=True,
        number_of_holder=holders,
        cliff_percentage=cliff_percentage,
        morph_alpha=morph_alpha,
        morph_beta=morph_beta,
    )
    __x.pop(0)
    return pd.DataFrame(__x, columns=list(df.columns))


def add_iter_and_test(__df_cliff, df_test, index):

    __df_cliff["iteration"] = str(index)
    __df_cliff["type"] = "train"
    df_test["iteration"] = str(index)
    df_test["type"] = "test"

    return pd.concat([__df_cliff, df_test])


if __name__ == '__main__':

    base_project_name = sys.argv[1]
    base_project_name_file = base_project_name + ".csv"

    project_name_split_train = "data_split_" + base_project_name +  "_train.csv"
    project_name_split_test = "data_split_" + base_project_name  + "_test.csv"

    logging.info(base_project_name)
    logging.info(base_project_name_file)
    logging.info(project_name_split_train)

    df_base = pd.read_csv("lace_data/" + base_project_name_file)

    iteration_col_name = "iteration_id"
    type_col_name = "type_id"
    technique_name = "technique_id"
    cols = list(df_base.columns)
    cols.extend([iteration_col_name, type_col_name, technique_name])
    df_all = pd.DataFrame(columns=cols)

    df_split_train = pd.read_csv(project_name_split_train)
    df_split_test = pd.read_csv(project_name_split_test)

    # take all the data and anonymise it, name it

    logging.info("total length of the data: " + str(len(df_base)))
    logging.info("total length of the train data: "+ str( len(df_split_train)))
    logging.info("total length of the test data: "+ str(len(df_split_test)))

    df_base = df_base.dropna(subset=["commit_id"])
    df_base.fillna(0, inplace=True)
    # drop two columns 0 and Unnamed: 0
    df_base = df_base.drop(columns=["Unnamed: 0", "fixcount", "project",
                                    "buggy", "fix", "year",
                                    "osawr", 'author_date', 'asawr', 'rsawr'], errors="ignore")
    small_columns = list(df_base.columns)
    small_columns.remove("bugcount")
    small_columns.remove("commit_id")

    # remove_bugcount = list(df_base.columns)
    # remove_bugcount.remove("bugcount")
    # remove_bugcount.remove("commit_id")

    vals_used = "vals_used"

    # ["do_lace2", "do_lace1", "do_morph"]


    for __func in [do_lace1, do_morph, do_lace2]:

        logging.info(__func.__name__)
        iteration_val = 0

        if __func.__name__ == "do_cliff":
            vals = [{"cliff_percentage" : float(x)} for x in np.arange(0.2, 1, 0.2)]

        elif __func.__name__ == "do_morph":

            vals = []
            vals.append({"morph_alpha": 0.15, "morph_beta": 0.35})
            for x in np.arange(0.2, 1.2, 0.2):
                for y in np.arange(0.2, 1.2, 0.2):
                    vals.append({"morph_alpha" : float(x), "morph_beta" : float(y)})


            # vals = [{"morph_alpha" : f,loat(x), "morph_beta" : float(x)} for x in np.arange(0.2, 1, 0.2)]

        elif __func.__name__ == "do_lace1":

            vals = []
            vals.append({"alpha": 0.15, "beta": 0.35, "cliff_percentage": 0.4})
            vals.append({"alpha": 0.15, "beta": 0.35, "cliff_percentage": 0.8})
            for x in np.arange(0.2, 1.2, 0.2):
                for y in np.arange(0.2, 1.2, 0.2):
                    for z in [0.4, 0.8]:
                        vals.append({"alpha" : float(x), "beta" : float(y), "cliff_percentage" : float(z)})



        elif __func.__name__ == "do_lace2":

            vals = []
            vals.append({"morph_alpha": 0.15, "morph_beta": 0.35, "cliff_percentage": 0.4})
            vals.append({"morph_alpha": 0.15, "morph_beta": 0.35, "cliff_percentage": 0.8})
            for x in np.arange(0.2, 1.2, 0.2):
                for y in np.arange(0.2, 1.2, 0.2):
                    for z in [0.4, 0.8]:
                        vals.append({"morph_alpha" : float(x), "morph_beta" : float(y), "cliff_percentage" :  float(z)})


        for i, v in enumerate(vals):
            iteration_val = i

            logging.info("iteration: " + str(iteration_val))
            logging.info("values: " + v.__str__())
            objective = "bugcount"

            __priv_df = df_base[df_base["commit_id"].isin(df_split_train["commit_id"])]
            __df = __func(__priv_df, small_columns, objective, **v)
            __df[iteration_col_name] = iteration_val
            __df[type_col_name] = "privacy_check"
            __df[vals_used] = str(v)
            __df[technique_name] = __func.__name__

            df_all = pd.concat([df_all, __df], sort=False)

            __train_df = df_base[df_base["commit_id"].isin(df_split_train["commit_id"])]
            __test_df = df_base[df_base["commit_id"].isin(df_split_test["commit_id"])]

            # anonymise the train data
            __df = __func(__train_df, small_columns, objective, **v)
            __df[iteration_col_name] = iteration_val
            __df[type_col_name] = "train"
            __df[vals_used] = str(v)
            __df[technique_name] = __func.__name__

            df_all = pd.concat([df_all, __df], sort=False)

            __test_df[iteration_col_name] = iteration_val
            __test_df[type_col_name] = "test"
            __test_df[vals_used] = str(v)
            __test_df[iteration_col_name] = iteration_val
            __test_df[technique_name] = __func.__name__

            df_all = pd.concat([df_all, __test_df], sort=False)

        #     break
        # break


    print("done")
    df_all.to_csv(base_project_name + "_all_comp.csv", index=False)


