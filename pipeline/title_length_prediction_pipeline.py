import json
import pandas as pd
import statsmodels.formula.api as sm
# import numpy as np


def load_data(filepath="../data/notable_works_by_charles_dickens.csv"):
    df = pd.read_csv(filepath)

    return df


def add_columns(df):
    new_df = df.copy()

    new_df["title_len"] = df.Title.map(len)
    new_df["year"] = df["Year completed"]
    new_df["is_novel"] = df.Type == "Novel"

    return new_df


def compute_model_parameters(df):
    result = sm.ols(formula="title_len ~ year", data=df).fit()

    return {
        "r_squared": result.rsquared,
        "year_beta": result.params["year"],
    }


def __main__():
    df = load_data()
    df = add_columns(df)
    params = compute_model_parameters(df)

    print( json.dumps(params, indent=2) )

if __name__=="__main__":
    __main__()