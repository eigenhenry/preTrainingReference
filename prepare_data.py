"""Download the ten datasets and save them as CSV files in data/ (run once before the notebook).

Each dataset is cleaned and reduced to its binary target and the features used in the paper.
Files that already exist in data/ are kept as they are; delete one to rebuild it.
"""
import io, zipfile, urllib.request
from pathlib import Path

import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)

def _cached(name, build):
    p = DATA_DIR / f"{name}.csv"
    if p.exists():
        return pd.read_csv(p)
    df = build(); df.to_csv(p, index=False)
    return df

def load_titanic():
    def _b():
        d = sns.load_dataset("titanic")
        d["sex"] = d["sex"].map({"male": 1, "female": 0})
        d["embarked"] = LabelEncoder().fit_transform(d["embarked"].astype(str))
        d["age"] = d["age"].fillna(d["age"].median())
        return d[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]
    return _cached("titanic", _b)

def load_diabetes():
    def _b():  # Pima Indians Diabetes
        d = fetch_openml("diabetes", version=1, as_frame=True, parser="auto").frame.copy()
        d = d.rename(columns={"preg": "Pregnancies", "plas": "Glucose", "pres": "BloodPressure",
            "skin": "SkinThickness", "insu": "Insulin", "mass": "BMI",
            "pedi": "DiabetesPedigreeFunction", "age": "Age"})
        d["Outcome"] = (d["class"] == "tested_positive").astype(int)
        return d.drop(columns=["class"])
    return _cached("diabetes", _b)

def load_adult():
    def _b():
        r = fetch_openml("adult", version=2, as_frame=True, parser="auto").frame.copy()
        r = r.dropna().reset_index(drop=True)
        r["income"] = (r["class"].str.strip().str.replace(".", "", regex=False) == ">50K").astype(int)
        r = r.rename(columns={"education-num": "education_num", "capital-gain": "capital_gain",
            "capital-loss": "capital_loss", "hours-per-week": "hours_per_week"})
        r["married"] = (r["marital-status"] == "Married-civ-spouse").astype(int)
        r["relationship_grp"] = r["relationship"].map({"Husband": "spouse", "Wife": "spouse",
            "Not-in-family": "independent", "Unmarried": "independent",
            "Own-child": "dependent", "Other-relative": "dependent"})
        high = {"Exec-managerial", "Prof-specialty", "Protective-serv", "Tech-support"}
        low  = {"Other-service", "Handlers-cleaners", "Farming-fishing", "Priv-house-serv",
                "Machine-op-inspct", "Armed-Forces"}
        r["occupation_grp"] = r["occupation"].map(
            lambda x: "high" if x in high else ("low" if x in low else "mid"))
        r["workclass_grp"] = r["workclass"].map({"Federal-gov": "government", "Local-gov": "government",
            "State-gov": "government", "Self-emp-inc": "self_employed", "Self-emp-not-inc": "self_employed",
            "Private": "private", "Without-pay": "private"})
        r["race_grp"] = r["race"].map({"White": "white", "Black": "black",
            "Asian-Pac-Islander": "other_race", "Amer-Indian-Eskimo": "other_race", "Other": "other_race"})
        r["native_us"] = (r["native-country"] == "United-States").astype(int)
        keep = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week", "sex",
                "married", "workclass_grp", "occupation_grp", "relationship_grp", "race_grp",
                "native_us", "income"]
        return r[keep]
    return _cached("adult", _b)

def load_compas():
    def _b():
        url = ("https://raw.githubusercontent.com/propublica/compas-analysis/"
               "master/compas-scores-two-years.csv")
        r = pd.read_csv(url)
        r = r[(r["days_b_screening_arrest"] <= 30) & (r["days_b_screening_arrest"] >= -30) &
              (r["is_recid"] != -1) & (r["c_charge_degree"] != "O") &
              (r["score_text"] != "N/A")].copy()
        r["race_black"]      = (r["race"] == "African-American").astype(int)
        r["juv_any"]         = ((r["juv_fel_count"] + r["juv_misd_count"] + r["juv_other_count"]) > 0).astype(int)
        r["sex"]             = (r["sex"] == "Male").astype(int)
        r["c_charge_degree"] = (r["c_charge_degree"] == "F").astype(int)
        cols = ["age", "sex", "race_black", "priors_count", "c_charge_degree", "juv_any", "two_year_recid"]
        return r[cols].dropna().reset_index(drop=True)
    return _cached("compas", _b)

def load_breast_cancer_df():
    return _cached("breast_cancer", lambda: load_breast_cancer(as_frame=True).frame.copy())

def load_german():
    def _b():
        d = fetch_openml("credit-g", version=1, as_frame=True, parser="auto").frame.copy()
        d["target"] = (d["class"] == "bad").astype(int)   # positive = bad credit
        return d.drop(columns=["class"])
    return _cached("german", _b)

_BANK_RENAME = {"V1": "age", "V2": "job", "V3": "marital", "V4": "education", "V5": "default",
    "V6": "balance", "V7": "housing", "V8": "loan", "V9": "contact", "V10": "day", "V11": "month",
    "V12": "duration", "V13": "campaign", "V14": "pdays", "V15": "previous", "V16": "poutcome"}
def load_bank():
    def _b():
        d = fetch_openml("bank-marketing", version=1, as_frame=True, parser="auto").frame.copy()
        d = d.rename(columns=_BANK_RENAME)
        d["target"] = (d["Class"].astype(int) == 2).astype(int)   # 2 = subscribed
        return d.drop(columns=["Class", "duration"])              # duration = leakage
    return _cached("bank", _b)

def load_heart():
    def _b():  # Statlog (Heart), UCI; 270 patients, no missing values
        d = fetch_openml("heart-statlog", version=1, as_frame=True, parser="auto").frame.copy()
        d["target"] = (d["class"] == "present").astype(int)       # 1 = heart disease present
        return d.drop(columns=["class"])
    return _cached("heart", _b)

def load_student():
    def _b():  # Portuguese-language course; target = final grade G3 >= 10 (pass)
        url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
        raw = urllib.request.urlopen(url, timeout=60).read()
        inner = zipfile.ZipFile(io.BytesIO(zipfile.ZipFile(io.BytesIO(raw)).read("student.zip")))
        d = pd.read_csv(io.BytesIO(inner.read("student-por.csv")), sep=";")
        d["target"] = (d["G3"] >= 10).astype(int)
        return d.drop(columns=["G1", "G2", "G3"])                 # period grades = leakage
    return _cached("student", _b)

_TAIWAN_RENAME = {"x1": "LIMIT_BAL", "x2": "SEX", "x3": "EDUCATION", "x4": "MARRIAGE", "x5": "AGE",
    **{f"x{i}": f"PAY_{i - 5}" for i in range(6, 12)},
    **{f"x{i}": f"BILL_AMT{i - 11}" for i in range(12, 18)},
    **{f"x{i}": f"PAY_AMT{i - 17}" for i in range(18, 24)}}
def load_taiwan():
    def _b():
        d = fetch_openml("default-of-credit-card-clients", version=1, as_frame=True, parser="auto").frame.copy()
        d = d.rename(columns=_TAIWAN_RENAME)
        d["target"] = d["y"].astype(int)
        return d.drop(columns=["y"])
    return _cached("taiwan", _b)


if __name__ == "__main__":
    for load in (load_titanic, load_diabetes, load_adult, load_compas, load_breast_cancer_df, load_german,
                 load_bank, load_heart, load_student, load_taiwan):
        df = load()
        print(f"{load.__name__:22s} {len(df):6d} rows")
