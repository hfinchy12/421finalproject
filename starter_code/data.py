import pandas as pd
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_data(x_path):
    pd.read_csv(x_path)


def split_data(x, y, split=0.8, random_state=42):
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=split, random_state=random_state)
    return train_x, train_y, test_x, test_y


def find_splits(df):
    """
    Find the splits in an unprocessed dataset
    """
    columns_per_split = [
        ["admissionheight", "admissionweight", "age",
            "ethnicity", "gender", "unitvisitnumber"],
        ["cellattributevalue", "celllabel"],
        ["labmeasurenamesystem", "labname", "labresult"],
        ["nursingchartcelltypevalname", "nursingchartvalue"]
    ]
    splits = {"primary": {}, "capillary": {}, "labs": {}, "charts": {}}

    for split, cols in zip(splits.keys(), columns_per_split):
        # find the earliest and latest beginnings for columns (i.e., where they are not null)
        min_row = len(df) - 1
        max_row = 0
        splits[split]["columns"] = cols

        for col in cols:
            rows = df[df[col].notna()].iloc[:, 0]
            if (len(rows)) == 0:
                col_min = 0
                col_max = 0
                min_row = 0
                max_row = 0
                continue
            col_min = rows.iloc[0]
            col_max = rows.iloc[-1]
            min_row = min(col_min, min_row)
            max_row = max(col_max, max_row)

        splits[split]["start"] = min_row
        splits[split]["stop"] = max_row

    return splits


def process_patient_primary(df, start=0, stop=2015, estimateData=False):
    """
    stay id, admission height, weight, age, ethnicity, gender, visit number
    N.B.: stop is inclusive (whereas Python's stop is usually exclusive, e.g., for range() or list slicing)
    """
    primary = df.iloc[start:(stop+1)]
    primary = primary[["patientunitstayid", "admissionheight",
                       "admissionweight", "age", "ethnicity", "gender", "unitvisitnumber"]]

    # NOTE: do we want to just get rid of other/unknown race column in the table?
    primary["ethnicity"].fillna(value="Other/Unknown", inplace=True)

    primary["age"] = primary["age"].replace("> 89", "90")
    primary["age"] = pd.to_numeric(primary["age"], downcast="float")

    dummy_gender = pd.get_dummies(primary["gender"])
    dummy_ethnicity = pd.get_dummies(primary["ethnicity"])

    if ("Native American" not in dummy_ethnicity.columns):
        dummy_ethnicity["Native American"] = 0

    primary = pd.concat([primary, dummy_gender, dummy_ethnicity], axis=1)
    primary = primary.drop(["gender", "ethnicity"], axis=1)

    primary["patientunitstayid"] = primary["patientunitstayid"].astype("int32")

    if estimateData:
        # set NaN height/weight to mean
        mean_height = primary[primary["admissionheight"].notna(
        )]["admissionheight"].mean()
        primary["admissionheight"].fillna(value=mean_height, inplace=True)

        mean_weight = primary[primary["admissionweight"].notna(
        )]["admissionweight"].mean()
        primary["admissionweight"].fillna(value=mean_weight, inplace=True)

    else:
        primary = primary.dropna(subset=["admissionheight", "admissionweight"])

    return primary


def process_patient_stats(df):
    """
    For each type of lab or chart value that a patient does, determine mean, standard deviation,
    number of measurements, line of best fit slope and intercept with respect to time, and total time.
    For GCS total, we will also calculate number of times the patient was unable
    to be scored due to medication; these instances will not be included in the
    mean/stdev/# values. This value in the dataset is "Unable to score due to medication".

    Relevant values from the dataset: 
    glucose, pH, 
    Respiratory Rate, O2 Saturation, Heart Rate, Non-Invasive BP Systolic, Non-Invasive BP Diastolic,
    Invasive BP Diastolic, Invasive BP Systolic, GCS Total,
    Non-Invasive BP Mean, Invasive BP Mean
    """
    unique_stays = df["patientunitstayid"].unique()
    labs = df["labname"].dropna().unique()
    charts = df["nursingchartcelltypevalname"].dropna().unique()
    measurements = np.concatenate((labs, charts))

    feature_types = ["mean", "stdev", "number", "slope", "intercept", "time"]
    features = ["patientunitstayid"]
    for m in measurements:
        for f in feature_types:
            features.append(m + " " + f)

    features.append("GCS unable to score")

    stats = pd.DataFrame(columns=features)

    for stay in unique_stays:

        patient_info = df[df["patientunitstayid"] == stay]

        row = {f: [0] for f in features}
        row["GCS unable to score"] = [0]
        row["patientunitstayid"] = [stay]

        if len(patient_info) == 0:
            df_row = pd.DataFrame(row)
            stats = pd.concat((stats, df_row))
            continue

        GCS_unscored_rows = patient_info[patient_info["nursingchartvalue"]
                                         == "Unable to score due to medication"].index
        GCS_unscored = len(GCS_unscored_rows)
        row["GCS unable to score"] = [GCS_unscored]

        for measurement in measurements:

            patient_measurements = patient_info[patient_info["nursingchartcelltypevalname"] == measurement].copy(
            )
            if measurement == "GCS Total" and GCS_unscored > 0:
                string_rows = patient_info[patient_info["nursingchartvalue"]
                                           == "Unable to score due to medication"].index
                patient_measurements = patient_measurements.drop(
                    string_rows, axis=0)

            patient_measurements["nursingchartvalue"] = pd.to_numeric(
                patient_measurements["nursingchartvalue"], downcast="float")

            patient_measurements.dropna()

            if len(patient_measurements) > 0:

                values = patient_measurements["nursingchartvalue"]
                times = patient_measurements["offset"]
                times = pd.to_numeric(times, downcast="float")

                line_data = pd.concat((times, values), axis=1)
                line_data = line_data.dropna()

                row[measurement + " mean"] = [values.mean()]
                row[measurement + " stdev"] = [values.std()]
                row[measurement + " number"] = [len(values)]

                model = LinearRegression().fit(
                    line_data["offset"].to_numpy().reshape(-1, 1), line_data["nursingchartvalue"])

                slope = model.coef_[0]
                intercept = model.intercept_

                row[measurement + " slope"] = [slope]
                row[measurement + " intercept"] = [intercept]
                row[measurement + " time"] = [times.max() - times.min()]

        df_row = pd.DataFrame(row).fillna(value=0)
        stats = pd.concat((stats, df_row))

    return stats


def scale_data(df):
    ss = StandardScaler()
    # mm = MinMaxScaler()

    no_norm_columns = ["patientunitstayid", "Female", "Male",
                       "African American", "Asian", "Caucasian", "Hispanic", "Other/Unknown"]
    no_normalize = df[no_norm_columns]
    normalize = df[df.columns.difference(no_norm_columns)]

    scaled = pd.DataFrame(ss.fit_transform(normalize),
                          columns=normalize.columns, index=normalize.index)

    merged = no_normalize.merge(scaled, left_index=True, right_index=True)

    return merged


def preprocess_x(df, estimateData=False, savePath=None):
    """
    Combines all the preprocessing steps
    """

    splits = find_splits(df)

    df = df.drop(df.columns[[0]], axis=1)
    df = df.drop(["cellattributevalue", "celllabel",
                 "labmeasurenamesystem"], axis=1)

    primary = process_patient_primary(
        df, start=splits["primary"]["start"], stop=splits["primary"]["stop"], estimateData=estimateData)
    stats = process_patient_stats(df)

    data = primary.merge(stats, on="patientunitstayid")

    data = scale_data(data)

    data = data.fillna(value=0)

    if savePath != None:
        pd.to_csv(savePath, header=True, index=False)

    return data


def match_up_data(df_x, df_y):
    """
    Match patient stats to hospital discharge status, remove patientunitstayid from both,
    and split up data again.
    """
    matched = df_x.merge(df_y, on="patientunitstayid")

    x = matched[matched.columns.difference(
        ["patientunitstayid", "hospitaldischargestatus", "Unnamed: 0"])]
    y = matched["hospitaldischargestatus"]

    return x, y
