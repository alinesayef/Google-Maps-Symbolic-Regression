# (c) Copyright 2024 - Present, Ali Nesayef
from random import random, randint
import googlemaps
from datetime import datetime
from pprint import pprint
import sys
import json
import pandas as pd
import numpy as np

from sklearn.utils.validation import check_random_state
from gplearn.genetic import SymbolicClassifier, SymbolicRegressor

# API key to be used
API_KEY = "INSERT_GOOGLE_API_KEY_HERE"

# initialize google maps api client
def initialize_google_api_client():
    # https://googlemaps.github.io/google-maps-services-python/docs/index.html
    # initialize the google maps client and return it
    gmaps = googlemaps.Client(API_KEY)
    return gmaps


# get directions function
def get_directions(gmaps: googlemaps.client.Client, start_location, end_location):
    # defining time for query equal current time
    now = datetime.now()
    # get the directions for the specified start and end points
    # https://googlemaps.github.io/google-maps-services-python/docs/index.html
    directions_result = gmaps.directions(
        start_location,
        end_location,
        mode="driving",
        departure_time=now,
        alternatives=True,
    )
    # extract the directions dict from the returned list
    return directions_result


# function to write the data into an excel file
def get_df(data: dict, est_gp, index, params):
    # https://docs.python.org/3/tutorial/datastructures.html
    # extract steps from returned dictionary
    steps = data["legs"][0]["steps"]
    # length of steps list
    steps_length = len(steps)
    keys = []
    # populate list with keys according to list length
    for i in range(steps_length):
        keys.append(f"step_{i}")
    # define empty dict
    steps_dict = {}
    # populate dict with values
    for i in range(steps_length):
        steps_dict[keys[i]] = steps[i]


    # read into dataframe
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.from_dict.html
    df = pd.DataFrame.from_dict(steps_dict)
    df = pd.DataFrame.from_dict(steps_dict)

    df.loc["CO2_omissions"] = [randint(0, 99)] * df.shape[1]
    df.loc["scenic_score"] = [randint(1, 9)] * df.shape[1]
    df.loc["safety_score"] = [randint(1, 9)] * df.shape[1]

    dis = []
    total_dis = 0
    total_time = 0
    i = 0
    _co2 = 0
    _scene = 0
    _safe = 0
    for r in df.loc["distance"]:
        total_dis += r["value"]
        _co2 += df.loc["CO2_omissions"][i] * r["value"]
        _scene += df.loc["scenic_score"][i] * r["value"]
        _safe += df.loc["safety_score"][i] * r["value"]
        i += 1

    for r in df.loc["distance"]:
        dis.append(r["value"] / total_dis)

    _air = []
    for s in df.loc["CO2_omissions"]:
        _air.append(100 - s)

    _co2 = int(_co2 / total_dis * 100) / 100
    _scene = int(_scene / total_dis * 100) / 100
    _safe = int(_safe / total_dis * 100) / 100

    for r in df.loc["duration"]:
        total_time += r["value"]

    y_gp = est_gp.predict(np.c_[_air, df.loc["scenic_score"].ravel(), df.loc["safety_score"].ravel(), dis]).reshape(df.loc["CO2_omissions"].shape)

    spec = 0;
    for r in y_gp:
        spec += r

    tmp_time = total_time
    _hour = int(total_time / 3600)
    total_time = total_time % 3600;
    _min = int(total_time / 60) + 1
    _time = ""
    if _hour != 0:
        _time = str(_hour) + " h "
    if _min != 0:
        _time += str(_min) + " min"
    _dis = ""
    if total_dis >= 1000:
        _dis = str(int((total_dis / 1000) * 100) / 100)  + " km"
    else:
        _dis = str(total_dist) + " m"

    data = {}
    data["Distance"] = {}
    data["Time"] = {}
    data["CO2"] = {}
    data["Scene"] = {}
    data["Safety"] = {}
    data["Preference"] = {}
    data["Distance"][f"route{index}"] = _dis
    data["Time"][f"route{index}"] = tmp_time
    data["CO2"][f"route{index}"] = _co2
    data["Scene"][f"route{index}"] = _scene
    data["Safety"][f"route{index}"] = _safe

    data["Preference"][f"route{index}"] = spec

    data_save = pd.DataFrame.from_dict(data)

    dt = {}
    dt["Time"] = tmp_time
    dt["time_ch"] = _time
    dt["Preference"] = spec
    dt["route"] = f"route {index}"

    files = {}
    files["google"] = df
    files["result"] = data_save
    files["time"] = dt

    return files


def SRAlgorithm(w1, w2, w3, criteria):
    rng = check_random_state(0)
    X_train = rng.uniform(1, 5, 200).reshape(50, 4)
    y_train = (X_train[:, 0] * w1 + X_train[:, 1]*w2*6 + X_train[:, 2]*8*w3) / (w1 + w2*6 + 8*w3 + 1) * X_train[:, 3]
    est_gp = SymbolicRegressor(population_size=5000,
                           generations=20, stopping_criteria=criteria,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)
    est_gp.fit(X_train, y_train)
    return est_gp

    #https://gplearn.readthedocs.io/en/stable/examples.html#symbolic-regressor

def sortResultTime(val):
    return int(val["Time"])

def sortResultPre(val):
    return 1 / (float(val["Preference"]) + 1)

def sortPre(val):
    return 1 / (float(val["value"]) + 1)

def save_file(directions, filename, est_gp, params):

    total = []
    data = []
    result = []
    time = []
    index = 1;
    for r in directions:
        total = get_df(r, est_gp, index, params)
        data.append(total["google"])
        time.append(total["time"])
        result.append(total["result"])
        index += 1

    if params["time"] == 1:
        result.sort(key=sortResultTime)
        time.sort(key=sortResultTime)
    else:
        result.sort(key=sortResultPre)
        time.sort(key=sortResultPre)

    for i in range(index - 1):
        result[i]["Time"] = time[i]["time_ch"]

    str_str = ""
    if params["time"] == 1:
        str_str = time[0]["route"] + " is the fastest of the routes"
    else:
        _pre = []
        _pre_tmp0 = {}
        _pre_tmp0["value"] = params["air"]
        _pre_tmp0["name"] = "air"
        _pre.append(_pre_tmp0)
        _pre_tmp1 = {}
        _pre_tmp1["value"] = params["scene"]
        _pre_tmp1["name"] = "scene"
        _pre.append(_pre_tmp1)
        _pre_tmp2 = {}
        _pre_tmp2["value"] = params["safe"]
        _pre_tmp2["name"] = "safety"
        _pre.append(_pre_tmp2)
        _pre.sort(key=sortPre)

        str_str = time[0]["route"] + " is the best route"
        i = 0

    print(str_str)

    pd.concat(result).to_excel("result.xlsx")
    return pd.concat(data).to_excel(f"{filename}.xlsx")


# function to parse command line arguments.
def get_runtime_arguments():
    # n = arguments passed
    n = len(sys.argv)

    params = {}
    params["count"] = n
    params["air"] = 0
    params["scene"] = 0
    params["safe"] = 0
    params["time"] = 0
    # if arguments does not equal six return false
    if n > 2:
        params["start_location"] = sys.argv[1]
        params["end_location"] = sys.argv[2]

    if n > 3:
        for i in range(n - 1):
            params[sys.argv[i]] = sys.argv[i + 1]
            i += 1

    return params


# define main function
def main():
    params = get_runtime_arguments()

    if (params["count"] < 3):
        print("Please enter start location and end location and your preferences")
        return

    #file_name = start_location + "_" + end_location
    file_name = "API_data"
    gmaps = initialize_google_api_client()
    directions = get_directions(gmaps, params["start_location"], params["end_location"])

    w1 = int(params["air"])
    w2 = int(params["scene"])
    w3 = int(params["safe"])
    criteria = 0.1

    if w1 == 0:
        if w2 == 0:
            if w3 == 0:
                params["time"] = 1

    if params["time"] != 1:
        _pre_total = w1 + w2 + w3
        w1 /= _pre_total
        w2 /= _pre_total
        w3 /= _pre_total
        params["air"] = w1
        params["scene"] = w2
        params["safe"] = w3

    if (params["count"] == 5):
        criteria = 0.018

    est_gp = SRAlgorithm(w1, w2, w3, criteria)

    save_file(directions, file_name, est_gp, params)

if __name__ == "__main__":
    main()
