import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sensors import *
from pathlib import Path


TRAIN_PATH = Path("train")
TEST_PATH = Path("test")


def load_targets(raw_data_path, index_step_size=200):
    label_path = os.path.join(raw_data_path, TRAIN_PATH, "labels.train.csv")
    train_labels = pd.read_csv(label_path, header=None)

    train_labels["subject"] = train_labels[0].apply(lambda x: x[:3])
    train_labels["trial"] = train_labels[0].apply(lambda x: x[3:6])
    train_labels["arm"] = train_labels[0].apply(lambda x: x.split(".")[1])
    train_labels["start"] = train_labels[1].values
    train_labels["end"] = train_labels[2].values
    train_labels["action"] = train_labels[3].values
    train_labels = train_labels.drop([1, 2, 3], axis=1)

    left_arm_actions = train_labels.loc[train_labels["arm"] == "la"]["action"].values
    right_arm_actions = train_labels.loc[train_labels["arm"] == "ra"]["action"].values
    left_label_encoder = LabelEncoder().fit(left_arm_actions)
    right_label_encoder = LabelEncoder().fit(right_arm_actions)

    targets = {}
    labels_grouped = train_labels.groupby(["subject", "trial", "arm"])
    unique_combinations = train_labels.groupby(["subject", "trial", "arm"]).first().index.values

    for (subject, trial, arm) in unique_combinations:
        current_group = labels_grouped.get_group((subject, trial, arm))

        win_end = index_step_size
        group_targets = {}
        for (ix, row) in current_group.iterrows():
            while win_end < (row["end"] * 1000):
                group_targets[win_end] = row["action"]
                win_end += index_step_size

        if subject not in targets:
            targets[subject] = {trial: {arm: group_targets}}

        elif trial not in targets[subject]:
            targets[subject][trial] = {arm: group_targets}

        else:
            targets[subject][trial][arm] = group_targets

    return targets, left_label_encoder, right_label_encoder


def load_emg_data(raw_data_path, data_type="train", scale_trial=None, 
                  fit_and_return_scaler=None, index_step_size=200, 
                  fit_scaler_on_absolute=False):
    if data_type == "train":
        emg_path = os.path.join(raw_data_path, TRAIN_PATH, "emg")
    elif data_type == "test":
        emg_path = os.path.join(raw_data_path, TEST_PATH, "emg")
    else:
        raise ValueError(f"unknown data type when loading emg data {data_type}")

    global_scaler = None

    if fit_and_return_scaler is not None:
        if fit_and_return_scaler == "standard":
            global_scaler = StandardScaler()

        elif fit_and_return_scaler == "minmax":
            global_scaler = MinMaxScaler()

        elif fit_and_return_scaler == "zero_mean":
            global_scaler = StandardScaler(with_std=False)
        
        elif fit_and_return_scaler == "robust":
            global_scaler = RobustScaler(quantile_range=(15.0, 85.0))

        else:
            raise ValueError(f"unknown global scaler type {fit_and_return_scaler}")

    emg_data = {}
    for fn in os.listdir(emg_path):
        subject = fn[:3]
        trial = fn[3:6]
        print("Loading EMG: ", subject, trial)

        if subject not in emg_data:
            emg_data[subject] = {}

        emg_data[subject][trial] = {}
        emg_trial_data = pd.read_csv(os.path.join(emg_path, fn))
        emg_trial_data = emg_trial_data.interpolate(method="linear", axis=0).fillna(0)
        sensors = [c for c in emg_trial_data.columns if c != "ts"]

        if scale_trial is not None:
            if scale_trial == "standard":
                trial_scaler = StandardScaler()

            elif scale_trial == "minmax":
                trial_scaler = MinMaxScaler()

            elif scale_trial == "zero_mean":
                trial_scaler = StandardScaler(with_std=False)
            
            elif scale_trial == "robust":
                trial_scaler = RobustScaler(quantile_range=(15.0, 85.0))

            else:
                raise ValueError(f"unknown trial scaling {scale_trial}")

            emg_trial_data[sensors] = trial_scaler.fit_transform(emg_trial_data[sensors].values)

        if fit_and_return_scaler is not None:
            if fit_scaler_on_absolute:
                global_scaler.partial_fit(abs(emg_trial_data[sensors].values))
            else:
                global_scaler.partial_fit(emg_trial_data[sensors].values)

        emg_trial_data["window"] = emg_trial_data["ts"].apply(
            lambda x: int(x * 1000) // index_step_size * index_step_size)

        unique_windows = emg_trial_data["window"].unique()
        window_groups = emg_trial_data.groupby("window")

        for win in unique_windows:
            emg_data[subject][trial][win] = window_groups.get_group(win).drop(["ts", "window"], axis=1).values

    if fit_and_return_scaler:
        return emg_data, global_scaler
    else:
        return emg_data


def load_mocap_data(raw_data_path, data_type="train", index_step_size=200,
                    fit_and_return_scaler=None,
                    use_sensors="all"):
    if data_type == "train":
        mocap_path = os.path.join(raw_data_path, TRAIN_PATH, "mocap")
    elif data_type == "test":
        mocap_path = os.path.join(raw_data_path, TEST_PATH, "mocap")
    else:
        raise ValueError(f"what do you mean '{data_type}'")

    scaler = None
    if fit_and_return_scaler is not None:
        if fit_and_return_scaler == "standard":
            scaler = StandardScaler()
        elif fit_and_return_scaler == "minmax":
            scaler = MinMaxScaler()
        elif fit_and_return_scaler == "zero_mean":
            scaler = StandardScaler(with_std=False)
        elif fit_and_return_scaler == "robust":
            scaler = RobustScaler(quantile_range=(0.05, 0.95))
        else:
            raise ValueError(f"unknown mocap scaler type {fit_and_return_scaler}")

    mocap_data = {}
    for fn in os.listdir(mocap_path):
        subject = fn[:3]
        trial = fn[3:6]
        print("Loading MoCap: ", subject, trial)

        if subject not in mocap_data:
            mocap_data[subject] = {}

        mocap_data[subject][trial] = {}
        mocap_trial_data = pd.read_csv(os.path.join(mocap_path, fn))
        mocap_trial_data = mocap_trial_data.interpolate(method="linear", axis=0).fillna(0)

        if use_sensors == "all":
            use_sensors = {sens.split("_")[0]: "all" for sens in mocap_trial_data.columns
                           if sens.split("_")[0] != "ts" and sens.split("_")[0] not in IGNORE_SENSORS}

        elif use_sensors == "position":
                use_sensors = {sens.split("_")[0]: "position" for sens in mocap_trial_data.columns
                               if sens.split("_")[0] != "ts" and sens.split("_")[0] not in IGNORE_SENSORS}
        
        elif use_sensors == "rotation":
            use_sensors = {sens.split("_")[0]: "rotation" for sens in mocap_trial_data.columns
                           if sens.split("_")[0] != "ts" and sens.split("_")[0] not in IGNORE_SENSORS}

        sensors = []
        for sensor_position, sensor_types in use_sensors.items():
            if sensor_types == "all":
                sensor_types = ALL_SENSOR_TYPES
            elif sensor_types == "position":
                sensor_types = POSITION_SENSOR_TYPES
            elif sensor_types == "rotation":
                sensor_types = ROTATION_SENSOR_TYPES
            elif isinstance(sensor_types, (list, tuple)):
                pass
            else:
                raise ValueError(f"unknown sensor type(s) {sensor_types}")

            for sensor_type in sensor_types:
                sensor_name = sensor_position + "_" + sensor_type
                sensors.append(sensor_name)

                if sensor_position in SENSOR_REFERENCE:
                    reference_sensor = SENSOR_REFERENCE[sensor_position]
                    reference_sensor_name = reference_sensor + "_" + sensor_type

                    if reference_sensor_name == "Chest_Position_Y" or "Rotation" in reference_sensor_name:
                        pass
                    else:
                        print("Subtracting", reference_sensor_name, "from", sensor_name)

                        mocap_trial_data[sensor_name] = (mocap_trial_data[sensor_name]
                                                        - mocap_trial_data[reference_sensor_name])

        if fit_and_return_scaler:
            scaler.partial_fit(mocap_trial_data[sensors].values)

        mocap_trial_data["window"] = mocap_trial_data["ts"].apply(
            lambda x: int(x * 1000) // index_step_size * index_step_size)

        unique_windows = mocap_trial_data["window"].unique()
        window_groups = mocap_trial_data.groupby("window")
        for win in unique_windows:
            mocap_data[subject][trial][win] = window_groups.get_group(win)[sensors].values

    print(sensors)
    if fit_and_return_scaler:
        return mocap_data, sensors, scaler
    else:
        return mocap_data, sensors
