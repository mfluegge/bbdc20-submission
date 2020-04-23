import numpy as np
import pandas as pd
import os
import sys
from sklearn.preprocessing import LabelEncoder, RobustScaler, LabelBinarizer
import random
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

from utils import *
from load_data import *


def get_mean_diff(seqs):
    return np.mean(np.diff(seqs, axis=0), axis=0)

def get_start_end_diff(seqs, win_size=3):
    return np.sum(seqs[-win_size:], axis=0) - np.sum(seqs[:win_size], axis=0)

def get_mean_value(seqs):
    return np.mean(seqs, axis=0)

def get_min_value(seqs):
    return np.amin(seqs, axis=0)

def get_max_value(seqs):
    return np.amax(seqs, axis=0)

def build_all_features(seqs):
    start_end_diffs = get_start_end_diff(seqs, win_size=7)
    mean_values = get_mean_value(seqs)
    maxs = get_max_value(seqs)
    mins = get_min_value(seqs)
    mean_diffs = get_mean_diff(seqs)
    all_feats = [mean_values, mean_diffs, start_end_diffs, maxs, mins]
    return np.nan_to_num(np.concatenate(all_feats))

def add_mocap_feats(get_from, add_to, win_time, left_time=0, right_time=0, use=None):
    mocap_data = get_data_around_time_step(get_from, win_time,
                                           left_time=left_time, right_time=right_time,
                                           index_step_size=STEP_SIZE)
    if use is not None:
        mocap_data = mocap_data[:, use]
        
    add_to.append(build_all_features(mocap_data)) 

def train_models(features, targets, left_encoder, right_encoder, seed,
                 n_models=7):
    x_train = []
    y_train_left = []
    y_train_right = []
    sample_weight = []
    for subject in features:
        for trial_name, trial_data in targets[subject].items():
            y_train_left += list(trial_data["la"].values())
            y_train_right += list(trial_data["ra"].values())
            
            data = features[subject][trial_name]
            x_train += list(data.values())

            if subject == "s05":
                sample_weight += ([2] * len(data.values()))
            else:
                sample_weight += ([1] * len(data.values()))

    x_train = np.array(x_train)    
    y_train_left = left_encoder.transform(y_train_left)
    y_train_right = right_encoder.transform(y_train_right)
    sample_weight = np.array(sample_weight)

    base_params = {
        "objective": "multiclass",
        "learning_rate": 0.01, 
        "num_class": 6,
        "num_leaves": 50,
        "max_depth": 50,
        "min_data_in_leaf": 10,
        "feature_fraction": 0.2,
        "bagging_fraction": 0.1,
        "bagging_freq": 3,
        "n_jobs": 4,
        "seed": SEED,
        "max_bin": 12
    }
    
    print("---- Training left models")
    all_left_models = []
    for iteration in range(n_models):
        rounds = np.random.randint(600, 800)
        leaves = np.random.randint(15, 60)
        bins = np.random.randint(8, 16)
        min_data_in_leaf = np.random.randint(5, 25)
        learning_rate = np.random.randint(8, 12) / 1000
        
        base_params["num_leaves"] = leaves
        base_params["min_data_in_leaf"] = min_data_in_leaf
        base_params["learning_rate"] = learning_rate
        base_params["max_bin"] = bins
        base_params["seed"] = SEED + iteration

        print("left iteration", iteration + 1)      
        print("params", base_params)
        print("rounds", rounds)
        train_set = lgb.Dataset(x_train, y_train_left, weight=sample_weight)
        m = lgb.train(base_params, num_boost_round=rounds, train_set=train_set)
        all_left_models.append(m)
    
    print("---- Training right models")
    all_right_models = []
    for iteration in range(n_models):
        rounds = np.random.randint(600, 800)
        leaves = np.random.randint(15, 60)
        bins = np.random.randint(8, 16)
        min_data_in_leaf = np.random.randint(5, 25)
        learning_rate = np.random.randint(8, 12) / 1000
        
        base_params["num_leaves"] = leaves
        base_params["min_data_in_leaf"] = min_data_in_leaf
        base_params["learning_rate"] = learning_rate
        base_params["max_bin"] = bins
        base_params["seed"] = SEED + iteration

        print("right iteration", iteration + 1)      
        print("params", base_params)
        print("rounds", rounds)
        train_set = lgb.Dataset(x_train, y_train_right, weight=sample_weight)
        m = lgb.train(base_params, num_boost_round=rounds, train_set=train_set)
        all_right_models.append(m)
    
    return all_left_models, all_right_models


if __name__ == "__main__":
    # pass path to raw data pls
    if len(sys.argv) < 2:
        sys.exit("Speficy path to bbdc data as first input argument.")
    
    raw_data_path = sys.argv[1]

    SEED = 1337
    random.seed(SEED)
    np.random.seed(SEED)
    STEP_SIZE = 200

    print("Loading training data")
    train_targets, left_encoder, right_encoder = load_targets(raw_data_path, index_step_size=STEP_SIZE)
    subject_trial_steps = get_subject_trial_step_tuples(train_targets)

    mocap_data, sensors_used = load_mocap_data(
        raw_data_path,
        index_step_size=STEP_SIZE,
        fit_and_return_scaler=None,
        use_sensors={
            "LHand": "position",
            "RHand": "position",
            "Chest": ["Position_X", "Position_Z"] 
        }
    )

    emg_data = load_emg_data(raw_data_path, index_step_size=STEP_SIZE, fit_and_return_scaler=None)

    print("Fixing EMG for subjects 2/4")
    left_onehot = LabelBinarizer()
    right_onehot = LabelBinarizer()
    left_onehot.fit(left_encoder.transform(
        np.array(list(train_targets["s01"]["t01"]["la"].values()))))
    right_onehot.fit(right_encoder.transform(
        np.array(list(train_targets["s01"]["t01"]["ra"].values()))))


    # fixing emg 4 for subjects 2 + 4
    subject_emgs = {}
    for sub in ["s01", "s02", "s03", "s04", "s05"]:
        all_emg_sub = []
        sub_emg = emg_data[sub]
        all_labels_sub = []
        for t in sub_emg:
            data = np.concatenate(list(sub_emg[t].values()))
            labels_left = np.repeat(list(train_targets[sub][t]["la"].values()), 120)
            labels_right = np.repeat(list(train_targets[sub][t]["ra"].values()), 120)

            if len(labels_left) > len(data):
                labels_left = labels_left[:len(data)]
                labels_right = labels_right[:len(data)]
            elif len(labels_left) < len(data):
                labels_left = np.concatenate((labels_left, 
                                            np.repeat(labels_left[-1], 
                                                        len(data) - len(labels_left) )))
                labels_right = np.concatenate((labels_right,
                                            np.repeat(labels_right[-1], 
                                                        len(data) - len(labels_right) )))

            labels_left = left_onehot.transform(left_encoder.transform(labels_left))
            labels_right = right_onehot.transform(right_encoder.transform(labels_right))
            labels = np.concatenate((labels_left, labels_right), axis=1)
            all_labels_sub.append(labels)
            all_emg_sub.append(data)

        all_emg_sub = np.concatenate(all_emg_sub)
        all_labels_sub = np.concatenate(all_labels_sub)

        feats = all_emg_sub[:, [0, 1, 2, 3, 5, 6, 7]]
        feats = np.concatenate((feats, all_labels_sub), axis=1)
        targets = all_emg_sub[:, 4]
        subject_emgs[sub] = {"feats": feats, "targets": targets}

    train_data = np.concatenate((subject_emgs["s01"]["feats"], 
                                subject_emgs["s03"]["feats"],
                                subject_emgs["s05"]["feats"]))

    targets = np.concatenate((subject_emgs["s01"]["targets"], 
                                subject_emgs["s03"]["targets"],
                                subject_emgs["s05"]["targets"]))
    emg_4_model = LinearRegression().fit(train_data, targets)

    emg_4_fix_subjects = ["s02", "s04"]
    for sub in emg_4_fix_subjects:
        subject_emg_data = emg_data[sub]
        for trial_name, trial_data in subject_emg_data.items():
            for win_time, win_data in trial_data.items():
                if win_time + 200 in train_targets[sub][trial_name]["la"]:
                    left_label = np.repeat(train_targets[sub][trial_name]["la"][win_time + 200], len(win_data))
                    right_label = np.repeat(train_targets[sub][trial_name]["ra"][win_time+ 200], len(win_data))
                else:
                    left_label = ["la-nothing"] * len(win_data)
                    right_label = ["ra-nothing"] * len(win_data)
                
                win_data = win_data[:, [0, 1, 2, 3, 5, 6, 7]]
                left_label = left_onehot.transform(left_encoder.transform(left_label))
                right_label = right_onehot.transform(right_encoder.transform(right_label))
                win_data = np.concatenate((win_data, left_label, right_label), axis=1)
                
                emg_4_fixed = emg_4_model.predict(win_data)
                trial_data[win_time][:, 4] = emg_4_fixed
            
            
    # fixing emg 0 for subject 4
    subject_emgs = {}
    for sub in ["s01", "s02", "s03", "s04", "s05"]:
        all_emg_sub = []
        sub_emg = emg_data[sub]
        all_labels_sub = []
        for t in sub_emg:
            data = np.concatenate(list(sub_emg[t].values()))
            labels_left = np.repeat(list(train_targets[sub][t]["la"].values()), 120)
            labels_right = np.repeat(list(train_targets[sub][t]["ra"].values()), 120)

            if len(labels_left) > len(data):
                labels_left = labels_left[:len(data)]
                labels_right = labels_right[:len(data)]
            elif len(labels_left) < len(data):
                labels_left = np.concatenate((labels_left, 
                                            np.repeat(labels_left[-1], 
                                                        len(data) - len(labels_left) )))
                labels_right = np.concatenate((labels_right,
                                            np.repeat(labels_right[-1], 
                                                        len(data) - len(labels_right) )))

            labels_left = left_onehot.transform(left_encoder.transform(labels_left))
            labels_right = right_onehot.transform(right_encoder.transform(labels_right))
            labels = np.concatenate((labels_left, labels_right), axis=1)
            all_labels_sub.append(labels)
            all_emg_sub.append(data)

        all_emg_sub = np.concatenate(all_emg_sub)
        all_labels_sub = np.concatenate(all_labels_sub)

        feats = all_emg_sub[:, [1, 2, 4, 3, 5, 6, 7]]
        feats = np.concatenate((feats, all_labels_sub), axis=1)
        targets = all_emg_sub[:, 0]
        subject_emgs[sub] = {"feats": feats, "targets": targets}

    train_data = np.concatenate((subject_emgs["s01"]["feats"],
                                subject_emgs["s02"]["feats"],
                                subject_emgs["s03"]["feats"],
                                subject_emgs["s05"]["feats"]))

    targets = np.concatenate((subject_emgs["s01"]["targets"], 
                            subject_emgs["s02"]["targets"],
                            subject_emgs["s03"]["targets"],
                            subject_emgs["s05"]["targets"]))
    emg_0_model = LinearRegression().fit(train_data, targets)

    emg_0_fix_subjects = ["s04"]
    for sub in emg_4_fix_subjects:
        subject_emg_data = emg_data[sub]
        for trial_name, trial_data in subject_emg_data.items():
            for win_time, win_data in trial_data.items():
                if win_time + 200 in train_targets[sub][trial_name]["la"]:
                    left_label = np.repeat(train_targets[sub][trial_name]["la"][win_time + 200], len(win_data))
                    right_label = np.repeat(train_targets[sub][trial_name]["ra"][win_time+ 200], len(win_data))
                else:
                    left_label = ["la-nothing"] * len(win_data)
                    right_label = ["ra-nothing"] * len(win_data)
                
                win_data = win_data[:, [1, 2, 3, 4, 5, 6, 7]]
                left_label = left_onehot.transform(left_encoder.transform(left_label))
                right_label = right_onehot.transform(right_encoder.transform(right_label))
                win_data = np.concatenate((win_data, left_label, right_label), axis=1)
                
                emg_0_fixed = emg_0_model.predict(win_data)
                trial_data[win_time][:, 0] = emg_0_fixed

    subject_scalers = {}
    for subject in emg_data:
        subject_datas = []
        for trial_name, trial_data in emg_data[subject].items():
            for win_data in trial_data.values():
                subject_datas.append(win_data)

        subject_datas = np.concatenate(subject_datas, axis=0)
        scaler = RobustScaler(quantile_range=(25.0, 75.0)).fit(subject_datas)
        subject_scalers[subject] = scaler

    # Building features
    features = {}
    for s in set(train_targets.keys()):
        print("Building features fot subject", s)
        features[s] = {}
        
        for trial_name, trial_targets in train_targets[s].items():
            print("---- Building features for trial", trial_name)
            features[s][trial_name] = {}
            
            emg_trial_data = emg_data[s][trial_name]
            mocap_trial_data = mocap_data[s][trial_name]
            window_end_times = list(trial_targets["la"].keys())
            max_end_time = max(window_end_times)
            trial_window_features = {}
            for i, win_time in enumerate(window_end_times):
                current_window_features = []

                use = ["LHand_Position_X", "LHand_Position_Y", "LHand_Position_Z",
                    "RHand_Position_X", "RHand_Position_Y", "RHand_Position_Z",
                    "Chest_Position_X", "Chest_Position_Z"]
                
                hands = ["LHand_Position_X", "LHand_Position_Y", "LHand_Position_Z",
                        "RHand_Position_X", "RHand_Position_Y", "RHand_Position_Z"]
                
                chest = ["Chest_Position_X", "Chest_Position_Z"]
                
                use_ix = [ix for ix, s in enumerate(sensors_used) if s in use]
                
                add_mocap_feats(mocap_trial_data, current_window_features, 
                                win_time, left_time=0.4, use=use_ix)
                
                add_mocap_feats(mocap_trial_data, current_window_features, 
                                win_time, right_time=0.4, use=use_ix)
        
                add_mocap_feats(mocap_trial_data, current_window_features, 
                                win_time, left_time=1.2, use=use_ix)
                
                add_mocap_feats(mocap_trial_data, current_window_features, 
                                win_time, right_time=1.2, use=use_ix)  
                
                add_mocap_feats(mocap_trial_data, current_window_features, 
                                win_time, left_time=2, use=use_ix)
                
                add_mocap_feats(mocap_trial_data, current_window_features, 
                                win_time, right_time=2, use=use_ix)
                
                add_mocap_feats(mocap_trial_data, current_window_features, 
                                win_time, left_time=3.4, use=use_ix)
                
                add_mocap_feats(mocap_trial_data, current_window_features, 
                                win_time, right_time=3.4, use=use_ix)

                emg_2 = get_data_around_time_step(emg_trial_data, win_time, left_time=2,
                                                right_time=2, missing_value=0,
                                                index_step_size=STEP_SIZE)
        
                emg_timesteps = subject_scalers[s].transform(emg_2)
                emg_timesteps = np.nan_to_num(emg_timesteps)
                emg_2_wins = get_windows(emg_timesteps, win_size=200, shift=200, cut=0)
                emg_win_means = np.mean(abs(emg_2_wins), axis=1)
                current_window_features.append(emg_win_means.reshape(-1))
                
                current_window_features = np.concatenate(current_window_features)
                features[s][trial_name][win_time] = current_window_features
                
    print("Loading test data")
    mocap_test_data, _ = load_mocap_data(
        raw_data_path,
        index_step_size=STEP_SIZE,
        data_type="test",
        fit_and_return_scaler=None, 
        use_sensors={
            "LHand": "position",
            "RHand": "position",
            "Chest": ["Position_X", "Position_Z"]
        }
    )

    emg_test_data = load_emg_data(
        raw_data_path,
        index_step_size=STEP_SIZE,
        data_type="test",
        fit_and_return_scaler=None
    )


    print("Fitting test subject emg scalers")
    test_subject_datas = []
    for trial_name, trial_data in emg_test_data["s06"].items():
        trial_datas = []
        for win_data in trial_data.values():
            test_subject_datas.append(win_data)
            
    test_subject_datas = np.concatenate(test_subject_datas, axis=0)
    test_scaler = RobustScaler(quantile_range=(25.0, 75.0)).fit(test_subject_datas)


    print("Building test features")
    test_rows = []
    test_features = []
    s = "s06"
    for trial_name, trial in mocap_test_data[s].items():
        print("---- ", trial_name)
        emg_trial_data = emg_test_data[s][trial_name]
        mocap_trial_data = mocap_test_data[s][trial_name]
        window_end_times = list(mocap_trial_data.keys())
        
        for win_time in window_end_times:
            current_window_features = []

            use = ["LHand_Position_X", "LHand_Position_Y", "LHand_Position_Z",
                   "RHand_Position_X", "RHand_Position_Y", "RHand_Position_Z",
                   "Chest_Position_X", "Chest_Position_Z"]
            
            hands = ["LHand_Position_X", "LHand_Position_Y", "LHand_Position_Z",
                    "RHand_Position_X", "RHand_Position_Y", "RHand_Position_Z"]
            
            chest = ["Chest_Position_X", "Chest_Position_Z"]
            
            use_ix = [ix for ix, s in enumerate(sensors_used) if s in use]
            
            add_mocap_feats(mocap_trial_data, current_window_features, 
                            win_time, left_time=0.4, use=use_ix)
            
            add_mocap_feats(mocap_trial_data, current_window_features, 
                            win_time, right_time=0.4, use=use_ix)
    
            add_mocap_feats(mocap_trial_data, current_window_features, 
                            win_time, left_time=1.2, use=use_ix)
            
            add_mocap_feats(mocap_trial_data, current_window_features, 
                            win_time, right_time=1.2, use=use_ix)  
            
            add_mocap_feats(mocap_trial_data, current_window_features, 
                            win_time, left_time=2, use=use_ix)
            
            add_mocap_feats(mocap_trial_data, current_window_features, 
                            win_time, right_time=2, use=use_ix)
            
            add_mocap_feats(mocap_trial_data, current_window_features, 
                            win_time, left_time=3.4, use=use_ix)
            
            add_mocap_feats(mocap_trial_data, current_window_features, 
                            win_time, right_time=3.4, use=use_ix)

            emg_2 = get_data_around_time_step(emg_trial_data, win_time, left_time=2,
                                              right_time=2, missing_value=0,
                                              index_step_size=STEP_SIZE)
    
            emg_timesteps = test_scaler.transform(emg_2)
            emg_timesteps = np.nan_to_num(emg_timesteps)
            emg_2_wins = get_windows(emg_timesteps, win_size=200, shift=200, cut=0)
            emg_win_means = np.mean(abs(emg_2_wins), axis=1)
            current_window_features.append(emg_win_means.reshape(-1))
            
            current_window_features = np.concatenate(current_window_features)
            test_features.append(current_window_features)
            test_rows.append([s, trial_name, win_time])
            
    test_features = np.array(test_features)
    test_times = pd.DataFrame(test_rows, columns=["subject", "trial", "ts"])

    print("Training models")
    left_models, right_models = train_models(
                                    features, train_targets, left_encoder, 
                                    right_encoder, SEED,
                                    n_models=11)

    print("Predicting test data")
    left_preds = []
    for model in left_models:
        left_preds.append(model.predict(test_features).argmax(axis=1))
    
    right_preds = []
    for model in right_models:
        right_preds.append(model.predict(test_features).argmax(axis=1))

    left_preds = np.array(left_preds).swapaxes(0, 1)
    right_preds = np.array(right_preds).swapaxes(0, 1)

    left_test_predictions = []
    for p in left_preds:
        left_test_predictions.append(np.bincount(p).argmax())

    right_test_predictions = []
    for p in right_preds:
        right_test_predictions.append(np.bincount(p).argmax())


    print("Building submission file")

    test_df = test_times.copy()
    test_df["left_pred"] = left_encoder.inverse_transform(left_test_predictions)
    test_df["right_pred"] = right_encoder.inverse_transform(right_test_predictions)
    test_df["key"] = test_df["subject"] + test_df["trial"]

    trial_groups = test_df.groupby("key")
    unique_trials = test_df["key"].unique()

    new_left_rows = []
    new_right_rows = []
    for trial_key in unique_trials:
        print(trial_key)
        trial_group = trial_groups.get_group(trial_key)
        trial_group = trial_group.sort_values(by="ts")
        
        # generating left predictions
        current_action = None
        current_start = None
        for ix, (subject, trial, ts, left_pred, right_pred, key) in trial_group.iterrows():
            key = subject + trial + "." + "la"
            
            if current_action is None:
                current_action = left_pred
                current_start = ts
            
            elif current_action != left_pred:
                new_left_rows.append([key, current_start / 1000, ts / 1000, current_action])
                current_start = ts
                current_action = left_pred
            
            else:
                continue
        if current_start / 1000 != ts/1000:
            new_left_rows.append([key, current_start / 1000, ts / 1000, current_action])
        
                
        # generating right predictions
        current_action = None
        current_start = None
        for ix, (subject, trial, ts, left_pred, right_pred, key) in trial_group.iterrows():
            key = subject + trial + "." + "ra"
            
            if current_action is None:
                current_action = right_pred
                current_start = ts
            
            elif current_action != right_pred:
                new_right_rows.append([key, current_start / 1000, ts / 1000, current_action])
                current_start = ts
                current_action = right_pred
            
            else:
                continue

        if current_start / 1000 != ts/1000:
            new_right_rows.append([key, current_start / 1000, ts / 1000, current_action])


    test_rows_combined = new_left_rows + new_right_rows

    submission_df = pd.DataFrame(test_rows_combined)

    submission_df.to_csv("submission.csv", index=False, header=None)