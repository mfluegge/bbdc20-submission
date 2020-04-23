import numpy as np


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def get_windows(signal, win_size=20, shift=10, cut=1):
    if cut == 0:
        return np.array([signal[start:min(start + win_size, len(signal))]
                         for start in range(0, len(signal), shift)])
    else:
        return np.array([signal[start:min(start + win_size, len(signal))]
                         for start in range(0, len(signal), shift)][:-cut])


def downsample_signal(signal, method="mean", **kwargs):
    downsampled = get_windows(signal, **kwargs)

    if method == "mean":
        return np.mean(downsampled, axis=1)

    elif method == "abs_mean":
        return np.mean(abs(downsampled), axis=1)

    elif method == "abs_max_orig":
        return _abs_max_orig(downsampled, axis=1)

    elif method == "abs_max":
        return np.amax(abs(downsampled), axis=1)

    else:
        raise ValueError(f"unknown downsampling method {method}")


def _abs_max_orig(a, axis=None):
    """Returns value with max absolute value along an axis:

    https://stackoverflow.com/a/39152275
    """
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)


def get_subject_trial_step_tuples(targets):
    subject_trial_combos = [(subject, trial, window_time)
                            for subject, subject_trials in targets.items()
                            for trial, hand in subject_trials.items()
                            for window_time in hand["ra"]]
    return subject_trial_combos


def get_data_around_time_step(data, step_index, left_time=2, right_time=2,
                              missing_value=np.nan, index_step_size=200):
    left_min_index = int(step_index - left_time * 1000)
    right_max_index = int(step_index + right_time * 1000)

    signal_max_index = max(data.keys())

    data_shape = data[signal_max_index - index_step_size].shape

    step_data = []
    for data_index in range(left_min_index, right_max_index, index_step_size):
        if data_index < 0 or data_index >= signal_max_index:
            adding = np.empty(data_shape)
            adding[:] = missing_value
        else:
            adding = data[data_index]

        step_data.append(adding)

    step_data = np.concatenate(step_data, axis=0)
    return step_data

