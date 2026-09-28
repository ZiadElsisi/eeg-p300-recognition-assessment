import mne
import numpy as np
import pandas as pd
from preprocessing.pipeline import preprocess_eeg,PREPROCESSED_DATA_DIR
from loaders.bnci2014_008 import load_bnci2014_008
from pathlib import Path


#----------- try on a specific subject -----------
SUBJECT_ID= 2
#----------- feature selection parameters -----------
# Time in Milliseconds
CROPPING_WINDOW=[0,800]
WINDOW_RANGE=49 #how wide is this window
STEP=50 #the step between the starting values of each window
N_CHANNELS=8
S_Freq=256
#-------------------------------------------------
def get_subject_raw_data(subject_id:int=SUBJECT_ID,session_id:int|None=None,run_id:int|None=None):
    """
        Retrieve raw EEG data for a specific subject, session, and run from the benchmark dataset.

        Parameters:
        -----------
        subject_id : int, optional
            The identifier of the subject whose data is to be loaded. Defaults to SUBJECT_ID.
        session_id : int or None, optional
            The specific session identifier. If None, automatically selects the first available
            session for the subject.
        run_id : int or None, optional
            The specific run identifier. If None, automatically selects the first available
            run for the chosen session.

        Returns:
        --------
        raw_data : mne.io.RawArray
            The raw MNE time-series data object corresponding to the specified subject, session, and run.
        """
    dataset, raw_test_data = load_bnci2014_008(subjects=[subject_id])
    if session_id is None:
        session_id = next(iter(raw_test_data[subject_id]))
    if run_id is None:
        run_id = next(iter(raw_test_data[subject_id][session_id]))
    raw_data = raw_test_data[subject_id][session_id][run_id]
    return raw_data

def get_and_inspect_epochs(raw_data:mne.io.RawArray=get_subject_raw_data(),file_name:str|None=None):
    """
        Preprocess raw EEG data into clean epochs, optionally save them to disk,
        print key dataset inspection metrics, and return the resulting epochs object.

        This function executes the automated preprocessing pipeline on the input raw EEG data.
        If a `file_name` is provided, the processed epochs are saved to the preprocessed data directory.
        It then extracts the feature matrix (x) and target labels (y), prints structural diagnostics
        (such as dimensions, channels, sampling rate, and event mappings), and returns the epochs.

        Parameters:
        -----------
        raw_data : mne.io.RawArray, optional
            The raw MNE time-series data object to be preprocessed. if not provided get the raw EEG data from default subject,session and run.
        file_name : str or None, optional
            The filename for saving the cleaned epochs as a `.fif` file inside
            `data/preprocessed/`. If None, the epochs are processed in memory
            without saving to disk.

        Returns:
        --------
        final_epochs : mne.Epochs
            The cleaned, trial-segmented MNE Epochs object ready for feature extraction and modeling.
        """
    print("== start preprocessing ==")
    if file_name:
        final_epochs = preprocess_eeg(
            raw_data,
            output_path=Path.joinpath(PREPROCESSED_DATA_DIR, file_name)
        )
    else:
        final_epochs = preprocess_eeg(
            raw_data)
    print("\n")
    x = final_epochs.get_data()
    y = final_epochs.events[:, -1]
    print("== pre feature selection Epochs inspection ==")
    print("EEG shape:", x.shape)
    print("Labels shape:", y.shape)
    print("Channels:", final_epochs.ch_names)
    print("Sampling frequency:", final_epochs.info["sfreq"])
    print("Event mapping:", final_epochs.event_id)
    print("\n")
    return final_epochs

def feature_extraction(epochs:mne.Epochs)->pd.DataFrame:
    try:
        # Validate configuration
        if len(CROPPING_WINDOW) != 2:
            raise ValueError("CROPPING_WINDOW must contain (start, end).")

        start, end = CROPPING_WINDOW

        if start >= end:
            raise ValueError("CROPPING_WINDOW start must be less than end.")

        if STEP <= 0:
            raise ValueError("STEP must be greater than zero.")

        if WINDOW_RANGE <= 0:
            raise ValueError("WINDOW_RANGE must be greater than zero.")

        if len(epochs.ch_names) != N_CHANNELS:
            raise ValueError(
                f"Expected {N_CHANNELS} channels, "
                f"but received {len(epochs.ch_names)}."
            )
        #cropping the time samples
        start=CROPPING_WINDOW[0]
        end=CROPPING_WINDOW[1]
        shorter_epochs = epochs.copy().crop(tmin=round(start/1000,1), tmax=round(end/1000,1), include_tmax=True)
        # number of time samples after cropping
        print("number of time samples before and after cropping the time of epochs:")
        for name, obj in dict(Original=epochs, Cropped=shorter_epochs).items():
            print(f"{name} epochs has {obj.get_data(copy=False).shape[-1]} time samples")
        print("\n")
        # construct the windows
        feature_windows = [
            [x,min(x+WINDOW_RANGE,end)]
            for x in range(int(start), int(end),STEP)
            if x<end
        ]
        # in case the last window ends with "end-1" for ex:(499)
        if feature_windows[-1][-1]!=end:
            feature_windows[-1][-1] = feature_windows[-1][-1] + 1
        # transform to tuples
        feature_windows = [(x, y) for x, y in feature_windows]
        print(f"feature windows: {feature_windows}")
        print("\n")
        # creating the window averages pipeline
        epochs_windows_channel_means = [] #every epoch consists of 8 channels every channel has n windows
        for x, y in feature_windows:
            start_idx=round((x-start)*S_Freq/1000) #the index of each time in the range of (start-end)
            end_idx=round((y-start)*S_Freq/1000)
            data=shorter_epochs.get_data()
            epoch_window = data[:,:,start_idx:end_idx]
            epochs_windows_channel_means.append(epoch_window.mean(axis=2)) #epochs where every epoch consists of 8 channels every channel has 1 value (average of the window)
        data_2d = np.concatenate(epochs_windows_channel_means, axis=1)
        # construct the data frame
        column_names = [
            f'Ch{ch}_Window_{w}_mean'
            for w in range(1, len(feature_windows) + 1)
            for ch in range(1, N_CHANNELS + 1)
        ]
        print(column_names)
        print("\n")
        data_2d = data_2d * 1e6
        features = pd.DataFrame(data_2d, columns=column_names)
        print(f"feature shape: {features.shape}")
        print("\n")
        return features
    except ValueError as e:
        print(e)
        return pd.DataFrame()
def target_extraction(epochs:mne.Epochs)->pd.DataFrame:
    target_labels=epochs.events[:,-1]
    return pd.DataFrame(target_labels,columns=["target"])






