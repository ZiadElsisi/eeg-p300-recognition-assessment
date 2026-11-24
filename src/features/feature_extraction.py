import mne
import numpy as np
import pandas as pd
from src.preprocessing.pipeline import preprocess_eeg,PREPROCESSED_DATA_DIR
from src.loaders.bnci2014_008 import load_bnci2014_008
from pathlib import Path
import json
from mne.preprocessing import Xdawn

# FEATURE ENGINEERING TECHNIQUES METHODS AND CLASSES
def automatic_feature_window_construction(epochs:mne.Epochs,offset:int=3,threshold:float=0.8)->list:#
    """
    Constructs adaptive feature windows by comparing the spatial
    distributions of Target and NonTarget average EEG signals using
    cosine similarity.

    Parameters:
        epochs (mne.Epochs): Cropped EEG epochs containing Target and
                             NonTarget trials.
        offset (int): Number of samples between compared time points.
                      Default is 3.
        threshold (float): Cosine similarity threshold used to determine
                           when a new feature window is created.
                           Default is 0.8 .

    Returns:
        list: Feature windows, each containing the starting and ending
              sample indices.
    """
    if offset <= 0: ## The offset between considered samples
        raise ValueError("offset must be greater than zero.")

    if not -1 <= threshold <= 1: ## Cosine similarity threshold
        raise ValueError("threshold must be between -1 and 1.")


    target_epochs=epochs.copy()["Target"].get_data()
    nontarget_epochs = epochs.copy()["NonTarget"].get_data()

    # Each column in Average represents a time point, and each row represents an EEG channel.
    target_average_epoch=np.mean(target_epochs,axis=0)
    nontarget_average_epoch=np.mean(nontarget_epochs,axis=0)

    average_epoch=target_average_epoch-nontarget_average_epoch
    n_samples=average_epoch.shape[-1]

    current_start=0
    feature_window=[] # Stores the detected window boundaries.
    '''For example : 
    [
    [0, 24],
    [25, 60],
    [61, 110],
    [111, 205]
]
    '''
    for i in range(0,n_samples-offset,offset): ## jump on epochs by offset

        vector1=average_epoch[:,i] ## All channels values at sample i
        vector2=average_epoch[:,i+offset] ## all channel values at sample i+offset
        denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)

        # Skip comparisons where a vector is almost zero
        if denominator < 1e-12:
            continue

        similarity = np.dot(vector1, vector2) / denominator

        if similarity<=threshold:
            feature_window.append([current_start,i+offset-1])
            current_start=i+offset
    if current_start<n_samples:
        feature_window.append([current_start,n_samples-1])
    return feature_window

class SpatialFilter:
    def __init__(self,n_components:int=8):
        self.__spatial_filter=Xdawn(n_components=n_components,correct_overlap=False)
        self.is_fitted_=False
    def fit(self,epochs):
        self.__spatial_filter.fit(epochs)
        self.is_fitted_=True
        return self
    def transform(self,epochs):
        if not self.is_fitted_:
            raise RuntimeError("SpatialFilter must be fitted first.")
        # apply the spatial filter on the testing epochs
        transformed_data = self.__spatial_filter.transform(epochs)
        print("TRANSFORMED DATA ",transformed_data.shape)
        info = mne.create_info(
            ch_names=[
                f"Xdawn{i + 1}"
                for i in range(transformed_data.shape[1])
            ],
            sfreq=epochs.info["sfreq"],
            ch_types="eeg"
        )

        transformed_epochs = mne.EpochsArray(
            data=transformed_data,
            info=info,
            events=epochs.events.copy(),
            event_id=epochs.event_id,
            tmin=epochs.tmin,
            baseline=None
        )
        # The filter is learned using the ERP associated with an event type (in our case: Target), but it can be applied to every epoch regardless of its actual class.
        return transformed_epochs
    def fit_transform(self,epochs):
        return self.fit(epochs).transform(epochs)
    def set_n_components(self,n_components:int):
        self.__spatial_filter.n_components=n_components


# THE MAIN FEATURE EXTRACTION CLASS
class FeatureExtraction:
    def __init__(
            self,
            cropping_start=0,
            cropping_end=800,
            n_channels=8,
            n_components=8,
            threshold=0.8,
            offset=3
    ):
        # user defined attributes
        self.cropping_window = (cropping_start, cropping_end)
        self.n_channels = n_channels
        self.n_components = n_components
        self.threshold = threshold
        self.offset = offset
        # spatial filter
        self.spatial_filter = SpatialFilter(n_components)
        # learned attributes
        self.feature_windows_ = None
        self.feature_names_ = None
        self.is_fitted_ = False

    def _validate_epochs(self, epochs):
        start, end = self.cropping_window

        if start >= end:
            raise ValueError("Invalid cropping window.")

        if self.offset <= 0:
            raise ValueError("Offset must be positive.")

        if not -1 <= self.threshold <= 1:
            raise ValueError("Threshold must be between -1 and 1.")

    def _crop_epochs(self, epochs):

        start, end = self.cropping_window

        return epochs.copy().crop(
            tmin=start / 1000,
            tmax=end / 1000,
            include_tmax=True
        )

    def _construct_windows(self, epochs):

        return automatic_feature_window_construction(
            epochs,
            offset=self.offset,
            threshold=self.threshold
        )

    def _extract_averages(self, epochs):
        data = epochs.get_data()

        window_averages = []

        for start, end in self.feature_windows_:
            window_data = data[:, :, start:end + 1]

            window_averages.append(
                window_data.mean(axis=2)
            )

        data_2d = np.concatenate(
            window_averages,
            axis=1
        )

        data_2d *= 1e6

        return pd.DataFrame(
            data_2d,
            columns=self.feature_names_
        )

    def fit(self, epochs):
        self._validate_epochs(epochs)
        cropped_epochs = self._crop_epochs(epochs)
        # Learn spatial filters
        self.spatial_filter.fit(cropped_epochs)

        # Transform training epochs
        filtered_epochs = self.spatial_filter.transform(
            cropped_epochs
        )
        ## We can Replace the above two lines by fit transrom

        # Learn adaptive feature windows from training data
        self.feature_windows_ = self._construct_windows(
            filtered_epochs
        )
        n_components=filtered_epochs.get_data().shape[1]
        print("NUMBER OF COMPONENTS: ", n_components)
        self.feature_names_ = [
            f"Ch{ch}_Window_{w}_mean"
            for w in range(1, len(self.feature_windows_) + 1)
            for ch in range(1, n_components + 1)
        ]
        self.is_fitted_ = True
        return self

    def transform(self, epochs):

        if not self.is_fitted_:
            raise RuntimeError(
                "FeatureExtractor must be fitted first."
            )

        self._validate_epochs(epochs)
        cropped_epochs = self._crop_epochs(epochs)

        # Apply previously learned spatial filters
        filtered_epochs = self.spatial_filter.transform(
            cropped_epochs
        )
        # Extract features using previously learned windows
        return self._extract_averages(filtered_epochs)

    def fit_transform(self, epochs):

        self.fit(epochs)

        return self.transform(epochs)
# target extraction method
def target_extraction(epochs:mne.Epochs)->pd.DataFrame:
    epochs_copy=epochs.copy()
    target_labels=epochs_copy.events[:,-1]
    return pd.DataFrame(target_labels,columns=["target"])
