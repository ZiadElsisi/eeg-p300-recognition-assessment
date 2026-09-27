[![DOI](https://img.shields.io/badge/DOI-10.82901%2Fnemar.nm000169-blue)](https://doi.org/10.82901/nemar.nm000169)

# BNCI 2014-008 P300 dataset (ALS patients)

BNCI 2014-008 P300 dataset (ALS patients).

## Dataset Overview

- **Code**: BNCI2014-008
- **Paradigm**: p300
- **DOI**: 10.3389/fnhum.2013.00732
- **Subjects**: 8
- **Sessions per subject**: 1
- **Events**: Target=2, NonTarget=1
- **Trial interval**: [0, 1.0] s
- **File format**: Unknown
- **Data preprocessed**: True

## Acquisition

- **Sampling rate**: 256.0 Hz
- **Number of channels**: 8
- **Channel types**: eeg=8
- **Channel names**: Fz, Cz, Pz, Oz, P3, P4, PO7, PO8
- **Montage**: 10-10
- **Hardware**: g.MOBILAB
- **Software**: BCI2000
- **Reference**: right earlobe
- **Ground**: left mastoid
- **Sensor type**: active electrodes
- **Line frequency**: 50.0 Hz
- **Online filters**: 0.1-10 Hz bandpass, 50 Hz notch
- **Electrode type**: g.Ladybird
- **Electrode material**: Ag/AgCl

## Participants

- **Number of subjects**: 8
- **Health status**: ALS patients
- **Clinical population**: amyotrophic lateral sclerosis
- **Age**: mean=58.0, std=12.0, min=40, max=72
- **Gender distribution**: M=5, F=3
- **BCI experience**: naive
- **Species**: human

## Experimental Protocol

- **Paradigm**: p300
- **Number of classes**: 2
- **Class labels**: Target, NonTarget
- **Study design**: P300 speller with 6x6 matrix for copy-spelling task in ALS patients
- **Feedback type**: visual
- **Stimulus type**: row-column intensification
- **Stimulus modalities**: visual
- **Primary modality**: visual
- **Synchronicity**: synchronous
- **Mode**: online
- **Training/test split**: True
- **Instructions**: Copy spell seven predefined words of five characters each by focusing attention on desired letters

## HED Event Annotations

Schema: HED 8.4.0 | Browse: https://www.hedtags.org/hed-schema-browser

```
  Target
    ├─ Sensory-event
    ├─ Experimental-stimulus
    ├─ Visual-presentation
    └─ Target

  NonTarget
    ├─ Sensory-event
    ├─ Experimental-stimulus
    ├─ Visual-presentation
    └─ Non-target

```
## Paradigm-Specific Parameters

- **Detected paradigm**: p300
- **Number of targets**: 36
- **Number of repetitions**: 10
- **Inter-stimulus interval**: 125.0 ms
- **Stimulus onset asynchrony**: 250.0 ms

## Data Structure

- **Trials**: 35
- **Blocks per session**: 7
- **Trials context**: per subject (7 words, 5 characters each)

## Preprocessing

- **Data state**: preprocessed
- **Preprocessing applied**: True
- **Steps**: bandpass filtering, notch filtering, artifact rejection, baseline correction
- **Highpass filter**: 0.1 Hz
- **Lowpass filter**: 10.0 Hz
- **Bandpass filter**: {'low_cutoff_hz': 0.1, 'high_cutoff_hz': 10.0}
- **Notch filter**: [50] Hz
- **Filter type**: Butterworth
- **Filter order**: 4
- **Artifact methods**: amplitude threshold rejection
- **Re-reference**: right earlobe
- **Epoch window**: [0.0, 1.0]
- **Notes**: Epochs with peak amplitude >70 μV or <-70 μV were rejected. Baseline correction based on 200 ms preceding each epoch.

## Signal Processing

- **Classifiers**: SWLDA
- **Feature extraction**: temporal features, decimation

## Cross-Validation

- **Method**: 7-fold
- **Folds**: 7
- **Evaluation type**: within_subject

## Performance (Original Study)

- **Accuracy**: 97.5%
- **Binary Accuracy Offline**: 87.4
- **P300 Amplitude Mean Uv**: 3.3

## BCI Application

- **Applications**: communication
- **Environment**: laboratory
- **Online feedback**: True

## Tags

- **Pathology**: ALS
- **Modality**: P300
- **Type**: ERP

## Documentation

- **DOI**: 10.3389/fnhum.2013.00732
- **License**: CC-BY-NC-ND-4.0
- **Investigators**: Angela Riccio, Luca Simione, Francesca Schettini, Alessia Pizzimenti, Maurizio Inghilleri, Marta Olivetti Belardinelli, Donatella Mattia, Febo Cincotti
- **Senior author**: Febo Cincotti
- **Contact**: a.riccio@hsantalucia.it
- **Institution**: Fondazione Santa Lucia
- **Department**: Neuroelectrical Imaging and BCI Laboratory
- **Address**: Via Ardeatina, 306, 00179 Rome, Italy
- **Country**: Italy
- **Repository**: BNCI Horizon
- **Publication year**: 2013
- **Funding**: Italian Agency for Research on ALS-ARiSLA project 'Brindisys'; FARI project C26I12AJZZ at the Sapienza University of Rome
- **Ethics approval**: Fondazione Santa Lucia ethic committee
- **Keywords**: brain computer interface, amyotrophic lateral sclerosis, P300, attention, working memory

## References

Riccio, A., Simione, L., Schettini, F., Pizzimenti, A., Inghilleri, M., Belardinelli, M. O., & Mattia, D. (2013). Attention and P300-based BCI performance in people with amyotrophic lateral sclerosis. Frontiers in human neuroscience, 7, 732. https://doi.org/10.3389/fnhum.2013.00732

Notes

.. note::

``BNCI2014_008`` was previously named ``BNCI2014008``. ``BNCI2014008`` will be removed in version 1.1.

.. versionadded:: 0.4.0
Appelhoff, S., Sanderson, M., Brooks, T., Vliet, M., Quentin, R., Holdgraf, C., Chaumon, M., Mikulan, E., Tavabi, K., Hochenberger, R., Welke, D., Brunner, C., Rockhill, A., Larson, E., Gramfort, A. and Jas, M. (2019). MNE-BIDS: Organizing electrophysiological data into the BIDS format and facilitating their analysis. Journal of Open Source Software 4: (1896). https://doi.org/10.21105/joss.01896

Pernet, C. R., Appelhoff, S., Gorgolewski, K. J., Flandin, G., Phillips, C., Delorme, A., Oostenveld, R. (2019). EEG-BIDS, an extension to the brain imaging data structure for electroencephalography. Scientific Data, 6, 103. https://doi.org/10.1038/s41597-019-0104-8

---
Generated by MOABB 1.5.0 (Mother of All BCI Benchmarks)
https://github.com/NeuroTechX/moabb
