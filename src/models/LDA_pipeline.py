import json
import mne
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, roc_auc_score,classification_report,auc,precision_recall_curve)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import src.features.feature_extraction as ext
from src.preprocessing.pipeline import PROJECT_ROOT
#----------- Fix a Seed number -----------
SEED=42
TEST_SIZE=0.2
EVALUATION_RESULTS_DIR = PROJECT_ROOT/"results"/"lda_holdout_evaluation"
#----------- target/non target labels -----------
labels={
    "target":2,
    "nontarget":1
}

def lda_pipeline_holdout_evaluation(features:pd.DataFrame,target:pd.DataFrame | pd.Series,selected_subjects:dict[str,int]={},verbose:bool=False):

    #------------------------------------------------
    # Report number of Target/NonTarget
    # ------------------------------------------------
    if verbose:
        num_of_targets = (target["target"] == 2).sum()
        print(f"number of target samples: {num_of_targets}")
        print(f"number of non target samples: {len(target) - num_of_targets}")
        print("\n")

    #------------------------------------------------
    # Fixed splitting
    # ------------------------------------------------
    # 1d array is expected so we convert to 1d series
    y = target["target"]
    x = features.copy()
    x.drop(columns=["subject_id"],inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=SEED,stratify=target)  # split based on the population real distribution where the non target is the majority
    # save train,test indices
    indices={
        "train_idx":x_train.index.tolist(),
        "test_idx":x_test.index.tolist()
    }

    EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROJECT_ROOT/"results"/"lda_holdout_evaluation"/"train_test_split.json",'w') as file:
        json.dump(indices,file,indent=4)
    #------------------------------------------------
    # Create Pipeline
    # ------------------------------------------------
    pipe=Pipeline([("scaler",StandardScaler()),
                  ("classifier",LDA(priors=[0.5,0.5]))
                   ])
    #------------------------------------------------
    # Train the model
    # ------------------------------------------------
    pipe.fit(x_train,y_train)

    #------------------------------------------------
    # Predict
    # ------------------------------------------------

    # as probabilities (how confident are we that the sample is target)
    target_column = np.where(
        pipe.classes_ == labels["target"]
    )[0]
    predictions_prob=pipe.predict_proba(x_test)[:,target_column]

    # as labels
    prediction=pipe.predict(x_test)

    #------------------------------------------------
    # Holdout Evaluation
    # ------------------------------------------------
    evaluation_report(y_test,prediction,predictions_prob,selected_subjects)
    return prediction,predictions_prob

def evaluation_report(y_true,y_pred,y_pred_prob,selected_subjects:dict[str,int]={}):
    evaluation_figures_dir = PROJECT_ROOT / "figures" / "hold_out_evaluation"
    evaluation_figures_dir.mkdir(parents=True, exist_ok=True)
    #------------------------------------------------
    # PR-Auc curve
    # ------------------------------------------------
    pr_precision,pr_recall,pr_thresholds= precision_recall_curve(y_true,y_pred_prob,pos_label=labels["target"])
    pr_auc=round(auc(pr_recall,pr_precision), 2)
    plt.plot(pr_recall,pr_precision,label=f"AUC = {pr_auc}")
    baseline=(y_true==2).sum()/y_true.shape[0]
    plt.axhline(y=baseline, linestyle='--', color='gray',label="Baseline")
    plt.xlabel("Recall")#
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(PROJECT_ROOT/"figures"/"hold_out_evaluation"/"PR_AUC_curve.png")
    plt.close()
    #------------------------------------------------
    # EVALUATION DATA FRAME
    # ------------------------------------------------
    metrices={
    "accuracy" : round(accuracy_score(y_true, y_pred), 2),
    "precision" : round(precision_score(y_true, y_pred, pos_label=2), 2),
    "recall" : round(recall_score(y_true, y_pred, pos_label=2), 2),
    "f1_score" : round(f1_score(y_true, y_pred, pos_label=2), 2),
    "roc_auc" : round(roc_auc_score(y_true, y_pred_prob), 2),
    "pr_auc": pr_auc,
    }
    evaluation_df = pd.DataFrame.from_dict([selected_subjects | metrices])
    evaluation_df.to_csv(EVALUATION_RESULTS_DIR/"holdout_evaluation_selected_subjects_metrices.csv")
    #------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------
    cf=confusion_matrix(y_true,y_pred)
    disp=ConfusionMatrixDisplay(confusion_matrix=cf,display_labels=['not target','target'])
    disp.plot()
    plt.title("CONFUSION MATRIX")
    plt.savefig(PROJECT_ROOT/"figures"/"hold_out_evaluation"/"confusion_matrix.png")
    plt.close()
    #------------------------------------------------
    # Roc-Auc curve
    # ------------------------------------------------
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob, pos_label=labels["target"])
    plt.plot(fpr, tpr, label=f"AUC = {metrices['roc_auc']}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve — Target vs NonTarget")
    plt.legend()
    plt.grid(True)
    plt.savefig(PROJECT_ROOT/"figures"/"hold_out_evaluation"/"ROC_curve.png")
    plt.close()



    print(classification_report(y_true,y_pred,target_names=['Non Target','Target']))

def run_lda_pipeline_holdout_evaluation(subjects:list[int]):
    data_frames = {}
    for i in subjects:
        # epochs = ext.get_and_inspect_preprocessed_epochs(ext.get_subject_raw_data(subject_id=i))
        epochs=mne.read_epochs(PROJECT_ROOT/"data"/"preprocessed"/f"subject_{i}_clean-epo.fif",preload=True)
        data_frames[i] = {
            "features": ext.feature_extraction(epochs),
            "label": ext.target_extraction(epochs)
        }
    features,target=ext.create_data_set(data_frames)
    selected_subjects={}
    for idx in range(1,9):
        if idx in subjects:
            selected_subjects[f"subject_{idx}"]=1
        else:
            selected_subjects[f"subject_{idx}"]=0
    lda_pipeline_holdout_evaluation(features,target,selected_subjects)





if __name__ == "__main__":
    run_lda_pipeline_holdout_evaluation(subjects=[1,2])