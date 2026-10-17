import json
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, roc_auc_score,classification_report)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import src.features.feature_extraction as ext
from src.preprocessing.pipeline import PROJECT_ROOT
#----------- Fix a Seed number -----------
SEED=42
TEST_SIZE=0.2
#----------- target/non target labels -----------
labels={
    "target":2,
    "nontarget":1
}
def lda_pipeline_holdout_evaluation(features,target):

    #------------------------------------------------
    # Report number of Target/NonTarget
    # ------------------------------------------------
    # num_of_targets = (target["target"] == 2).sum()
    # print(f"number of target samples: {num_of_targets}")
    # print(f"number of non target samples: {len(target) - num_of_targets}")
    # print("\n")

    #------------------------------------------------
    # Fixed splitting
    # ------------------------------------------------
    # 1d array is expected so we convert to 1d series
    y = target["target"]
    x = features.copy()
    x.drop(columns=["epoch_id"],inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=SEED,stratify=target)  # split based on the population real distribution where the non target is the majority
    # save train,test indices
    indices={
        "train_idx":x_train.index.tolist(),
        "test_idx":x_test.index.tolist()
    }
    with open(PROJECT_ROOT/"data"/"train_test_split.json",'w') as file:
        json.dump(indices,file)
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
    evaluation_report(y_test,prediction,predictions_prob)
    return prediction,predictions_prob

def evaluation_report(y_true,y_pred,y_pred_prob):
    output_dir = PROJECT_ROOT / "figures" / "hold_out_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"accuracy: {round(accuracy_score(y_true,y_pred),2)}")
    print(f"precision: {round(precision_score(y_true,y_pred,pos_label=2),2)}")
    print(f"recall: {round(recall_score(y_true,y_pred,pos_label=2),2)}")
    print(f"f1 score: {round(f1_score(y_true,y_pred,pos_label=2),2)}")
    cf=confusion_matrix(y_true,y_pred)
    disp=ConfusionMatrixDisplay(confusion_matrix=cf,display_labels=['not target','target'])
    auc = roc_auc_score(y_true, y_pred_prob)
    disp.plot()
    plt.title("CONFUSION MATRIX")
    plt.savefig(PROJECT_ROOT/"figures"/"hold_out_evaluation"/"confusion_matrix.png")
    plt.close()
    print("ROC-AUC:", round(auc,2))
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob, pos_label=labels["target"])
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve — Target vs NonTarget")
    plt.legend()
    plt.savefig(PROJECT_ROOT/"figures"/"hold_out_evaluation"/"ROC_curve.png")
    plt.close()
    print(classification_report(y_true,y_pred,target_names=['Non Target','Target']))
def create_data_set(dataFrames:dict[str,dict[str,pd.DataFrame]])->tuple[pd.DataFrame,pd.DataFrame]:
    data_frames=[]
    for idx,data_frame in  dataFrames.items():
        feature_copy=data_frame['features'].copy()
        label_copy=data_frame['label'].copy()
        subject_id=[f'subj_{idx}' for _ in range(feature_copy.shape[0])]
        feature_copy['subject_id']=subject_id
        feature_copy.set_index("subject_id",inplace=True)
        label_copy['subject_id']=subject_id
        label_copy.set_index("subject_id",inplace=True)
        epochs_column=[f"ep_{i}" for i in range(1,feature_copy.shape[0]+1)]
        feature_copy.insert(0,"epoch_id",epochs_column)
        data_frames.append(pd.concat([feature_copy,label_copy],axis=1,sort=False))
    dataset=pd.concat(data_frames,sort=False)
    target=pd.DataFrame(dataset["target"],index=dataset.index.tolist(),columns=["target"])
    features=dataset.drop(columns=["target"])
    print(dataset)
    return features,target
def run_lda_pipeline_holdout_evaluation(subjects:list[int]):
    data_frames = {}
    for i in subjects:
        epochs = ext.get_and_inspect_preprocessed_epochs(ext.get_subject_raw_data(subject_id=i))
        data_frames[i] = {
            "features": ext.feature_extraction(epochs),
            "label": ext.target_extraction(epochs)
        }
    features,target=create_data_set(data_frames)
    lda_pipeline_holdout_evaluation(features,target)





if __name__ == "__main__":
    run_lda_pipeline_holdout_evaluation(subjects=[2,8])