import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, roc_auc_score,classification_report)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import features.feature_extraction as ext
from preprocessing.pipeline import PROJECT_ROOT
#----------- Fix a Seed number -----------
SEED=42
TEST_SIZE=0.2
#----------- target/non target labels -----------
labels={
    "target":2,
    "nontarget":1
}
def lda_pipeline(features,target):
    pipe=Pipeline([("scaler",StandardScaler()),
                  ("classifier",LDA(priors=[0.5,0.5]))
                   ])
    num_of_targets=(target["target"]==2).sum()
    print(f"number of target samples: {num_of_targets}")
    print(f"number of non target samples: {len(target)-num_of_targets}")
    print("\n")
    # 1d array is expected so we convert to 1d series
    y = target["target"]
    x_train,x_test,y_train,y_test=train_test_split(features,y,test_size=TEST_SIZE,random_state=SEED,stratify=target) #split based on the population real distribution where the non target is the majority
    # save train,test indices
    indices={
        "train_idx":x_train.index.tolist(),
        "test_idx":x_test.index.tolist()
    }
    with open(PROJECT_ROOT/"data"/"train_test_split",'w') as file:
        json.dump(indices,file)
    # train the model
    pipe.fit(x_train,y_train)
    # get predictions
    # as probabilities (how confident are we that the sample is target)
    target_column = np.where(
        pipe.classes_ == labels["target"]
    )[0][0]
    predictions_prob=pipe.predict_proba(x_test)[:,target_column]
    # as labels
    prediction=pipe.predict(x_test)
    evaluation_report(y_test,prediction,predictions_prob)
def evaluation_report(y_true,y_pred,y_pred_prob):
    print(f"accuracy: {accuracy_score(y_true,y_pred)}")
    print(f"precision: {precision_score(y_true,y_pred,pos_label=2)}")
    print(f"recall: {recall_score(y_true,y_pred,pos_label=2)}")
    print(f"f1 score: {f1_score(y_true,y_pred,pos_label=2)}")
    cf=confusion_matrix(y_true,y_pred)
    disp=ConfusionMatrixDisplay(confusion_matrix=cf,display_labels=['not target','target'])
    auc = roc_auc_score(y_true, y_pred_prob)
    disp.plot()
    plt.title("CONFUSION MATRIX")
    plt.savefig(PROJECT_ROOT/"figures"/"evaluation"/"confusion_matrix.png")
    plt.close()
    print("ROC-AUC:", auc)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob, pos_label=labels["target"])
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve — Target vs NonTarget")
    plt.legend()
    plt.savefig(PROJECT_ROOT/"figures"/"evaluation"/"ROC_curve.png")
    plt.close()
    print(classification_report(y_true,y_pred,target_names=['Non Target','Target']))

if __name__ == "__main__":
    epochs = ext.get_and_inspect_epochs()
    features = ext.feature_extraction(epochs)
    target = ext.target_extraction(epochs)
    lda_pipeline(features,target)