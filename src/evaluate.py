import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import pandas as pd
from utils import read_config, load_data, split_data, ensure_dir

def main(config_path="configs/config.yaml"):
    config = read_config(config_path)
    df = load_data(config["data"]["csv_path"])
    
    X_train, X_test, y_train, y_test = split_data(
        df, 
        config["data"]["target"], 
        config["data"]["test_size"], 
        config["data"]["random_state"]
    )
    
    model_path = os.path.join(config["artifacts"]["out_dir"], "best_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Train the model first.")
    
    model = joblib.load(model_path)
    
    # Prédictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    ensure_dir(config["artifacts"]["out_dir"])
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(config["artifacts"]["out_dir"], 'roc.png'))
    plt.close()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(config["artifacts"]["out_dir"], 'pr.png'))
    plt.close()
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(config["artifacts"]["out_dir"], 'confusion_matrix.png'))
    plt.close()
    
    print("Evaluation complete. Plots saved in artifacts/")

if __name__ == "__main__":
    main()
