import os
import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from pipeline import build_pipeline
from utils import read_config, load_data, split_data, ensure_dir

def main(config_path="configs/config.yaml"):
    config = read_config(config_path)
    df = load_data(config["data"]["csv_path"])
    X_train, X_test, y_train, y_test = split_data(df, config["data"]["target"], config["data"]["test_size"], config["data"]["random_state"])
    numeric_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    pipeline = build_pipeline(
        numeric=config["features"]["numeric"],
        categorical=config["features"]["categorical"],
        model_type=config["model"]["type"]
    )
    
    param_grid = config["model"]["params"]
    cv = StratifiedKFold(n_splits=config["cv"]["n_splits"], shuffle=True, random_state=config["data"]["random_state"])
    
    mlflow.sklearn.autolog()
    
    #grid = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring=config["cv"]["scoring"], n_jobs=-1)
    grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring=config["cv"]["scoring"],
    n_jobs=1  # <-- important pour Windows + MLflow autologging
    )
    
    with mlflow.start_run():
        grid.fit(X_train, y_train)
        ensure_dir(config["artifacts"]["out_dir"])
        # Sauvegarde modèle
        joblib.dump(grid.best_estimator_, os.path.join(config["artifacts"]["out_dir"], "best_model.joblib"))
        # Sauvegarde prédictions
        preds = grid.predict(X_test)
        pd.DataFrame({"y_true": y_test, "y_pred": preds}).to_csv(os.path.join(config["artifacts"]["out_dir"], "predictions.csv"), index=False)
        

if __name__ == "__main__":
    main()
