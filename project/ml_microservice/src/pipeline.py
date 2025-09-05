from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def build_pipeline(numeric, categorical, model_type="logreg"):
    # Pipeline numérique
    num = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Pipeline catégorielle
    cat = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num, numeric),
            ("cat", cat, categorical)
        ]
    )
    
    # Sélection du modèle
    if model_type == "logreg":
        model = LogisticRegression(max_iter=500)
    elif model_type == "random_forest":
        model = RandomForestClassifier()
    else:
        raise ValueError(f"Unsupported model_type {model_type}")
    
    # Pipeline final
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("logreg", model)])
    #pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    return pipeline
