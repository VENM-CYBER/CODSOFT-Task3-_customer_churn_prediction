import os
import warnings
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

warnings.filterwarnings("ignore")

# ======================================================
# PATHS
# ======================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "Churn_Modelling.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ======================================================
# LOAD DATA
# ======================================================

print("\nLoading dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}\n\n"
        "Place Churn_Modelling.csv inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)

# ======================================================
# CLEAN DATA
# ======================================================

drop_columns = ["RowNumber", "CustomerId", "Surname"]

for col in drop_columns:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# ======================================================
# FEATURES & TARGET
# ======================================================

TARGET = "Exited"

if TARGET not in df.columns:
    raise ValueError("Target column 'Exited' not found.")

X = df.drop(TARGET, axis=1)
y = df[TARGET]

# ======================================================
# FEATURE TYPES
# ======================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumeric Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)

# ======================================================
# PREPROCESSING
# ======================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ======================================================
# TRAIN TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# ======================================================
# MODELS
# ======================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

results = {}

# ======================================================
# TRAINING
# ======================================================

print("\nTraining Models...\n")

for name, model in models.items():

    print("=" * 60)
    print(name)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_auc = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="roc_auc"
    ).mean()

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    print(f"CV AUC   : {cv_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    results[name] = auc

# ======================================================
# HYPERPARAMETER TUNING
# ======================================================

print("\nHyperparameter Tuning...\n")

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        )
    )
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 5]
}

grid = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)

print("\nBest CV Score:")
print(grid.best_score_)

# ======================================================
# FINAL EVALUATION
# ======================================================

preds = best_model.predict(X_test)
probs = best_model.predict_proba(X_test)[:, 1]

final_auc = roc_auc_score(y_test, probs)

print("\nFinal ROC AUC:", round(final_auc, 4))

# ======================================================
# SAVE MODEL
# ======================================================

model_path = os.path.join(
    MODEL_DIR,
    "customer_churn_model.pkl"
)

joblib.dump(best_model, model_path)

# ======================================================
# SAVE RESULTS
# ======================================================

results_df = pd.DataFrame({
    "Model": list(results.keys()),
    "ROC_AUC": list(results.values())
})

results_path = os.path.join(
    MODEL_DIR,
    "model_comparison.csv"
)

results_df.to_csv(results_path, index=False)

# ======================================================
# DONE
# ======================================================

print("\nModel Saved:")
print(model_path)

print("\nResults Saved:")
print(results_path)

print("\nTraining Completed Successfully!")

print("\nModel Comparison:")
print(results_df.sort_values(
    by="ROC_AUC",
    ascending=False
))