import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pickle
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignore convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load dataset
file_path = "/home/alvaro/Documents/Workspace/midi/models/mean_var_rgb_training/data/midi_data.csv"
df = pd.read_csv(file_path)

# Select only the required columns
features = ["r_mean", "g_mean", "b_mean", "r_var", "g_var", "b_var"]
X = df[features].values
y = df["grain_type"].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define models
models = {
    "svm": SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "logistic_regression": LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
}

# Train and evaluate models
for model_name, model in models.items():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=10)
    print(f"{model_name.upper()} Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    # Train model
    pipeline.fit(X_train, y_train)

    # Test model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name.upper()} Test Accuracy: {accuracy:.2f}")

    # Confusion Matrix

    fig, ax = plt.subplots(figsize=(12, 10))  # Flexible figure size

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap="Blues", ax=ax)

    # Rotate labels automatically based on their length
    plt.xticks(rotation=90, ha="right", fontsize=10)  # Auto-align long labels
    plt.yticks(fontsize=10)
    plt.title(f"{model_name.upper()} Confusion Matrix")

    # Auto-adjust layout dynamically
    plt.tight_layout()

    
    plt.savefig(f"/home/alvaro/Documents/Workspace/midi/models/mean_var_rgb_training/results/3_models_{model_name}_cm.png")
    plt.close()

    # Save model
    with open(f"/home/alvaro/Documents/Workspace/midi/models/mean_var_rgb_training/models/3_models_{model_name}_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print(f"{model_name.upper()} model trained and saved successfully.")
