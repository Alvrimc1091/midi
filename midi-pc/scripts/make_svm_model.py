import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import pickle
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignore convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Function to normalize a vector (used in the custom transformer)
def normalizar_vector(vector):
    vector_np = np.array(vector)
    magnitud = np.linalg.norm(vector_np)
    if magnitud == 0:
        return vector_np  # Avoid division by zero
    return vector_np / magnitud

# Custom transformer to normalize vectors
class VectorNormalizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.apply_along_axis(normalizar_vector, 1, X)

# Function to extract numeric values from a given row
def extract_numeric_values(row):
    pattern = re.compile(r'\[ *(\d+)\]')
    numeric_values = pattern.findall(row[0])
    return [int(value) for value in numeric_values]

# Read grain data from CSV files
def read_grain_data(file_path, label):
    df = pd.read_csv(file_path, sep='\t', header=None)
    df['label'] = label
    return df

# Define CSV files and labels
file_paths = {
    'fosfato': "/home/pi/demo/data/training2/fosfato_data.csv",
    'lisina': "/home/pi/demo/data/training2/lisina_data.csv",
    'jpc': "/home/pi/demo/data/training2/jpc_data.csv",
    'gluten': "/home/pi/demo/data/training2/gluten_data.csv",
    'conchuela': "/home/pi/demo/data/training2/conchuela_data.csv"
}

# Read and combine all grain data
dfs = []
for key, path in file_paths.items():
    label = key.split('_')[0]
    df = read_grain_data(path, label)
    dfs.append(df)

df_combined = pd.concat(dfs, ignore_index=True)

# Function to clean and extract numeric values
def clean_and_extract(df):
    df_cleaned = df.apply(lambda row: pd.Series(extract_numeric_values(row)), axis=1)
    return df_cleaned

# Balance dataset
def balance_data(df):
    max_size = df['label'].value_counts().max()
    lst = [df]
    for class_index, group in df.groupby('label'):
        lst.append(group.sample(max_size - len(group), replace=True))
    df_new = pd.concat(lst)
    return df_new

# Clean and extract numeric values for the dataset
df_cleaned = clean_and_extract(df_combined)
df_cleaned['label'] = df_combined['label']

# Balance the dataset
df_cleaned_balanced = balance_data(df_cleaned)

# Split the data into train and test sets
df_train, df_test = train_test_split(df_cleaned_balanced, test_size=0.25, random_state=42)

# Create an imputer to replace NaN values with the column mean
imputer = SimpleImputer(strategy='mean')

# Separate features (X) and labels (y) for training and testing
X_train = df_train.drop(columns=['label']).values
y_train = df_train['label'].values

X_test = df_test.drop(columns=['label']).values
y_test = df_test['label'].values

# Hyperparameters for SVM
svm_params = {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'}

# Define the SVM classifier
svm_classifier = SVC(**svm_params, random_state=42)

# Create pipeline for SVM
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('normalizer', VectorNormalizer()),
    ('scaler', StandardScaler()),
    ('classifier', svm_classifier)
])

# Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=10)
print(f'SVM Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}')

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'SVM Test Accuracy: {accuracy:.2f}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
disp.plot(cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.savefig('/home/pi/demo/models/svm_confusion_matrix.png')
plt.close()

# Save the trained model
with open('/home/pi/demo/models/svm_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("SVM model trained and saved successfully.")
