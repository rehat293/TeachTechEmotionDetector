# Memory-Efficient Emotion Classifier in Google Colab
# Full workflow: train, save, load, and predict on emotions.csv (Drive mount only)

# 1) Mount Google Drive and verify file path
from google.colab import drive
import os
import pandas as pd

drive.mount('/content/drive', force_remount=True)
CSV_PATH = '/content/drive/MyDrive/emotions.csv'  # adjust path if needed
assert os.path.exists(CSV_PATH), f"File not found: {CSV_PATH}"

# 2) Inspect CSV headers to confirm column names
df_header = pd.read_csv(CSV_PATH, nrows=0)
print("CSV columns:", df_header.columns.tolist())

# 3) Import dependencies
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

# 4) Configuration (update these to match CSV)
CHUNKSIZE  = 1000             # rows per batch
TEXT_COL   = 'text'           # name of the text column in your CSV
LABEL_COL  = 'label'          # name of the label column in your CSV
N_FEATURES = 2**18            # number of hashed features
MODEL_PATH = '/content/drive/MyDrive/emotion_clf.pkl'

# 5) Training function (incremental)
def train_and_save(csv_path, model_path):
    # verify specified columns exist in CSV
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if TEXT_COL not in header or LABEL_COL not in header:
        raise KeyError(f"Required columns not found. Available: {header}")

    vectorizer = HashingVectorizer(
        n_features=N_FEATURES,
        alternate_sign=False,
        norm=None,
        binary=False
    )
    encoder = LabelEncoder()
    classifier = SGDClassifier(
        loss='log_loss',      # logistic regression
        max_iter=1,
        tol=None,
        learning_rate='optimal',
        random_state=42
    )

    first_pass = True
    classes = None
    for chunk in pd.read_csv(csv_path, chunksize=CHUNKSIZE):
        texts = chunk[TEXT_COL].astype(str).tolist()
        X = vectorizer.transform(texts)
        y_raw = chunk[LABEL_COL].values

        if first_pass:
            encoder.fit(y_raw)
            classes = encoder.transform(encoder.classes_)
            first_pass = False

        y = encoder.transform(y_raw)
        if not hasattr(classifier, 'classes_'):
            classifier.partial_fit(X, y, classes=classes)
        else:
            classifier.partial_fit(X, y)

    joblib.dump({'model': classifier, 'vectorizer': vectorizer, 'encoder': encoder}, model_path)
    print(f"Training complete. Model saved to: {model_path}")

# 6) Prediction function
def load_and_predict(text, model_path):
    data = joblib.load(model_path)
    model = data['model']
    vectorizer = data['vectorizer']
    encoder = data['encoder']

    X_new = vectorizer.transform([text])
    y_pred = model.predict(X_new)
    return encoder.inverse_transform(y_pred)[0]

# 7) Execute training and example prediction
train_and_save(CSV_PATH, MODEL_PATH)

sample_text = "I feel happy and energetic"
predicted_emotion = load_and_predict(sample_text, MODEL_PATH)
print(f"Input: {sample_text}\nPredicted emotion: {predicted_emotion}")
