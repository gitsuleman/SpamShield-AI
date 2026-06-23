print("Training started...")



import pandas as pd
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ====================================
# SMS MODEL
# ====================================

print("Loading SMS Dataset...")

sms_df = pd.read_csv(
    "data/spam_sms.csv",
    sep="\t",
    names=["label", "text"],
    encoding="latin-1"
)

print(sms_df.head())

sms_X = sms_df["text"]
sms_y = sms_df["label"]

sms_model = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)

sms_model.fit(sms_X, sms_y)

joblib.dump(
    sms_model,
    "spam_sms_model.pkl"
)

print("SMS Model Saved")

# ====================================
# EMAIL MODEL
# ====================================

email_df = pd.read_csv(
    "data/spam_email.csv"
)

email_X = email_df["text"]
email_y = email_df["spam"]

email_model = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)

email_model.fit(email_X, email_y)

joblib.dump(
    email_model,
    "models/spam_email_model.pkl"
)

print("Email Model Saved")