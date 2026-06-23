import joblib

sms_model = joblib.load("models/spam_sms_model.pkl")

message = "Congratulations! You won ₹50,000. Click here now."

prediction = sms_model.predict([message])[0]

print(prediction)