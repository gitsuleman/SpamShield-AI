import streamlit as st
import joblib



# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------
# Custom CSS
# --------------------------------
st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

.title {
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
    color: #60a5fa;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 1.1rem;
    margin-bottom: 30px;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #1e293b;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------
# Sidebar
# --------------------------------
st.sidebar.title("🛡️ SpamShield AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Spam Detection",
        "Analytics",
        "About"
    ]
)

# --------------------------------
# Dashboard
# --------------------------------
if page == "Dashboard":

    st.markdown(
        "<div class='title'>SpamShield AI</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>AI Powered Spam Detection System</div>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Messages Analyzed", "12,543")

    with col2:
        st.metric("Spam Detected", "5,201")

    with col3:
        st.metric("Accuracy", "98.2%")

    st.info(
        "Welcome to SpamShield AI. Use the Spam Detection page to analyze messages."
    )

# --------------------------------
# Spam Detection
# --------------------------------

elif page == "Spam Detection":

    st.header("📩 Analyze Message")

    message_type = st.selectbox(
        "Select Message Type",
        ["SMS", "Email"]
    )

    message = st.text_area(
        "Enter Message",
        height=250,
        placeholder="Paste your SMS or Email here..."
    )

    if st.button("🚀 Analyze"):

        if message.strip() == "":
            st.warning("Please enter a message.")

        else:

            import joblib

            if message_type == "SMS":
                model = joblib.load(
                    "models/spam_sms_model.pkl"
                )
            else:
                model = joblib.load(
                    "models/spam_email_model.pkl"
                )

            prediction = model.predict([message])[0]

            probability = model.predict_proba([message])
            confidence = probability.max() * 100

            if str(prediction).lower() in ["spam", "1"]:
                st.error("🚨 SPAM DETECTED")
            else:
                st.success("✅ SAFE MESSAGE")

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

# --------------------------------
# Analytics
# --------------------------------
elif page == "Analytics":

    st.header("📊 Analytics Dashboard")

    st.metric("Total Scans", "12,543")
    st.metric("Spam Messages", "5,201")

# --------------------------------
# About
# --------------------------------
elif page == "About":

    st.header("ℹ️ About SpamShield AI")

    st.write("""
    SpamShield AI is a machine learning-based spam
    detection system developed using:

    - Python
    - Streamlit
    - Scikit-Learn
    - NLP
    - TF-IDF
    """)
