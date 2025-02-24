import streamlit as st
import requests

# FastAPI URL
FASTAPI_URL = "http://127.0.0.1:8000/predict"

# Streamlit UI
st.set_page_config(page_title="Conversation Analysis", layout="centered")

st.title("💬 Conversation Outcome Predictor")
st.write("Enter conversation details to predict the outcome.")

# Layout Columns
col1, col2 = st.columns(2)

# Column 1 Inputs
with col1:
    utterance_count = st.number_input("Total Number of Utterances", min_value=1, step=1,
                                      help="Total exchanges in the conversation.")
    utterance_ratio = st.slider("Customer-to-Agent Utterance Ratio", 0.0, 5.0, step=0.01,
                                 help="Ratio of customer messages to agent messages.")
    response_time = st.number_input("Avg. Response Time (seconds)", min_value=0.0, step=0.1,
                                    help="Average time taken for responses in seconds.")
    conversation_duration = st.number_input("Total Conversation Duration (seconds)", min_value=1, step=1,
                                            help="Total time spent in the conversation.")
    sentiment_trend = st.slider("Sentiment Change (Start to End)", -1.0, 1.0, step=0.01,
                                 help="Difference in sentiment score from start to end of conversation.")

# Column 2 Inputs
with col2:
    customer_sentiment = st.slider("Customer Sentiment Score (VADER)", -1.0, 1.0, step=0.01,
                                   help="Sentiment of the customer (negative to positive).")
    agent_sentiment = st.slider("Agent Sentiment Score (VADER)", -1.0, 1.0, step=0.01,
                                help="Sentiment of the agent (negative to positive).")
    avg_word_count = st.number_input("Avg. Words Per Message", min_value=1, step=1,
                                     help="Average number of words per message.")
    avg_char_count = st.number_input("Avg. Characters Per Message", min_value=1, step=1,
                                     help="Average number of characters per message.")
    day_of_week = st.selectbox("Day of Conversation", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                               index=0, help="Day when the conversation took place.")
    is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0  # Auto-detect weekend

# Predict Button
if st.button("🚀 Predict Outcome"):
    # Map inputs to JSON payload
    input_data = {
        "utterance_count_first": utterance_count,
        "utterance_ratio_first": utterance_ratio,
        "response_time_mean": response_time,
        "conversation_duration_first": conversation_duration,
        "sentiment_trend_first": sentiment_trend,
        "customer_vader_sentiment_mean": customer_sentiment,
        "agent_vader_sentiment_mean": agent_sentiment,
        "word_count_mean": avg_word_count,
        "char_count_mean": avg_char_count,
        "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week),
        "is_weekend": is_weekend
    }

    # API Request
    response = requests.post(FASTAPI_URL, json=input_data)

    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.success(f"Predicted Outcome: **{prediction}**")
    else:
        st.error("Prediction failed. Please check FastAPI server")

