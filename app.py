from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# Load the trained model and scaler
model_path = "D:/IBA/model.pkl"
scaler_path = "D:/IBA/scaler.pkl"

# Load the model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Define the selected features
selected_features = [
    'utterance_count_first', 'utterance_ratio_first', 'response_time_mean',
    'conversation_duration_first', 'sentiment_trend_first', 'customer_vader_sentiment_mean',
    'agent_vader_sentiment_mean', 'word_count_mean', 'char_count_mean', 'day_of_week', 'is_weekend'
]

# Define the input data model
class ConversationFeatures(BaseModel):
    utterance_count_first: float
    utterance_ratio_first: float
    response_time_mean: float
    conversation_duration_first: float
    sentiment_trend_first: float
    customer_vader_sentiment_mean: float
    agent_vader_sentiment_mean: float
    word_count_mean: float
    char_count_mean: float
    day_of_week: int
    is_weekend: int

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Conversation Features Prediction API!"}

# Define the prediction endpoint
@app.post("/predict")
def predict(features: ConversationFeatures):
    # Convert input data to a NumPy array
    input_data = np.array([[
        features.utterance_count_first,
        features.utterance_ratio_first,
        features.response_time_mean,
        features.conversation_duration_first,
        features.sentiment_trend_first,
        features.customer_vader_sentiment_mean,
        features.agent_vader_sentiment_mean,
        features.word_count_mean,
        features.char_count_mean,
        features.day_of_week,
        features.is_weekend
    ]])

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make a prediction
    prediction = model.predict(input_data_scaled)

    # Return the prediction
    return {"prediction": int(prediction[0])}
