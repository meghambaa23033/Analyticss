# Clinical Trial Recruitment Rate Prediction

This repository contains a **Streamlit app** for predicting recruitment rates in clinical trials. The app uses a trained machine learning model (XGBoost) to estimate the recruitment rate (patients per site per month) based on various trial features such as enrollment, trial duration, sponsor experience, and more. The app is designed to assist clinical trial planners in optimizing recruitment strategies and improving trial efficiency.

---

## Features

- **Predict Recruitment Rate**: Input trial-specific features to predict the recruitment rate.
- **User-Friendly Interface**: Interactive Streamlit app with a sidebar for input fields.
- **Feature Scaling**: Ensures consistency with the training data using a pre-fitted scaler.
- **Model Integration**: Utilizes a pre-trained XGBoost model for accurate predictions.
- **Customizable Inputs**: Supports both numerical and categorical inputs for trial features.
- **Real-Time Predictions**: Provides instant predictions based on user inputs.

---

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- Required Python libraries:
  - `streamlit`
  - `scikit-learn`
  - `numpy`
  - `pandas`
  - `joblib`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/clinical-trial-recruitment-prediction.git
   cd clinical-trial-recruitment-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the following files are present in the repository:
   - `app.py` (Streamlit app code)
   - `best_xgboost_model.pkl` (trained XGBoost model)
   - `scaler.pkl` (fitted scaler for feature scaling)

---

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open the app in your default web browser. The app will be available at:
   ```
   http://localhost:8501
   ```

3. Use the sidebar to input trial-specific features:
   - **Numerical Inputs**: Enrollment, Trial Duration, Sponsor Experience Score, etc.
   - **Categorical Inputs**: Study Phase, Study Status, Multinational Trial Flag, etc.

4. Click the "Predict Recruitment Rate" button to get the predicted recruitment rate.

5. The app will display the predicted recruitment rate in patients per site per month.

---

## Example Input and Output

### Input:
- Enrollment: `200`
- Trial Duration (Days): `60`
- Time Since Last Update (Days): `15`
- Enrollment Rate Per Day: `3.5`
- Sponsor Experience Score: `10`
- Number of Locations: `5`
- Multinational Trial Flag: `1`
- Encoded Study Phase: `3` (Phase 3)

### Output:
- Predicted Recruitment Rate: `5.67 patients/site/month`

---

## File Structure

```
clinical-trial-recruitment-prediction/
│
├── app.py                  # Streamlit app code
├── best_xgboost_model.pkl  # Trained XGBoost model
├── scaler.pkl              # Fitted scaler for feature scaling
├── requirements.txt        # List of required Python libraries
└── README.md               # Project documentation
```

---

## Acknowledgments

- **Machine Learning Model**: The XGBoost model was trained on a cleaned and preprocessed dataset of clinical trials.
- **Streamlit**: For providing an easy-to-use framework for building interactive web apps.
- **Contributors**: Thanks to all contributors who helped in developing and testing this project.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For questions or feedback, please contact:
- **Name**: Megha Singh Panwar
- **Email**: megha.mbaa23033@iimkashipur.ac.in

```
