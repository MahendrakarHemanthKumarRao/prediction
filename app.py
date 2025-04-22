import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# App settings
st.set_page_config(page_title="NIFTY 50 Predictor", layout="wide")

st.title("📈 NIFTY 50 Close Price Predictor (LLM Style)")
st.markdown("Upload your NIFTY 50 historical dataset and get the next day's **predicted close** using an LLM-inspired model.")

# Upload CSV file
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Check for required columns
        if 'Date' not in df.columns or 'Close' not in df.columns:
            st.error("❌ Your CSV must have 'Date' and 'Close' columns.")
        else:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date')

            window_size = 30
            if len(df) < window_size:
                st.warning("⚠️ Need at least 30 rows of data to make a prediction.")
            else:
                close_prices = df['Close'].values
                dates = df['Date'].values

                # Fit linear model
                X = np.arange(window_size).reshape(-1, 1)
                y = close_prices[-window_size:]
                model = LinearRegression()
                model.fit(X, y)

                # Predict next close price
                next_day_index = np.array([[window_size]])
                predicted_close = model.predict(next_day_index)[0]
                predicted_date = dates[-1] + np.timedelta64(1, 'D')

                st.success(f"📅 **Predicted Close for {predicted_date}**: ₹{predicted_close:.2f}")

                # Plotting
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(dates[-window_size:], y, marker='o', label='Actual Closing Prices')
                ax.plot(predicted_date, predicted_close, 'ro', label='Predicted Next Close')
                ax.axhline(predicted_close, color='red', linestyle='--', alpha=0.5)
                ax.set_title("NIFTY 50 Closing Price Prediction")
                ax.set_xlabel("Date")
                ax.set_ylabel("Close Price")
                ax.legend()
                ax.grid(True)
                fig.autofmt_xdate()
                st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
else:
    st.info("👆 Upload your NIFTY 50 CSV file to begin.")
