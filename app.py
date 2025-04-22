import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Streamlit config
st.set_page_config(page_title="NIFTY 50 - 6 Day Predictor", layout="wide")

st.title("📈 NIFTY 50 - 6 Day Close Price Predictor (LLM Style)")
st.markdown("Upload your NIFTY 50 historical CSV file. The app will predict the **next 6 closing prices** using a simple linear regression approach.")

# Upload section
uploaded_file = st.file_uploader("📂 Upload your NIFTY 50 CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Ensure necessary columns exist
        if 'Date' not in df.columns or 'Close' not in df.columns:
            st.error("❌ CSV must have 'Date' and 'Close' columns.")
        else:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date')

            window_size = 30
            if len(df) < window_size:
                st.warning("⚠️ At least 30 rows are required.")
            else:
                original_dates = list(df['Date'].values)
                original_close = list(df['Close'].values)

                X = np.arange(window_size).reshape(-1, 1)
                y = np.array(original_close[-window_size:])
                predicted_dates = []
                predicted_values = []

                last_known_date = df['Date'].iloc[-1]

                for i in range(6):
                    model = LinearRegression()
                    model.fit(X, y)
                    next_value = model.predict([[window_size]])[0]

                    predicted_values.append(next_value)
                    last_known_date += pd.Timedelta(days=1)
                    predicted_dates.append(last_known_date)

                    # Update input window
                    y = np.append(y[1:], next_value)

                # Display predictions
                pred_df = pd.DataFrame({
                    "Date": predicted_dates,
                    "Predicted Close": predicted_values
                })

                st.success("✅ Successfully predicted the next 6 days!")
                st.dataframe(pred_df)

                # Plot
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(original_dates[-window_size:], original_close[-window_size:], label="Actual", marker='o')
                ax.plot(predicted_dates, predicted_values, label="Predicted", marker='o', color='red')
                ax.set_title("NIFTY 50 - Actual vs Next 6 Day Prediction")
                ax.set_xlabel("Date")
                ax.set_ylabel("Close Price")
                ax.legend()
                ax.grid(True)
                fig.autofmt_xdate()
                st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
else:
    st.info("📂 Please upload a CSV file with at least 'Date' and 'Close' columns.")
