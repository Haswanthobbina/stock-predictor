
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


st.title("📈 Stock Price Predictor")

# 🔧 Custom CSS Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #2a5d84;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Inputs
ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL, TSLA, MSFT, AMZN, GOOGL, META, NVDA, JPM, V)", value="AAPL")
days_to_predict = st.slider("📆 Days in Future to Predict", 1, 80, 35)

if st.button("Predict"):
    # Download stock data
    data = yf.download(ticker, start="2018-01-01", end="2024-12-31")
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data = data.dropna()

    # Features and target
    features = ['Open', 'High', 'Low', 'Close', 'SMA_5']
    X = data[features]
    y = data['Close']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prepare for future prediction
    last_row = X.iloc[-1].copy()
    future_preds = []
    close_series = X['Close'].squeeze().copy()  # Ensure it's a Series

    for _ in range(days_to_predict):
        next_pred = model.predict([last_row])[0]
        future_preds.append(next_pred)

        # Update SMA_5 with last 4 closes + next_pred
        recent_closes = close_series.iloc[-4:].tolist() + [next_pred]
        sma_5 = sum(recent_closes[-5:]) / 5

        # Create new row for next prediction
        new_row = last_row.copy()
        new_row['Close'] = next_pred
        new_row['SMA_5'] = sma_5

        # Append to close_series for future SMA calculations
        close_series = pd.concat([close_series, pd.Series([next_pred])], ignore_index=True)

        # Update last_row for next prediction step
        last_row = new_row

    # Display result
    st.success(f"📌 Predicted price after {days_to_predict} days: ₹{float(future_preds[-1]):.2f}")
    st.markdown(f"<h3 style='color:#00cc44;'>📌 Predicted price after {days_to_predict} days: ₹{float(future_preds[-1]):.2f}</h3>", unsafe_allow_html=True)
     # 🧾 Final predicted price
    st.markdown(
        f"""
        <div style="padding: 1rem; background-color: #dfffe0; border-radius: 10px; border: 1px solid #a6e3af;">
            <h3 style='color: #1a5d1a;'>💰 Predicted stock price after {days_to_predict} days:</h3>
            <h1 style='color: #004d00;'>₹{float(future_preds[-1]):.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Plot prediction
    st.subheader("📊 Future Price Trend")
    future_df = pd.DataFrame(future_preds, columns=["Predicted Close"])
    future_df.index.name = "Days Ahead"
    st.line_chart(future_df)  
    
     # 📊 Show prediction line chart
    fig, ax = plt.subplots()
    ax.plot(range(1, days_to_predict + 1), future_preds, color='#ff6600', linewidth=2)
    ax.set_title('📊 Stock Price Forecast')
    ax.set_xlabel('Days from Today')
    ax.set_ylabel('Price (₹)')
    st.pyplot(fig)