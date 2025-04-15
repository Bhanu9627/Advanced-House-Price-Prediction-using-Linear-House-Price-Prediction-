import tkinter as tk
from tkinter import messagebox, filedialog
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Exchange rate (as of April 14, 2025)
USD_TO_INR = 86.04
PRICE_INCREASE_PERCENTAGE = 15  # Increase the predicted USD price by 15%

# =============================================
# GROUP DETAILS (Now at the top for better visibility)
# =============================================
GROUP_INFO = """
🏛 University: Chandigarh University
📚 Department: MCA
🎓 Program:  (AI & Machine Learning)

👥 Group Members:
---------------------------------
1. Ashutosh Sarin
   - UID: 24MCI10022
   
2. Bhanu Saini
   - UID: 24MCI10026

📅 Academic Details:
---------------------------------
- Section/Group: 24MAM1-A
- Subject: Machine Learning Lab
- Project Title: Advanced House Price Prediction
- Date: April 2025

💡 Features:
---------------------------------
- California Housing Dataset
- Linear Regression Model
- USD to INR Conversion
- 15% Price Adjustment
- Feature Importance Visualization
"""

# =============================================
# MAIN PROGRAM FUNCTIONS
# =============================================

def load_data():
    california = fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    df['Price_USD'] = california.target  # Original price in USD

    features_to_exclude = ['Longitude', 'Latitude', 'AveOccup']
    feature_names = [col for col in california.feature_names if col not in features_to_exclude]
    return df[feature_names + ['Price_USD']], feature_names

def train_model(df):
    X = df.drop('Price_USD', axis=1)
    y = df['Price_USD'] # Train on USD prices

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    y_pred_usd = model.predict(X_scaled)
    mse = mean_squared_error(y, y_pred_usd)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred_usd)
    print(f"Model Evaluation (USD) - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    return model, scaler

def predict_price(model, scaler, features, user_input):
    user_input = np.array(user_input).reshape(1, -1)
    user_input_scaled = scaler.transform(user_input)
    predicted_price_usd = model.predict(user_input_scaled)[0]

    # Increase the predicted USD price
    increased_price_usd = predicted_price_usd * (1 + PRICE_INCREASE_PERCENTAGE / 100)

    # Convert the increased price to INR
    predicted_price_inr = increased_price_usd * USD_TO_INR
    return increased_price_usd, predicted_price_inr

def save_model(model, scaler, filename='house_price_model_increased.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump({'model': model, 'scaler': scaler, 'increase_percentage': PRICE_INCREASE_PERCENTAGE}, file)
    messagebox.showinfo("Success", f"Model saved as {filename} with a {PRICE_INCREASE_PERCENTAGE}% price increase.")

def load_saved_model(filename='house_price_model_increased.pkl'):
    try:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        loaded_model = data['model']
        loaded_scaler = data['scaler']
        loaded_increase_percentage = data.get('increase_percentage', PRICE_INCREASE_PERCENTAGE)
        PRICE_INCREASE_PERCENTAGE = loaded_increase_percentage
        return loaded_model, loaded_scaler
    except FileNotFoundError:
        messagebox.showerror("Error", f"Model file '{filename}' not found. Using default settings.")
        df, _ = load_data()
        model, scaler = train_model(df)
        return model, scaler

def plot_feature_importance(model, feature_names):
    coefficients = model.coef_
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': coefficients
    })
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

# =============================================
# IMPROVED GROUP DETAILS DISPLAY (SPLASH SCREEN)
# =============================================
def show_splash_screen():
    splash = tk.Tk()
    splash.title("Group Information")
    splash.geometry("600x500")
    splash.configure(bg="#2c3e50")  # Dark blue background
    
    # Header
    header = tk.Label(
        splash, 
        text="Machine Learning Project", 
        font=("Helvetica", 20, "bold"), 
        fg="white", 
        bg="#2c3e50"
    )
    header.pack(pady=(20, 10))
    
    # Project Title
    title = tk.Label(
        splash, 
        text="🏡 California House Price Prediction", 
        font=("Helvetica", 16), 
        fg="#f39c12",  # Orange color
        bg="#2c3e50"
    )
    title.pack(pady=(0, 20))
    
    # Group Info in a frame
    info_frame = tk.Frame(splash, bg="#34495e", bd=2, relief="groove")
    info_frame.pack(padx=20, pady=10, fill="both", expand=True)
    
    info_label = tk.Label(
        info_frame, 
        text=GROUP_INFO, 
        font=("Consolas", 11), 
        fg="#ecf0f1",  # Light gray
        bg="#34495e",  # Darker blue
        justify="left"
    )
    info_label.pack(padx=10, pady=10)
    
    # Countdown label
    countdown_label = tk.Label(
        splash, 
        text="Application will start in 5 seconds...", 
        font=("Helvetica", 10), 
        fg="#bdc3c7",  # Light gray
        bg="#2c3e50"
    )
    countdown_label.pack(pady=(10, 20))
    
    # Countdown function
    def update_countdown(seconds=5):
        if seconds > 0:
            countdown_label.config(text=f"Application will start in {seconds} seconds...")
            splash.after(1000, update_countdown, seconds-1)
        else:
            splash.destroy()
            create_main_gui()
    
    splash.after(500, update_countdown)
    splash.mainloop()

# =============================================
# MAIN GUI FUNCTION (renamed from create_gui)
# =============================================
def create_main_gui():
    df, feature_names = load_data()
    model, scaler = train_model(df)

    root = tk.Tk()
    root.title("🏡 House Price Prediction (Increased INR)")
    root.geometry("580x800")
    root.configure(bg="#f0f4f7")  # Light background color

    header = tk.Label(root, 
                     text=f"Enter Feature Values (Price will be increased by {PRICE_INCREASE_PERCENTAGE}%)", 
                     font=("Helvetica", 16, "bold"), 
                     bg="#f0f4f7", 
                     fg="#333")
    header.pack(pady=15)

    user_inputs = {}

    for feature in feature_names:
        frame = tk.Frame(root, bg="#f0f4f7")
        frame.pack(pady=5)

        label = tk.Label(frame, 
                        text=f"{feature}:", 
                        font=("Arial", 12), 
                        bg="#f0f4f7", 
                        fg="#222", 
                        width=18, 
                        anchor="w")
        label.pack(side=tk.LEFT, padx=10)

        entry = tk.Entry(frame, 
                        font=("Arial", 12), 
                        width=20, 
                        bd=2, 
                        relief="groove")
        entry.pack(side=tk.RIGHT, padx=10)

        user_inputs[feature] = entry

    def on_predict():
        try:
            user_values = [float(user_inputs[feature].get()) for feature in feature_names]
            increased_price_usd, predicted_price_inr = predict_price(model, scaler, feature_names, user_values)
            messagebox.showinfo("Prediction", 
                              f"The predicted house price (increased by {PRICE_INCREASE_PERCENTAGE}%) is:\n"
                              f"USD: ${increased_price_usd:.2f}\n"
                              f"INR: ₹{predicted_price_inr:.2f}")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for all fields")

    def on_save_model():
        save_model(model, scaler)

    def on_load_model():
        loaded_model, loaded_scaler = load_saved_model()
        nonlocal model, scaler
        model = loaded_model
        scaler = loaded_scaler
        messagebox.showinfo("Model Loaded", 
                           f"The model has been loaded successfully! "
                           f"Price will be increased by {PRICE_INCREASE_PERCENTAGE}%.")

    def on_feature_importance():
        plot_feature_importance(model, feature_names)

    def change_increase_percentage():
        def set_percentage():
            global PRICE_INCREASE_PERCENTAGE
            try:
                new_percentage = float(percentage_entry.get())
                if 0 <= new_percentage <= 100:
                    PRICE_INCREASE_PERCENTAGE = new_percentage
                    percentage_window.destroy()
                    header.config(text=f"Enter Feature Values (Price will be increased by {PRICE_INCREASE_PERCENTAGE}%)")
                    messagebox.showinfo("Success", 
                                       f"Price increase percentage updated to {PRICE_INCREASE_PERCENTAGE}%.")
                else:
                    messagebox.showerror("Error", "Percentage must be between 0 and 100.")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number.")

        percentage_window = tk.Toplevel(root)
        percentage_window.title("Set Increase Percentage")
        tk.Label(percentage_window, 
                text="Enter desired increase percentage (%):").pack(padx=10, pady=10)
        percentage_entry = tk.Entry(percentage_window)
        percentage_entry.pack(padx=10, pady=5)
        tk.Button(percentage_window, 
                 text="Set Percentage", 
                 command=set_percentage).pack(pady=10)

    # Stylish Button Creator
    def styled_button(text, command):
        return tk.Button(
            root, text=text, command=command,
            font=("Helvetica", 12, "bold"), 
            bg="#4287f5", fg="white",
            activebackground="#2e62c0", activeforeground="white",
            relief="raised", bd=3, padx=10, pady=5, width=30
        )

    # Buttons
    styled_button(f"Predict House Price (+{PRICE_INCREASE_PERCENTAGE}%)", on_predict).pack(pady=12)
    styled_button("Set Price Increase Percentage", change_increase_percentage).pack(pady=8)
    styled_button(f"Save Model (+{PRICE_INCREASE_PERCENTAGE}%)", on_save_model).pack(pady=5)
    styled_button("Load Model", on_load_model).pack(pady=5)
    styled_button("Show Feature Importance", on_feature_importance).pack(pady=12)

    # Disclaimer with current exchange rate and increase
    disclaimer_text = (f"Note: Predicted USD price is increased by {PRICE_INCREASE_PERCENTAGE}%. "
                      f"The INR price is based on an exchange rate of 1 USD = ₹{USD_TO_INR:.2f} "
                      f"as of April 14, 2025.")
    disclaimer_label = tk.Label(root, 
                               text=disclaimer_text, 
                               font=("Arial", 10), 
                               bg="#f0f4f7", 
                               fg="#777", 
                               wraplength=550, 
                               justify="center")
    disclaimer_label.pack(pady=10)

    root.mainloop()

# =============================================
# PROGRAM ENTRY POINT
# =============================================
if __name__ == "_main_":
    show_splash_screen()  # First show the splash screen