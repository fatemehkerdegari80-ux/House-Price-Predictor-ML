# House Price Predictor (ML)

A Python project that predicts house prices using Linear Regression.

# 🚀 How it Works & Setup

### 1. WHAT IT DOES:
 - Cleans data (removes Area > 2000)
 - Encodes 'Address' and features (Parking, Elevator, etc.) to numbers
 - Trains a LinearRegression model on 80% of the dataset
 - Estimates price based on Area, Room, and Location

### 2. REQUIREMENTS:
pip install pandas numpy matplotlib scikit-learn

### 3. QUICK START:
 - Put 'housePrice.csv' in the same folder as the script
 - Make sure the code reads: pd.read_csv("housePrice.csv")
 - Run: python main.py

### 4. RESULTS:
 - Current Variance Score: 0.55
