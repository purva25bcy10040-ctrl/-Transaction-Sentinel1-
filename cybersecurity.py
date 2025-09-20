import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
import time
import random
from datetime import datetime

# -----------------------------
# User Database
# -----------------------------
USERS = {
    "purva": "mypassword",
    "admin": "admin123"
}

# -----------------------------
# Login Menu
# -----------------------------
def login_menu():
    print("=== Welcome to Transaction Sentinel ===")
    print("Please login to continue.\n")
    for attempt in range(3):
        username = input("Username: ").strip()
        password = input("Password: ")
        if username in USERS and USERS[username] == password:
            print(f"\n✅ Welcome, {username}! Access granted.\n")
            return True
        else:
            print("❌ Invalid credentials. Try again.\n")
    print("❌ Too many failed attempts. Exiting program.")
    return False

# -----------------------------
# Configuration
# -----------------------------
MERCHANT_CATEGORIES = ["grocery", "electronics", "fashion", "travel", "fuel"]
DEVICE_TYPES        = ["mobile", "desktop", "tablet"]

# -----------------------------
# Data Generation
# -----------------------------
def generate_training_data(n=500):
    np.random.seed(42)
    return pd.DataFrame({
        "amount": np.random.normal(500, 150, n),
        "hour": np.random.randint(0, 24, n),
        "location_id": np.random.randint(1, 50, n),
        "merchant_category": np.random.choice(MERCHANT_CATEGORIES, n),
        "device_type": np.random.choice(DEVICE_TYPES, n)
    })

def generate_test_data(n=100):
    np.random.seed(100)
    random.seed(100)
    X, y = [], []
    for _ in range(n):
        if random.random() < 0.85:
            X.append([
                np.random.normal(500, 150),
                np.random.randint(0, 24),
                np.random.randint(1, 50),
                random.choice(MERCHANT_CATEGORIES),
                random.choice(DEVICE_TYPES)
            ])
            y.append(1)
        else:
            X.append([
                np.random.uniform(3000, 10000),
                np.random.choice([1, 2, 3]),
                np.random.randint(1, 100),
                random.choice(MERCHANT_CATEGORIES + ["luxury", "crypto"]),
                random.choice(DEVICE_TYPES + ["unknown"])
            ])
            y.append(-1)
    return pd.DataFrame(X, columns=["amount","hour","location_id","merchant_category","device_type"]), np.array(y)

def generate_transaction():
    if random.random() < 0.9:
        amt = np.random.normal(500, 150)
        hour = np.random.randint(0, 24)
        loc = np.random.randint(1, 50)
        mc = random.choice(MERCHANT_CATEGORIES)
        dev = random.choice(DEVICE_TYPES)
    else:
        amt = np.random.uniform(3000, 10000)
        hour = np.random.choice([1, 2, 3])
        loc = np.random.randint(1, 100)
        mc = random.choice(MERCHANT_CATEGORIES + ["luxury", "crypto"])
        dev = random.choice(DEVICE_TYPES + ["unknown"])
    return pd.DataFrame([[amt, hour, loc, mc, dev]], columns=["amount","hour","location_id","merchant_category","device_type"])

# -----------------------------
# Model Training
# -----------------------------
def train_model(contamination=0.05):
    print("🔄 Training model...")
    data = generate_training_data()
    num_features = ["amount", "hour", "location_id"]
    cat_features = ["merchant_category", "device_type"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    model = IsolationForest(contamination=contamination, random_state=42)
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipe.fit(data)
    print("✅ Model training complete!\n")
    return pipe

# -----------------------------
# Model Evaluation
# -----------------------------
def evaluate_model(pipe):
    print("📊 Evaluating model...")
    X_test, y_true = generate_test_data()
    y_pred = pipe.predict(X_test)

    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
    print("\n=== Confusion Matrix ===")
    print("               Pred Normal   Pred Fraud")
    print(f"Actual Normal      {cm[0,0]:>5}          {cm[0,1]:>5}")
    print(f"Actual Fraud       {cm[1,0]:>5}          {cm[1,1]:>5}")

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, labels=[1, -1], target_names=["Normal","Fraud"]))

# -----------------------------
# Real-Time Monitoring
# -----------------------------
transaction_history = []

def start_monitoring(pipe):
    print("\n🖥 Starting real-time monitoring. Press Ctrl+C to stop.\n")
    try:
        while True:
            txn = generate_transaction()
            pred = pipe.predict(txn)
            status = "⚠ FRAUD ALERT" if pred[0]==-1 else "✅ Normal"

            # Add to history
            transaction_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "amount": txn["amount"][0],
                "hour": txn["hour"][0],
                "location": txn["location_id"][0],
                "category": txn["merchant_category"][0],
                "device": txn["device_type"][0],
                "prediction": status
            })

            print(f"Transaction: {txn.values.tolist()[0]} --> {status}", flush=True)
            # Display summary every 5 transactions
            if len(transaction_history) % 5 == 0:
                normal_count = sum(1 for t in transaction_history if t["prediction"]=="✅ Normal")
                fraud_count = sum(1 for t in transaction_history if t["prediction"]=="⚠ FRAUD ALERT")
                print(f"\n📊 Summary: Normal={normal_count}, Fraud={fraud_count}, Total={len(transaction_history)}\n")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped.")

# -----------------------------
# Manual Transaction
# -----------------------------
def manual_transaction(pipe):
    print("\n📝 Enter manual transaction details:")
    try:
        amt = float(input("Amount: "))
        hour = int(input("Hour (0-23): "))
        loc = int(input("Location ID (1-100): "))
        mc = input(f"Merchant Category {MERCHANT_CATEGORIES}: ").strip()
        dev = input(f"Device Type {DEVICE_TYPES}: ").strip()

        txn = pd.DataFrame([[amt, hour, loc, mc, dev]], columns=["amount","hour","location_id","merchant_category","device_type"])
        pred = pipe.predict(txn)
        status = "⚠ FRAUD ALERT" if pred[0]==-1 else "✅ Normal"

        transaction_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "amount": amt,
            "hour": hour,
            "location": loc,
            "category": mc,
            "device": dev,
            "prediction": status
        })

        print(f"\nTransaction Prediction: {status}\n")
    except Exception as e:
        print(f"❌ Error: {e}")

# -----------------------------
# View History
# -----------------------------
def view_history():
    if not transaction_history:
        print("\n📂 No transactions yet.\n")
        return
    df = pd.DataFrame(transaction_history)
    print("\n=== Transaction History ===")
    print(df)

# -----------------------------
# Save History
# -----------------------------
def save_history():
    if transaction_history:
        df = pd.DataFrame(transaction_history)
        df.to_csv("transaction_history.csv", index=False)
        print("💾 Transaction history saved to 'transaction_history.csv'.")

# -----------------------------
# Adjust Model Sensitivity
# -----------------------------
def adjust_sensitivity():
    try:
        c = float(input("Enter new contamination (0.01-0.2, default 0.08): "))
        if 0.01 <= c <= 0.2:
            print(f"✅ Contamination set to {c}")
            return c
        else:
            print("❌ Invalid range, keeping previous value.")
            return None
    except:
        print("❌ Invalid input, keeping previous value.")
        return None

# -----------------------------
# Main Menu
# -----------------------------
def main():
    if not login_menu():
        return

    contamination = 0.08
    pipe = train_model(contamination=contamination)

    while True:
        print("\n=== Main Menu ===")
        print("1. Evaluate Model")
        print("2. Start Real-Time Monitoring")
        print("3. Enter Manual Transaction")
        print("4. View Transaction History")
        print("5. Adjust Model Sensitivity")
        print("6. Save Transaction History")
        print("7. Exit")
        choice = input("Enter your choice (1-7): ").strip()

        if choice=="1":
            evaluate_model(pipe)
        elif choice=="2":
            start_monitoring(pipe)
        elif choice=="3":
            manual_transaction(pipe)
        elif choice=="4":
            view_history()
        elif choice=="5":
            new_c = adjust_sensitivity()
            if new_c:
                contamination = new_c
                pipe = train_model(contamination=contamination)
        elif choice=="6":
            save_history()
        elif choice=="7":
            print("👋 Exiting program. Goodbye!")
            break
        else:
            print("❌ Invalid choice. Enter 1-7.")

# Run the program
main()
