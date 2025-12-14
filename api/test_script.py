import pandas as pd
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlencode

# === НАСТРОЙКИ ===
CSV_PATH = "train.csv"
BASE_URL = "http://0.0.0.0:8000/predict"

# === ЗАГРУЗКА ДАННЫХ ===
df = pd.read_csv(CSV_PATH)
df = df[5_000_000:]

y_true = []
y_pred = []

try:
    # === ОБХОД СТРОК ===
    for _, row in df.iterrows():
        params = {
            "Gender": row["Gender"],
            "Age": int(row["Age"]),
            "Driving_License": int(row["Driving_License"]),
            "Region_Code": int(row["Region_Code"]),
            "Previously_Insured": int(row["Previously_Insured"]),
            "Vehicle_Age": row["Vehicle_Age"],
            "Vehicle_Damage": row["Vehicle_Damage"],
            "Annual_Premium": float(row["Annual_Premium"]),
            "Policy_Sales_Channel": int(row["Policy_Sales_Channel"]),
            "Vintage": int(row["Vintage"]),
        }

        url = f"{BASE_URL}?{urlencode(params)}"

        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()

            data = response.json()
            prediction = int(data["prediction"])

            y_pred.append(prediction)
            y_true.append(int(row["Response"]))

            print(f"id={row['id']}, response={data}")

        except Exception as e:
            print(f"Ошибка для id={row['id']}: {e}")
except KeyboardInterrupt:
    ...

# === МЕТРИКИ ===
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# === ВЫВОД ===
print("=== METRICS ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
