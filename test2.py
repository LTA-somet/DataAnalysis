import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Bước 1: Đọc dữ liệu từ tệp CSV
df = pd.read_csv('WineQTAfter.csv')

# Bước 2: Xác định các biến độc lập và biến phụ thuộc
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
            'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
y = df['quality']

# Bước 3: Tính các chỉ số đánh giá cho từng biến độc lập
for feature in features:
    print(f"\nĐánh giá cho biến độc lập: {feature}")

    # Lấy dữ liệu biến độc lập
    X = df[[feature]]

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Xây dựng và huấn luyện mô hình hồi quy tuyến tính
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Dự đoán giá trị
    y_pred = linear_model.predict(X_test)

    # Tính các chỉ số đánh giá
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Hiển thị kết quả
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    print("\nCoefficients:", linear_model.coef_)
    print("Intercept:", linear_model.intercept_)