import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Bước 1: Đọc dữ liệu từ tệp CSV
pd.set_option('display.max_columns', None)  # Hiển thị tất cả các cột
pd.set_option('display.width', 100)  # Tăng chiều rộng hiển thị của bảng
df = pd.read_csv('WineQT.csv')
print("Dữ liệu ban đầu:")
print(df.head())

# plt.figure(figsize=(8, 6))
# sns.countplot(x='quality', data=df, palette='viridis')
# plt.xlabel('Quality')
# plt.ylabel('Count')
# plt.title('Distribution of Quality Ratings')
# plt.show()
# Bước 2: Chuẩn bị dữ liệu
# Xác định các biến độc lập (các thuộc tính) và biến phụ thuộc
X = df[['fixed acidity', 'volatile acidity', 'citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]  # Thay 'feature1', 'feature2', 'feature3' bằng tên các cột của bạn
y = df['quality']  # Thay 'target' bằng tên cột biến phụ thuộc của bạn




data_describe = df.describe(include='all')
print(data_describe)


# Bước 3: Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Bước 4: Xây dựng và huấn luyện mô hình hồi quy tuyến tính
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Bước 5: Dự đoán và đánh giá mô hình
y_pred = linear_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Bước 6: Phân tích kết quả
print("\nCoefficients:", linear_model.coef_)
print("Intercept:", linear_model.intercept_)

# Bước 7: Lưu mô hình đã huấn luyện
joblib.dump(linear_model, 'linear_model.pkl')
print("\nMô hình đã được lưu vào tệp 'linear_model.pkl'.")

# Bước 8: Tải và sử dụng mô hình đã lưu (ví dụ)
loaded_model = joblib.load('linear_model.pkl')
new_data = pd.DataFrame([[7, 0.5, 0.03, 2, 0.07, 11, 34, 0.996, 3.2, 0.5, 9.5]],
                        columns=['fixed acidity', 'volatile acidity', 'citric acid',
                                 'residual sugar', 'chlorides', 'free sulfur dioxide',
                                 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'])

# Dự đoán với mô hình
predicted = loaded_model.predict(new_data)
print("\nDự đoán mới:", predicted)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nMean Absolute Error: {mae:.2f}")
# Bước 9: Vẽ biểu đồ scatter
# plt.figure(figsize=(18, 20))
#
# # Danh sách các tên biến độc lập
# features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
#              'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
#
# for feature in features:
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(x=X_test[feature], y=y_pred, color='blue', alpha=0.5)
#     sns.regplot(x=X_test[feature], y=y_pred, scatter=False, color='red')
#     plt.xlabel(feature)
#     plt.ylabel('Predicted Values')
#     plt.title(f'{feature} vs Predicted Values')
#     plt.show()  # Hiển thị biểu đồ hiện tại và chờ người dùng tương tác

# Bước 9: Vẽ biểu đồ scatter tổng hợp

# Danh sách các tên biến độc lập
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
            'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']


# Danh sách các tên biến độc lập
# features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
#             'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
#
# # Thiết lập số hàng và cột cho lưới biểu đồ
# n_rows = len(features) // 3 + (len(features) % 3 > 0)  # 3 cột mỗi hàng
#
# fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 6))
#
# # Vẽ từng biểu đồ scatter cho từng biến
# for i, feature in enumerate(features):
#     row, col = divmod(i, 3)
#     sns.scatterplot(x=X_test[feature], y=y_pred, alpha=0.5, ax=axes[row, col], color='blue')
#     sns.regplot(x=X_test[feature], y=y_pred, scatter=False, color='red', ax=axes[row, col],
#                 line_kws={"linestyle": "--", "linewidth": 2}, ci=0)  # Loại bỏ dải tin cậy
#     axes[row, col].set_xlabel(feature)
#     axes[row, col].set_ylabel('Predicted Values')
#     axes[row, col].set_title(f'{feature} vs Predicted Values')
#
# # Ẩn các subplot trống nếu có
# for i in range(len(features), n_rows * 3):
#     fig.delaxes(axes.flatten()[i])
#
# plt.tight_layout()
# plt.show()  # Hiển thị tất cả các biểu đồ cùng một lúc
# Bước 10: Vẽ biểu đồ heatmap của ma trận tương quan
# plt.subplot(1, 2, 2)
# corr_matrix = df.corr()  # Tính toán ma trận tương quan
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Heatmap of Correlation Matrix')
#
# features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
#              'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
#
# for feature in features:
#     plt.figure(figsize=(8, 6))
#     sns.boxplot(x=df[feature])
#     plt.xlabel(feature)
#     plt.title(f'Box Plot of {feature}')
#     plt.show()  # Hiển thị biểu đồ hiện tại và chờ người dùng tương tác
#
# df_long = df.melt(value_vars=features, var_name='Feature', value_name='Value')

# # Vẽ box plot tổng hợp
# plt.figure(figsize=(10, 5))
# sns.boxplot(x='Feature', y='Value', data=df_long)
# plt.xticks(rotation=90)  # Xoay nhãn trục x để dễ đọc hơn
# plt.xlabel('Feature')
# plt.ylabel('Values')
# plt.title('Box Plot of All Features')
# plt.show()
# plt.tight_layout()  # Điều chỉnh để tránh chồng lấn
# plt.show()

# features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
#             'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
#
# # Tạo một figure và các subplot
# n_features = len(features)
# n_cols = 3  # Số cột trong lưới subplot
# n_rows = (n_features + n_cols - 1) // n_cols  # Tính số hàng cần thiết
#
# plt.figure(figsize=(15, n_rows * 5))
#
# for i, feature in enumerate(features):
#     plt.subplot(n_rows, n_cols, i + 1)
#     plt.hist(df[feature], bins=30, color='blue', edgecolor='black')
#     plt.xlabel(feature)
#     plt.ylabel('Frequency')
#     plt.title(f'Histogram of {feature}')
#
# plt.tight_layout()  # Điều chỉnh layout để tránh chồng lấn
# plt.show()
# Hiển thị một vài giá trị thực tế và giá trị dự đoán
# comparison = pd.DataFrame({'Giá trị thực tế': y_test, 'Giá trị dự đoán': y_pred})
# print(comparison.head(10))
#
# # Hiển thị kết quả dự đoán dưới dạng biểu đồ
# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(y_test)), y_test, color='blue', label='Giá trị thực tế')
# plt.scatter(range(len(y_pred)), y_pred, color='red', label='Giá trị dự đoán')
# plt.xlabel('Samples')
# plt.ylabel('Quality')
# plt.title('So sánh giữa giá trị thực tế và giá trị dự đoán')
# plt.legend()
# plt.show()
