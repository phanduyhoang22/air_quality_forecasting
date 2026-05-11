# Vietnam Air Quality PM2.5 Forecasting
### Dự báo nồng độ bụi mịn PM2.5 bằng mô hình Ridge Regression đa biến

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Tổng quan dự án
Ô nhiễm không khí, đặc biệt là bụi mịn PM2.5, đang là vấn đề nhức nhối tại các đô thị lớn ở Việt Nam. Dự án này được xây dựng nhằm **dự báo nồng độ PM2.5 theo từng giờ** dựa trên các dữ liệu động lực học khí tượng (nhiệt độ, độ ẩm, hướng gió...) và các chỉ số không khí trong quá khứ.

Dự án tập trung vào 4 thành phố trọng điểm: **Hà Nội, Hải Phòng, Đà Nẵng, và TP. Hồ Chí Minh**.

## Điểm nổi bật về kỹ thuật
Thay vì sử dụng các thư viện có sẵn như Scikit-learn cho mô hình, mình đã **tự triển khai thuật toán Ridge Regression từ đầu bằng NumPy** để hiểu sâu hơn về cơ chế Optimization (Gradient Descent) và Regularization.

* **Anti-Data Leakage:** Xử lý dữ liệu nghiêm ngặt, đảm bảo không sử dụng thông tin tương lai (như chênh lệch áp suất hiện tại) để dự báo hiện tại.
* **Feature Engineering:** Tối ưu hóa các biến đầu vào bằng phương pháp lượng giác (Trigonometric) cho hướng gió và thời gian, cùng các kỹ thuật Lag/Rolling features.
* **Mục tiêu thực tế:** Ưu tiên chỉ số **Recall cho Class 1 (Unhealthy)** để giảm thiểu sai sót trong việc cảnh báo khi chất lượng không khí xấu.

## Cấu trúc mã nguồn
```text
air_quality_forecasting/
├── src/
│   ├── models/        # Chứa Ridge Regression tự viết bằng NumPy
│   ├── features/      # Xử lý đặc trưng & tránh rò rỉ dữ liệu
│   └── visualization/ # Xuất biểu đồ chuẩn báo cáo khoa học
├── data/              # Dữ liệu từ Open-Meteo API
├── main.py            # Luồng thực thi chính (Pipeline)
└── report/            # Báo cáo chi tiết bằng LaTeX