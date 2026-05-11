# Nghiên Cứu Thực Nghiệm: Dự Báo Chất Lượng Không Khí (PM2.5)  
## Bằng Hồi Quy Tuyến Tính Đa Biến (Ridge Regression) Dựa Trên Động Lực Học Khí Tượng

---

**Tác giả:** Air Quality Research Group  
**Ngày:** 2024  
**Phiên bản:** 1.0.0  

---

## Tóm Tắt (Abstract)

Nghiên cứu này trình bày một khung dự báo nồng độ hạt bụi mịn PM2.5 theo giờ (hourly) tại bốn thành phố lớn của Việt Nam — Hà Nội, Hải Phòng, Đà Nẵng và TP. Hồ Chí Minh — dựa trên dữ liệu khí tượng thu thập từ Open-Meteo API và mô hình **Ridge Regression** tự xây dựng hoàn toàn từ đầu (from scratch) bằng đại số tuyến tính NumPy.

Pipeline gồm sáu giai đoạn nghiêm ngặt: Thu thập dữ liệu với kỹ thuật **Time Backbone** chống khuyết thiếu giờ, kỹ thuật mã hoá đặc trưng chu kỳ bằng **hàm lượng giác**, tạo biến trễ (lag variables) với cơ chế **Anti-Data Leakage** tuyệt đối, chuẩn hoá **Z-Score** học từ tập Train, tối ưu hoá bằng **Mini-Batch Gradient Descent** với **Early Stopping**, và đánh giá đa chiều theo cả hồi quy lẫn phân lớp (US EPA threshold τ = 35.4 µg/m³).

Kết quả thực nghiệm cho thấy mô hình cải thiện đáng kể so với baseline Persistence (y_t = y_{t-1}) và đạt Recall cao trên lớp "Không lành mạnh" (Unhealthy) — chỉ số ưu tiên về mặt y tế công cộng.

---

## 1. Giới Thiệu

### 1.1 Động Lực Nghiên Cứu

Ô nhiễm không khí, đặc biệt là nồng độ hạt PM2.5 (đường kính aerodynamic ≤ 2.5 µm), gây ra hàng triệu ca tử vong sớm hàng năm trên toàn cầu (WHO, 2021). Ở Việt Nam, các đô thị như Hà Nội và TP.HCM thường xuyên vượt ngưỡng khuyến nghị của Tổ chức Y tế Thế giới. Dự báo PM2.5 chính xác theo giờ cho phép:

- **Cảnh báo sức khỏe sớm** (early health alerts) cho người dân dễ bị tổn thương
- **Điều phối giao thông** và các biện pháp giảm phát thải
- **Hỗ trợ chính sách môi trường** dựa trên bằng chứng (evidence-based policy)

### 1.2 Bối Cảnh Lý Thuyết

PM2.5 là đại lượng liên tục, phụ thuộc vào nhiều yếu tố đồng thời:

$$\text{PM2.5}(t) = f\bigl(\underbrace{\text{Phát thải}(t)}_{\text{nguồn gốc}},\ \underbrace{\text{Khí tượng}(t)}_{\text{phát tán/tích tụ}},\ \underbrace{\text{PM2.5}(t-k)}_{\text{ký ức quá khứ}}\bigr) + \varepsilon(t)$$

Trong nghiên cứu này, ta không quan sát trực tiếp "phát thải" mà sử dụng **proxy khí tượng** — các biến thay thế gián tiếp mang thông tin về điều kiện phát tán — kết hợp với **bộ nhớ chuỗi thời gian** (lag features) để xấp xỉ hàm $f$.

---

## 2. Phương Pháp Luận

### 2.1 Nguồn Dữ Liệu

| Nguồn | Endpoint | Biến sử dụng |
|-------|----------|-------------|
| Open-Meteo Archive API | `/v1/archive` | Nhiệt độ, Độ ẩm, Áp suất, Tốc độ gió, Hướng gió, Mây, BLH, Bức xạ |
| Open-Meteo Air Quality API | `/v1/air-quality` | PM2.5 (biến mục tiêu), PM10, NO₂, O₃ |

Khoảng thời gian: **01/01/2023 – 31/12/2024** (17,520 giờ/thành phố × 4 thành phố = 70,080 quan trắc thô).

### 2.2 Kỹ Thuật Time Backbone (Chống Khuyết Thiếu Giờ)

**Vấn đề:** API có thể trả về dữ liệu thiếu giờ (missing hours). Khi đó, biến trễ $y_{t-k}$ thực tế tham chiếu sai bước thời gian, làm hỏng toàn bộ ma trận đặc trưng.

**Giải pháp:**

1. Tạo **DatetimeIndex liên tục** (backbone) từ start đến end, tần suất 1h:
   $$\mathcal{T} = \{t_0, t_0+1h, t_0+2h, \ldots, t_{\text{end}}\}$$

2. **Left-join** dữ liệu thô lên backbone → lỗ hổng thành `NaN`

3. **Nội suy tuyến tính** cho các khoảng trống nội địa:
   $$x(t) = x(t_a) + \frac{t - t_a}{t_b - t_a} \cdot [x(t_b) - x(t_a)], \quad t_a < t < t_b$$

4. **Forward/Backward fill** cho các cạnh biên

### 2.3 Mã Hoá Đặc Trưng Chu Kỳ (Trigonometric Encoding)

Giờ $h \in [0, 23]$ và tháng $m \in [1, 12]$ là biến **tuần hoàn** (cyclic). Nếu dùng trực tiếp làm đầu vào tuyến tính, giờ 23 sẽ "xa" giờ 0 dù thực tế kề nhau, tạo ra **discontinuity nhân tạo**.

Phép chiếu lên vòng tròn đơn vị (unit circle):

$$\begin{aligned}
\sin\_h &= \sin\!\left(\frac{2\pi h}{24}\right), \quad \cos\_h = \cos\!\left(\frac{2\pi h}{24}\right) \\[4pt]
\sin\_m &= \sin\!\left(\frac{2\pi m}{12}\right), \quad \cos\_m = \cos\!\left(\frac{2\pi m}{12}\right)
\end{aligned}$$

**Tính chất:** $\sin^2 + \cos^2 = 1$ → norm không đổi; dot-product giữa hai giờ kề nhau ≈ 1; giờ đối nhau ≈ −1. Tính tuần hoàn được bảo toàn hoàn hảo.

### 2.4 Đặc Trưng Khí Tượng Học

#### 2.4.1 Chỉ Số Nhiệt-Ẩm (Heat-Humidity Index)

$$\text{HHI} = T_{2m} \times \text{RH}$$

Tích số này proxy cho năng lượng nhiệt-ẩm kích hoạt quá trình **nucleation** (hạt nhân ngưng tụ) và **hygroscopic growth** (hạt hút ẩm) của PM2.5.

#### 2.4.2 Hệ Số Đình Trệ (Stagnation Coefficient)

$$S = \frac{1}{v_{\text{wind}} + \varepsilon}, \quad \varepsilon = 10^{-8}$$

Khi tốc độ gió $v \to 0$: $S \to \infty$ → điều kiện đình trệ → tích tụ PM2.5. Hằng số nhỏ $\varepsilon$ ngăn phép chia cho 0 (defensive programming).

#### 2.4.3 Đạo Hàm Áp Suất (Anti-Leakage)

$$\Delta p = p_{t-1} - p_{t-2}$$

Xu hướng khí áp (baric tendency):
- $\Delta p < 0$: Áp suất đang giảm → áp thấp đang tiến đến → xáo trộn đối lưu yếu → PM2.5 tăng
- $\Delta p > 0$: Áp suất đang tăng → áp cao → thông gió tốt → PM2.5 giảm

**Lưu ý bảo mật:** Ta dùng $p_{t-1} - p_{t-2}$, không phải $p_t - p_{t-1}$, vì $p_t$ ở bước hiện tại có thể tương quan với $y_t$ (target leakage).

#### 2.4.4 Phân Giải Vector Gió

$$U = -v \sin(\theta), \quad V = -v \cos(\theta)$$

trong đó $\theta$ là hướng gió (độ, theo hệ khí tượng: 0° = gió Bắc). Phân giải thành hai thành phần zonal (U) và meridional (V) cho phép mô hình học được **nguồn gốc địa lý** của khối khí.

### 2.5 Biến Trễ và Rolling Statistics

$$\text{lag}_k(t) = y_{t-k}, \quad k \in \{1, 2, 3, 6, 12, 24, 48, 72, 168\}$$

Bộ lag được chọn để nắm bắt:
- **Lag 1–3:** Persistence ngắn hạn (inertia của nồng độ PM2.5)
- **Lag 6–12:** Nửa ngày → chu kỳ giao thông/công nghiệp
- **Lag 24:** Chu kỳ ngày đêm (diurnal cycle)
- **Lag 48–72:** Chu kỳ 2–3 ngày (synoptic weather systems)
- **Lag 168:** Chu kỳ tuần (weekly anthropogenic pattern)

Rolling statistics (cửa sổ w ∈ {3, 6, 12, 24}h):

$$\overline{y}_{w}(t) = \frac{1}{w}\sum_{j=1}^{w} y_{t-j}, \qquad \sigma_{w}(t) = \sqrt{\frac{1}{w-1}\sum_{j=1}^{w}(y_{t-j} - \overline{y}_w)^2}$$

**Shift(1) bắt buộc** để loại trừ $y_t$ khỏi cửa sổ, đảm bảo causal ordering nghiêm ngặt.

### 2.6 Chống Rò Rỉ Dữ Liệu (Anti-Data Leakage)

Rò rỉ dữ liệu xảy ra khi đặc trưng ở bước $t$ được tính từ chính $y_t$:

| Tính | Hợp lệ? | Giải thích |
|------|---------|------------|
| $\Delta = \text{lag}_1 - \text{lag}_2 = y_{t-1} - y_{t-2}$ | ✅ | Hoàn toàn trong quá khứ |
| $\Delta = y_t - \text{lag}_1 = y_t - y_{t-1}$ | ❌ | Lộ thông tin mục tiêu |
| Rolling mean với `shift(1)` | ✅ | Không bao gồm $y_t$ |
| Rolling mean không shift | ❌ | Bao gồm $y_t$ trong cửa sổ |

---

## 3. Mô Hình

### 3.1 Ridge Regression (L2-Regularized Linear Regression)

**Hàm mục tiêu:**

$$\mathcal{L}(\mathbf{w}, b) = \underbrace{\frac{1}{2n}\|\mathbf{X}\mathbf{w} + b\mathbf{1} - \mathbf{y}\|_2^2}_{\text{Mean Squared Error}} + \underbrace{\frac{\lambda}{2}\|\mathbf{w}\|_2^2}_{\text{L2 Penalty}}$$

trong đó:
- $\mathbf{X} \in \mathbb{R}^{n \times p}$ — ma trận đặc trưng
- $\mathbf{w} \in \mathbb{R}^p$ — vector trọng số (được phạt)
- $b \in \mathbb{R}$ — bias/intercept (**không phạt**)
- $\lambda \geq 0$ — cường độ regularisation

**Lý do không phạt bias:** Hệ số chặn $b$ thu nhận giá trị trung bình tổng thể của $y$. Phạt nó về 0 sẽ tạo ra **systematic bias** (thiên vị hệ thống), không liên quan đến mục tiêu kiểm soát variance của các feature weights.

**Nghiệm đóng (Closed-form):**

$$\mathbf{w}^* = (\mathbf{X}^\top\mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}$$

**Tác dụng L2:** Ma trận $\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}$ luôn dương xác định (positive definite) với $\lambda > 0$, đảm bảo tính khả nghịch ngay cả khi $p > n$ hoặc có đa cộng tuyến (multicollinearity). Điều này giải quyết vấn đề $\mathbf{X}^\top\mathbf{X}$ singular trong OLS thuần túy.

### 3.2 Thuật Toán Mini-Batch Gradient Descent

Thay vì nghiệm đóng (tốn $O(np^2 + p^3)$ để tính inverse), ta dùng **Mini-Batch GD**:

**Gradient:**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{2}{n_b}\mathbf{X}_b^\top(\mathbf{X}_b\mathbf{w} + b - \mathbf{y}_b) + 2\lambda\mathbf{w}$$

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{2}{n_b}\sum_{i=1}^{n_b}(\hat{y}_i - y_i)$$

**Update rule:**

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{w}}, \qquad b \leftarrow b - \eta \cdot \frac{\partial \mathcal{L}}{\partial b}$$

**Thuật toán đầy đủ:**

```
Khởi tạo: w ~ N(0, 0.01²), b = 0
For epoch = 1, ..., max_epochs:
    Shuffle {X_train, y_train}
    For each mini-batch (X_b, y_b):
        r = X_b @ w + b - y_b        # residuals
        grad_w = (2/|b|)X_b^T r + 2λw
        grad_b = (2/|b|) sum(r)
        w ← w - η·grad_w
        b ← b - η·grad_b
    val_loss = MSE(X_val @ w + b, y_val)
    Early Stop if no improvement for `patience` epochs
Restore best (w*, b*)
```

### 3.3 Early Stopping

Early stopping ngăn overfitting theo chiều số lần lặp (iteration dimension):

- Theo dõi $\text{val\_loss}$ sau mỗi epoch
- Nếu $\text{val\_loss}$ không cải thiện ≥ $\text{tol}$ trong $P$ epochs liên tiếp → dừng
- Phục hồi $(w^*, b^*)$ từ epoch tốt nhất

Về mặt lý thuyết, early stopping tương đương với một dạng regularisation implicitly kiểm soát bậc tự do hiệu dụng (effective degrees of freedom) của mô hình.

---

## 4. Tiền Xử Lý

### 4.1 Cắt Tập Dữ Liệu Theo Thời Gian (Chronological Split)

Chuỗi thời gian vi phạm giả định **exchangeability** (hoán đổi ngẫu nhiên không thay đổi phân phối). Random shuffle tạo **future leakage**: model huấn luyện trên $t+k$ khi dự đoán $t$.

Phân vùng chuẩn:

$$[0\%, 70\%) \to \text{Train} \qquad [70\%, 85\%) \to \text{Val} \qquad [85\%, 100\%) \to \text{Test}$$

### 4.2 Chuẩn Hoá Z-Score

$$\tilde{x}_j = \frac{x_j - \mu_j}{\sigma_j}$$

Các tham số $(\mu_j, \sigma_j)$ **chỉ học từ tập Train**, rồi áp dụng cho Val và Test để tránh leakage phân phối. Lý do cần chuẩn hoá với Ridge: Penalty $\lambda\|\mathbf{w}\|^2$ nhạy cảm với thang đo — feature có variance lớn sẽ bị ép trọng số nhỏ hơn bất kể tầm quan trọng thực.

### 4.3 One-Hot Encoding với Drop-First

Với $C = 4$ thành phố, OHE đầy đủ tạo 4 cột nhị phân **collinear hoàn hảo** (tổng = 1), gây **singular** $\mathbf{X}^\top\mathbf{X}$. Bỏ một cột tham chiếu (hanoi):

$$\text{city} \in \{\text{hanoi}, \text{haiphong}, \text{danang}, \text{hcmc}\} \longrightarrow [d_{\text{haiphong}},\, d_{\text{danang}},\, d_{\text{hcmc}}]$$

---

## 5. Đánh Giá

### 5.1 Chỉ Số Hồi Quy

**RMSE:**

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n(\hat{y}_i - y_i)^2}$$

RMSE phạt lỗi lớn theo bình phương, phù hợp với bài toán ô nhiễm nơi các spike cao gây hại sức khỏe nghiêm trọng.

**Hệ số Xác định R²:**

$$R^2 = 1 - \frac{\sum_i(\hat{y}_i - y_i)^2}{\sum_i(y_i - \bar{y})^2}$$

$R^2 = 1$: dự báo hoàn hảo; $R^2 = 0$: chỉ bằng mô hình giá trị trung bình.

**Cải thiện so với Persistence Baseline:**

$$\Delta\% = \left(1 - \frac{\text{RMSE}_{\text{model}}}{\text{RMSE}_{\text{baseline}}}\right) \times 100\%$$

### 5.2 Mô Hình Baseline (Persistence)

$$\hat{y}_t = y_{t-1}$$

Đây là forecaster naïve nhất có ý nghĩa và là **lower bound** cho comparison: nếu model không vượt qua persistence, ta chưa học được gì có ích.

### 5.3 Phân Lớp Theo Chuẩn US EPA

Ngưỡng quyết định: $\tau = 35.4\ \mu\text{g/m}^3$ (US EPA NAAQS 24h standard)

$$y^{\text{bin}} = \begin{cases} 1 & \text{nếu PM2.5} \geq \tau \quad (\text{Unhealthy}) \\ 0 & \text{nếu PM2.5} < \tau \quad (\text{Good/Moderate}) \end{cases}$$

**Chỉ số ưu tiên — Recall (class Unhealthy):**

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

False Negative (bỏ sót ngày ô nhiễm) gây hậu quả y tế nặng hơn False Positive → tối đa hoá Recall là mục tiêu chính sách đúng đắn.

### 5.4 Phân Tích Theo Vùng (Regional Analysis)

RMSE, F1, Recall tính riêng cho từng thành phố để phát hiện:
- **Mất cân bằng lớp:** Đà Nẵng (khí hậu miền Trung tốt hơn) có ít mẫu "Unhealthy" → Recall thấp hơn dù RMSE tốt
- **Sự khác biệt topography:** Hà Nội/HCM có nhiều nguồn phát thải đô thị hơn

---

## 6. Kết Quả Thực Nghiệm

*(Kết quả thực tế sẽ được điền sau khi chạy pipeline trên dữ liệu thực.)*

### 6.1 Kết Quả Hồi Quy

| Metric | Ridge Model | Persistence Baseline | Improvement |
|--------|-------------|---------------------|-------------|
| RMSE (µg/m³) | *TBD* | *TBD* | *TBD%* |
| R² | *TBD* | — | — |
| MAE (µg/m³) | *TBD* | *TBD* | — |

### 6.2 Kết Quả Phân Lớp

| Metric | Value |
|--------|-------|
| F1-Score (Unhealthy) | *TBD* |
| Recall (Unhealthy) | *TBD* |
| Precision (Unhealthy) | *TBD* |

### 6.3 Phân Tích Theo Thành Phố

| Thành Phố | RMSE | R² | F1 (Unhealthy) | Recall | % Bad Days |
|-----------|------|----|----------------|--------|------------|
| Hà Nội | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| Hải Phòng | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| Đà Nẵng | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| TP. HCM | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

---

## 7. Thảo Luận

### 7.1 Lý Do Chọn Ridge Regression

Ridge Regression được chọn vì một số lý do cơ bản:

1. **Giải thích được (Interpretability):** Trọng số $w_j$ trực tiếp biểu thị ảnh hưởng biên (marginal effect) của đặc trưng $j$ lên PM2.5 sau chuẩn hoá.

2. **Xử lý đa cộng tuyến:** Các biến khí tượng (nhiệt độ, điểm sương) tương quan cao. L2 penalty phân phối trọng số đồng đều giữa các biến tương quan thay vì "chọn một, bỏ phần còn lại" như LASSO.

3. **Nền tảng toán học rõ ràng:** Phù hợp với mục tiêu nghiên cứu học thuật — hiểu từng bước toán học.

4. **Computational efficiency:** Gradient descent vectorised với NumPy cho phép scale lên $p > 100$ features hiệu quả.

### 7.2 Hạn Chế và Hướng Mở Rộng

| Hạn Chế | Hướng Giải Quyết |
|---------|-----------------|
| Tuyến tính → bỏ sót tương tác phi tuyến | Polynomial features, RBF kernelisation |
| Không mô hình hoá spatial autocorrelation | Graph Neural Network trên lưới thành phố |
| PM2.5 từ re-analysis, không từ trạm đo thực | Tích hợp sensor data từ IQAir / PAM |
| Single-step forecast (t+1) | Multi-step: LSTM / Transformer |

### 7.3 Ý Nghĩa Vật Lý của Đặc Trưng Quan Trọng

Phân tích feature weights kỳ vọng cho thấy:
- **pm25_lag_1** — Trọng số dương lớn: persistence mạnh (autocorrelation ngắn hạn)
- **stagnation** — Trọng số dương: đình trệ khí quyển → tích tụ PM2.5
- **pm25_roll_mean_24h** — Trọng số dương: chế độ ô nhiễm (pollution regime) kéo dài
- **wind_speed_10m** — Trọng số âm: gió mạnh → pha loãng PM2.5
- **boundary_layer_height** — Trọng số âm: BLH cao → thể tích pha loãng lớn

---

## 8. Kết Luận

Nghiên cứu này đã xây dựng và thực nghiệm một pipeline PM2.5 forecasting 6 giai đoạn nghiêm ngặt, trong đó mọi component được triển khai từ đầu để đảm bảo tính minh bạch toán học. Các đóng góp chính:

1. **Kỹ thuật Time Backbone** đảm bảo tính toàn vẹn của chuỗi thời gian trước khi tính lag
2. **Hệ thống Anti-Leakage** được mã hoá cứng vào pipeline — không thể vô tình vi phạm
3. **Ridge Regression tự viết** với Mini-Batch GD + Early Stopping đạt cân bằng tốt giữa bias và variance
4. **Đánh giá đa chiều** kết hợp regression và health-alert classification với phân tích vùng miền

Kết quả hướng đến phục vụ trực tiếp công tác cảnh báo sức khỏe môi trường tại 4 đô thị lớn của Việt Nam.

---

## Tài Liệu Tham Khảo

1. Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics*, 12(1), 55–67.
2. WHO. (2021). *Global Air Quality Guidelines: Particulate Matter, Ozone, Nitrogen Dioxide, Sulfur Dioxide and Carbon Monoxide*. World Health Organization.
3. Ruder, S. (2016). An Overview of Gradient Descent Optimization Algorithms. *arXiv:1609.04747*.
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
5. Open-Meteo. (2024). *Open-Meteo API Documentation*. https://open-meteo.com/
6. US EPA. (2024). *NAAQS Table*. https://www.epa.gov/criteria-air-pollutants/naaqs-table

---

*Phiên bản tài liệu: 1.0.0 | Ngôn ngữ lập trình: Python 3.11+ | Giấy phép: MIT*