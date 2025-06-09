# 📚 HƯỚNG DẪN CÁC THUẬT TOÁN KHAI THÁC DỮ LIỆU

## 📋 Mục lục

1. [Apriori - Luật Kết Hợp](#1-apriori---luật-kết-hợp)
2. [Rough Set - Lý Thuyết Tập Thô](#2-rough-set---lý-thuyết-tập-thô)
3. [Decision Tree ID3 - Cây Quyết Định](#3-decision-tree-id3---cây-quyết-định)
4. [Naive Bayes - Phân Loại Xác Suất](#4-naive-bayes---phân-loại-xác-suật)
5. [K-Means - Phân Cụm](#5-k-means---phân-cụm)
6. [Kohonen SOM - Bản Đồ Tự Tổ Chức](#6-kohonen-som---bản-đồ-tự-tổ-chức)
7. [Hướng Dẫn Sử Dụng App](#7-hướng-dẫn-sử-dụng-app)

## 🚀 Giới thiệu dự án

Đây là ứng dụng **Streamlit** triển khai 6 thuật toán khai thác dữ liệu chính với giao diện tương tác thân thiện. Mỗi thuật toán được thiết kế để giải quyết các bài toán khác nhau trong phân tích dữ liệu.

### 📁 Cấu trúc dự án:

```
📂 Dự án Data Mining
├── 📄 app.py              # Giao diện Streamlit chính
├── 📄 func.py             # Logic thuật toán
├── 📄 sample_data.csv     # Dữ liệu mẫu
├── 📄 run_demo.bat        # Script khởi chạy
├── 📄 DEMO_CHEATSHEET.md  # Tham khảo demo
└── 📄 readme.md           # Tài liệu này
```

---

# 1. APRIORI - LUẬT KẾT HỢP

## 🌟 Phiên bản DỄ HIỂU

### Apriori là gì?

Hãy tưởng tượng bạn là chủ siêu thị và muốn biết khách hàng thường mua những gì cùng nhau!

**Ví dụ đời thường:**

```
🛒 Giỏ hàng 1: Bánh mì + Sữa + Bơ
🛒 Giỏ hàng 2: Bánh mì + Sữa
🛒 Giỏ hàng 3: Sữa + Phô mai
🛒 Giỏ hàng 4: Bánh mì + Sữa + Phô mai
```

**Apriori sẽ tìm ra:**

- "Nếu ai mua **Bánh mì** thì **80%** cũng mua **Sữa**"
- "Nếu ai mua **Sữa** và **Bánh mì** thì **60%** cũng mua **Bơ**"

### Tại sao quan trọng?

- 🎯 **Bố trí hàng hóa**: Đặt sữa gần bánh mì
- 📈 **Khuyến mãi**: Giảm giá combo bánh mì + sữa
- 💡 **Gợi ý mua hàng**: "Khách hàng đã mua bánh mì, có thể thích sữa"

### Các khái niệm đơn giản:

- **Support (Hỗ trợ)**: Có bao nhiều % khách hàng mua item này?
  - VD: 70% khách mua bánh mì → Support(Bánh mì) = 0.7
- **Confidence (Tin cậy)**: Nếu mua A thì khả năng mua B là bao nhiều?
  - VD: Trong số người mua bánh mì, 80% cũng mua sữa → Conf(Bánh mì → Sữa) = 0.8

## ⚙️ CÁC TÙYY CHỌN TRONG APP

### 📊 Tùy chọn tiền xử lý dữ liệu:

1. **One-hot encode các cột được chọn**

   - Dùng cho dữ liệu giao dịch (TransactionID, Product)
   - Chuyển categorical data thành dạng binary matrix
   - **Khi nào dùng**: Có dữ liệu dạng giao dịch rõ ràng

2. **Sử dụng các cột số gốc**

   - Dùng trực tiếp dữ liệu số
   - **Lưu ý**: Chỉ phù hợp khi dữ liệu đã ở dạng 0/1

3. **Chuyển đổi toàn bộ sang boolean**
   - Biến tất cả thành True/False
   - **Khi nào dùng**: Dữ liệu hỗn hợp nhiều loại

### 🎚️ Tham số chính:

- **Min Support (0.01-1.0)**: Tỷ lệ tối thiểu xuất hiện của itemset
  - Thấp (0.1-0.3): Nhiều pattern, có thể nhiều noise
  - Cao (0.5-0.8): Ít pattern, chỉ những mẫu rất phổ biến
- **Min Confidence (0.01-1.0)**: Độ tin cậy tối thiểu của luật
  - Thấp (0.3-0.5): Nhiều luật, độ tin cậy thấp
  - Cao (0.7-0.9): Ít luật, độ tin cậy cao

## 🔬 Phiên bản TECHNICAL

### Định nghĩa chính thức:

Apriori là thuật toán tìm **frequent itemsets** và **association rules** từ cơ sở dữ liệu giao dịch.

### 📋 CÁC BƯỚC GIẢI QUYẾT CHI TIẾT:

#### Bước 1: Tiền xử lý dữ liệu

**🔸 Ví dụ thực tế từ file `data_apriori_to_OHE.csv`:**

```
Dữ liệu gốc (giao dịch):
Mã hóa đơn | Mã hàng
o1        | i1, i2, i3
o2        | i2, i3, i4
o3        | i2, i3, i4
o4        | i1, i2, i3
o5        | i3, i4

Sau khi chuyển đổi thành Binary Matrix:
     | i1 | i2 | i3 | i4
-----|----|----|----|----|
o1   | 1  | 1  | 1  | 0
o2   | 0  | 1  | 1  | 1
o3   | 0  | 1  | 1  | 1
o4   | 1  | 1  | 1  | 0
o5   | 0  | 0  | 1  | 1
```

**🔸 Ví dụ từ file `data_apriori.csv` (dữ liệu đã ở dạng binary):**

```
     | i1 | i2 | i3 | i4
-----|----|----|----|----|
o1   | 1  | 1  | 1  | 0
o2   | 0  | 1  | 1  | 1
o3   | 0  | 1  | 1  | 1
o4   | 1  | 1  | 1  | 0
o5   | 0  | 0  | 1  | 1
```

#### Bước 2: Tìm Frequent 1-itemsets (L₁)

```python
# Đếm frequency của từng item (với min_support = 0.4)
Tổng số giao dịch: 5

support(i1) = 2/5 = 0.4 ≥ 0.4 → Keep
support(i2) = 3/5 = 0.6 ≥ 0.4 → Keep
support(i3) = 5/5 = 1.0 ≥ 0.4 → Keep
support(i4) = 3/5 = 0.6 ≥ 0.4 → Keep

L₁ = {i1, i2, i3, i4}  # Tất cả đều frequent
```

#### Bước 3: Sinh Candidate 2-itemsets (C₂)

```python
# Kết hợp các frequent 1-itemsets
L₁ = {i1, i2, i3, i4}
C₂ = {(i1,i2), (i1,i3), (i1,i4), (i2,i3), (i2,i4), (i3,i4)}
```

#### Bước 4: Tính Support và lọc → L₂

```python
# Đếm support của từng 2-itemset
support(i1,i2) = 2/5 = 0.4 ≥ 0.4 → Keep (o1,o4)
support(i1,i3) = 2/5 = 0.4 ≥ 0.4 → Keep (o1,o4)
support(i1,i4) = 0/5 = 0.0 < 0.4 → Remove
support(i2,i3) = 3/5 = 0.6 ≥ 0.4 → Keep (o1,o2,o3,o4)
support(i2,i4) = 2/5 = 0.4 ≥ 0.4 → Keep (o2,o3)
support(i3,i4) = 3/5 = 0.6 ≥ 0.4 → Keep (o2,o3,o5)

L₂ = {(i1,i2), (i1,i3), (i2,i3), (i2,i4), (i3,i4)}
```

#### Bước 5: Sinh Candidate 3-itemsets (C₃)

```python
# Kết hợp các 2-itemsets có chung 1 item
C₃ = {(i1,i2,i3), (i2,i3,i4)}

support(i1,i2,i3) = 2/5 = 0.4 ≥ 0.4 → Keep (o1,o4)
support(i2,i3,i4) = 2/5 = 0.4 ≥ 0.4 → Keep (o2,o3)

L₃ = {(i1,i2,i3), (i2,i3,i4)}
```

#### Bước 6: Thử sinh 4-itemsets

```python
C₄ = {(i1,i2,i3,i4)}
support(i1,i2,i3,i4) = 0/5 = 0.0 < 0.4 → Remove

L₄ = {} → Dừng thuật toán
```

#### Bước 7: Sinh Association Rules (với min_confidence = 0.6)

```python
# Từ frequent itemsets, sinh các luật:

Từ (i1,i2,i3):
- i1 → (i2,i3): conf = 0.4/0.4 = 1.0 ≥ 0.6 ✓
- i2 → (i1,i3): conf = 0.4/0.6 = 0.67 ≥ 0.6 ✓
- i3 → (i1,i2): conf = 0.4/1.0 = 0.4 < 0.6 ✗
- (i1,i2) → i3: conf = 0.4/0.4 = 1.0 ≥ 0.6 ✓

Từ (i2,i3,i4):
- i2 → (i3,i4): conf = 0.4/0.6 = 0.67 ≥ 0.6 ✓
- i3 → (i2,i4): conf = 0.4/1.0 = 0.4 < 0.6 ✗
- i4 → (i2,i3): conf = 0.4/0.6 = 0.67 ≥ 0.6 ✓
```

### 🎯 **KẾT QUẢ CUỐI CÙNG APRIORI:**

```
📊 FREQUENT ITEMSETS:
- 1-itemsets: {i1}, {i2}, {i3}, {i4}
- 2-itemsets: {i1,i2}, {i1,i3}, {i2,i3}, {i2,i4}, {i3,i4}
- 3-itemsets: {i1,i2,i3}, {i2,i3,i4}

🎯 ASSOCIATION RULES (confidence ≥ 0.6):
1. i1 → {i2,i3} (conf: 1.0, lift: 1.67)
2. i2 → {i1,i3} (conf: 0.67, lift: 1.67)
3. {i1,i2} → i3 (conf: 1.0, lift: 1.0)
4. i2 → {i3,i4} (conf: 0.67, lift: 1.11)
5. i4 → {i2,i3} (conf: 0.67, lift: 1.11)

💡 INSIGHTS:
- Item i3 xuất hiện trong tất cả giao dịch (support = 1.0)
- Nếu mua i1 thì chắc chắn sẽ mua cả i2 và i3
- Items i2, i3, i4 thường xuất hiện cùng nhau
```

### Các bước thuật toán:

1. **Tìm frequent 1-itemsets** (L₁)
2. **Sinh candidate 2-itemsets** (C₂) từ L₁
3. **Tính support và lọc** → L₂
4. **Lặp lại** cho đến khi không còn frequent itemsets

### Công thức toán học:

**Support:**

```
Support(A) = |T(A)| / |T|
```

Trong đó: |T(A)| = số giao dịch chứa A, |T| = tổng số giao dịch

**Confidence:**

```
Confidence(A → B) = Support(A ∪ B) / Support(A)
```

**Lift:**

```
Lift(A → B) = Support(A ∪ B) / (Support(A) × Support(B))
```

### Pseudocode:

```python
def apriori(transactions, min_support):
    L1 = find_frequent_1_itemsets(transactions, min_support)
    L = [L1]
    k = 2

    while L[k-2] is not empty:
        Ck = apriori_gen(L[k-2])  # Generate candidates
        for transaction in transactions:
            Ct = subset(Ck, transaction)  # Candidates in transaction
            for candidate in Ct:
                candidate.count++

        Lk = {c in Ck | c.support >= min_support}
        L.append(Lk)
        k = k + 1

    return union(L)
```

---

# 2. ROUGH SET - LÝ THUYẾT TẬP THÔ

## 🌟 Phiên bản DỄ HIỂU

### Rough Set là gì?

Hãy tưởng tượng bạn muốn phân loại học sinh "Giỏi" hay "Yếu" dựa trên điểm số, nhưng có một số trường hợp không rõ ràng!

**Ví dụ:**

```
Học sinh A: Toán=8, Lý=7, Hóa=9 → Giỏi ✓
Học sinh B: Toán=5, Lý=4, Hóa=6 → Yếu ✓
Học sinh C: Toán=7, Lý=6, Hóa=5 → ??? (Không chắc)
```

### Rough Set giúp gì?

- 🎯 **Tìm thuộc tính quan trọng**: Môn nào quyết định "Giỏi/Yếu"?
- 🔍 **Xử lý không chắc chắn**: Phân loại những trường hợp mơ hồ
- ✂️ **Giảm thuộc tính**: Loại bỏ thông tin thừa

### Các khái niệm đơn giản:

- **Lower Approximation (Xấp xỉ dưới)**: Những trường hợp **CHẮC CHẮN** thuộc nhóm
- **Upper Approximation (Xấp xỉ trên)**: Những trường hợp **CÓ THỂ** thuộc nhóm
- **Boundary Region (Vùng biên)**: Những trường hợp **KHÔNG CHẮC**

### Minh họa trực quan:

```
🎯 Mục tiêu: Phân loại "Học sinh giỏi"

Lower Approximation (Chắc chắn giỏi):
👨‍🎓 Toán≥8 AND Lý≥8 → 100% Giỏi

Upper Approximation (Có thể giỏi):
👨‍🎓 Toán≥6 OR Lý≥6 → Có thể Giỏi

Boundary Region (Không chắc):
👨‍🎓 6≤Toán<8 AND 6≤Lý<8 → Cần xem thêm
```

## ⚙️ CÁC TÙYY CHỌN TRONG APP

### 🎯 Chọn thuộc tính:

- **Thuộc tính quyết định**: Cột chứa kết quả cần dự đoán (VD: "Phân loại", "Kết quả")
- **Thuộc tính điều kiện**: Các cột đặc trưng để phân tích (VD: "Điểm toán", "Điểm lý")

### 🏆 Chọn lớp mục tiêu:

- Chọn giá trị cụ thể muốn phân tích (VD: "Giỏi", "Yếu", "Trung bình")

### 💡 Lưu ý quan trọng:

- **Dữ liệu categorical**: App tự động chuyển sang string
- **Cảnh báo rời rạc hóa**: Nếu cột số có >15 giá trị unique sẽ được cảnh báo

## 🔬 Phiên bản TECHNICAL

### Định nghĩa toán học:

Cho tập vũ trụ U, quan hệ tương đương R, và tập mục tiêu X ⊆ U.

**Lower Approximation:**

```
R*(X) = {x ∈ U | [x]R ⊆ X}
```

**Upper Approximation:**

```
R*(X) = {x ∈ U | [x]R ∩ X ≠ ∅}
```

**Boundary Region:**

```
BND_R(X) = R*(X) - R*(X)
```

### Các thước đo chất lượng:

**Accuracy (Độ chính xác):**

```
α_R(X) = |R*(X)| / |R*(X)|
```

**Dependency (Mức độ phụ thuộc):**

```
γ_R(D) = |POS_R(D)| / |U|
```

### 📋 CÁC BƯỚC GIẢI QUYẾT CHI TIẾT:

#### Bước 1: Xây dựng Information Table

**🔸 Ví dụ thực tế từ file `data_rough_set.csv`:**

```
| ID | Kích thước | Màu sắc | Hình dạng   | Lớp |
|----|------------|---------|-------------|-----|
| 1  | Vừa        | Xanh    | Viên gạch   | A   |
| 2  | Nhỏ        | Đỏ      | Hình nêm    | B   |
| 3  | Nhỏ        | Đỏ      | Hình cầu    | A   |
| 4  | Lớn        | Đỏ      | Hình nêm    | B   |
| 5  | Lớn        | Lục     | Hình trụ    | A   |
| 6  | Lớn        | Đỏ      | Hình trụ    | B   |
| 7  | Lớn        | Lục     | Hình cầu    | A   |
```

#### Bước 2: Tạo Equivalence Classes

```python
# Phân tích cho Target = "A" với thuộc tính điều kiện [Kích thước, Màu sắc]

Nhóm theo [Kích thước, Màu sắc]:
E1: {1} → [Vừa, Xanh] → {A}
E2: {2,3} → [Nhỏ, Đỏ] → {B, A}  # Hỗn hợp!
E3: {4,6} → [Lớn, Đỏ] → {B, B}
E4: {5,7} → [Lớn, Lục] → {A, A}
```

#### Bước 3: Tính Lower Approximation

```python
# Tìm classes chắc chắn thuộc target "A"
Target_set = {1, 3, 5, 7}  # Các object có Lớp = "A"

Lower_Approx = {}  # Chỉ những class hoàn toàn trong target
# E1: {1} → tất cả thuộc "A" → Thêm vào Lower
# E2: {2,3} → có object 2 thuộc "B" → Không thêm
# E3: {4,6} → tất cả thuộc "B" → Không thêm
# E4: {5,7} → tất cả thuộc "A" → Thêm vào Lower

Lower_Approx = {1, 5, 7}
```

#### Bước 4: Tính Upper Approximation

```python
# Tìm classes có giao khác rỗng với target "A"
Upper_Approx = {}
# E1: {1} ∩ {1,3,5,7} = {1} ≠ ∅ → Thêm vào Upper
# E2: {2,3} ∩ {1,3,5,7} = {3} ≠ ∅ → Thêm vào Upper
# E3: {4,6} ∩ {1,3,5,7} = {} = ∅ → Không thêm
# E4: {5,7} ∩ {1,3,5,7} = {5,7} ≠ ∅ → Thêm vào Upper

Upper_Approx = {1, 2, 3, 5, 7}
```

#### Bước 5: Tính Accuracy và Dependency

```python
Accuracy = |Lower_Approx| / |Upper_Approx| = 3/5 = 0.6
Dependency = |Lower_Approx| / |Total_Objects| = 3/7 = 0.43

Boundary_Region = Upper_Approx - Lower_Approx = {2, 3}
```

#### Bước 6: Tìm Reducts

```python
# Test từng subset của attributes với full dependency = 3/7 = 0.43

Test [Kích thước]:
- Vừa: {1} → A
- Nhỏ: {2,3} → B,A
- Lớn: {4,5,6,7} → B,A,B,A
dependency = 1/7 = 0.14 ≠ 0.43

Test [Màu sắc]:
- Xanh: {1} → A
- Đỏ: {2,3,4,6} → B,A,B,B
- Lục: {5,7} → A,A
dependency = 3/7 = 0.43 = 0.43 ✓

Test [Hình dạng]:
- Viên gạch: {1} → A
- Hình nêm: {2,4} → B,B
- Hình cầu: {3,7} → A,A
- Hình trụ: {5,6} → A,B
dependency = 4/7 = 0.57 > 0.43

→ Reduct tối thiểu: {Màu sắc}
```

### 🎯 **KẾT QUẢ CUỐI CÙNG ROUGH SET:**

```
📊 PHÂN TÍCH CHO LỚP "A":

🎯 Lower Approximation: {1, 5, 7}
   - Object 1: [Vừa, Xanh, Viên gạch] → Chắc chắn A
   - Object 5: [Lớn, Lục, Hình trụ] → Chắc chắn A
   - Object 7: [Lớn, Lục, Hình cầu] → Chắc chắn A

🎯 Upper Approximation: {1, 2, 3, 5, 7}
   - Bao gồm cả những object có thể thuộc A

🎯 Boundary Region: {2, 3}
   - Object 2,3: [Nhỏ, Đỏ] → Không chắc chắn

📊 ĐỘ ĐO CHẤT LƯỢNG:
- Accuracy: 0.6 (60% objects trong upper đã được xác định)
- Dependency: 0.43 (43% objects có thể phân loại chắc chắn)

🔧 REDUCTS:
- Reduct tối thiểu: {Màu sắc}
- Core: {Màu sắc} (thuộc tính không thể thiếu)

💡 INSIGHTS:
- Màu sắc là thuộc tính quan trọng nhất để phân biệt lớp A
- Objects có màu Lục luôn thuộc lớp A
- Objects có màu Đỏ cần thêm thông tin để phân loại
```

### Reduct và Core:

- **Reduct**: Tập con tối thiểu của thuộc tính vẫn giữ nguyên khả năng phân loại
- **Core**: Giao của tất cả các reduct (thuộc tính không thể thiếu)

---

# 3. DECISION TREE ID3 - CÂY QUYẾT ĐỊNH

## 🌟 Phiên bản DỄ HIỂU

### Decision Tree là gì?

Như một cây câu hỏi để đưa ra quyết định! Mỗi nút là một câu hỏi, mỗi nhánh là một đáp án.

**Ví dụ: Có nên đi chơi không?**

```
🌤️ Thời tiết như thế nào?
├── ☀️ Nắng
│   └── 😊 Đi chơi!
├── 🌧️ Mưa
│   └── 🏠 Ở nhà
└── ☁️ Nhiều mây
    └── 💰 Có tiền không?
        ├── 💸 Có → 😊 Đi chơi!
        └── 💸 Không → 🏠 Ở nhà
```

### Tại sao dùng Decision Tree?

- 📖 **Dễ hiểu**: Như sách hướng dẫn từng bước
- 🚀 **Nhanh**: Quyết định trong vài giây
- 🎯 **Chính xác**: Dựa trên dữ liệu thực tế

### Cách hoạt động:

1. **Chọn câu hỏi tốt nhất** (thuộc tính quan trọng nhất)
2. **Chia dữ liệu** theo câu trả lời
3. **Lặp lại** cho đến khi có kết quả rõ ràng

## ⚙️ CÁC TÙYY CHỌN TRONG APP

### 🎯 Chọn thuộc tính:

- **Thuộc tính mục tiêu**: Cột cần dự đoán (VD: "Buy", "Class", "Result")
- **Thuộc tính đầu vào**: Các cột đặc trưng (VD: "Age", "Income", "Gender")

### 📊 Phương pháp chia nhánh:

1. **Information Gain (Entropy)**

   - Dựa trên lý thuyết thông tin
   - Ưu tiên thuộc tính giảm entropy nhiều nhất
   - **Khi nào dùng**: Dữ liệu cân bằng, nhiều class

2. **Gini Gain**
   - Dựa trên độ bất thuần Gini
   - Nhanh hơn entropy
   - **Khi nào dùng**: Dữ liệu lớn, cần tốc độ

### 🚫 Lưu ý khi sử dụng:

- App tự động chuyển tất cả dữ liệu sang string
- Loại bỏ các dòng có giá trị 'nan'
- Hiển thị điểm số của từng thuộc tính trước khi xây dựng cây

## 🔬 Phiên bản TECHNICAL

### Thuật toán ID3:

ID3 sử dụng **Information Gain** để chọn thuộc tính tốt nhất cho mỗi nút.

**Entropy:**

```
Entropy(S) = -∑(p_i × log₂(p_i))
```

**Information Gain:**

```
Gain(S,A) = Entropy(S) - ∑((|S_v|/|S|) × Entropy(S_v))
```

### 📋 CÁC BƯỚC GIẢI QUYẾT CHI TIẾT:

#### Bước 1: Chuẩn bị dữ liệu

**🔸 Ví dụ thực tế từ file `data_tree.csv` (Tennis Dataset):**

```
| ID  | Outlook  | Temperature | Humidity | Wind   | Play ball |
|-----|----------|-------------|----------|--------|-----------|
| D1  | Sunny    | Hot         | High     | Weak   | No        |
| D2  | Sunny    | Hot         | High     | Strong | No        |
| D3  | Overcast | Hot         | High     | Weak   | Yes       |
| D4  | Rainy    | Mild        | High     | Weak   | Yes       |
| D5  | Rainy    | Cool        | Normal   | Weak   | Yes       |
| D6  | Rainy    | Cool        | Normal   | Strong | No        |
| D7  | Overcast | Cool        | Normal   | Strong | Yes       |
| D8  | Sunny    | Mild        | High     | Weak   | No        |
| D9  | Sunny    | Cool        | Normal   | Weak   | Yes       |
| D10 | Rainy    | Mild        | Normal   | Weak   | Yes       |
| D11 | Sunny    | Mild        | Normal   | Strong | Yes       |
| D12 | Overcast | Mild        | High     | Strong | Yes       |
| D13 | Overcast | Hot         | Normal   | Weak   | Yes       |
| D14 | Rainy    | Mild        | High     | Strong | No        |
```

#### Bước 2: Tính Entropy của tập gốc

```python
Total = 14, Yes = 9, No = 5
p(Yes) = 9/14 = 0.643
p(No) = 5/14 = 0.357

Entropy(S) = -[9/14 × log₂(9/14) + 5/14 × log₂(5/14)]
           = -[0.643 × (-0.637) + 0.357 × (-1.485)]
           = 0.940
```

#### Bước 3: Tính Information Gain cho từng thuộc tính

**Thuộc tính Outlook:**

```python
# Chia theo Outlook
Sunny: {D1,D2,D8,D9,D11} → 2 Yes, 3 No → Entropy = 0.971
Overcast: {D3,D7,D12,D13} → 4 Yes, 0 No → Entropy = 0
Rainy: {D4,D5,D6,D10,D14} → 3 Yes, 2 No → Entropy = 0.971

Gain(Outlook) = 0.940 - [5/14×0.971 + 4/14×0 + 5/14×0.971] = 0.246
```

**Thuộc tính Temperature:**

```python
# Chia theo Temperature
Hot: {D1,D2,D3,D13} → 2 Yes, 2 No → Entropy = 1.0
Mild: {D4,D8,D10,D11,D12,D14} → 4 Yes, 2 No → Entropy = 0.918
Cool: {D5,D6,D7,D9} → 3 Yes, 1 No → Entropy = 0.811

Gain(Temperature) = 0.940 - [4/14×1.0 + 6/14×0.918 + 4/14×0.811] = 0.029
```

**Thuộc tính Humidity:**

```python
# Chia theo Humidity
High: {D1,D2,D3,D4,D8,D12,D14} → 3 Yes, 4 No → Entropy = 0.985
Normal: {D5,D6,D7,D9,D10,D11,D13} → 6 Yes, 1 No → Entropy = 0.592

Gain(Humidity) = 0.940 - [7/14×0.985 + 7/14×0.592] = 0.151
```

**Thuộc tính Wind:**

```python
# Chia theo Wind
Weak: {D1,D3,D4,D5,D8,D9,D10,D13} → 6 Yes, 2 No → Entropy = 0.811
Strong: {D2,D6,D7,D11,D12,D14} → 3 Yes, 3 No → Entropy = 1.0

Gain(Wind) = 0.940 - [8/14×0.811 + 6/14×1.0] = 0.048
```

#### Bước 4: Chọn thuộc tính tốt nhất

```python
Gain(Outlook) = 0.246 (cao nhất)
Gain(Humidity) = 0.151
Gain(Wind) = 0.048
Gain(Temperature) = 0.029

→ Chọn Outlook làm root node
```

#### Bước 5: Xây dựng cây đệ quy

```
                    Outlook
            ┌─────────┼─────────┐
         Sunny    Overcast    Rainy
            │         │         │
          [?]       Yes       [?]

# Sunny branch cần chia tiếp (2 Yes, 3 No)
# Rainy branch cần chia tiếp (3 Yes, 2 No)
```

**Chia nhánh Sunny:** (D1,D2,D8,D9,D11)

```python
# Chỉ xét subset Sunny
Gain(Humidity) = 0.971 - [4/5×0 + 1/5×0] = 0.971
Gain(Wind) = 0.971 - [3/5×0 + 2/5×0] = 0.971
→ Chọn Humidity (hoặc Wind, cả hai đều perfect)

Sunny → Humidity:
├── High: No (D1,D2,D8)
└── Normal: Yes (D9,D11)
```

**Chia nhánh Rainy:** (D4,D5,D6,D10,D14)

```python
# Chỉ xét subset Rainy
Gain(Wind) = 0.971 - [3/5×0 + 2/5×0] = 0.971
→ Chọn Wind

Rainy → Wind:
├── Weak: Yes (D4,D5,D10)
└── Strong: No (D6,D14)
```

### 🎯 **CÂY QUYẾT ĐỊNH HOÀN CHỈNH:**

```
                    Outlook
            ┌─────────┼─────────┐
         Sunny    Overcast    Rainy
            │         │         │
        Humidity     Yes      Wind
        ┌───┴───┐           ┌───┴───┐
      High   Normal      Weak    Strong
        │       │         │        │
       No      Yes       Yes       No
```

### 🎯 **KẾT QUẢ CUỐI CÙNG DECISION TREE:**

```
📊 CÁC LUẬT RÚT RA:

1. IF Outlook = Overcast THEN Play = Yes
2. IF Outlook = Sunny AND Humidity = High THEN Play = No
3. IF Outlook = Sunny AND Humidity = Normal THEN Play = Yes
4. IF Outlook = Rainy AND Wind = Weak THEN Play = Yes
5. IF Outlook = Rainy AND Wind = Strong THEN Play = No

📈 ĐIỂM SỐ CÁC THUỘC TÍNH:
- Outlook: 0.246 (quan trọng nhất)
- Humidity: 0.151 (quan trọng thứ 2)
- Wind: 0.048 (quan trọng thứ 3)
- Temperature: 0.029 (ít quan trọng nhất)

🎯 ACCURACY: 100% (14/14 cases phân loại đúng)

💡 INSIGHTS:
- Outlook là yếu tố quyết định chính
- Khi Overcast → luôn chơi tennis
- Khi Sunny → phụ thuộc độ ẩm
- Khi Rainy → phụ thuộc gió
- Temperature không ảnh hưởng nhiều đến quyết định
```

### Pseudocode:

```python
def ID3(examples, target_attribute, attributes):
    # Base cases
    if all examples have same target value:
        return leaf node with that value

    if attributes is empty:
        return leaf node with majority target value

    # Choose best attribute
    best_attr = argmax(information_gain(examples, attr, target))

    # Create decision node
    tree = DecisionNode(best_attr)

    # Recursively build subtrees
    for value in values(best_attr):
        subset = examples where best_attr = value
        if subset is empty:
            add leaf with majority value
        else:
            remaining_attrs = attributes - {best_attr}
            subtree = ID3(subset, target_attribute, remaining_attrs)
            tree.add_branch(value, subtree)

    return tree
```

### Ưu và nhược điểm:

**Ưu điểm:**

- Dễ hiểu và giải thích
- Không cần giả định về phân phối dữ liệu
- Xử lý được dữ liệu categorical

**Nhược điểm:**

- Dễ overfitting
- Thiên vị với thuộc tính có nhiều giá trị
- Không xử lý được dữ liệu thiếu

---

# 4. NAIVE BAYES - PHÂN LOẠI XÁC SUẤT

## 🌟 Phiên bản DỄ HIỂU

### Naive Bayes là gì?

Như một bác sĩ chẩn đoán bệnh dựa trên triệu chứng!

**Ví dụ: Phân loại email spam**

```
📧 Email có từ "FREE" → 80% là spam
📧 Email có từ "MONEY" → 70% là spam
📧 Email có từ "URGENT" → 60% là spam

📩 Email mới: "FREE MONEY URGENT!!!"
→ Xác suất spam rất cao!
```

### Tại sao gọi là "Naive" (Ngây thơ)?

Vì thuật toán **giả định** các đặc trưng độc lập với nhau (điều này thường không đúng trong thực tế).

**Ví dụ ngây thơ:**

- Giả định: Từ "FREE" và "MONEY" xuất hiện độc lập
- Thực tế: Spam thường dùng cả hai từ cùng lúc

### Các loại Naive Bayes:

- **Gaussian NB**: Cho dữ liệu số (chiều cao, cân nặng...)
- **Multinomial NB**: Cho dữ liệu đếm (số từ trong email...)

## ⚙️ CÁC TÙYY CHỌN TRONG APP

### 🎯 Chọn cột mục tiêu:

- Chọn từ sidebar trước khi chạy thuật toán
- **Lưu ý**: Bắt buộc phải chọn cho Naive Bayes

### 🔧 Tùy chọn loại Naive Bayes:

#### 1. **GaussianNB**

- **Khi nào dùng**: Dữ liệu số liên tục (tuổi, lương, điểm số...)
- **Giả định**: Features tuân theo phân phối Gaussian (normal)
- **Ưu điểm**: Không cần tham số, tự động tính mean và variance
- **Nhược điểm**: Giả định phân phối có thể không đúng

#### 2. **MultinomialNB**

- **Khi nào dùng**: Dữ liệu đếm, frequency (text classification, word count...)
- **Tham số Alpha**: Laplace smoothing (0.01-3.0, thường 0.5-1.0)
  - Alpha thấp: Ít smoothing, có thể overfitting
  - Alpha cao: Nhiều smoothing, có thể underfitting
- **Yêu cầu**: Tất cả features ≥ 0
- **Xử lý giá trị âm**: Tích chọn "Áp dụng MinMaxScaler"

### 📊 Kết quả hiển thị:

- **Training Accuracy**: Độ chính xác trên tập huấn luyện
- **Model Type**: Loại model đã train
- **Feature Info**: Thông tin về features đã xử lý

## 🔬 Phiên bản TECHNICAL

### Định lý Bayes:

```
P(C|X) = P(X|C) × P(C) / P(X)
```

Trong đó:

- P(C|X): Xác suất thuộc class C khi biết features X
- P(X|C): Likelihood - xác suất có features X khi thuộc class C
- P(C): Prior - xác suất ban đầu của class C
- P(X): Evidence - xác suất có features X

### Giả định độc lập:

```
P(x₁,x₂,...,xₙ|C) = P(x₁|C) × P(x₂|C) × ... × P(xₙ|C)
```

### Công thức tổng quát:

```
ĉ = argmax P(c) ∏ P(xᵢ|c)
    c∈C         i=1
```

### Các biến thể:

### 📊 VÍ DỤ CHI TIẾT VỚI DỮ LIỆU THỰC TẾ:

**🔸 Dữ liệu từ file `data_nb.csv` (Tennis Dataset):**

```
Dự đoán: Có chơi tennis hay không dựa trên thời tiết
Target: Play ball (Yes/No)
Features: Outlook, Temperature, Humidity, Wind
```

#### Bước 1: Chuẩn bị dữ liệu và tính Prior

```python
Total samples: 14
P(Yes) = 9/14 = 0.643
P(No) = 5/14 = 0.357
```

#### Bước 2: Tính Likelihood cho từng feature

**Feature: Outlook**

```python
# P(Outlook|Play=Yes)
P(Sunny|Yes) = 2/9 = 0.222      # D9,D11
P(Overcast|Yes) = 4/9 = 0.444   # D3,D7,D12,D13
P(Rainy|Yes) = 3/9 = 0.333      # D4,D5,D10

# P(Outlook|Play=No)
P(Sunny|No) = 3/5 = 0.600       # D1,D2,D8
P(Overcast|No) = 0/5 = 0.000    # Không có
P(Rainy|No) = 2/5 = 0.400       # D6,D14
```

**Feature: Temperature**

```python
# P(Temperature|Play=Yes)
P(Hot|Yes) = 2/9 = 0.222        # D3,D13
P(Mild|Yes) = 4/9 = 0.444       # D4,D10,D11,D12
P(Cool|Yes) = 3/9 = 0.333       # D5,D7,D9

# P(Temperature|Play=No)
P(Hot|No) = 2/5 = 0.400         # D1,D2
P(Mild|No) = 2/5 = 0.400        # D8,D14
P(Cool|No) = 1/5 = 0.200        # D6
```

**Feature: Humidity**

```python
# P(Humidity|Play=Yes)
P(High|Yes) = 3/9 = 0.333       # D3,D4,D12
P(Normal|Yes) = 6/9 = 0.667     # D5,D7,D9,D10,D11,D13

# P(Humidity|Play=No)
P(High|No) = 4/5 = 0.800        # D1,D2,D8,D14
P(Normal|No) = 1/5 = 0.200      # D6
```

**Feature: Wind**

```python
# P(Wind|Play=Yes)
P(Weak|Yes) = 6/9 = 0.667       # D3,D4,D5,D9,D10,D13
P(Strong|Yes) = 3/9 = 0.333     # D7,D11,D12

# P(Wind|Play=No)
P(Weak|No) = 2/5 = 0.400        # D1,D8
P(Strong|No) = 3/5 = 0.600      # D2,D6,D14
```

#### Bước 3: Dự đoán cho sample mới

**Test case: [Outlook=Sunny, Temperature=Cool, Humidity=High, Wind=Strong]**

```python
# Tính P(Yes|features)
P(Yes|features) = P(Yes) × P(Sunny|Yes) × P(Cool|Yes) × P(High|Yes) × P(Strong|Yes)
                = 0.643 × 0.222 × 0.333 × 0.333 × 0.333
                = 0.006

# Tính P(No|features)
P(No|features) = P(No) × P(Sunny|No) × P(Cool|No) × P(High|No) × P(Strong|No)
               = 0.357 × 0.600 × 0.200 × 0.800 × 0.600
               = 0.021

# Kết luận: P(No) > P(Yes) → Prediction: No
```

### 🎯 **KẾT QUẢ CUỐI CÙNG NAIVE BAYES:**

```
📊 BẢNG XÁC SUẤT LIKELIHOOD:

🌤️ OUTLOOK:
       | Yes     | No
-------|---------|--------
Sunny  | 0.222   | 0.600
Overcast| 0.444  | 0.000
Rainy  | 0.333   | 0.400

🌡️ TEMPERATURE:
       | Yes     | No
-------|---------|--------
Hot    | 0.222   | 0.400
Mild   | 0.444   | 0.400
Cool   | 0.333   | 0.200

💧 HUMIDITY:
       | Yes     | No
-------|---------|--------
High   | 0.333   | 0.800
Normal | 0.667   | 0.200

💨 WIND:
       | Yes     | No
-------|---------|--------
Weak   | 0.667   | 0.400
Strong | 0.333   | 0.600

🎯 PRIOR PROBABILITIES:
- P(Yes) = 0.643
- P(No) = 0.357

💡 INSIGHTS:
- Overcast → 100% chơi tennis (P(Overcast|No) = 0)
- High Humidity → thiên về không chơi (P(High|No) = 0.8)
- Weak Wind → thiên về chơi (P(Weak|Yes) = 0.667)
- Model accuracy trên training data: ~93% (13/14 correct)

🔮 LUẬT DỰ ĐOÁN MẠNH:
1. IF Outlook = Overcast → Chắc chắn Play = Yes
2. IF Humidity = High AND Outlook = Sunny → Thiên về Play = No
3. IF Wind = Weak AND Humidity = Normal → Thiên về Play = Yes
```

**Gaussian Naive Bayes:**

```
P(xᵢ|c) = (1/√(2πσ²c)) × exp(-(xᵢ-μc)²/(2σ²c))
```

**Multinomial Naive Bayes:**

```
P(xᵢ|c) = (Nᵢc + α) / (Nc + α|V|)
```

---

# 5. K-MEANS - PHÂN CỤM

## 🌟 Phiên bản DỄ HIỂU

### K-Means là gì?

Như việc chia học sinh thành các nhóm dựa trên điểm số!

**Ví dụ: Chia khách hàng thành 3 nhóm**

```
👥 Nhóm 1: Khách hàng VIP (thu nhập cao, mua nhiều)
👥 Nhóm 2: Khách hàng thường (thu nhập trung bình)
👥 Nhóm 3: Khách hàng mới (thu nhập thấp, ít mua)
```

### Cách hoạt động (như chơi game):

1. **🎯 Đặt K tâm cụm ngẫu nhiên** (K = 3 tâm)
2. **👥 Gán mỗi người vào nhóm gần nhất**
3. **📍 Di chuyển tâm về giữa nhóm**
4. **🔄 Lặp lại** cho đến khi ổn định

### Minh họa trực quan với dữ liệu tranh:

```
**🔸 Sử dụng dữ liệu từ file `data_k-means_kohonen.csv`:**

Initial data: Các bức tranh với đặc trưng [Màu, Nét, Khối]

Bước 1: Random centroids     Bước 2: Assign to clusters
    T1[16,124,19]  •••          C1{T1,T2,T3,T5} • • •
  •  T2[6,13,70]    •             •           ○
T3[10,22,59] •    T5[21,97,23]    •   C2{T4,T6} ○ ○
    • T4[5,81,92]  •              •           ○
      T6[7,94,88] •                            ○

Bước 3: Update centroids     Final result:
  C1_new[13.25,64,37.75] •       "Abstract Art" • • •
     •         •                      •
     •   C2_new[6,87.5,90] ○           •
     •         ○               "Geometric Art" ○ ○
               ○                              ○
```

## 🔬 Phiên bản TECHNICAL

### Objective Function:

K-Means tối thiểu hóa **Within-Cluster Sum of Squares (WCSS)**:

```
J = ∑∑ ||xᵢ - μⱼ||²
    j=1 i∈Cⱼ
```

### Thuật toán Lloyd:

```python
def kmeans(X, k, max_iters=100):
    # Initialize centroids randomly
    centroids = initialize_random(X, k)

    for iteration in range(max_iters):
        # Assignment step
        clusters = assign_points_to_centroids(X, centroids)

        # Update step
        new_centroids = compute_centroids(clusters)

        # Check convergence
        if converged(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids
```

### Độ phức tạp:

- **Time**: O(n × k × i × d)
  - n: số điểm dữ liệu
  - k: số cụm
  - i: số iteration
  - d: số chiều

### 📋 CÁC BƯỚC GIẢI QUYẾT CHI TIẾT:

#### Bước 1: Chuẩn bị và chuẩn hóa dữ liệu

**🔸 Ví dụ thực tế từ file `data_k-means_kohonen.csv` (Art Dataset):**

```python
# Dữ liệu gốc - Phân tích các bức tranh qua 3 đặc trưng
Original Data:
ID | Số màu | Số đường nét | Số hình khối
1  | 16     | 124         | 19
2  | 6      | 13          | 70
3  | 10     | 22          | 59
4  | 5      | 81          | 92
5  | 21     | 97          | 23
6  | 7      | 94          | 88

# Chuẩn hóa bằng MinMaxScaler (0-1)
Scaled Data:
ID | Số màu | Số đường nét | Số hình khối
1  | 0.688  | 1.000       | 0.000
2  | 0.063  | 0.000       | 0.699
3  | 0.313  | 0.081       | 0.562
4  | 0.000  | 0.611       | 1.000
5  | 1.000  | 0.757       | 0.055
6  | 0.125  | 0.729       | 0.945
```

#### Bước 2: Khởi tạo K centroids ngẫu nhiên

```python
K = 2  # Chia tranh thành 2 nhóm phong cách
# Initial centroids (random)
C1 = [0.4, 0.5, 0.3]  # Centroid 1
C2 = [0.8, 0.2, 0.7]  # Centroid 2
```

#### Bước 3: Assignment Step (Iteration 1)

```python
# Tính khoảng cách Euclidean từ mỗi tranh đến centroids

Tranh 1: [0.688, 1.000, 0.000]
- dist_to_C1 = √[(0.688-0.4)² + (1.0-0.5)² + (0.0-0.3)²] = 0.625
- dist_to_C2 = √[(0.688-0.8)² + (1.0-0.2)² + (0.0-0.7)²] = 1.081
→ Assign to Cluster 1

Tranh 2: [0.063, 0.000, 0.699]
- dist_to_C1 = √[(0.063-0.4)² + (0.0-0.5)² + (0.699-0.3)²] = 0.694
- dist_to_C2 = √[(0.063-0.8)² + (0.0-0.2)² + (0.699-0.7)²] = 0.762
→ Assign to Cluster 1

Tranh 3: [0.313, 0.081, 0.562]
- dist_to_C1 = √[(0.313-0.4)² + (0.081-0.5)² + (0.562-0.3)²] = 0.511
- dist_to_C2 = √[(0.313-0.8)² + (0.081-0.2)² + (0.562-0.7)²] = 0.550
→ Assign to Cluster 1

# Kết quả Iteration 1:
Cluster 1: [Tranh 1, 2, 3, 4, 5]
Cluster 2: [Tranh 6]
```

#### Bước 4: Update Step (Iteration 1)

```python
# Tính centroid mới = trung bình của các điểm trong cluster

Cluster 1 (Tranh 1,2,3,4,5):
new_C1 = mean([[0.688,1.000,0.000], [0.063,0.000,0.699],
               [0.313,0.081,0.562], [0.000,0.611,1.000],
               [1.000,0.757,0.055]])
new_C1 = [0.413, 0.490, 0.463]

Cluster 2 (Tranh 6):
new_C2 = [0.125, 0.729, 0.945]
```

#### Bước 5: Assignment Step (Iteration 2)

```python
# Dùng centroids mới để gán lại

Với new_C1=[0.413, 0.490, 0.463], new_C2=[0.125, 0.729, 0.945]:

Tranh 4: [0.000, 0.611, 1.000]
- dist_to_new_C1 = 0.605
- dist_to_new_C2 = 0.150
→ Reassign to Cluster 2!

# Kết quả Iteration 2:
Cluster 1: [Tranh 1, 2, 3, 5]
Cluster 2: [Tranh 4, 6]
```

#### Bước 6: Update Step (Iteration 2)

```python
# Update centroids again
final_C1 = [0.516, 0.460, 0.329]  # Mean of Tranh 1,2,3,5
final_C2 = [0.063, 0.670, 0.973]  # Mean of Tranh 4,6

# Kiểm tra hội tụ - centroids thay đổi < threshold → STOP
```

### 🎯 **KẾT QUẢ CUỐI CÙNG K-MEANS:**

```
📊 PHÂN CỤM TRANH NGHỆ THUẬT (K=2):

🎨 CLUSTER 1 - "Tranh phức tạp, nhiều màu":
- Tranh 1: [16 màu, 124 nét, 19 khối] → Phong cách phức tạp
- Tranh 2: [6 màu, 13 nét, 70 khối] → Tối giản nhưng nhiều khối
- Tranh 3: [10 màu, 22 nét, 59 khối] → Cân bằng
- Tranh 5: [21 màu, 97 nét, 23 khối] → Rất nhiều màu và nét

🖼️ CLUSTER 2 - "Tranh đơn giản, ít màu":
- Tranh 4: [5 màu, 81 nét, 92 khối] → Ít màu nhưng nhiều hình khối
- Tranh 6: [7 màu, 94 nét, 88 khối] → Tương tự tranh 4

📈 CENTROIDS CUỐI CÙNG:
- Cluster 1: [0.516, 0.460, 0.329] (Nhiều màu, ít khối)
- Cluster 2: [0.063, 0.670, 0.973] (Ít màu, nhiều khối)

📊 WCSS = 0.847 (tổng độ lệch trong cluster)

💡 INSIGHTS:
- Tranh được nhóm chủ yếu theo số lượng hình khối
- Cluster 1: Tập trung vào sự đa dạng màu sắc và đường nét
- Cluster 2: Tập trung vào hình khối, ít quan tâm màu sắc
- Có thể mô tả như "Abstract Art" vs "Geometric Art"
```

### Cách chọn K:

1. **Elbow Method**: Tìm "khuỷu tay" trên đồ thị WCSS

   ```python
   # Chạy K-Means với K = 1,2,3,4,5...
   # Plot WCSS vs K
   # Chọn K tại điểm "elbow" (giảm chậm lại)
   ```

2. **Silhouette Analysis**: Đo độ tách biệt của các cụm

   ```python
   # Silhouette score từ -1 đến 1
   # Score cao = clusters tách biệt tốt
   # Chọn K có silhouette score cao nhất
   ```

3. **Gap Statistic**: So sánh với phân phối ngẫu nhiên
   ```python
   # So sánh WCSS thực tế vs WCSS của dữ liệu random
   # Chọn K có gap lớn nhất
   ```

---

# 6. KOHONEN SOM - BẢN ĐỒ TỰ TỔ CHỨC

## 🌟 Phiên bản DỄ HIỂU

### SOM là gì?

Như việc vẽ bản đồ thế giới trên giấy phẳng! Biến dữ liệu phức tạp nhiều chiều thành bản đồ 2D dễ nhìn.

**Ví dụ thực tế:**

```
🎵 Phân loại nhạc:
- Góc trên trái: Rock, Metal
- Góc trên phải: Classical, Jazz
- Góc dưới trái: Pop, Dance
- Góc dưới phải: Country, Folk

→ Nhạc tương tự sẽ ở gần nhau trên bản đồ!
```

### Tại sao hữu ích?

- 👁️ **Trực quan hóa**: Thấy được pattern trong dữ liệu
- 🗺️ **Giảm chiều**: Từ 100 thuộc tính → bản đồ 2D
- 🔍 **Tìm nhóm**: Các vùng tương tự nhau

### Cách hoạt động (như học bản đồ):

1. **🗺️ Tạo lưới trống** (như bản đồ trắng)
2. **📍 Đặt dữ liệu lên lưới** (mỗi điểm tìm vị trí phù hợp)
3. **🔄 Điều chỉnh dần** (vùng xung quanh cũng thay đổi)
4. **✅ Hoàn thành bản đồ** (dữ liệu tương tự ở gần nhau)

## 🔬 Phiên bản TECHNICAL

### Cấu trúc SOM:

- **Input Layer**: Vector đầu vào x ∈ ℝⁿ
- **Output Layer**: Lưới 2D các neuron với weight vector wᵢⱼ ∈ ℝⁿ

### Thuật toán training:

**1. Tìm Best Matching Unit (BMU):**

```
c = argmin ||x(t) - wᵢⱼ(t)||
    i,j
```

**2. Update weights:**

```
wᵢⱼ(t+1) = wᵢⱼ(t) + α(t) × h(c,i,j,t) × [x(t) - wᵢⱼ(t)]
```

**3. Neighborhood function:**

```
h(c,i,j,t) = exp(-||rc - rᵢⱼ||² / (2σ(t)²))
```

**4. Learning rate decay:**

```
α(t) = α₀ × exp(-t/τ₁)
σ(t) = σ₀ × exp(-t/τ₂)
```

### Pseudocode:

```python
def train_som(data, map_size, epochs):
    # Initialize weight vectors randomly
    weights = initialize_weights(map_size, input_dim)

    for epoch in range(epochs):
        for x in data:
            # Find BMU
            bmu = find_best_matching_unit(x, weights)

            # Update weights in neighborhood
            for i, j in map_coordinates:
                distance = euclidean_distance(bmu, (i,j))
                if distance <= neighborhood_radius(epoch):
                    influence = neighborhood_function(distance, epoch)
                    learning_rate = get_learning_rate(epoch)

                    weights[i][j] += learning_rate * influence * (x - weights[i][j])

    return weights
```

---

# 📊 SO SÁNH CÁC THUẬT TOÁN

| Thuật toán        | Loại           | Ứng dụng chính        | Ưu điểm           | Nhược điểm           |
| ----------------- | -------------- | --------------------- | ----------------- | -------------------- |
| **Apriori**       | Association    | Basket Analysis       | Tìm pattern ẩn    | Chậm với dữ liệu lớn |
| **Rough Set**     | Classification | Feature Selection     | Xử lý uncertainty | Phức tạp tính toán   |
| **Decision Tree** | Classification | Rule Extraction       | Dễ hiểu           | Dễ overfitting       |
| **Naive Bayes**   | Classification | Text Classification   | Nhanh, đơn giản   | Giả định độc lập     |
| **K-Means**       | Clustering     | Customer Segmentation | Đơn giản, nhanh   | Cần biết trước K     |
| **SOM**           | Visualization  | Data Exploration      | Trực quan hóa tốt | Khó diễn giải        |

---

# 🎯 KẾT LUẬN

Mỗi thuật toán có điểm mạnh riêng:

- **🛒 Muốn tìm pattern mua hàng** → Dùng **Apriori**
- **🎯 Muốn phân loại chính xác** → Dùng **Decision Tree** hoặc **Naive Bayes**
- **👥 Muốn chia nhóm khách hàng** → Dùng **K-Means**
- **🗺️ Muốn khám phá dữ liệu** → Dùng **SOM**
- **⚖️ Muốn loại bỏ thuộc tính thừa** → Dùng **Rough Set**

**Lời khuyên:** Hãy thử nhiều thuật toán và so sánh kết quả để chọn phương pháp phù hợp nhất!

---

# 7. HƯỚNG DẪN SỬ DỤNG APP

## 🚀 Khởi chạy ứng dụng

### Cách 1: Sử dụng file batch (Windows)

```bash
# Double-click file run_demo.bat
# Hoặc chạy trong command prompt:
run_demo.bat
```

### Cách 2: Sử dụng terminal

```bash
# Cài đặt dependencies (chỉ cần làm 1 lần)
pip install streamlit pandas numpy matplotlib seaborn scikit-learn mlxtend minisom graphviz

# Chạy app
streamlit run app.py
```

## 📊 Quy trình sử dụng từng thuật toán

### 1️⃣ **APRIORI - Luật kết hợp**

#### Bước 1: Upload dữ liệu

- Tải file CSV có dữ liệu giao dịch
- VD: Cột "TransactionID" và "Product"

#### Bước 2: Chọn phương pháp tiền xử lý

```
🔹 One-hot encode: Cho dữ liệu giao dịch chuẩn
   - Chọn 2 cột: [TransactionID, Product]

🔹 Dữ liệu số gốc: Cho dữ liệu đã ở dạng 0/1

🔹 Chuyển boolean: Cho dữ liệu hỗn hợp
```

#### Bước 3: Thiết lập tham số

- **Support**: 0.1-0.5 (thường dùng 0.3)
- **Confidence**: 0.5-0.8 (thường dùng 0.6)

#### Bước 4: Phân tích kết quả

```
✅ Tập phổ biến: Các item xuất hiện thường xuyên
✅ Tập tối đại: Tập phổ biến không chứa trong tập nào lớn hơn
✅ Luật kết hợp: If A then B với confidence
```

---

### 2️⃣ **ROUGH SET - Tập thô**

#### Bước 1: Chuẩn bị dữ liệu

- Đảm bảo có cột kết quả rõ ràng
- Dữ liệu categorical hoặc đã rời rạc hóa

#### Bước 2: Chọn thuộc tính

```
🎯 Thuộc tính quyết định: Cột kết quả (VD: "Buy")
📊 Thuộc tính điều kiện: Các cột đặc trưng (VD: "Age", "Income")
```

#### Bước 3: Chọn lớp mục tiêu

- Chọn giá trị cụ thể cần phân tích (VD: "Yes")

#### Bước 4: Đọc kết quả

```
📍 Xấp xỉ dưới: Chắc chắn thuộc lớp
📍 Xấp xỉ trên: Có thể thuộc lớp
📊 Độ chính xác: Tỷ lệ dưới/trên
🔧 Rút gọn: Thuộc tính tối thiểu cần thiết
```

---

### 3️⃣ **DECISION TREE - Cây quyết định**

#### Bước 1: Chọn thuộc tính

```
🎯 Thuộc tính mục tiêu: Cột cần dự đoán
📊 Thuộc tính đầu vào: Các cột đặc trưng
```

#### Bước 2: Chọn phương pháp

- **Gain (Entropy)**: Cho dữ liệu cân bằng
- **Gini Gain**: Cho dữ liệu lớn, cần tốc độ

#### Bước 3: Phân tích kết quả

```
🌳 Biểu đồ cây: Visualization dễ hiểu
📋 Luật rút ra: Dạng IF-THEN
📊 Điểm thuộc tính: Information Gain/Gini của từng feature
```

---

### 4️⃣ **NAIVE BAYES - Phân loại xác suất**

#### Bước 1: Chọn cột mục tiêu

- Chọn cột cần phân loại từ sidebar

#### Bước 2: Chọn loại Naive Bayes

```
🔹 GaussianNB: Cho dữ liệu số liên tục
🔹 MultinomialNB: Cho dữ liệu đếm/frequency
   - Cài đặt Alpha (Laplace smoothing): 0.1-3.0
   - Có thể cần MinMaxScaler nếu có giá trị âm
```

#### Bước 3: Đọc kết quả

```
📊 Accuracy: Độ chính xác trên tập training
⚡ Tốc độ: Rất nhanh, phù hợp dữ liệu lớn
```

---

### 5️⃣ **K-MEANS - Phân cụm**

#### Bước 1: Chuẩn bị dữ liệu

- Không cần chọn cột mục tiêu
- Sử dụng tất cả cột số

#### Bước 2: Chọn số cụm K

- Thường chọn 2-5 cụm
- Có thể thử nhiều giá trị để so sánh

#### Bước 3: Phân tích kết quả

```
📊 Bảng gán cụm: Mỗi dòng thuộc cụm nào
📈 Scatter plot: Visualization 2D (nếu ≥2 thuộc tính)
🎯 Tâm cụm: Vị trí trung tâm của từng cụm
```

---

### 6️⃣ **KOHONEN SOM - Bản đồ tự tổ chức**

#### Bước 1: Thiết lập tham số

```
🔧 Kích thước lưới:
   - Rows: 3-10 (thường 4)
   - Cols: 3-10 (thường 4)

🎚️ Tham số training:
   - Sigma: 0.5-2.0 (thường 1.0)
   - Learning Rate: 0.1-0.8 (thường 0.5)
   - Iterations: 100-1000 (thường 500)
```

#### Bước 2: Đọc kết quả

```
🗺️ Hit Map: Màu đậm = nhiều data points
📊 Class Maps: Phân bố từng lớp trên bản đồ
🔍 Pattern: Vùng tương tự gần nhau
```

## 🎯 TIPS SỬ DỤNG HIỆU QUẢ

### ✅ Nên làm:

- **Test nhiều tham số**: Thử các giá trị khác nhau để so sánh
- **Chuẩn bị dữ liệu tốt**: Clean data trước khi upload
- **Đọc cảnh báo**: App có thông báo hữu ích
- **So sánh thuật toán**: Dùng nhiều phương pháp cho cùng bài toán

### ❌ Tránh:

- **Dữ liệu quá ít**: <10 dòng sẽ khó phân tích
- **Quá nhiều thuộc tính**: Có thể gây overfitting
- **Tham số cực đoan**: Min support = 0.01 hoặc 0.99
- **Bỏ qua validation**: Không check kết quả có hợp lý

### 🔧 Xử lý lỗi thường gặp:

**Lỗi: "File seems to be binary"**

- Upload file CSV, không phải Excel/Word

**Lỗi: "MultinomialNB không thể xử lý giá trị âm"**

- Tích chọn "Áp dụng MinMaxScaler"

**Cảnh báo: "Cột có nhiều giá trị duy nhất"**

- Cân nhắc rời rạc hóa cho Rough Set

**Không có kết quả Apriori**

- Giảm min_support xuống 0.1-0.2

---

## 📋 CHECKLIST TRƯỚC KHI SỬ DỤNG

- [ ] Đã cài đặt Python và pip
- [ ] Đã install các thư viện cần thiết
- [ ] File dữ liệu ở định dạng CSV
- [ ] Dữ liệu đã được làm sạch (không quá nhiều missing values)
- [ ] Hiểu rõ mục tiêu phân tích
- [ ] Đọc kỹ hướng dẫn thuật toán tương ứng

---

_📚 Tài liệu này được tạo để hỗ trợ học tập môn Khai thác Dữ liệu. Chúc bạn học tốt! 🚀_
