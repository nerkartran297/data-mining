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

```python
# Chuyển đổi dữ liệu thành transaction matrix
# VD: DataFrame → Binary Matrix
TransactionID | Product  →  Bread | Milk | Butter
T001         | Bread       1    | 0    | 0
T001         | Milk        1    | 1    | 0
T002         | Butter      0    | 0    | 1
```

#### Bước 2: Tìm Frequent 1-itemsets (L₁)

```python
# Đếm frequency của từng item
support(Bread) = 3/4 = 0.75 ≥ min_support → Keep
support(Milk) = 2/4 = 0.5 ≥ min_support → Keep
support(Butter) = 1/4 = 0.25 < min_support → Remove
```

#### Bước 3: Sinh Candidate 2-itemsets (C₂)

```python
# Kết hợp các frequent 1-itemsets
L₁ = {Bread, Milk}
C₂ = {Bread,Milk}
```

#### Bước 4: Tính Support và lọc → L₂

```python
support(Bread,Milk) = 2/4 = 0.5 ≥ min_support → Keep
L₂ = {Bread,Milk}
```

#### Bước 5: Sinh Association Rules

```python
# Từ itemset {Bread,Milk}
Rule 1: Bread → Milk
confidence = support(Bread,Milk)/support(Bread) = 0.5/0.75 = 0.67

Rule 2: Milk → Bread
confidence = support(Bread,Milk)/support(Milk) = 0.5/0.5 = 1.0
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

```
| ID | Toán | Lý | Hóa | Kết quả |
|----|------|----|----|---------|
| 1  | 8    | 9  | 7  | Giỏi    |
| 2  | 6    | 7  | 8  | Giỏi    |
| 3  | 4    | 5  | 6  | Yếu     |
| 4  | 7    | 6  | 7  | ?       |
```

#### Bước 2: Tạo Equivalence Classes

```python
# Nhóm theo thuộc tính điều kiện [Toán, Lý]
Class 1: {ID1} → [8,9] → Giỏi
Class 2: {ID2} → [6,7] → Giỏi
Class 3: {ID3} → [4,5] → Yếu
Class 4: {ID4} → [7,6] → ?
```

#### Bước 3: Tính Lower Approximation

```python
# Tìm classes chắc chắn thuộc target "Giỏi"
Target_set = {ID1, ID2}  # Các object có Kết quả = "Giỏi"
Lower_Approx = {}  # Chỉ những class hoàn toàn trong target
# Class 1: {ID1} ⊆ {ID1,ID2} → Thêm vào Lower
# Class 2: {ID2} ⊆ {ID1,ID2} → Thêm vào Lower
Lower_Approx = {ID1, ID2}
```

#### Bước 4: Tính Upper Approximation

```python
# Tìm classes có giao khác rỗng với target
Upper_Approx = {}
# Class 1: {ID1} ∩ {ID1,ID2} ≠ ∅ → Thêm vào Upper
# Class 2: {ID2} ∩ {ID1,ID2} ≠ ∅ → Thêm vào Upper
Upper_Approx = {ID1, ID2}
```

#### Bước 5: Tính Accuracy và Dependency

```python
Accuracy = |Lower_Approx| / |Upper_Approx| = 2/2 = 1.0
Dependency = |Lower_Approx| / |Total_Objects| = 2/4 = 0.5
```

#### Bước 6: Tìm Reducts

```python
# Test từng subset của attributes
Full_dependency = dependency([Toán,Lý,Hóa], Kết quả) = 0.75
Test: dependency([Toán,Lý], Kết quả) = 0.5 ≠ 0.75
Test: dependency([Toán,Hóa], Kết quả) = 0.5 ≠ 0.75
Test: dependency([Lý,Hóa], Kết quả) = 0.5 ≠ 0.75
→ Không có reduct, cần tất cả 3 thuộc tính
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

```
| Age | Income | Gender | Buy |
|-----|--------|--------|-----|
| 25  | 30000  | Male   | Yes |
| 30  | 45000  | Female | Yes |
| 35  | 55000  | Male   | No  |
| 28  | 38000  | Female | Yes |
```

#### Bước 2: Tính Entropy của tập gốc

```python
Total = 4, Yes = 3, No = 1
Entropy(S) = -[3/4 × log₂(3/4) + 1/4 × log₂(1/4)]
           = -[0.75 × (-0.415) + 0.25 × (-2)]
           = 0.811
```

#### Bước 3: Tính Information Gain cho từng thuộc tính

**Thuộc tính Age:**

```python
# Chia theo Age
Age=25: {Yes} → Entropy = 0
Age=30: {Yes} → Entropy = 0
Age=35: {No} → Entropy = 0
Age=28: {Yes} → Entropy = 0

Gain(Age) = 0.811 - [1/4×0 + 1/4×0 + 1/4×0 + 1/4×0] = 0.811
```

**Thuộc tính Income:**

```python
# Nhóm theo khoảng income
Low(≤35000): {Yes} → Entropy = 0
High(>35000): {Yes,Yes,No} → Entropy = 0.918

Gain(Income) = 0.811 - [1/4×0 + 3/4×0.918] = 0.122
```

#### Bước 4: Chọn thuộc tính tốt nhất

```python
Gain(Age) = 0.811 (cao nhất)
Gain(Income) = 0.122
Gain(Gender) = 0.311

→ Chọn Age làm root node
```

#### Bước 5: Xây dựng cây đệ quy

```
       Age
    ┌─────┼─────┐
   25    30    35    28
    │     │     │     │
   Yes   Yes   No    Yes
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

### Minh họa trực quan:

```
Bước 1: Tâm ngẫu nhiên        Bước 2: Gán nhóm
   *     •  •                   *●●   ○ ○
 •   •     •                  ●   ●   ○ ○
   •   *     • •                ●   *○   ○ ○
     •   •  *                    ●   ○○  *○

Bước 3: Di chuyển tâm          Kết quả cuối:
   *●●   ○ ○                    *●●   ○ ○
 ●   ●   ○ ○                  ●   ●   ○ ○
   ●   *○   ○ ○                 ●   *○   ○ ○
     ●   ○○  *○                   ●   ○○  *○
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

```python
# Dữ liệu gốc
data = [[25, 30000], [30, 45000], [35, 55000], [22, 25000]]

# Chuẩn hóa bằng MinMaxScaler
scaled_data = [[0.0, 0.0], [0.38, 0.67], [1.0, 1.0], [0.0, 0.0]]
```

#### Bước 2: Khởi tạo K centroids ngẫu nhiên

```python
K = 3
centroids = [[0.2, 0.3], [0.6, 0.8], [0.9, 0.1]]  # Random positions
```

#### Bước 3: Assignment Step

```python
# Tính khoảng cách từ mỗi điểm đến centroids
for each point:
    distances = [euclidean(point, centroid) for centroid in centroids]
    assign point to closest centroid

# Kết quả iteration 1:
Cluster 0: [point1, point4]
Cluster 1: [point2]
Cluster 2: [point3]
```

#### Bước 4: Update Step

```python
# Tính centroid mới = trung bình của các điểm trong cluster
new_centroids = []
for cluster in clusters:
    new_centroid = mean(cluster_points)
    new_centroids.append(new_centroid)

# New centroids: [[0.1, 0.1], [0.38, 0.67], [1.0, 1.0]]
```

#### Bước 5: Kiểm tra hội tụ

```python
# Nếu centroids không thay đổi đáng kể → Dừng
# Ngược lại → Lặp lại Bước 3 và 4

convergence_threshold = 0.001
if max(distance(old_centroid, new_centroid)) < threshold:
    break
```

#### Bước 6: Tính WCSS (Within-Cluster Sum of Squares)

```python
wcss = 0
for cluster in clusters:
    for point in cluster:
        wcss += euclidean_distance(point, cluster_centroid)**2
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
