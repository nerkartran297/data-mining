# 📚 HƯỚNG DẪN CÁC THUẬT TOÁN KHAI THÁC DỮ LIỆU

## 📋 Mục lục

1. [Apriori - Luật Kết Hợp](#1-apriori---luật-kết-hợp)
2. [Rough Set - Lý Thuyết Tập Thô](#2-rough-set---lý-thuyết-tập-thô)
3. [Decision Tree ID3 - Cây Quyết Định](#3-decision-tree-id3---cây-quyết-định)
4. [Naive Bayes - Phân Loại Xác Suất](#4-naive-bayes---phân-loại-xác-suật)
5. [K-Means - Phân Cụm](#5-k-means---phân-cụm)
6. [Kohonen SOM - Bản Đồ Tự Tổ Chức](#6-kohonen-som---bản-đồ-tự-tổ-chức)

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

## 🔬 Phiên bản TECHNICAL

### Định nghĩa chính thức:

Apriori là thuật toán tìm **frequent itemsets** và **association rules** từ cơ sở dữ liệu giao dịch.

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

### Cách chọn K:

1. **Elbow Method**: Tìm "khuỷu tay" trên đồ thị WCSS
2. **Silhouette Analysis**: Đo độ tách biệt của các cụm
3. **Gap Statistic**: So sánh với phân phối ngẫu nhiên

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

_📚 Tài liệu này được tạo để hỗ trợ học tập môn Khai thác Dữ liệu. Chúc bạn học tốt! 🚀_
