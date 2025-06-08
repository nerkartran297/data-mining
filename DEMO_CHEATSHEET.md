# 🚀 DEMO CHEATSHEET - THAM KHẢO NHANH

## 📋 Danh sách file đã tạo:

- ✅ `app.py` - Giao diện chính
- ✅ `func.py` - Logic thuật toán
- ✅ `sample_data.csv` - Dữ liệu mẫu cho demo
- ✅ `run_demo.bat` - Script khởi chạy nhanh
- ✅ `ALGORITHMS_GUIDE.md` - Hướng dẫn chi tiết các thuật toán

## 🎯 DEMO SCRIPT CHO TỪNG THUẬT TOÁN

### 1️⃣ APRIORI (Luật kết hợp)

```
🎯 Mục đích: Tìm sản phẩm thường mua cùng nhau
📊 Dữ liệu: TransactionID + Product
⚙️ Tham số: Support=0.3, Confidence=0.6
💡 Giải thích: "Nếu mua Bread → 80% cũng mua Milk"
```

### 2️⃣ ROUGH SET (Tập thô)

```
🎯 Mục đích: Tìm thuộc tính quan trọng, xử lý uncertainty
📊 Dữ liệu: Các thuộc tính điều kiện + thuộc tính quyết định
💡 Giải thích: "Income và Age đủ để quyết định Buy/No Buy"
🔍 Kết quả: Lower/Upper approximation, Accuracy, Reducts
```

### 3️⃣ DECISION TREE (Cây quyết định)

```
🎯 Mục đích: Tạo luật if-then dễ hiểu
📊 Dữ liệu: Features (Age, Income, Gender) → Target (Buy)
⚙️ Phương pháp: Information Gain hoặc Gini
💡 Giải thích: "IF Income>40000 AND Age<35 THEN Buy=Yes"
```

### 4️⃣ NAIVE BAYES (Phân loại xác suất)

```
🎯 Mục đích: Phân loại dựa trên xác suất
📊 Dữ liệu: Features → Target class
⚙️ Loại: GaussianNB (số) hoặc MultinomialNB (đếm)
💡 Giải thích: "Dựa trên Age=25, Income=30k → 75% khả năng Buy=Yes"
```

### 5️⃣ K-MEANS (Phân cụm)

```
🎯 Mục đích: Chia khách hàng thành nhóm
📊 Dữ liệu: Tất cả features (không cần target)
⚙️ Tham số: K=3 (số cụm)
💡 Giải thích: "Nhóm 1: VIP, Nhóm 2: Thường, Nhóm 3: Mới"
```

### 6️⃣ KOHONEN SOM (Bản đồ tự tổ chức)

```
🎯 Mục đích: Trực quan hóa dữ liệu nhiều chiều
📊 Dữ liệu: Tất cả features
⚙️ Tham số: Grid 4x4, Sigma=1.0, LR=0.5, Iter=500
💡 Giải thích: "Biến data nhiều chiều thành bản đồ 2D"
```

---

## 🗣️ CÂU NÓI DEMO CHUẨN

### Mở đầu:

> "Thưa thầy/cô, em xin phép demo ứng dụng khai thác dữ liệu với 6 thuật toán chính..."

### Cho mỗi thuật toán:

1. **Giới thiệu**: "Thuật toán X dùng để..."
2. **Upload data**: "Em sẽ dùng dữ liệu mẫu về giao dịch bán hàng..."
3. **Thiết lập**: "Em chọn tham số như sau vì..."
4. **Chạy**: "Bấm nút chạy và chờ kết quả..."
5. **Giải thích**: "Kết quả cho thấy... có ý nghĩa thực tế là..."

### Kết thúc:

> "Qua demo em thấy mỗi thuật toán có ưu nhược riêng, phù hợp với từng mục đích khác nhau..."

---

## ❓ CHUẨN BỊ TRẢ LỜI CÂU HỎI

### Q: Tại sao chọn tham số này?

**A:** "Em chọn dựa trên đặc điểm dữ liệu và mục đích phân tích. Tham số quá thấp/cao sẽ..."

### Q: Thuật toán nào tốt nhất?

**A:** "Tùy thuộc mục đích. Apriori cho association, Decision Tree cho classification dễ hiểu, K-Means cho segmentation..."

### Q: Có thể áp dụng thực tế không?

**A:** "Dạ có. VD: Apriori cho recommendation system, K-Means cho customer segmentation, Naive Bayes cho spam detection..."

### Q: Làm sao đánh giá hiệu quả?

**A:** "Dùng accuracy, precision, recall cho classification. Silhouette score cho clustering. Support/confidence cho association rules..."

---

## 🚨 LƯU Ý QUAN TRỌNG

### ✅ PHẢI LÀM:

- Giải thích TẠI SAO dùng thuật toán này
- Nói rõ Ý NGHĨA kết quả
- Kết nối với THỰC TẾ
- Chuẩn bị sẵn dữ liệu backup

### ❌ TRÁNH:

- Chỉ chạy mà không giải thích
- Dùng tham số random không có lý do
- Không hiểu kết quả đang nói gì
- Quên check lỗi trước khi demo

---

## 🎯 THỨ TỰ DEMO GỢI Ý

1. **Apriori** - Dễ hiểu nhất, kết quả trực quan
2. **Decision Tree** - Có biểu đồ cây đẹp
3. **K-Means** - Có scatter plot màu sắc
4. **Naive Bayes** - Nhanh, cho thấy accuracy
5. **Rough Set** - Phức tạp hơn, để cuối
6. **SOM** - Đẹp nhất, để kết thúc ấn tượng

---

## 📱 CHECKLIST TRƯỚC DEMO

- [ ] Đã test chạy tất cả thuật toán
- [ ] Chuẩn bị 2-3 bộ dữ liệu khác nhau
- [ ] Đọc kỹ file `ALGORITHMS_GUIDE.md`
- [ ] Luyện thuyết trình 2-3 lần
- [ ] Backup code trên USB/cloud
- [ ] Test trên máy demo thật

**🍀 Chúc bạn demo thành công! 🚀**
