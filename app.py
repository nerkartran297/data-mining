import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import MinMaxScaler

# Import các hàm từ func.py
import func 

# Cấu hình trang Streamlit
st.set_page_config(
    layout="wide", 
    page_title="🔍 Data Mining Toolkit", 
    page_icon="🔍",
    initial_sidebar_state="expanded"
)

# Custom CSS để làm đẹp giao diện và support dark mode
st.markdown("""
<style>
    /* Main header - adaptable to theme */
    .main-header {
        font-size: 3rem;
        color: var(--text-color, #1f77b4);
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Algorithm cards - always visible */
    .algorithm-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white !important;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Info boxes - dark mode compatible */
    .info-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
        background-color: rgba(33, 150, 243, 0.1);
        color: inherit;
    }
    
    /* Explanation boxes */
    .explanation-box {
        background: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: inherit;
    }
    
    /* Warning boxes */
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #FFC107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: inherit;
    }
    
    /* Success boxes */
    .success-box {
        background: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: inherit;
    }
    
    /* Error boxes */
    .error-box {
        background: rgba(244, 67, 54, 0.1);
        border-left: 4px solid #f44336;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: inherit;
    }
    
    /* Step-by-step explanation */
    .step-explanation {
        background: rgba(156, 39, 176, 0.1);
        border-left: 4px solid #9C27B0;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: inherit;
    }
    
    /* Variable explanation */
    .variable-explanation {
        background: rgba(255, 87, 34, 0.1);
        border-left: 4px solid #FF5722;
        padding: 12px;
        border-radius: 5px;
        margin: 8px 0;
        color: inherit;
        font-family: 'Courier New', monospace;
    }
    
    /* Formula display */
    .formula-box {
        background: rgba(63, 81, 181, 0.1);
        border: 2px solid #3F51B5;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        text-align: center;
        font-family: 'Times New Roman', serif;
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# HEADER & NAVIGATION
# ==============================================================================
st.markdown('<h1 class="main-header">🔍 Data Mining Toolkit</h1>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div class="info-box" style="text-align: center; margin-bottom: 30px;">
        <h3>🎓 Đồ án cuối kỳ: Khai thác Dữ liệu</h3>
        <p><strong>👨‍💻 Nhóm thực hiện:</strong> Trần Nhật Khánh & Trần Nhật Huy</p>
        <p>📚 6 thuật toán Data Mining với giao diện thân thiện</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.markdown("## 📁 Cấu hình dữ liệu")
    
    # File upload with better styling
    uploaded_file = st.file_uploader(
        "📤 Chọn file CSV để phân tích", 
        type=["csv"],
        help="Hỗ trợ các file CSV có cấu trúc dữ liệu phù hợp"
    )
    
    # Load demo data option
    st.markdown("---")
    st.markdown("### 📊 Hoặc sử dụng dữ liệu mẫu")
    demo_options = {
        "Không sử dụng": None,
        "🛒 Apriori - Giao dịch": "data_apriori.csv",
        "⚖️ Rough Set - Phân loại": "data_rough_set.csv", 
        "🌳 Decision Tree - Tennis": "data_tree.csv",
        "🎯 Naive Bayes - Tennis": "data_nb.csv",
        "🎨 Clustering - Art": "data_k-means_kohonen.csv"
    }
    
    demo_choice = st.selectbox("Chọn dữ liệu mẫu:", list(demo_options.keys()))
    
    if demo_choice != "Không sử dụng" and demo_options[demo_choice]:
        try:
            uploaded_file = demo_options[demo_choice]
            st.success(f"✅ Đã chọn: {demo_choice}")
        except:
            st.error("❌ Không tìm thấy file mẫu")

# Load data
df_original = None
if uploaded_file is not None:
    try:
        if isinstance(uploaded_file, str):  # Demo file
            df_original = pd.read_csv(uploaded_file)
        else:  # Uploaded file
            df_original = pd.read_csv(uploaded_file)
        
        with st.sidebar:
            st.markdown("### 📋 Thông tin dữ liệu")
            st.info(f"📊 Kích thước: {df_original.shape[0]} hàng x {df_original.shape[1]} cột")
            st.info(f"💾 Bộ nhớ: {df_original.memory_usage(deep=True).sum() / 1024:.1f} KB")
    except Exception as e:
        st.error(f"❌ Lỗi đọc file: {e}")
        df_original = None

# ==============================================================================
# MAIN CONTENT AREA
# ==============================================================================
if df_original is not None:
    # Data preview section
    with st.expander("👀 Xem trước dữ liệu", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(df_original.head(10), use_container_width=True)
        with col2:
            st.markdown("### 📈 Thống kê cơ bản")
            st.metric("Số dòng", df_original.shape[0])
            st.metric("Số cột", df_original.shape[1])
            st.metric("Giá trị null", df_original.isnull().sum().sum())
    
    # Algorithm selection with tabs
    st.markdown("## 🛠️ Chọn thuật toán phân tích")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🛒 Apriori", "⚖️ Rough Set", "🌳 Decision Tree", 
        "🎯 Naive Bayes", "👥 K-Means", "🗺️ SOM"
    ])
    
    # ==============================================================================
    # APRIORI TAB
    # ==============================================================================
    with tab1:
        st.markdown("### 🛒 Thuật toán Apriori - Tìm luật kết hợp")
        
        # Detailed algorithm explanation
        with st.expander("📚 Giải thích chi tiết thuật toán Apriori", expanded=False):
            st.markdown("""
            <div class="explanation-box">
                <h4>🎯 Mục đích thuật toán</h4>
                <p><strong>Apriori</strong> là thuật toán kinh điển để tìm các <strong>tập mục phổ biến</strong> (frequent itemsets) 
                và <strong>luật kết hợp</strong> (association rules) trong dữ liệu giao dịch.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="variable-explanation">
                <h4>📖 Giải thích ký hiệu:</h4>
                <p><strong>• i1, i2, i3, i4...</strong> = Item (sản phẩm): Đại diện cho các sản phẩm riêng lẻ</p>
                <p><strong>• T1, T2, T3...</strong> = Transaction (giao dịch): Mỗi giao dịch chứa nhiều items</p>
                <p><strong>• L1, L2, L3...</strong> = Level k itemsets: Tập phổ biến có k phần tử</p>
                <p><strong>• C1, C2, C3...</strong> = Candidate itemsets: Tập ứng viên cần kiểm tra</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="formula-box">
                <h4>📐 Công thức tính toán:</h4>
                <p><strong>Support(X) = |T(X)| / |D|</strong></p>
                <p>Trong đó: T(X) = số giao dịch chứa itemset X, D = tổng số giao dịch</p>
                <hr>
                <p><strong>Confidence(X → Y) = Support(X ∪ Y) / Support(X)</strong></p>
                <p>Độ tin cậy của luật: X xuất hiện thì Y cũng xuất hiện</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="step-explanation">
                <h4>🔄 Các bước thực hiện:</h4>
                <p><strong>Bước 1:</strong> Quét dữ liệu, đếm tần suất xuất hiện của từng item đơn lẻ</p>
                <p><strong>Bước 2:</strong> Loại bỏ items có support < min_support → Tạo L1</p>
                <p><strong>Bước 3:</strong> Tạo tập ứng viên C2 từ L1 (kết hợp 2 items)</p>
                <p><strong>Bước 4:</strong> Tính support cho C2, loại bỏ itemsets không đạt → Tạo L2</p>
                <p><strong>Bước 5:</strong> Lặp lại cho đến khi không tìm được tập phổ biến mới</p>
                <p><strong>Bước 6:</strong> Sinh luật kết hợp từ các tập phổ biến</p>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col2:
            st.markdown("""
            <div class="algorithm-card">
                <h4>📝 Thông tin thuật toán</h4>
                <p>• Tìm các tập mục phổ biến</p>
                <p>• Sinh luật kết hợp</p>
                <p>• Ứng dụng: Market Basket Analysis</p>
                <p>• Phân tích mua sắm khách hàng</p>
                <p>• Gợi ý sản phẩm</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col1:
            df_apriori_input = df_original.copy()
            df_apriori_input = df_apriori_input.loc[:, ~df_apriori_input.columns.str.startswith('Unnamed')]
            
            st.markdown("#### ⚙️ Cấu hình tiền xử lý")
            preprocess_method = st.radio(
                "Chọn phương pháp xử lý dữ liệu:",
                ("🔢 One-hot encode", "📊 Sử dụng dữ liệu số gốc", "🔄 Chuyển đổi boolean"),
                horizontal=True
            )
            
            df_processed = None
            if "One-hot" in preprocess_method:
                cols_to_encode = st.multiselect(
                    "Chọn cột để mã hóa:",
                    df_apriori_input.columns.tolist(),
                    help="Chọn các cột categorical để chuyển thành dạng binary"
                )
                if cols_to_encode:
                    df_processed = func.apriori_one_hot_encode_data(df_apriori_input, cols_to_encode)
            elif "số gốc" in preprocess_method:
                df_processed = func.apriori_original_process_data(df_apriori_input)
            else:
                df_processed = func.apriori_general_transactional_conversion(df_apriori_input)
            
            if df_processed is not None:
                with st.expander("👀 Xem dữ liệu đã xử lý"):
                    st.dataframe(df_processed.head())
                
                st.markdown("#### 🎚️ Thiết lập tham số")
                col_a, col_b = st.columns(2)
                with col_a:
                    min_support = st.slider("🎯 Min Support", 0.01, 1.0, 0.3, 0.01, 
                                          help="Ngưỡng hỗ trợ tối thiểu (0.3 = 30%)")
                with col_b:
                    min_confidence = st.slider("💪 Min Confidence", 0.01, 1.0, 0.6, 0.01,
                                             help="Ngưỡng tin cậy tối thiểu (0.6 = 60%)")
                
                if st.button("🚀 Chạy Apriori", type="primary", use_container_width=True):
                    with st.spinner("⏳ Đang phân tích..."):
                        try:
                            f_itemsets, rules, max_f_itemsets = func.run_apriori_calculations(
                                df_processed, min_support, min_confidence
                            )
                            
                            st.markdown("### 📊 Kết quả phân tích")
                            
                            # Metrics row
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("🎯 Tập phổ biến", len(f_itemsets) if not f_itemsets.empty else 0)
                            with col2:
                                st.metric("⭐ Tập tối đại", len(max_f_itemsets) if not max_f_itemsets.empty else 0)
                            with col3:
                                st.metric("📜 Luật kết hợp", len(rules) if not rules.empty else 0)
                            
                            # Explanation of results
                            st.markdown("""
                            <div class="explanation-box">
                                <h4>📖 Giải thích kết quả:</h4>
                                <p><strong>• Tập phổ biến:</strong> Các tập items xuất hiện cùng nhau với tần suất ≥ min_support</p>
                                <p><strong>• Tập tối đại:</strong> Tập phổ biến không là tập con của tập phổ biến nào khác</p>
                                <p><strong>• Luật kết hợp:</strong> Luật dạng "X → Y" với độ tin cậy ≥ min_confidence</p>
                                <p><strong>• Support:</strong> Tỷ lệ giao dịch chứa itemset trong tổng số giao dịch</p>
                                <p><strong>• Confidence:</strong> Tỷ lệ giao dịch chứa Y trong số giao dịch chứa X</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Results display
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("#### 🎯 Tập phổ biến")
                                if not f_itemsets.empty:
                                    display_df = f_itemsets.copy()
                                    display_df['itemsets'] = display_df['itemsets'].apply(lambda x: ', '.join(list(x)))
                                    st.dataframe(display_df[['support', 'itemsets']], use_container_width=True)
                                else:
                                    st.info("Không tìm thấy tập phổ biến nào")
                            
                            with col2:
                                st.markdown("#### 📜 Luật kết hợp")
                                if not rules.empty:
                                    display_rules = rules.copy()
                                    display_rules['Rule'] = (display_rules['antecedents'].apply(lambda x: ', '.join(list(x))) + 
                                                           ' → ' + 
                                                           display_rules['consequents'].apply(lambda x: ', '.join(list(x))))
                                    st.dataframe(display_rules[['Rule', 'confidence', 'support']], use_container_width=True)
                                else:
                                    st.info("Không tìm thấy luật kết hợp nào")
                            
                            # Detailed calculation explanation
                            with st.expander("🧮 Giải thích chi tiết từng bước tính toán", expanded=False):
                                detailed_explanation = func.get_apriori_detailed_explanation(
                                    df_apriori_input, min_support, min_confidence
                                )
                                st.markdown(detailed_explanation)
                                    
                        except Exception as e:
                            st.error(f"❌ Lỗi: {e}")
    
    # ==============================================================================
    # ROUGH SET TAB
    # ==============================================================================
    with tab2:
        st.markdown("### ⚖️ Thuật toán Rough Set - Lý thuyết tập thô")
        
        # Detailed algorithm explanation
        with st.expander("📚 Giải thích chi tiết lý thuyết Rough Set", expanded=False):
            st.markdown("""
            <div class="explanation-box">
                <h4>🎯 Mục đích thuật toán</h4>
                <p><strong>Rough Set Theory</strong> được phát triển bởi Pawlak (1982) để xử lý thông tin 
                không chắc chắn và không đầy đủ trong việc phân loại dữ liệu.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="variable-explanation">
                <h4>📖 Giải thích ký hiệu:</h4>
                <p><strong>• U</strong> = Universe (vũ trụ): Tập hợp tất cả các đối tượng</p>
                <p><strong>• A</strong> = Attributes (thuộc tính): Tập thuộc tính mô tả đối tượng</p>
                <p><strong>• D</strong> = Decision attribute (thuộc tính quyết định): Thuộc tính phân loại</p>
                <p><strong>• [x]ᵣ</strong> = Equivalence class: Lớp tương đương của đối tượng x</p>
                <p><strong>• R(X)</strong> = Lower approximation: Xấp xỉ dưới của tập X</p>
                <p><strong>• R̄(X)</strong> = Upper approximation: Xấp xỉ trên của tập X</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="formula-box">
                <h4>📐 Công thức tính toán:</h4>
                <p><strong>Lower Approximation: R(X) = {x ∈ U | [x]ᵣ ⊆ X}</strong></p>
                <p>Các đối tượng chắc chắn thuộc lớp X</p>
                <hr>
                <p><strong>Upper Approximation: R̄(X) = {x ∈ U | [x]ᵣ ∩ X ≠ ∅}</strong></p>
                <p>Các đối tượng có thể thuộc lớp X</p>
                <hr>
                <p><strong>Accuracy: α(X) = |R(X)| / |R̄(X)|</strong></p>
                <p>Độ chính xác của việc phân loại tập X</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="step-explanation">
                <h4>🔄 Các bước thực hiện:</h4>
                <p><strong>Bước 1:</strong> Xác định thuộc tính điều kiện và thuộc tính quyết định</p>
                <p><strong>Bước 2:</strong> Tạo các lớp tương đương dựa trên thuộc tính điều kiện</p>
                <p><strong>Bước 3:</strong> Tính Lower Approximation (các đối tượng chắc chắn)</p>
                <p><strong>Bước 4:</strong> Tính Upper Approximation (các đối tượng có thể)</p>
                <p><strong>Bước 5:</strong> Tính độ chính xác và mức độ phụ thuộc</p>
                <p><strong>Bước 6:</strong> Tìm các Reduct (tập thuộc tính tối thiểu)</p>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col2:
            st.markdown("""
            <div class="algorithm-card">
                <h4>📝 Thông tin thuật toán</h4>
                <p>• Xử lý uncertainty trong dữ liệu</p>
                <p>• Tìm attribute reducts</p>
                <p>• Lower/Upper approximation</p>
                <p>• Giảm chiều dữ liệu</p>
                <p>• Feature selection tự động</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col1:
            df_rs_input = df_original.copy()
            df_rs_input = df_rs_input.loc[:, ~df_rs_input.columns.str.startswith('Unnamed')]
            
            # Convert to string for categorical analysis
            for col in df_rs_input.columns:
                if pd.api.types.is_numeric_dtype(df_rs_input[col]) and df_rs_input[col].nunique() > 15:
                    st.warning(f"⚠️ Cột '{col}' có nhiều giá trị - nên rời rạc hóa")
                df_rs_input[col] = df_rs_input[col].astype(str)
            
            st.markdown("#### ⚙️ Chọn thuộc tính")
            col_a, col_b = st.columns(2)
            with col_a:
                decision_attr = st.selectbox(
                    "🎯 Thuộc tính quyết định:",
                    df_rs_input.columns.tolist(),
                    index=len(df_rs_input.columns)-1
                )
            
            available_attrs = [col for col in df_rs_input.columns if col != decision_attr]
            with col_b:
                condition_attrs = st.multiselect(
                    "📊 Thuộc tính điều kiện:",
                    available_attrs,
                    default=available_attrs[:2] if len(available_attrs) >= 2 else available_attrs
                )
            
            if decision_attr and condition_attrs:
                target_values = df_rs_input[decision_attr].unique().tolist()
                target_class = st.selectbox("🎯 Lớp mục tiêu:", target_values)
                
                if st.button("🚀 Phân tích Rough Set", type="primary", use_container_width=True):
                    with st.spinner("⏳ Đang tính toán..."):
                        try:
                            lower_approx = func.rs_lower_approximation(df_rs_input, target_class, condition_attrs, decision_attr)
                            upper_approx = func.rs_upper_approximation(df_rs_input, target_class, condition_attrs, decision_attr)
                            accuracy = func.rs_accuracy(df_rs_input, target_class, condition_attrs, decision_attr)
                            dependency = func.rs_dependency(df_rs_input, condition_attrs, decision_attr)
                            reducts = func.rs_find_all_reducts(df_rs_input, available_attrs, decision_attr)
                            
                            st.markdown("### 📊 Kết quả phân tích")
                            
                            # Metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("🎯 Lower Approx", len(lower_approx))
                            with col2:
                                st.metric("🔍 Upper Approx", len(upper_approx))
                            with col3:
                                st.metric("📊 Accuracy", f"{accuracy:.3f}")
                            with col4:
                                st.metric("🔗 Dependency", f"{dependency:.3f}")
                            
                            # Detailed results
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("#### 📋 Chi tiết kết quả")
                                st.info(f"🎯 **Lớp mục tiêu:** {target_class}")
                                st.info(f"📊 **Thuộc tính điều kiện:** {', '.join(condition_attrs)}")
                                st.success(f"✅ **Lower Approximation:** {len(lower_approx)} objects")
                                st.warning(f"⚠️ **Upper Approximation:** {len(upper_approx)} objects")
                            
                            with col2:
                                st.markdown("#### 🔧 Attribute Reducts")
                                if reducts:
                                    for i, reduct in enumerate(reducts, 1):
                                        st.success(f"**Reduct {i}:** {', '.join(reduct)}")
                                else:
                                    st.info("Không tìm thấy reduct nào")
                            
                            # Detailed calculation explanation
                            with st.expander("🧮 Giải thích chi tiết từng bước tính toán", expanded=False):
                                detailed_explanation = func.get_rough_set_detailed_explanation(
                                    df_rs_input, target_class, condition_attrs, decision_attr
                                )
                                st.markdown(detailed_explanation)
                                    
                        except Exception as e:
                            st.error(f"❌ Lỗi: {e}")
    
    # ==============================================================================
    # DECISION TREE TAB
    # ==============================================================================
    with tab3:
        st.markdown("### 🌳 Decision Tree - Cây quyết định ID3")
        
        # Detailed algorithm explanation
        with st.expander("📚 Giải thích chi tiết thuật toán Decision Tree ID3", expanded=False):
            st.markdown("""
            <div class="explanation-box">
                <h4>🎯 Mục đích thuật toán</h4>
                <p><strong>ID3 (Iterative Dichotomiser 3)</strong> được phát triển bởi Quinlan để xây dựng 
                cây quyết định từ dữ liệu huấn luyện, tạo ra các luật phân loại dễ hiểu.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="variable-explanation">
                <h4>📖 Giải thích ký hiệu:</h4>
                <p><strong>• S</strong> = Training set (tập huấn luyện): Dữ liệu để xây dựng cây</p>
                <p><strong>• A</strong> = Attribute (thuộc tính): Các đặc trưng đầu vào</p>
                <p><strong>• v₁, v₂, v₃...</strong> = Values (giá trị): Các giá trị có thể của thuộc tính</p>
                <p><strong>• E(S)</strong> = Entropy: Đo độ hỗn loạn/không đồng nhất của tập dữ liệu</p>
                <p><strong>• IG(S,A)</strong> = Information Gain: Lượng thông tin thu được khi chia theo A</p>
                <p><strong>• Root</strong> = Node gốc, <strong>Leaf</strong> = Node lá (kết quả phân loại)</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="formula-box">
                <h4>📐 Công thức tính toán:</h4>
                <p><strong>Entropy: E(S) = -Σᵢ pᵢ × log₂(pᵢ)</strong></p>
                <p>Trong đó: pᵢ = tỷ lệ samples thuộc lớp i trong tập S</p>
                <hr>
                <p><strong>Information Gain: IG(S,A) = E(S) - Σᵥ (|Sᵥ|/|S|) × E(Sᵥ)</strong></p>
                <p>Trong đó: Sᵥ = tập con của S có giá trị thuộc tính A = v</p>
                <hr>
                <p><strong>Gini Index: G(S) = 1 - Σᵢ pᵢ²</strong></p>
                <p>Phương pháp thay thế cho Entropy</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="step-explanation">
                <h4>🔄 Các bước thực hiện ID3:</h4>
                <p><strong>Bước 1:</strong> Tính Entropy của tập dữ liệu gốc S</p>
                <p><strong>Bước 2:</strong> Với mỗi thuộc tính A, tính Information Gain(S,A)</p>
                <p><strong>Bước 3:</strong> Chọn thuộc tính có IG cao nhất làm root node</p>
                <p><strong>Bước 4:</strong> Chia dữ liệu theo các giá trị của thuộc tính đã chọn</p>
                <p><strong>Bước 5:</strong> Đệ quy cho mỗi nhánh với tập con tương ứng</p>
                <p><strong>Bước 6:</strong> Dừng khi tất cả samples trong nhánh cùng lớp</p>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col2:
            st.markdown("""
            <div class="algorithm-card">
                <h4>📝 Thông tin thuật toán</h4>
                <p>• Phân loại dựa trên rules</p>
                <p>• Dễ hiểu và giải thích</p>
                <p>• Sử dụng Information Gain</p>
                <p>• Tạo luật IF-THEN</p>
                <p>• Xử lý dữ liệu categorical tốt</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col1:
            df_dt_input = df_original.copy()
            df_dt_input = df_dt_input.loc[:, ~df_dt_input.columns.str.startswith('Unnamed')]
            
            st.markdown("#### ⚙️ Cấu hình thuộc tính")
            col_a, col_b = st.columns(2)
            with col_a:
                target_attr = st.selectbox(
                    "🎯 Thuộc tính mục tiêu:",
                    df_dt_input.columns.tolist(),
                    index=len(df_dt_input.columns)-1
                )
            
            available_features = [col for col in df_dt_input.columns if col != target_attr]
            with col_b:
                method = st.radio("📊 Phương pháp:", ["Information Gain", "Gini Index"], horizontal=True)
                method_param = 'Gain' if 'Information' in method else 'Gini'
            
            selected_features = st.multiselect(
                "📋 Chọn thuộc tính đầu vào:",
                available_features,
                default=available_features
            )
            
            if target_attr and selected_features:
                if st.button("🚀 Xây dựng Decision Tree", type="primary", use_container_width=True):
                    with st.spinner("⏳ Đang xây dựng cây..."):
                        try:
                            # Prepare data
                            df_subset = df_dt_input[selected_features + [target_attr]].copy()
                            for col in df_subset.columns:
                                df_subset[col] = df_subset[col].astype(str)
                            # Remove rows with NaN values in any column
                            df_subset = df_subset.dropna()
                            # Remove rows with 'nan' string values (from str conversion)
                            df_subset = df_subset[~(df_subset == 'nan').any(axis=1)]
                            
                            if df_subset.empty:
                                st.error("❌ Dữ liệu trống sau khi xử lý. Vui lòng kiểm tra dữ liệu đầu vào.")
                            else:
                                st.markdown("### 📊 Kết quả phân tích")
                                
                                # Feature importance
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("#### 📈 Điểm số thuộc tính")
                                    for attr in selected_features:
                                        if method_param == 'Gain':
                                            score = func.dt_info_gain(df_subset, attr, target_attr)
                                        else:
                                            score = func.dt_gini_gain(df_subset, attr, target_attr)
                                        st.metric(f"{attr}", f"{score:.4f}")
                                
                                # Build tree
                                decision_tree = func.dt_build_tree(df_subset, selected_features, target_attr, method_param)
                                
                                with col2:
                                    st.markdown("#### 🌳 Visualization")
                                    try:
                                        graphviz_obj = func.dt_draw_tree_graphviz(decision_tree, target_attr)
                                        st.graphviz_chart(graphviz_obj)
                                    except Exception as e_graph:
                                        st.error(f"Lỗi vẽ cây: {e_graph}")
                                
                                # Extract rules
                                st.markdown("#### 📜 Các luật rút ra")
                                rules = func.dt_extract_rules(decision_tree, target_attr)
                                if rules:
                                    for i, rule in enumerate(rules, 1):
                                        st.success(f"**Luật {i}:** {rule}")
                                else:
                                    st.info("Không có luật nào được rút ra")
                                
                                # Detailed calculation explanation
                                with st.expander("🧮 Giải thích chi tiết từng bước tính toán", expanded=False):
                                    detailed_explanation = func.get_decision_tree_detailed_explanation(
                                        df_subset, selected_features, target_attr, method_param
                                    )
                                    st.markdown(detailed_explanation)
                                
                        except Exception as e:
                            import traceback
                            st.error(f"❌ Lỗi Decision Tree: {e}")
                            st.error(f"Chi tiết lỗi: {traceback.format_exc()}")
    
    # ==============================================================================
    # NAIVE BAYES TAB
    # ==============================================================================
    with tab4:
        st.markdown("### 🎯 Naive Bayes - Phân loại xác suất")
        
        # Detailed algorithm explanation
        with st.expander("📚 Giải thích chi tiết thuật toán Naive Bayes", expanded=False):
            st.markdown("""
            <div class="explanation-box">
                <h4>🎯 Mục đích thuật toán</h4>
                <p><strong>Naive Bayes</strong> là thuật toán phân loại dựa trên <strong>Định lý Bayes</strong> 
                với giả định "naive" rằng các features độc lập với nhau.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="variable-explanation">
                <h4>📖 Giải thích ký hiệu:</h4>
                <p><strong>• X = (x₁, x₂, ..., xₙ)</strong> = Feature vector: Vector đặc trưng đầu vào</p>
                <p><strong>• y</strong> = Class label: Nhãn lớp cần dự đoán</p>
                <p><strong>• P(y|X)</strong> = Posterior probability: Xác suất y xảy ra khi biết X</p>
                <p><strong>• P(X|y)</strong> = Likelihood: Xác suất X xảy ra khi biết y</p>
                <p><strong>• P(y)</strong> = Prior probability: Xác suất tiên nghiệm của lớp y</p>
                <p><strong>• P(X)</strong> = Evidence: Xác suất của X (hằng số normalizing)</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="formula-box">
                <h4>📐 Công thức tính toán:</h4>
                <p><strong>Bayes Theorem: P(y|X) = P(X|y) × P(y) / P(X)</strong></p>
                <p>Xác suất hậu nghiệm = Likelihood × Prior / Evidence</p>
                <hr>
                <p><strong>Naive assumption: P(X|y) = ∏ᵢ P(xᵢ|y)</strong></p>
                <p>Giả định features độc lập điều kiện</p>
                <hr>
                <p><strong>Classification: ŷ = argmax_y P(y) × ∏ᵢ P(xᵢ|y)</strong></p>
                <p>Chọn lớp có xác suất cao nhất</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="step-explanation">
                <h4>🔄 Các bước thực hiện:</h4>
                <p><strong>Bước 1:</strong> Tính Prior P(y) cho mỗi lớp từ tập huấn luyện</p>
                <p><strong>Bước 2:</strong> Tính Likelihood P(xᵢ|y) cho mỗi feature với mỗi lớp</p>
                <p><strong>Bước 3:</strong> Với sample mới X, tính P(y|X) cho mỗi lớp y</p>
                <p><strong>Bước 4:</strong> Chọn lớp có P(y|X) cao nhất làm kết quả</p>
                <p><strong>Smoothing:</strong> Áp dụng Laplace smoothing tránh P=0</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
                <h4>🔄 Các biến thể Naive Bayes:</h4>
                <p><strong>• Gaussian NB:</strong> Features liên tục, tuân theo phân phối chuẩn</p>
                <p><strong>• Multinomial NB:</strong> Features rời rạc, count data (text classification)</p>
                <p><strong>• Bernoulli NB:</strong> Features binary (0/1)</p>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col2:
            st.markdown("""
            <div class="algorithm-card">
                <h4>📝 Thông tin thuật toán</h4>
                <p>• Phân loại dựa trên xác suất</p>
                <p>• Nhanh và hiệu quả</p>
                <p>• Giả định độc lập features</p>
                <p>• Hoạt động tốt với dữ liệu ít</p>
                <p>• Ứng dụng: Text classification, Spam filter</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col1:
            df_nb_input = df_original.copy()
            df_nb_input = df_nb_input.loc[:, ~df_nb_input.columns.str.startswith('Unnamed')]
            
            st.markdown("#### ⚙️ Cấu hình mô hình")
            col_a, col_b = st.columns(2)
            with col_a:
                target_col = st.selectbox(
                    "🎯 Cột mục tiêu:",
                    df_nb_input.columns.tolist(),
                    index=len(df_nb_input.columns)-1
                )
            
            with col_b:
                nb_type = st.radio(
                    "📊 Loại Naive Bayes:",
                    ["GaussianNB", "MultinomialNB"],
                    horizontal=True
                )
            
            # Additional parameters
            alpha = 1.0
            if nb_type == "MultinomialNB":
                alpha = st.slider("🎚️ Alpha (Smoothing):", 0.01, 3.0, 1.0, 0.1)
            
            if target_col:
                if st.button("🚀 Huấn luyện Naive Bayes", type="primary", use_container_width=True):
                    with st.spinner("⏳ Đang huấn luyện..."):
                        try:
                            # Preprocess data
                            X_processed, y_processed, le_y, feature_encoders = func.general_preprocess_data_nbkm(
                                df_nb_input, target_col, st_instance=st
                            )
                            
                            if X_processed is not None and y_processed is not None:
                                # Handle negative values for MultinomialNB
                                X_final = X_processed.copy()
                                
                                if nb_type == "MultinomialNB" and (X_final < 0).any().any():
                                    st.warning("⚠️ MultinomialNB yêu cầu giá trị không âm. Áp dụng MinMaxScaler...")
                                    scaler = MinMaxScaler()
                                    X_final = pd.DataFrame(
                                        scaler.fit_transform(X_final), 
                                        columns=X_final.columns
                                    )
                                
                                method_internal = "gaussian" if nb_type == "GaussianNB" else "multinomial"
                                model, accuracy = func.train_naive_bayes_model(
                                    X_final, y_processed, method_internal, alpha, st_instance=st
                                )
                                
                                if model:
                                    st.markdown("### 📊 Kết quả huấn luyện")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("🎯 Loại mô hình", nb_type)
                                    with col2:
                                        st.metric("📊 Accuracy", f"{accuracy:.4f}")
                                    with col3:
                                        st.metric("🔢 Alpha", f"{alpha:.2f}" if nb_type == "MultinomialNB" else "N/A")
                                    
                                    st.success("✅ Mô hình đã được huấn luyện thành công!")
                                    
                                    # Show class distribution
                                    if le_y:
                                        st.markdown("#### 📈 Phân bố lớp")
                                        class_dist = pd.Series(y_processed).value_counts()
                                        class_names = le_y.classes_
                                        
                                        chart_data = pd.DataFrame({
                                            'Class': [class_names[i] for i in class_dist.index],
                                            'Count': class_dist.values
                                        })
                                        st.bar_chart(chart_data.set_index('Class'))
                                    
                                    # Detailed calculation explanation
                                    with st.expander("🧮 Giải thích chi tiết từng bước tính toán", expanded=False):
                                        detailed_explanation = func.get_naive_bayes_detailed_explanation(
                                            X_final, y_processed, method_internal
                                        )
                                        st.markdown(detailed_explanation)
                                    
                        except Exception as e:
                            st.error(f"❌ Lỗi: {e}")
    
    # ==============================================================================
    # K-MEANS TAB
    # ==============================================================================
    with tab5:
        st.markdown("### 👥 K-Means Clustering - Phân cụm")
        
        # Detailed algorithm explanation
        with st.expander("📚 Giải thích chi tiết thuật toán K-Means", expanded=False):
            st.markdown("""
            <div class="explanation-box">
                <h4>🎯 Mục đích thuật toán</h4>
                <p><strong>K-Means</strong> là thuật toán phân cụm unsupervised learning, chia dữ liệu thành 
                <strong>K cụm</strong> sao cho các điểm trong cùng cụm giống nhau nhất.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="variable-explanation">
                <h4>📖 Giải thích ký hiệu:</h4>
                <p><strong>• K</strong> = Số cụm: Tham số đầu vào, số cụm mong muốn</p>
                <p><strong>• X = {x₁, x₂, ..., xₙ}</strong> = Dataset: Tập dữ liệu đầu vào</p>
                <p><strong>• μₖ</strong> = Centroid: Tâm của cụm k (trọng tâm)</p>
                <p><strong>• Cₖ</strong> = Cluster k: Tập các điểm thuộc cụm k</p>
                <p><strong>• d(xᵢ, μₖ)</strong> = Distance: Khoảng cách từ điểm xᵢ đến centroid μₖ</p>
                <p><strong>• J</strong> = Cost function: Hàm mục tiêu cần tối thiểu hóa</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="formula-box">
                <h4>📐 Công thức tính toán:</h4>
                <p><strong>Euclidean Distance: d(xᵢ, μₖ) = √Σⱼ(xᵢⱼ - μₖⱼ)²</strong></p>
                <p>Khoảng cách Euclidean trong không gian nhiều chiều</p>
                <hr>
                <p><strong>Centroid Update: μₖ = (1/|Cₖ|) × Σ_{xᵢ∈Cₖ} xᵢ</strong></p>
                <p>Cập nhật centroid = trung bình của các điểm trong cụm</p>
                <hr>
                <p><strong>Objective Function: J = Σₖ Σ_{xᵢ∈Cₖ} ||xᵢ - μₖ||²</strong></p>
                <p>Tối thiểu hóa tổng bình phương khoảng cách trong cụm</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="step-explanation">
                <h4>🔄 Các bước thực hiện K-Means:</h4>
                <p><strong>Bước 1:</strong> Khởi tạo K centroids ngẫu nhiên μ₁, μ₂, ..., μₖ</p>
                <p><strong>Bước 2:</strong> Gán mỗi điểm xᵢ vào cụm gần nhất (theo khoảng cách)</p>
                <p><strong>Bước 3:</strong> Cập nhật centroids = trung bình các điểm trong cụm</p>
                <p><strong>Bước 4:</strong> Lặp bước 2-3 cho đến khi centroids không đổi</p>
                <p><strong>Điều kiện dừng:</strong> Centroids hội tụ hoặc đạt số iterations tối đa</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="warning-box">
                <h4>⚠️ Lưu ý quan trọng:</h4>
                <p><strong>• Chọn K:</strong> Cần biết trước số cụm hoặc dùng Elbow method</p>
                <p><strong>• Khởi tạo:</strong> Kết quả phụ thuộc vào centroid ban đầu</p>
                <p><strong>• Scaling:</strong> Nên chuẩn hóa dữ liệu trước khi clustering</p>
                <p><strong>• Giả định:</strong> Cụm có dạng hình cầu, kích thước tương đương</p>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col2:
            st.markdown("""
            <div class="algorithm-card">
                <h4>📝 Thông tin thuật toán</h4>
                <p>• Phân chia dữ liệu thành K cụm</p>
                <p>• Unsupervised learning</p>
                <p>• Dựa trên khoảng cách Euclidean</p>
                <p>• Ứng dụng: Segmentation, Data exploration</p>
                <p>• Tối ưu hóa centroids</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col1:
            df_km_input = df_original.copy()
            df_km_input = df_km_input.loc[:, ~df_km_input.columns.str.startswith('Unnamed')]
            
            st.markdown("#### ⚙️ Cấu hình clustering")
            col_a, col_b = st.columns(2)
            with col_a:
                k_clusters = st.slider("🎯 Số cụm (K):", 2, 10, 3, 1)
            with col_b:
                st.metric("📊 Số thuộc tính", df_km_input.select_dtypes(include=[np.number]).shape[1])
            
            if st.button("🚀 Chạy K-Means", type="primary", use_container_width=True):
                with st.spinner("⏳ Đang phân cụm..."):
                    try:
                        # Preprocess data
                        X_processed, _, _, _ = func.general_preprocess_data_nbkm(
                            df_km_input, None, st_instance=st
                        )
                        
                        if X_processed is not None:
                            labels, centers, df_clustered, X_scaled = func.run_kmeans_clustering_analysis(
                                X_processed, k_clusters
                            )
                            
                            if labels is not None:
                                st.markdown("### 📊 Kết quả phân cụm")
                                
                                # Metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("🎯 Số cụm", k_clusters)
                                with col2:
                                    st.metric("📊 Số điểm", len(labels))
                                with col3:
                                    unique_labels, counts = np.unique(labels, return_counts=True)
                                    st.metric("📈 Cụm lớn nhất", counts.max())
                                
                                # Show clustered data
                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    st.markdown("#### 📋 Dữ liệu đã phân cụm")
                                    st.dataframe(df_clustered.head(10), use_container_width=True)
                                
                                with col2:
                                    if X_processed.shape[1] >= 2:
                                        st.markdown("#### 📈 Visualization 2D")
                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                                                           c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
                                        ax.scatter(centers[:, 0], centers[:, 1], 
                                                 marker='X', s=200, color='red', label='Centroids')
                                        ax.set_xlabel(f"{X_processed.columns[0]} (normalized)")
                                        ax.set_ylabel(f"{X_processed.columns[1]} (normalized)")
                                        ax.set_title(f"K-Means Clustering (K={k_clusters})")
                                        ax.legend()
                                        plt.colorbar(scatter)
                                        st.pyplot(fig)
                                    else:
                                        st.info("Cần ít nhất 2 thuộc tính để visualization 2D")
                                
                                # Cluster analysis
                                st.markdown("#### 📊 Phân tích từng cụm")
                                for i in range(k_clusters):
                                    cluster_data = df_clustered[df_clustered['Cluster'] == i]
                                    with st.expander(f"Cụm {i} ({len(cluster_data)} điểm)"):
                                        st.dataframe(cluster_data.head())
                                
                                # Detailed calculation explanation
                                with st.expander("🧮 Giải thích chi tiết từng bước tính toán", expanded=False):
                                    detailed_explanation = func.get_kmeans_detailed_explanation(
                                        X_processed, k_clusters, random_state=42
                                    )
                                    st.markdown(detailed_explanation)
                                        
                    except Exception as e:
                        st.error(f"❌ Lỗi: {e}")
    
    # ==============================================================================
    # SOM TAB
    # ==============================================================================
    with tab6:
        st.markdown("### 🗺️ Kohonen SOM - Bản đồ tự tổ chức")
        
        # Detailed algorithm explanation
        with st.expander("📚 Giải thích chi tiết thuật toán Kohonen SOM", expanded=False):
            st.markdown("""
            <div class="explanation-box">
                <h4>🎯 Mục đích thuật toán</h4>
                <p><strong>Self-Organizing Map (SOM)</strong> được phát triển bởi Kohonen, là mạng neural 
                không giám sát để <strong>ánh xạ dữ liệu nhiều chiều xuống 2D</strong> và phát hiện patterns.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="variable-explanation">
                <h4>📖 Giải thích ký hiệu:</h4>
                <p><strong>• X = (x₁, x₂, ..., xₙ)</strong> = Input vector: Vector đầu vào n chiều</p>
                <p><strong>• W_ij = (w₁, w₂, ..., wₙ)</strong> = Weight vector: Vector trọng số của neuron (i,j)</p>
                <p><strong>• BMU</strong> = Best Matching Unit: Neuron có trọng số gần nhất với input</p>
                <p><strong>• σ(t)</strong> = Neighborhood radius: Bán kính láng giềng tại thời điểm t</p>
                <p><strong>• α(t)</strong> = Learning rate: Tốc độ học tại thời điểm t</p>
                <p><strong>• h_ij(t)</strong> = Neighborhood function: Hàm láng giềng</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="formula-box">
                <h4>📐 Công thức tính toán:</h4>
                <p><strong>BMU: c = argmin_i ||X - W_i||</strong></p>
                <p>Tìm neuron có trọng số gần nhất với input X</p>
                <hr>
                <p><strong>Distance: d_ij = ||r_i - r_j||</strong></p>
                <p>Khoảng cách vị trí giữa neuron i và j trên lưới</p>
                <hr>
                <p><strong>Neighborhood: h_ij(t) = exp(-d_ij² / 2σ(t)²)</strong></p>
                <p>Hàm Gaussian xác định ảnh hưởng láng giềng</p>
                <hr>
                <p><strong>Weight Update: W_i(t+1) = W_i(t) + α(t) × h_ci(t) × (X - W_i(t))</strong></p>
                <p>Cập nhật trọng số theo hướng input</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="step-explanation">
                <h4>🔄 Các bước thực hiện SOM:</h4>
                <p><strong>Bước 1:</strong> Khởi tạo lưới neurons với trọng số ngẫu nhiên</p>
                <p><strong>Bước 2:</strong> Chọn ngẫu nhiên input vector X từ dataset</p>
                <p><strong>Bước 3:</strong> Tìm BMU - neuron có trọng số gần X nhất</p>
                <p><strong>Bước 4:</strong> Cập nhật trọng số BMU và các neuron láng giềng</p>
                <p><strong>Bước 5:</strong> Giảm learning rate α(t) và neighborhood radius σ(t)</p>
                <p><strong>Bước 6:</strong> Lặp lại cho đến hội tụ hoặc đạt max iterations</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
                <h4>🎨 Ứng dụng SOM:</h4>
                <p><strong>• Data Visualization:</strong> Hiển thị dữ liệu nhiều chiều trên 2D map</p>
                <p><strong>• Clustering:</strong> Phát hiện các nhóm dữ liệu tương tự</p>
                <p><strong>• Feature Detection:</strong> Tìm patterns ẩn trong dữ liệu</p>
                <p><strong>• Dimensionality Reduction:</strong> Giảm chiều dữ liệu</p>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col2:
            st.markdown("""
            <div class="algorithm-card">
                <h4>📝 Thông tin thuật toán</h4>
                <p>• Neural network không giám sát</p>
                <p>• Giảm chiều dữ liệu</p>
                <p>• Visualization patterns</p>
                <p>• Topology preserving mapping</p>
                <p>• Competitive learning</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col1:
            df_som_input = df_original.copy()
            df_som_input = df_som_input.loc[:, ~df_som_input.columns.str.startswith('Unnamed')]
            
            st.markdown("#### ⚙️ Cấu hình SOM")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                som_rows = st.number_input("📏 Số hàng lưới:", 3, 10, 4, 1)
                som_cols = st.number_input("📐 Số cột lưới:", 3, 10, 4, 1)
            with col_b:
                sigma = st.slider("🎚️ Sigma:", 0.1, 3.0, 1.0, 0.1)
                learning_rate = st.slider("📈 Learning Rate:", 0.01, 1.0, 0.5, 0.01)
            with col_c:
                iterations = st.number_input("🔄 Iterations:", 100, 2000, 500, 100)
                num_classes = st.number_input("🏷️ Số lớp hiển thị:", 1, 5, 3, 1)
            
            if st.button("🚀 Huấn luyện SOM", type="primary", use_container_width=True):
                with st.spinner("⏳ Đang huấn luyện SOM..."):
                    try:
                        # Preprocess data
                        X_processed, _, _, _ = func.general_preprocess_data_nbkm(
                            df_som_input, None, st_instance=st
                        )
                        
                        if X_processed is not None:
                            som_model, X_scaled = func.train_kohonen_som_model(
                                X_processed, som_rows, som_cols, sigma, learning_rate, iterations, st_instance=st
                            )
                            
                            if som_model and X_scaled is not None:
                                st.markdown("### 📊 Kết quả SOM")
                                
                                # Metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("🗺️ Kích thước lưới", f"{som_rows}x{som_cols}")
                                with col2:
                                    st.metric("📊 Sigma", f"{sigma:.2f}")
                                with col3:
                                    st.metric("📈 Learning Rate", f"{learning_rate:.2f}")
                                with col4:
                                    st.metric("🔄 Iterations", iterations)
                                
                                # Visualization
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("#### 🔥 Hit Map")
                                    activation_response = som_model.activation_response(X_scaled).T
                                    fig_hit, ax_hit = plt.subplots(figsize=(som_cols*0.8, som_rows*0.8))
                                    im = ax_hit.pcolor(activation_response, cmap='viridis')
                                    ax_hit.set_title('SOM Hit Map')
                                    plt.colorbar(im, ax=ax_hit)
                                    st.pyplot(fig_hit)
                                
                                with col2:
                                    st.markdown("#### 🎨 Component Maps")
                                    # Random class visualization
                                    random_labels = np.random.randint(0, num_classes, size=X_scaled.shape[0])
                                    
                                    n_cols = min(num_classes, 2)
                                    n_rows = math.ceil(num_classes / n_cols)
                                    fig_comp, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
                                    
                                    if num_classes == 1:
                                        axes = [axes]
                                    else:
                                        axes = axes.flatten()
                                    
                                    for i in range(num_classes):
                                        if i < len(axes):
                                            class_data = X_scaled[random_labels == i]
                                            if len(class_data) > 0:
                                                win_map = som_model.win_map(class_data)
                                                heatmap = np.zeros((som_rows, som_cols))
                                                for pos, neurons in win_map.items():
                                                    heatmap[pos[0], pos[1]] = len(neurons)
                                                
                                                sns.heatmap(heatmap.T, ax=axes[i], cmap="coolwarm", 
                                                          annot=True, fmt=".0f", cbar=False)
                                                axes[i].set_title(f"Class {i}")
                                            else:
                                                axes[i].set_title(f"Class {i} (Empty)")
                                                axes[i].axis('off')
                                        else:
                                            axes[i].axis('off')
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig_comp)
                                
                                st.success("✅ SOM đã được huấn luyện thành công!")
                                
                                # Detailed calculation explanation
                                with st.expander("🧮 Giải thích chi tiết từng bước tính toán", expanded=False):
                                    detailed_explanation = func.get_som_detailed_explanation(
                                        X_processed, grid_size=(som_rows, som_cols), 
                                        learning_rate=learning_rate, epochs=iterations
                                    )
                                    st.markdown(detailed_explanation)
                                
                    except Exception as e:
                        st.error(f"❌ Lỗi: {e}")

else:
    # Welcome screen when no data is loaded
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h2>🚀 Chào mừng đến với Data Mining Toolkit!</h2>
        <p style="font-size: 18px;">
            Tải lên file CSV hoặc chọn dữ liệu mẫu từ sidebar để bắt đầu
        </p>
        <div style="margin: 30px 0;">
            <h3>📚 6 thuật toán được hỗ trợ:</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Algorithm overview cards using CSS classes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box" style="border-left-color: #007bff;">
            <h4>🛒 Apriori</h4>
            <p>Tìm luật kết hợp trong dữ liệu giao dịch</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box" style="border-left-color: #dc3545;">
            <h4>🎯 Naive Bayes</h4>
            <p>Phân loại dựa trên xác suất</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box" style="border-left-color: #28a745;">
            <h4>⚖️ Rough Set</h4>
            <p>Xử lý dữ liệu không chắc chắn</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box" style="border-left-color: #6f42c1;">
            <h4>👥 K-Means</h4>
            <p>Phân cụm dữ liệu không giám sát</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box" style="border-left-color: #ffc107;">
            <h4>🌳 Decision Tree</h4>
            <p>Xây dựng cây quyết định ID3</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box" style="border-left-color: #20c997;">
            <h4>🗺️ Kohonen SOM</h4>
            <p>Bản đồ tự tổ chức neural network</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🎓 <strong>Data Mining Toolkit</strong> - Đồ án cuối kỳ Khai thác Dữ liệu</p>
    <p>Được phát triển bởi: Trần Nhật Khánh & Trần Nhật Huy</p>
</div>
""", unsafe_allow_html=True) 