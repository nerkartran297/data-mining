import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math # Cần cho math.ceil trong SOM plot layout
from sklearn.preprocessing import MinMaxScaler

# Import các hàm từ func.py
import func 

# Cấu hình trang Streamlit
st.set_page_config(layout="wide", page_title="Đồ án cuối kỳ: Khai thác Dữ liệu")


# ==============================================================================
# GIAO DIỆN STREAMLIT CHÍNH
# ==============================================================================
st.title("Đồ án cuối kỳ: Khai thác Dữ liệu")
st.markdown("""
**Nhóm sinh viên thực hiện:**
1. Trần Nhật Khánh
2. Trần Nhật Huy
""")

# --- Tải dữ liệu ---
st.sidebar.header("📁 1. Tải Dữ Liệu Lên")
uploaded_file = st.sidebar.file_uploader("Chọn một tệp CSV", type=["csv"])

df_original = None
if uploaded_file is not None:
    try:
        df_original = pd.read_csv(uploaded_file)
        st.subheader("📄 Dữ liệu gốc (5 dòng đầu)")
        st.dataframe(df_original.head())
    except Exception as e:
        st.error(f"Lỗi khi đọc tệp CSV: {e}")
        df_original = None

# --- Chọn thuật toán ---
st.sidebar.header("⚙️ 2. Chọn Thuật Toán Phân Tích")
analysis_options = [
    "--- Chọn một thuật toán ---",
    "Apriori (Luật Kết Hợp)",
    "Rough Set (Lý Thuyết Tập Thô)",
    "Decision Tree (ID3)",
    "Naive Bayes",
    "K-Means Clustering",
    "Kohonen SOM"
]
selected_analysis = st.sidebar.selectbox("Chọn một phương pháp:", analysis_options, key="main_analysis_choice")

if df_original is not None and selected_analysis != analysis_options[0]:
    st.markdown("---")
    st.header(f" Phân tích bằng: {selected_analysis.split('(')[0].strip()}")

    # APRIORI
    if selected_analysis == analysis_options[1]:
        
        df_apriori_input = df_original.copy()
        df_apriori_input = df_apriori_input.loc[:, ~df_apriori_input.columns.str.startswith('Unnamed')]
        st.subheader("Tiền xử lý dữ liệu cho Apriori")
        preprocess_method_apriori = st.radio(
            "Chọn phương pháp tiền xử lý:",
            ("One-hot encode các cột được chọn", 
             "Sử dụng các cột số gốc", 
             "Chuyển đổi toàn bộ dữ liệu sang dạng boolean"),
            index=0, 
            key="apriori_preprocess_method"
        )

        df_apriori_processed = None
        if preprocess_method_apriori == "One-hot encode các cột được chọn":
            st.info("Phương pháp này sẽ one-hot encode các cột bạn chọn và sử dụng chúng cho Apriori.")
            apriori_cols_to_encode = st.multiselect(
                "Chọn các cột (categorical) để one-hot encode:",
                options=df_apriori_input.columns.tolist(),
                key="apriori_cols_ohe"
            )
            if apriori_cols_to_encode:
                 df_apriori_processed = func.apriori_one_hot_encode_data(df_apriori_input, apriori_cols_to_encode)
            else:
                st.info("Vui lòng chọn ít nhất một cột để one-hot encode, hoặc chọn phương pháp khác.")
        elif preprocess_method_apriori == "Sử dụng các cột số gốc (cảnh báo!)":
            st.warning("Đang sử dụng `process_data` gốc cho Apriori: chỉ chọn các cột số. Điều này có thể không phù hợp trừ khi các cột số của bạn đã là dạng 0/1 (giao dịch).")
            df_apriori_processed = func.apriori_original_process_data(df_apriori_input)
        else:
            st.info("Đang chuyển đổi dữ liệu sang dạng boolean cho Apriori (True nếu giá trị tồn tại và khác 0/False/chuỗi rỗng).")
            df_apriori_processed = func.apriori_general_transactional_conversion(df_apriori_input)

        if df_apriori_processed is not None:
            st.write("Dữ liệu đã xử lý cho Apriori (5 dòng đầu):")
            st.dataframe(df_apriori_processed.head())

            st.subheader("Thiết lập tham số Apriori")
            min_sup_apriori = st.slider("Ngưỡng Hỗ Trợ (Min Support)", 0.01, 1.0, 0.5, 0.01, key="apriori_minsup_slider")
            min_conf_apriori = st.slider("Ngưỡng Tin Cậy (Min Confidence)", 0.01, 1.0, 0.3, 0.01, key="apriori_minconf_slider")

            if st.button("🚀 Chạy Phân Tích Apriori", key="apriori_run_button"):
                st.subheader("Kết quả Phân Tích Apriori")
                try:
                    with st.spinner("Đang chạy Apriori..."):
                        f_itemsets, rules, max_f_itemsets = func.run_apriori_calculations(df_apriori_processed, min_sup_apriori, min_conf_apriori)
                    
                    st.markdown("##### Tập phổ biến:")
                    if not f_itemsets.empty:
                        f_itemsets_display = f_itemsets.copy()
                        f_itemsets_display['itemsets'] = f_itemsets_display['itemsets'].apply(lambda x: tuple(x))
                        st.dataframe(f_itemsets_display[['support', 'itemsets']].reset_index(drop=True))
                    else:
                        st.info("Không tìm thấy tập phổ biến nào.")

                    st.markdown("##### Tập phổ biến tối đại:")
                    if not max_f_itemsets.empty:
                        max_f_itemsets_display = max_f_itemsets.copy()
                        max_f_itemsets_display['itemsets'] = max_f_itemsets_display['itemsets'].apply(lambda x: tuple(x))
                        st.dataframe(max_f_itemsets_display[['support', 'itemsets']].reset_index(drop=True))
                    else:
                        st.info("Không tìm thấy tập phổ biến tối đại nào.")

                    st.markdown("##### Các luật kết hợp:")
                    if not rules.empty:
                        rules_display = rules.copy()
                        rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: tuple(x))
                        rules_display['consequents'] = rules_display['consequents'].apply(lambda x: tuple(x))

                        st.dataframe(rules_display[['antecedents', 'consequents', 'support', 'confidence']].reset_index(drop=True))

                    else:
                        st.info("Không tìm thấy luật kết hợp nào.")

                except ValueError as ve: 
                    st.error(f"Lỗi Apriori: {ve}")
                except Exception as e:
                    st.error(f"Lỗi không xác định khi chạy Apriori: {e}")
                
                st.markdown("---")

        else:
            st.info("Dữ liệu chưa được xử lý cho Apriori. Vui lòng hoàn tất Bước 1.")

    # ROUGH SET

    elif selected_analysis == analysis_options[2]:

        df_rs_input = df_original.copy()
        df_rs_input = df_rs_input.loc[:, ~df_rs_input.columns.str.startswith('Unnamed')]
        st.dataframe(df_rs_input)
        for col in df_rs_input.columns:
            if pd.api.types.is_numeric_dtype(df_rs_input[col]) and df_rs_input[col].nunique() > 15:
                st.sidebar.warning(f"Cột số '{col}' có nhiều giá trị duy nhất. Cân nhắc rời rạc hóa cho Rough Set.")
            df_rs_input[col] = df_rs_input[col].astype(str) 

        st.subheader("Chọn Thuộc Tính cho Rough Set")
        all_cols_rs = df_rs_input.columns.tolist()
        decision_attribute_rs_name = st.selectbox("Chọn Thuộc Tính Quyết Định:", options=all_cols_rs, index=len(all_cols_rs)-1 if all_cols_rs else 0, key="rs_decision_attr_select")
        available_condition_attributes_rs_names = [col for col in all_cols_rs if col != decision_attribute_rs_name]
        
        if not available_condition_attributes_rs_names:
            st.warning("Không còn thuộc tính điều kiện.")
        else:
            selected_condition_attributes_rs_names = st.multiselect(
                "Chọn các Thuộc Tính Điều Kiện (cho xấp xỉ/phụ thuộc ban đầu):",
                options=available_condition_attributes_rs_names,
                default=available_condition_attributes_rs_names[:min(2, len(available_condition_attributes_rs_names))],
                key="rs_condition_attrs_select"
            )
            st.subheader("Chọn Lớp Mục Tiêu")
            if decision_attribute_rs_name and df_rs_input[decision_attribute_rs_name].nunique() > 0:
                target_class_values_rs_options = df_rs_input[decision_attribute_rs_name].unique().tolist()
                selected_target_class_value_rs = st.selectbox(f"Chọn Lớp Mục Tiêu (của '{decision_attribute_rs_name}'):", options=target_class_values_rs_options, key="rs_target_class_select")

                if st.button(" Chạy Phân Tích Rough Set", key="rs_run_button"):
                    st.subheader("Kết quả Phân Tích Rough Set")
                    if not selected_condition_attributes_rs_names:
                        st.warning("Vui lòng chọn ít nhất một thuộc tính điều kiện.")
                    else:
                        with st.spinner("Đang tính toán Rough Set..."):
                            lower_approx_rs = func.rs_lower_approximation(df_rs_input, selected_target_class_value_rs, selected_condition_attributes_rs_names, decision_attribute_rs_name)
                            upper_approx_rs = func.rs_upper_approximation(df_rs_input, selected_target_class_value_rs, selected_condition_attributes_rs_names, decision_attribute_rs_name)
                            accuracy_rs_val = func.rs_accuracy(df_rs_input, selected_target_class_value_rs, selected_condition_attributes_rs_names, decision_attribute_rs_name)
                            dependency_rs_val = func.rs_dependency(df_rs_input, selected_condition_attributes_rs_names, decision_attribute_rs_name)
                            all_possible_cond_attrs_for_reduct = available_condition_attributes_rs_names
                            reducts_rs_list = func.rs_find_all_reducts(df_rs_input, all_possible_cond_attrs_for_reduct, decision_attribute_rs_name)

                        st.write(f"**Lớp mục tiêu:** `{selected_target_class_value_rs}` (từ `{decision_attribute_rs_name}`)")
                        st.write(f"**Các thuộc tính điều kiện đang xét:** `{', '.join(selected_condition_attributes_rs_names)}`")
                        st.write(f"**Xấp xỉ dưới:** `{len(lower_approx_rs)}` đối tượng. (Chỉ số: `{sorted(list(lower_approx_rs)) if lower_approx_rs else 'Rỗng'}`)")
                        st.write(f"**Xấp xỉ trên:** `{len(upper_approx_rs)}` đối tượng. (Chỉ số: `{sorted(list(upper_approx_rs)) if upper_approx_rs else 'Rỗng'}`)")
                        st.write(f"**Độ chính xác:** `{accuracy_rs_val:.4f}`")
                        st.write(f"**Mức độ phụ thuộc:** `{dependency_rs_val:.4f}`")
                        st.subheader("Các Rút Gọn Tối Thiểu")
                        if reducts_rs_list:
                            for r_idx, r_item in enumerate(reducts_rs_list): st.write(f"- Rút gọn {r_idx+1}: `{r_item}`")
                        else: st.info("Không tìm thấy rút gọn nào.")
                        st.markdown("---")

            else:
                st.info(f"Thuộc tính quyết định '{decision_attribute_rs_name}' cần giá trị để phân tích.")

    # DECISION TREE (ID3)
    elif selected_analysis == analysis_options[3]:

        df_dt_input = df_original.copy()
        df_dt_input = df_dt_input.loc[:, ~df_dt_input.columns.str.startswith('Unnamed')]
        st.subheader("Chọn Thuộc Tính cho Cây Quyết Định")
        all_cols_dt = df_dt_input.columns.tolist()
        
        target_attribute_dt_global = st.selectbox("Chọn Thuộc Tính Mục Tiêu (Lớp):", options=all_cols_dt, index=len(all_cols_dt)-1 if all_cols_dt else 0, key="dt_target_attr_select")
        available_features_dt_names = [col for col in all_cols_dt if col != target_attribute_dt_global]
        
        if not available_features_dt_names:
            st.warning("Không còn thuộc tính đầu vào.")
        else:
            selected_feature_attributes_dt_names = st.multiselect("Chọn các Thuộc Tính Đầu Vào:", options=available_features_dt_names, default=available_features_dt_names, key="dt_feature_attrs_select")
            st.subheader("Chọn Phương Pháp Chia Nhánh")
            method_dt_choice = st.radio("Chọn phương pháp:", ('Gain (Entropy)', 'Gini Gain'), key="dt_method_radio")
            method_param_dt = 'Gain' if 'Gain (Entropy)' in method_dt_choice else 'Gini'

            if st.button(" Xây Dựng Cây Quyết Định", key="dt_run_button"):
                st.subheader("Kết quả Cây Quyết Định")
                if not selected_feature_attributes_dt_names:
                    st.warning("Vui lòng chọn ít nhất một thuộc tính đầu vào.")
                elif target_attribute_dt_global:
                    with st.spinner("Đang xây dựng cây..."):
                        try:
                            df_subset_for_dt = df_dt_input[selected_feature_attributes_dt_names + [target_attribute_dt_global]].copy()
                            st.write("Chuyển đổi các cột đã chọn sang dạng chuỗi cho ID3:")
                            for col in df_subset_for_dt.columns:
                                df_subset_for_dt[col] = df_subset_for_dt[col].astype(str)
                                df_subset_for_dt = df_subset_for_dt[~df_subset_for_dt['Outlook'].isin(['nan'])]
                                st.caption(f"- Cột '{col}' đã sang chuỗi.")
                            
                            st.markdown("##### Điểm thuộc tính (bước đầu):")
                            for attr in selected_feature_attributes_dt_names:
                                score = func.dt_info_gain(df_subset_for_dt, attr, target_attribute_dt_global) if method_param_dt == 'Gain' else func.dt_gini_gain(df_subset_for_dt, attr, target_attribute_dt_global)
                                st.write(f"- **{attr}**: {method_param_dt} = `{score:.4f}`")
                            
                            decision_tree_model = func.dt_build_tree(df_subset_for_dt, selected_feature_attributes_dt_names, target_attribute_dt_global, method_param_dt)
                            st.success("✅ Cây đã được xây dựng!")
                            st.markdown("##### Hình ảnh Cây:")
                            try:
                                graphviz_dot_obj = func.dt_draw_tree_graphviz(decision_tree_model, target_attribute_dt_global) 
                                st.graphviz_chart(graphviz_dot_obj)
                            except Exception as e_graph: st.error(f"Lỗi vẽ cây: {e_graph}.")
                            st.markdown("##### Các Luật Rút Ra:")
                            extracted_rules_dt_list = func.dt_extract_rules(decision_tree_model, target_attribute_dt_global) 
                            if extracted_rules_dt_list:
                                for rule_item_str in extracted_rules_dt_list: st.markdown(f"- {rule_item_str}")
                            else: st.info("Không có luật nào.")
                        except Exception as e_dt_build: st.error(f"Lỗi xây dựng cây: {e_dt_build}")
                    st.markdown("---")
                else:
                    st.warning("Vui lòng chọn thuộc tính mục tiêu.")
    
    # NAIVE BAYES, K-MEANS, KOHONEN SOM (Từ script thứ 4)
    df_nbkm_input = df_original.copy()
    df_nbkm_input = df_nbkm_input.loc[:, ~df_nbkm_input.columns.str.startswith('Unnamed')]
    X_processed_nbkm, y_processed_for_nb_only, le_y_for_nb_only, feature_encoders_nbkm = None, None, None, None
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Tùy chọn cho Naive Bayes / K-Means / SOM")
    target_column_for_nb_som = st.sidebar.selectbox("Chọn Cột Mục Tiêu (cho Naive Bayes, tùy chọn cho SOM):", options=[None] + df_nbkm_input.columns.tolist(), index=0, key="nbkm_target_col_select")

    try:
        X_processed_nbkm, y_processed_for_nb_only, le_y_for_nb_only, feature_encoders_nbkm = \
            func.general_preprocess_data_nbkm(df_nbkm_input, target_column_for_nb_som, st_instance=st)

        if X_processed_nbkm is None and selected_analysis != analysis_options[0] and selected_analysis != analysis_options[1] and selected_analysis != analysis_options[2] and selected_analysis != analysis_options[3]:
            st.error("Lỗi tiền xử lý dữ liệu chung. Không thể tiếp tục.")
        else:
            if selected_analysis != analysis_options[0] and selected_analysis != analysis_options[1] and selected_analysis != analysis_options[2] and selected_analysis != analysis_options[3]: 
                st.markdown("#### Dữ liệu thuộc tính đã xử lý (X - 5 dòng đầu):")
                st.dataframe(X_processed_nbkm.head())
                if y_processed_for_nb_only is not None and target_column_for_nb_som:
                    st.markdown(f"#### Nhãn mục tiêu '{target_column_for_nb_som}' đã xử lý (y - 5 giá trị đầu):")
                    st.write(y_processed_for_nb_only[:5])
                    if le_y_for_nb_only: st.caption(f"Các lớp gốc: {list(le_y_for_nb_only.classes_)}")
            
            # --- NAIVE BAYES ---
            if selected_analysis == analysis_options[4]:

                if y_processed_for_nb_only is None or target_column_for_nb_som is None:
                    st.warning("⚠️ Vui lòng chọn cột mục tiêu ở thanh bên cho Naive Bayes.")
                else:
                    st.subheader("Chọn loại Naive Bayes")
                    nb_method_type_choice = st.radio("Chọn phương pháp:", ("GaussianNB", "MultinomialNB"), key="nb_method_radio_choice")
                    method_map = {"GaussianNB": "gaussian", "MultinomialNB": "multinomial"}
                    nb_method_internal = method_map[nb_method_type_choice]

                    alpha_nb = 1.0
                    X_nb_final = X_processed_nbkm.copy() 
                    can_train_multinomial = True

                    if nb_method_type_choice == "MultinomialNB":
                        alpha_nb = st.slider("Alpha (Laplace smoothing):", 0.01, 3.0, 0.5, 0.1, key="nb_alpha_slider")
                        if (X_nb_final < 0).any().any():
                            st.warning("MultinomialNB yêu cầu đặc trưng không âm.")
                            if st.checkbox("Áp dụng MinMaxScaler?", key="nb_minmax_multi_check"):
                                scaler_nb_multi = MinMaxScaler()
                                X_nb_final = scaler_nb_multi.fit_transform(X_nb_final)
                                st.info("Đã áp dụng MinMaxScaler.")
                            else:
                                st.error("Không thể huấn luyện MultinomialNB với giá trị âm.")
                                can_train_multinomial = False
                    
                    if st.button(" Huấn Luyện Naive Bayes", key="nb_run_button"):
                        st.subheader("Kết quả Naive Bayes")
                        if nb_method_type_choice == "MultinomialNB" and not can_train_multinomial:
                            st.error("Huấn luyện MultinomialNB bị hủy do có giá trị âm và không áp dụng scaling.")
                        else:
                            with st.spinner(f"Đang huấn luyện {nb_method_type_choice}..."):
                                try:
                                    model_nb_trained, acc_nb_trained = func.train_naive_bayes_model(
                                        pd.DataFrame(X_nb_final, columns=X_processed_nbkm.columns) if isinstance(X_nb_final, np.ndarray) else X_nb_final, 
                                        y_processed_for_nb_only, 
                                        nb_method_internal, 
                                        alpha_nb,
                                        st_instance=st 
                                    )
                                    if model_nb_trained:
                                        st.success(f"✅ {nb_method_type_choice} đã huấn luyện.")
                                        st.metric(label=f"Độ chính xác (trên tập huấn luyện)", value=f"{acc_nb_trained:.4f}")
                                    else: st.error(f"Không thể huấn luyện {nb_method_type_choice}.")
                                except ValueError as ve_nb_train: 
                                    st.error(f"Lỗi huấn luyện Naive Bayes: {ve_nb_train}")
                                except Exception as e_nb_train:
                                    st.error(f"Lỗi không xác định khi huấn luyện: {e_nb_train}")
                        st.markdown("---")
            
            # --- K-MEANS CLUSTERING ---
            elif selected_analysis == analysis_options[5]:
                st.subheader("Chọn Số Cụm (K)")
                num_k_clusters_choice = st.slider("Chọn số cụm (K):", 2, 10, 3, 1, key="kmeans_k_slider")
                if st.button(" Chạy K-Means", key="kmeans_run_button"):
                    st.subheader("Kết quả K-Means")
                    with st.spinner(f"Đang chạy K-Means (K={num_k_clusters_choice})..."):
                        try:
                            labels_km_res, centers_km_res, df_clustered_km_res, X_scaled_for_kmeans_plot = \
                                func.run_kmeans_clustering_analysis(X_processed_nbkm, num_k_clusters_choice)
                            
                            if labels_km_res is not None:
                                st.write(f"Kết quả gán nhãn cụm:")
                                st.dataframe(df_clustered_km_res.head())
                                if X_processed_nbkm.shape[1] >= 2 and X_scaled_for_kmeans_plot is not None:
                                    fig_km, ax_km = plt.subplots(figsize=(8, 6))
                                    ax_km.scatter(X_scaled_for_kmeans_plot[:, 0], X_scaled_for_kmeans_plot[:, 1], c=labels_km_res, cmap='viridis', alpha=0.7, edgecolors='k')
                                    ax_km.scatter(centers_km_res[:, 0], centers_km_res[:, 1], marker='X', s=100, color='red', label='Tâm cụm')
                                    ax_km.set_xlabel(f"{X_processed_nbkm.columns[0]} (chuẩn hóa)")
                                    ax_km.set_ylabel(f"{X_processed_nbkm.columns[1]} (chuẩn hóa)")
                                    ax_km.set_title(f"K-Means (K={num_k_clusters_choice})")
                                    ax_km.legend()
                                    st.pyplot(fig_km)
                                else: st.info("Cần ít nhất 2 thuộc tính để vẽ 2D.")
                            else: st.error("Lỗi K-Means.")
                        except ValueError as ve_km: st.error(f"Lỗi K-Means: {ve_km}")
                        except Exception as e_km: st.error(f"Lỗi không xác định K-Means: {e_km}")
                    st.markdown("---")

            # --- KOHONEN SOM ---
            elif selected_analysis == analysis_options[6]:
                st.subheader("Thiết lập Tham số SOM")
                som_grid_r = st.number_input("Số hàng lưới SOM:", 3, 25, 4, 1, key="som_r")
                som_grid_c = st.number_input("Số cột lưới SOM:", 3, 25, 4, 1, key="som_c")
                som_sig = st.slider("Sigma:", 0.1, 5.0, 1.0, 0.1, key="som_sig")
                som_lrn = st.slider("Learning Rate:", 0.01, 1.0, 0.5, 0.01, key="som_lrn")
                som_iter = st.number_input("Số vòng lặp:", 100, 10000, 500, 100, key="som_iter")
                num_som_rand_cls = st.number_input("Số 'lớp gán ngẫu nhiên' (script gốc):", 1, 10, 3, 1, key="som_rand_cls")

                if st.button(" Huấn Luyện SOM", key="som_run_button"):
                    st.subheader("Kết quả SOM")
                    with st.spinner("Đang huấn luyện SOM..."):
                        try:
                            som_model, X_scaled_som = func.train_kohonen_som_model(
                                X_processed_nbkm, som_grid_r, som_grid_c, som_sig, som_lrn, som_iter, st_instance=st
                            )
                            if som_model and X_scaled_som is not None:
                                st.markdown("### Hit Map (SOM)")
                                act_resp_som = som_model.activation_response(X_scaled_som).T
                                fig_som_h, ax_som_h = plt.subplots(figsize=(som_grid_c*0.5, som_grid_r*0.5))
                                im_h = ax_som_h.pcolor(act_resp_som, cmap='viridis')
                                ax_som_h.set_title('SOM - Hit Map')
                                fig_som_h.colorbar(im_h, ax=ax_som_h)
                                st.pyplot(fig_som_h)

                                st.markdown(f"### Bản đồ theo {num_som_rand_cls} 'Lớp Gán'")
                                som_rand_lbls = np.random.randint(0, num_som_rand_cls, size=X_scaled_som.shape[0])
                                n_c_fig_sr = min(num_som_rand_cls, 3)
                                n_r_fig_sr = math.ceil(num_som_rand_cls / n_c_fig_sr)
                                fig_sr, axs_sr = plt.subplots(n_r_fig_sr, n_c_fig_sr, figsize=(n_c_fig_sr*4, n_r_fig_sr*3.5))
                                if num_som_rand_cls == 1: axs_sr = [axs_sr]
                                else: axs_sr = axs_sr.flatten()
                                uniq_rand_lbls = np.unique(som_rand_lbls)
                                for i_p, ax_p_sr in enumerate(axs_sr):
                                    if i_p < num_som_rand_cls:
                                        cur_r_lbl = uniq_rand_lbls[i_p] if i_p < len(uniq_rand_lbls) else -1
                                        if cur_r_lbl != -1:
                                            data_r_lbl = X_scaled_som[som_rand_lbls == cur_r_lbl]
                                            if data_r_lbl.shape[0]>0:
                                                win_map_r = som_model.win_map(data_r_lbl)
                                                heatmap_r = np.zeros((som_grid_r, som_grid_c))
                                                for r_i_h, c_i_h in win_map_r.keys(): heatmap_r[r_i_h,c_i_h] = len(win_map_r[(r_i_h,c_i_h)])
                                                sns.heatmap(heatmap_r.T, ax=ax_p_sr, cmap="coolwarm", annot=True, fmt=".0f", cbar=i_p==0)
                                                ax_p_sr.set_title(f"Lớp: {cur_r_lbl}")
                                            else: ax_p_sr.set_title(f"Lớp: {cur_r_lbl} (Rỗng)")
                                        else: ax_p_sr.axis("off")
                                    else: ax_p_sr.axis("off")
                                plt.tight_layout()
                                st.pyplot(fig_sr)
                            else: st.error("Lỗi huấn luyện SOM.")
                        except ValueError as ve_som: st.error(f"Lỗi SOM: {ve_som}")
                        except Exception as e_som: st.error(f"Lỗi không xác định SOM: Điều chỉnh hàng và cột phù hợp.")
                    st.markdown("---")
            
            elif selected_analysis in [analysis_options[4], analysis_options[5], analysis_options[6]] and X_processed_nbkm is None:
                 st.error("Lỗi tiền xử lý dữ liệu chung tại NB, KM, SOM.")

    except Exception as e_general_preprocess: 
        st.error(f"Lỗi trong quá trình tiền xử lý dữ liệu chung: {e_general_preprocess}")


elif uploaded_file is None and selected_analysis != analysis_options[0]:
    st.info(" Vui lòng tải lên một tệp CSV từ thanh bên để bắt đầu phân tích.")
elif selected_analysis == analysis_options[0] and uploaded_file is not None:
    st.info("Dữ liệu đã được tải. Vui lòng chọn một thuật toán phân tích từ thanh bên.")
elif selected_analysis == analysis_options[0] and uploaded_file is None:
    st.info("Hãy tải tệp CSV và chọn một thuật toán để bắt đầu.")

st.sidebar.markdown("---")

