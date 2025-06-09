import pandas as pd
import numpy as np
from collections import Counter
import math
from itertools import combinations
import streamlit as st 

# Thư viện cho các thuật toán cụ thể
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from minisom import MiniSom
import graphviz 

# ==============================================================================
# PHẦN 1: LOGIC TỪ SCRIPT APRIORI
# ==============================================================================
def apriori_original_process_data(df_input):
    """Hàm process_data gốc từ script Apriori của bạn (chỉ chọn cột số)."""
    # st.warning("Đang sử dụng `process_data` gốc cho Apriori: chỉ chọn các cột số...") # Thông báo này nên ở app.py
    return df_input.select_dtypes(include=['number'])

def apriori_general_transactional_conversion(df_input):
    """Chuyển đổi DataFrame chung thành dạng boolean cho Apriori."""
    # st.info("Đang chuyển đổi dữ liệu sang dạng boolean cho Apriori...") # Thông báo này nên ở app.py
    return df_input.applymap(lambda x: True if pd.notna(x) and x != 0 and x != '' and x is not False else False)

def apriori_one_hot_encode_data(df_input, selected_columns):
    """Hàm tiền xử lý bằng one-hot encoding cho Apriori."""
    if not selected_columns:
        raise ValueError("Phải chọn ít nhất một cột để one-hot encode.")
    
    # Nếu có đúng 2 cột, sử dụng crosstab (transaction format)
    if len(selected_columns) == 2:
        invoice_col, item_col = selected_columns
        df_encoded = pd.crosstab(df_input[invoice_col], df_input[item_col]).T.astype(bool)
        return df_encoded
    
    # Nếu có nhiều cột, sử dụng one-hot encoding thông thường
    df_encoded = pd.get_dummies(df_input[selected_columns], prefix_sep='_').astype(bool)
    return df_encoded

def apriori_get_maximal_frequent_itemsets(frequent_itemsets_df):
    maximal_itemsets_list = []
    if frequent_itemsets_df.empty:
        return pd.DataFrame(maximal_itemsets_list, columns=frequent_itemsets_df.columns if not frequent_itemsets_df.empty else ['support', 'itemsets'])

    for i, row in frequent_itemsets_df.iterrows():
        current_itemset = row['itemsets']
        is_maximal = True
        for j, other_row in frequent_itemsets_df.iterrows():
            if i == j:
                continue
            other_itemset = other_row['itemsets']
            if current_itemset.issubset(other_itemset) and current_itemset != other_itemset:
                is_maximal = False
                break
        if is_maximal:
            maximal_itemsets_list.append(row)
    return pd.DataFrame(maximal_itemsets_list) if maximal_itemsets_list else pd.DataFrame(columns=frequent_itemsets_df.columns)


def run_apriori_calculations(df_processed, min_sup, min_conf):
    """Chạy thuật toán Apriori và trả về kết quả."""
    frequent_itemsets_df = pd.DataFrame()
    rules_df = pd.DataFrame()
    maximal_frequent_itemsets_df = pd.DataFrame()

    if df_processed.empty or not any(df_processed.any()):
        raise ValueError("Dữ liệu xử lý cho Apriori rỗng hoặc không chứa giá trị True nào.")

    frequent_itemsets_df = apriori(df_processed, min_support=min_sup, use_colnames=True)

    if frequent_itemsets_df.empty:
        return frequent_itemsets_df, rules_df, maximal_frequent_itemsets_df 

    maximal_frequent_itemsets_df = apriori_get_maximal_frequent_itemsets(frequent_itemsets_df.copy())
    
    # Sinh luật từ tất cả tập phổ biến
    if not frequent_itemsets_df.empty:
        rules_df = association_rules(frequent_itemsets_df, metric="confidence", min_threshold=min_conf)
    
    return frequent_itemsets_df, rules_df, maximal_frequent_itemsets_df

# ==============================================================================
# PHẦN 2: LOGIC TỪ SCRIPT ROUGH SET
# ==============================================================================
def rs_lower_approximation(df, target_class_value, condition_attributes, decision_attribute_name):
    X = set(df[df[decision_attribute_name] == target_class_value].index)
    condition_attributes_list = list(condition_attributes)
    if not condition_attributes_list:
        return X if all(df[decision_attribute_name] == target_class_value) else set()

    partitions = df.groupby(condition_attributes_list).groups.values()
    lower_approx = set()
    for block_indices in partitions:
        block_set = set(block_indices)
        if block_set.issubset(X):
            lower_approx.update(block_set)
    return lower_approx

def rs_upper_approximation(df, target_class_value, condition_attributes, decision_attribute_name):
    X = set(df[df[decision_attribute_name] == target_class_value].index)
    condition_attributes_list = list(condition_attributes)
    if not condition_attributes_list:
        return set(df.index) if X else set()

    partitions = df.groupby(condition_attributes_list).groups.values()
    upper_approx = set()
    for block_indices in partitions:
        block_set = set(block_indices)
        if block_set & X:
            upper_approx.update(block_set)
    return upper_approx

def rs_accuracy(df, target_class_value, condition_attributes, decision_attribute_name):
    lower = rs_lower_approximation(df, target_class_value, condition_attributes, decision_attribute_name)
    upper = rs_upper_approximation(df, target_class_value, condition_attributes, decision_attribute_name)
    return len(lower) / len(upper) if len(upper) > 0 else 0

def rs_dependency(df, condition_attributes, decision_attribute_name):
    condition_attributes_list = list(condition_attributes)
    if not condition_attributes_list:
        return 0
        
    partitions = df.groupby(condition_attributes_list).groups.values()
    total_len = len(df)
    if total_len == 0: return 0
    
    dependency_sum = sum(len(list(block_indices)) for block_indices in partitions if len(df.loc[list(block_indices), decision_attribute_name].unique()) == 1)
    return dependency_sum / total_len

def rs_find_all_reducts(df, all_possible_condition_attributes, decision_attribute_name):
    all_possible_condition_attributes = list(all_possible_condition_attributes)
    all_reducts = []
    full_dependency = rs_dependency(df, all_possible_condition_attributes, decision_attribute_name)

    for r_len in range(1, len(all_possible_condition_attributes) + 1):
        for subset_tuple in combinations(all_possible_condition_attributes, r_len):
            subset_list = list(subset_tuple)
            if rs_dependency(df, subset_list, decision_attribute_name) == full_dependency:
                is_minimal = True
                for existing_reduct in all_reducts:
                    if set(existing_reduct).issubset(set(subset_list)) and len(existing_reduct) < len(subset_list):
                        is_minimal = False
                        break
                if is_minimal:
                    all_reducts = [red for red in all_reducts if not (set(subset_list).issubset(set(red)) and len(subset_list) < len(red))]
                    if subset_list not in all_reducts:
                         all_reducts.append(subset_list)
    return all_reducts if all_reducts else []

# ==============================================================================
# PHẦN 3: LOGIC TỪ SCRIPT CÂY QUYẾT ĐỊNH (ID3)
# ==============================================================================
def dt_entropy(column_values):
    counts = Counter(column_values)
    total_count = len(column_values)
    if total_count == 0: return 0
    return -sum((count / total_count) * math.log2(count / total_count) for count in counts.values() if count > 0)

def dt_gini_impurity_for_attribute_split(df_subset, target_attribute_name):
    total_count = len(df_subset)
    if total_count == 0: return 0
    impurity = 1.0
    for class_val in df_subset[target_attribute_name].unique():
        p_i = len(df_subset[df_subset[target_attribute_name] == class_val]) / total_count
        impurity -= p_i ** 2
    return impurity

def dt_info_gain(df, split_attribute_name, target_attribute_name):
    total_entropy_val = dt_entropy(df[target_attribute_name])
    weighted_entropy_after_split = 0
    total_len = len(df)
    if total_len == 0: return 0
    for value in df[split_attribute_name].unique():
        subset = df[df[split_attribute_name] == value]
        subset_len = len(subset)
        if subset_len > 0:
            weighted_entropy_after_split += (subset_len / total_len) * dt_entropy(subset[target_attribute_name])
    return total_entropy_val - weighted_entropy_after_split

def dt_gini_gain(df, split_attribute_name, target_attribute_name):
    parent_gini = dt_gini_impurity_for_attribute_split(df, target_attribute_name)
    weighted_child_gini = 0
    total_len = len(df)
    if total_len == 0: return 0
    for value in df[split_attribute_name].unique():
        subset = df[df[split_attribute_name] == value]
        subset_len = len(subset)
        if subset_len > 0:
            weighted_child_gini += (subset_len / total_len) * dt_gini_impurity_for_attribute_split(subset, target_attribute_name)
    return parent_gini - weighted_child_gini

class DecisionTreeNode:
    def __init__(self, attribute=None, results=None, branches=None, attribute_value_from_parent=None):
        self.attribute = attribute
        self.results = results
        self.branches = branches if branches else {}
        self.attribute_value_from_parent = attribute_value_from_parent

def dt_build_tree(df, current_attributes, target_attribute_name, method='Gain'):
    if len(df[target_attribute_name].unique()) == 1:
        return DecisionTreeNode(results=df[target_attribute_name].iloc[0])
    if not current_attributes or df.empty:
        if df.empty:
            return DecisionTreeNode(results="Không xác định (nhánh rỗng)") 
        mode_values = df[target_attribute_name].mode()
        majority_class = mode_values.iloc[0] if not mode_values.empty else df[target_attribute_name].iloc[0]
        return DecisionTreeNode(results=majority_class)

    best_attribute_to_split = None
    best_score = -1
    for attr in current_attributes:
        if method == 'Gain':
            score = dt_info_gain(df, attr, target_attribute_name)
        else:
            score = dt_gini_gain(df, attr, target_attribute_name)
        if score > best_score:
            best_score = score
            best_attribute_to_split = attr
    
    if best_attribute_to_split is None or best_score <= 0:
        mode_values = df[target_attribute_name].mode()
        majority_class = mode_values.iloc[0] if not mode_values.empty else df[target_attribute_name].iloc[0]
        return DecisionTreeNode(results=majority_class)

    tree_node = DecisionTreeNode(attribute=best_attribute_to_split)
    remaining_attributes_for_children = [attr for attr in current_attributes if attr != best_attribute_to_split]
    for value in df[best_attribute_to_split].dropna().unique():
        subset_df = df[df[best_attribute_to_split] == value].drop(columns=[best_attribute_to_split])
        subtree = dt_build_tree(subset_df, remaining_attributes_for_children, target_attribute_name, method)
        subtree.attribute_value_from_parent = value
        tree_node.branches[value] = subtree
    return tree_node

def dt_draw_tree_graphviz(node, target_attribute_name_for_display, dot=None, parent_id=None):
    def traverse(current_node, current_dot, current_parent_id=None, edge_label=""):
        node_id = str(id(current_node))
        if current_node.results is not None:
            # Leaf nodes (results) - green background with black text
            current_dot.node(node_id, f"{target_attribute_name_for_display} = {current_node.results}", 
                           shape="box", style="filled", color="lightgreen", fontcolor="black")
        else:
            # Internal nodes (attributes) - blue background with black text
            current_dot.node(node_id, str(current_node.attribute), 
                           shape="ellipse", style="filled", color="skyblue", fontcolor="black")
        if current_parent_id:
            # Edges with black text for labels
            current_dot.edge(current_parent_id, node_id, label=str(edge_label), fontcolor="black")
        for val, branch_node in current_node.branches.items():
            traverse(branch_node, current_dot, node_id, str(val))
        return current_dot

    if dot is None:
        dot = graphviz.Digraph(comment='Decision Tree')
        # Set graph background to white to ensure good contrast
        dot.attr(bgcolor='white')
        # Set default node attributes
        dot.attr('node', shape='ellipse', style='filled', color='skyblue', fontcolor='black')
        # Set default edge attributes  
        dot.attr('edge', color='black', fontcolor='black')
    return traverse(node, dot, parent_id)

def dt_extract_rules(node, target_attribute_name_for_display, current_rule_parts=None, rules_list=None, attribute_name_map=None):
    if current_rule_parts is None: current_rule_parts = []
    if rules_list is None: rules_list = []
    
    node_attribute_display = node.attribute
    if attribute_name_map and node.attribute in attribute_name_map:
        node_attribute_display = attribute_name_map[node.attribute]

    if node.results is not None:
        rule_str = "NẾU " + " VÀ ".join(current_rule_parts) + f" THÌ {target_attribute_name_for_display} = {node.results}" if current_rule_parts else f"Tất cả đều: {target_attribute_name_for_display} = {node.results}"
        rules_list.append(rule_str)
        return rules_list

    for value, subtree_node in node.branches.items():
        condition = f"{node_attribute_display} = '{value}'"
        dt_extract_rules(subtree_node, target_attribute_name_for_display, current_rule_parts + [condition], rules_list, attribute_name_map)
    return rules_list

# ==============================================================================
# PHẦN 4: LOGIC TỪ SCRIPT NAIVE BAYES, K-MEANS, KOHONEN
# ==============================================================================
def general_preprocess_data_nbkm(df_input, target_column_name_for_nb=None, st_instance=None):

    df_processed = df_input.copy()
    
    if "Day" in df_processed.columns:
        df_processed.drop(columns=["Day"], inplace=True, errors='ignore')
    if "Tranh" in df_processed.columns and df_processed['Tranh'].nunique() == len(df_processed):
        if st_instance: st_instance.sidebar.warning("Cột 'Tranh' được phát hiện và loại bỏ (coi như ID).")
        df_processed.drop(columns=["Tranh"], inplace=True, errors='ignore')
    
    df_processed.dropna(inplace=True)

    if df_processed.empty:
        
        return None, None, None, None


    X_temp = None
    y_processed_nb = None
    le_y_for_nb = None
    
    if target_column_name_for_nb and target_column_name_for_nb in df_processed.columns:
        X_temp = df_processed.drop(columns=[target_column_name_for_nb])
        y_series_nb = df_processed[target_column_name_for_nb]
        if y_series_nb.dtype == "object" or pd.api.types.is_categorical_dtype(y_series_nb):
            le_y_for_nb = LabelEncoder()
            y_processed_nb = le_y_for_nb.fit_transform(y_series_nb)
        else:
            y_processed_nb = y_series_nb.values 
    else:
        X_temp = df_processed.copy()

    feature_label_encoders = {}
    X_processed_features = X_temp.copy()

    for col in X_temp.columns:
        if X_temp[col].dtype == "object" or pd.api.types.is_categorical_dtype(X_temp[col]):
            try:
                X_processed_features[col] = pd.to_numeric(X_temp[col])
                if st_instance: st_instance.info(f"Cột '{col}' (object/category) đã được chuyển đổi thành số.")
            except ValueError:
                le = LabelEncoder()
                X_processed_features[col] = le.fit_transform(X_temp[col])
                feature_label_encoders[col] = le
                if st_instance: st_instance.info(f"Cột '{col}' (object/category) đã được mã hóa bằng LabelEncoder.")
        elif not pd.api.types.is_numeric_dtype(X_temp[col]):
            try:
                X_processed_features[col] = X_temp[col].astype(float)
            except ValueError:
                 if st_instance: st_instance.error(f"Cột '{col}' có kiểu {X_temp[col].dtype} không thể chuyển đổi thành dạng số.")

                 return None, None, None, None
    
    try:
        X_processed_features = X_processed_features.astype(float)
    except Exception as e:
        if st_instance: st_instance.error(f"Lỗi khi ép kiểu tất cả các cột thuộc tính sang float: {e}")
        return None, None, None, None

    return X_processed_features, y_processed_nb, le_y_for_nb, feature_label_encoders


def train_naive_bayes_model(X_train_data, y_train_data, nb_method_type='gaussian', alpha_val=1.0, st_instance=None):
    model = None
    X_train_data_copy = X_train_data.copy() 

    if nb_method_type == 'gaussian':
        model = GaussianNB()
    elif nb_method_type == 'multinomial':
        if (X_train_data_copy < 0).any().any():
            if st_instance: st_instance.warning("⚠️ Dữ liệu có giá trị âm. MultinomialNB yêu cầu các đặc trưng không âm.")
            raise ValueError("MultinomialNB không thể xử lý giá trị âm. Cần chuẩn hóa trước.")
        model = MultinomialNB(alpha=alpha_val)
    else:
        if st_instance: st_instance.error("Phương pháp Naive Bayes không hợp lệ.")
        raise ValueError("Phương pháp Naive Bayes không hợp lệ.")

    try:
        model.fit(X_train_data_copy, y_train_data)
        accuracy = model.score(X_train_data_copy, y_train_data)
        return model, accuracy
    except Exception as e:
        if st_instance: st_instance.error(f"Lỗi khi huấn luyện mô hình Naive Bayes ({nb_method_type}): {e}")
        raise e 


def run_kmeans_clustering_analysis(X_data_for_kmeans, num_k_clusters=3):
    if X_data_for_kmeans.shape[1] < 1:
        raise ValueError("Dữ liệu cần ít nhất 1 thuộc tính để chạy K-Means.")
    
    scaler_kmeans = MinMaxScaler()
    X_scaled_for_kmeans = scaler_kmeans.fit_transform(X_data_for_kmeans)
    
    kmeans_model = KMeans(n_clusters=num_k_clusters, random_state=42, n_init="auto")
    cluster_labels = kmeans_model.fit_predict(X_scaled_for_kmeans)
    cluster_centers = kmeans_model.cluster_centers_

    df_clustered_results = X_data_for_kmeans.copy()
    df_clustered_results["Cluster"] = cluster_labels
    
    return cluster_labels, cluster_centers, df_clustered_results, X_scaled_for_kmeans


def train_kohonen_som_model(X_data_for_som, som_rows, som_cols, sigma_val=1.0, lr_val=0.5, iterations_val=100, st_instance=None):
    if X_data_for_som.shape[0] == 0 or X_data_for_som.shape[1] == 0:
        raise ValueError("Dữ liệu rỗng hoặc không có thuộc tính. Không thể huấn luyện SOM.")
    
    scaler_som = MinMaxScaler()
    X_scaled_for_som = scaler_som.fit_transform(X_data_for_som)
    input_length = X_scaled_for_som.shape[1]
    
    som_instance = MiniSom(
        x=som_cols, y=som_rows, input_len=input_length, 
        sigma=sigma_val, learning_rate=lr_val, random_seed=42 
    )
    som_instance.random_weights_init(X_scaled_for_som)
    if st_instance: st_instance.write(f"Đang huấn luyện SOM với {X_scaled_for_som.shape[0]} điểm, {iterations_val} vòng lặp...")
    som_instance.train_random(X_scaled_for_som, iterations_val) 
    if st_instance: st_instance.success("✅ Huấn luyện SOM hoàn tất!")
    return som_instance, X_scaled_for_som

# ===== DETAILED CALCULATION EXPLANATIONS =====

def get_apriori_detailed_explanation(df, min_support=0.4, min_confidence=0.6):
    """
    Trả về giải thích chi tiết từng bước tính toán Apriori
    """
    explanations = []
    
    # Chuẩn bị dữ liệu
    df_encoded = pd.get_dummies(df.iloc[:, 1], prefix='', prefix_sep='')
    total_transactions = len(df_encoded)
    
    explanations.append(f"**📊 Dữ liệu đầu vào:**")
    explanations.append(f"- Tổng số giao dịch: {total_transactions}")
    explanations.append(f"- Min Support: {min_support} ({min_support*100}%)")
    explanations.append(f"- Min Confidence: {min_confidence} ({min_confidence*100}%)")
    explanations.append("")
    
    # Tính support cho từng item
    explanations.append(f"**🔍 Bước 1: Tính Support cho từng item**")
    item_support = {}
    for col in df_encoded.columns:
        count = df_encoded[col].sum()
        support = count / total_transactions
        item_support[col] = support
        
        if support >= min_support:
            status = "✅ ĐẠT"
        else:
            status = "❌ LOẠI"
        
        explanations.append(f"- Item '{col}': {count}/{total_transactions} = {support:.3f} {status}")
    
    # L1 - Frequent 1-itemsets
    L1 = {item: support for item, support in item_support.items() if support >= min_support}
    explanations.append("")
    explanations.append(f"**📋 L1 (Frequent 1-itemsets): {len(L1)} items**")
    for item, support in L1.items():
        explanations.append(f"- {{{item}}}: {support:.3f}")
    
    # Generate C2 candidates
    explanations.append("")
    explanations.append(f"**🔍 Bước 2: Tạo C2 candidates và tính Support**")
    
    items = list(L1.keys())
    pair_support = {}
    
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            item1, item2 = items[i], items[j]
            # Tính support cho cặp
            count = ((df_encoded[item1] == 1) & (df_encoded[item2] == 1)).sum()
            support = count / total_transactions
            pair_support[(item1, item2)] = support
            
            if support >= min_support:
                status = "✅ ĐẠT"
            else:
                status = "❌ LOẠI"
            
            explanations.append(f"- {{{item1}, {item2}}}: {count}/{total_transactions} = {support:.3f} {status}")
    
    # L2 - Frequent 2-itemsets
    L2 = {pair: support for pair, support in pair_support.items() if support >= min_support}
    explanations.append("")
    explanations.append(f"**📋 L2 (Frequent 2-itemsets): {len(L2)} pairs**")
    for pair, support in L2.items():
        explanations.append(f"- {{{pair[0]}, {pair[1]}}}: {support:.3f}")
    
    # Tính Association Rules
    explanations.append("")
    explanations.append(f"**🔍 Bước 3: Tạo Association Rules**")
    
    rules = []
    for pair, pair_support in L2.items():
        item1, item2 = pair
        
        # Rule: item1 -> item2
        conf1 = pair_support / L1[item1]
        if conf1 >= min_confidence:
            rules.append((item1, item2, conf1))
            status1 = "✅ MẠNH"
        else:
            status1 = "❌ YẾU"
        
        explanations.append(f"- {item1} → {item2}: {pair_support:.3f}/{L1[item1]:.3f} = {conf1:.3f} {status1}")
        
        # Rule: item2 -> item1  
        conf2 = pair_support / L1[item2]
        if conf2 >= min_confidence:
            rules.append((item2, item1, conf2))
            status2 = "✅ MẠNH"
        else:
            status2 = "❌ YẾU"
        
        explanations.append(f"- {item2} → {item1}: {pair_support:.3f}/{L1[item2]:.3f} = {conf2:.3f} {status2}")
    
    # Tổng kết
    explanations.append("")
    explanations.append(f"**📊 Tổng kết kết quả:**")
    explanations.append(f"- Frequent 1-itemsets: {len(L1)}")
    explanations.append(f"- Frequent 2-itemsets: {len(L2)}")
    explanations.append(f"- Strong Association Rules: {len(rules)}")
    
    return "\n".join(explanations)

def get_rough_set_detailed_explanation(df, target_class, condition_attrs, decision_attr):
    """
    Trả về giải thích chi tiết từng bước tính toán Rough Set
    """
    explanations = []
    
    # Thông tin cơ bản
    total_objects = len(df)
    target_objects = df[df[decision_attr] == target_class].index.tolist()
    
    explanations.append(f"**📊 Dữ liệu đầu vào:**")
    explanations.append(f"- Tổng số đối tượng: {total_objects}")
    explanations.append(f"- Thuộc tính điều kiện: {', '.join(condition_attrs)}")
    explanations.append(f"- Thuộc tính quyết định: {decision_attr}")
    explanations.append(f"- Lớp mục tiêu: {target_class}")
    explanations.append(f"- Số đối tượng thuộc lớp '{target_class}': {len(target_objects)}")
    explanations.append("")
    
    # Tạo equivalence classes
    explanations.append(f"**🔍 Bước 1: Tạo Equivalence Classes**")
    equivalence_classes = {}
    
    for idx, row in df.iterrows():
        # Tạo key từ các thuộc tính điều kiện
        key = tuple(row[attr] for attr in condition_attrs)
        if key not in equivalence_classes:
            equivalence_classes[key] = []
        equivalence_classes[key].append(idx)
    
    for i, (key, objects) in enumerate(equivalence_classes.items(), 1):
        key_str = ', '.join([f"{condition_attrs[j]}={key[j]}" for j in range(len(key))])
        explanations.append(f"- Lớp tương đương {i}: [{key_str}] = {{{', '.join(map(str, objects))}}} ({len(objects)} objects)")
    
    explanations.append("")
    
    # Tính Lower Approximation
    explanations.append(f"**🔍 Bước 2: Tính Lower Approximation R(X)**")
    lower_approx = []
    
    for key, eq_class in equivalence_classes.items():
        # Kiểm tra tất cả objects trong equivalence class có thuộc target class không
        all_in_target = all(df.loc[obj, decision_attr] == target_class for obj in eq_class)
        
        key_str = ', '.join([f"{condition_attrs[j]}={key[j]}" for j in range(len(key))])
        
        if all_in_target:
            lower_approx.extend(eq_class)
            explanations.append(f"- Lớp [{key_str}]: TẤT CẢ thuộc '{target_class}' → ✅ THÊM VÀO R(X)")
        else:
            explanations.append(f"- Lớp [{key_str}]: KHÔNG phải tất cả thuộc '{target_class}' → ❌ LOẠI")
    
    explanations.append(f"- **Lower Approximation R(X) = {{{', '.join(map(str, sorted(lower_approx)))}}} ({len(lower_approx)} objects)**")
    explanations.append("")
    
    # Tính Upper Approximation
    explanations.append(f"**🔍 Bước 3: Tính Upper Approximation R̄(X)**")
    upper_approx = []
    
    for key, eq_class in equivalence_classes.items():
        # Kiểm tra có ít nhất 1 object thuộc target class không
        any_in_target = any(df.loc[obj, decision_attr] == target_class for obj in eq_class)
        
        key_str = ', '.join([f"{condition_attrs[j]}={key[j]}" for j in range(len(key))])
        
        if any_in_target:
            upper_approx.extend(eq_class)
            explanations.append(f"- Lớp [{key_str}]: CÓ ÍT NHẤT 1 thuộc '{target_class}' → ✅ THÊM VÀO R̄(X)")
        else:
            explanations.append(f"- Lớp [{key_str}]: KHÔNG có object nào thuộc '{target_class}' → ❌ LOẠI")
    
    explanations.append(f"- **Upper Approximation R̄(X) = {{{', '.join(map(str, sorted(upper_approx)))}}} ({len(upper_approx)} objects)**")
    explanations.append("")
    
    # Tính Accuracy
    accuracy = len(lower_approx) / len(upper_approx) if upper_approx else 0
    explanations.append(f"**🔍 Bước 4: Tính Accuracy**")
    explanations.append(f"- Accuracy α(X) = |R(X)| / |R̄(X)| = {len(lower_approx)} / {len(upper_approx)} = {accuracy:.3f}")
    
    if accuracy == 1.0:
        explanations.append(f"- **Accuracy = 1.0**: Tập X hoàn toàn xác định (crisp set)")
    elif accuracy == 0.0:
        explanations.append(f"- **Accuracy = 0.0**: Tập X hoàn toàn không xác định")
    else:
        explanations.append(f"- **Accuracy = {accuracy:.3f}**: Tập X một phần xác định (rough set)")
    
    explanations.append("")
    
    # Tính Dependency
    total_lower = len(lower_approx)
    total_universe = len(df)
    dependency = total_lower / total_universe
    
    explanations.append(f"**🔍 Bước 5: Tính Dependency**")
    explanations.append(f"- Dependency γ = |R(X)| / |U| = {total_lower} / {total_universe} = {dependency:.3f}")
    explanations.append(f"- **Giải thích**: {dependency*100:.1f}% đối tượng có thể phân loại chắc chắn")
    
    return "\n".join(explanations)

def get_decision_tree_detailed_explanation(df, selected_features, target_attr, method='Gain'):
    """
    Trả về giải thích chi tiết từng bước tính toán Decision Tree
    """
    explanations = []
    
    # Thông tin cơ bản
    total_samples = len(df)
    class_counts = df[target_attr].value_counts()
    
    # Debug: Kiểm tra type của class_counts
    print(f"DEBUG: class_counts type: {type(class_counts)}")
    print(f"DEBUG: class_counts: {class_counts}")
    
    explanations.append(f"**📊 Dữ liệu đầu vào:**")
    explanations.append(f"- Tổng số samples: {total_samples}")
    explanations.append(f"- Số features: {len(selected_features)}")
    explanations.append(f"- Target attribute: {target_attr}")
    explanations.append(f"- Method: {method}")
    explanations.append("")
    
    explanations.append(f"**📈 Phân bố classes:**")
    for cls, count in class_counts.items():
        proportion = count / total_samples
        explanations.append(f"- {cls}: {count}/{total_samples} = {proportion:.3f}")
    explanations.append("")
    
    # Tính Entropy của tập gốc
    import math
    entropy_total = 0
    # Xử lý an toàn cho class_counts
    try:
        if hasattr(class_counts, 'values') and callable(getattr(class_counts, 'values', None)):
            count_values = class_counts.values()
        elif hasattr(class_counts, 'values'):
            count_values = class_counts.values
        else:
            count_values = class_counts
        
        for count in count_values:
            if count > 0:
                p = count / total_samples
                entropy_total -= p * math.log2(p)
    except Exception as e:
        print(f"DEBUG: Error in entropy calculation: {e}")
        # Fallback: tính entropy thủ công
        for cls, count in class_counts.items():
            if count > 0:
                p = count / total_samples
                entropy_total -= p * math.log2(p)
    
    explanations.append(f"**🔍 Bước 1: Tính Entropy của tập gốc**")
    calculation_parts = []
    for cls, count in class_counts.items():
        if count > 0:
            p = count / total_samples
            part = f"({p:.3f} × log₂({p:.3f}))"
            calculation_parts.append(part)
    
    explanations.append(f"- E(S) = -Σᵢ pᵢ × log₂(pᵢ)")
    explanations.append(f"- E(S) = -[{' + '.join(calculation_parts)}]")
    explanations.append(f"- **E(S) = {entropy_total:.4f}**")
    explanations.append("")
    
    # Tính Information Gain cho mỗi feature
    explanations.append(f"**🔍 Bước 2: Tính Information Gain cho mỗi feature**")
    
    feature_gains = {}
    for feature in selected_features:
        feature_values = df[feature].unique()
        weighted_entropy = 0
        
        explanations.append(f"**Feature: {feature}**")
        
        for value in feature_values:
            subset = df[df[feature] == value]
            subset_size = len(subset)
            weight = subset_size / total_samples
            
            # Tính entropy của subset
            subset_class_counts = subset[target_attr].value_counts()
            subset_entropy = 0
            
            entropy_parts = []
            for cls, count in subset_class_counts.items():
                if count > 0:
                    p = count / subset_size
                    subset_entropy -= p * math.log2(p)
                    entropy_parts.append(f"({p:.3f} × log₂({p:.3f}))")
            
            weighted_entropy += weight * subset_entropy
            
            explanations.append(f"  - {feature}={value}: {subset_size} samples")
            explanations.append(f"    Classes: {dict(subset_class_counts)}")
            explanations.append(f"    E(S_{value}) = -[{' + '.join(entropy_parts)}] = {subset_entropy:.4f}")
        
        information_gain = entropy_total - weighted_entropy
        feature_gains[feature] = information_gain
        
        explanations.append(f"  - **IG({feature}) = {entropy_total:.4f} - {weighted_entropy:.4f} = {information_gain:.4f}**")
        explanations.append("")
    
    # Chọn feature tốt nhất
    best_feature = max(feature_gains, key=feature_gains.get)
    best_gain = feature_gains[best_feature]
    
    explanations.append(f"**🔍 Bước 3: Chọn Root Node**")
    explanations.append(f"- Feature có Information Gain cao nhất: **{best_feature}** (IG = {best_gain:.4f})")
    explanations.append(f"- **{best_feature}** được chọn làm Root Node")
    explanations.append("")
    
    # Ranking các features
    sorted_features = sorted(feature_gains.items(), key=lambda x: x[1], reverse=True)
    explanations.append(f"**📊 Ranking Information Gain:**")
    for i, (feat, gain) in enumerate(sorted_features, 1):
        explanations.append(f"{i}. {feat}: {gain:.4f}")
    
    return "\n".join(explanations)

def get_naive_bayes_detailed_explanation(X, y, nb_type='gaussian'):
    """
    Trả về giải thích chi tiết từng bước tính toán Naive Bayes
    """
    explanations = []
    
    # Thông tin cơ bản
    n_samples, n_features = X.shape
    classes = np.unique(y)
    
    explanations.append(f"**📊 Dữ liệu đầu vào:**")
    explanations.append(f"- Số samples: {n_samples}")
    explanations.append(f"- Số features: {n_features}")
    explanations.append(f"- Số classes: {len(classes)}")
    explanations.append(f"- Classes: {list(classes)}")
    explanations.append(f"- Naive Bayes type: {nb_type}")
    explanations.append("")
    
    # Tình Prior Probabilities
    explanations.append(f"**🔍 Bước 1: Tính Prior Probabilities P(y)**")
    class_counts = pd.Series(y).value_counts()
    
    for cls in classes:
        count = class_counts.get(cls, 0)
        prior = count / n_samples
        explanations.append(f"- P(y={cls}) = {count}/{n_samples} = {prior:.4f}")
    explanations.append("")
    
    # Tính Likelihood probabilities cho mỗi feature
    explanations.append(f"**🔍 Bước 2: Tính Likelihood P(xᵢ|y) cho mỗi feature**")
    
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    for i, feature_name in enumerate(feature_names[:3]):  # Chỉ hiển thị 3 features đầu
        explanations.append(f"**Feature: {feature_name}**")
        
        if nb_type == 'gaussian':
            # Gaussian NB - tính mean và std cho mỗi class
            for cls in classes:
                mask = (y == cls)
                feature_values = X.iloc[:, i] if isinstance(X, pd.DataFrame) else X[:, i]
                cls_values = feature_values[mask]
                
                mean_val = np.mean(cls_values)
                std_val = np.std(cls_values)
                
                explanations.append(f"  - Class {cls}: μ={mean_val:.4f}, σ={std_val:.4f}")
        
        else:  # Multinomial NB
            # Tính tần suất cho mỗi giá trị
            unique_values = np.unique(X.iloc[:, i] if isinstance(X, pd.DataFrame) else X[:, i])
            
            for cls in classes:
                mask = (y == cls)
                feature_values = X.iloc[:, i] if isinstance(X, pd.DataFrame) else X[:, i]
                cls_values = feature_values[mask]
                
                explanations.append(f"  - Class {cls}:")
                for val in unique_values[:3]:  # Chỉ hiển thị 3 giá trị đầu
                    count = np.sum(cls_values == val)
                    total = len(cls_values)
                    prob = count / total
                    explanations.append(f"    P({feature_name}={val}|{cls}) = {count}/{total} = {prob:.4f}")
        
        explanations.append("")
    
    # Ví dụ dự đoán
    explanations.append(f"**🔍 Bước 3: Ví dụ dự đoán cho sample đầu tiên**")
    first_sample = X.iloc[0] if isinstance(X, pd.DataFrame) else X[0]
    true_label = y[0]
    
    explanations.append(f"- Sample: {list(first_sample)}")
    explanations.append(f"- True label: {true_label}")
    explanations.append("")
    
    # Tính posterior cho mỗi class
    for cls in classes:
        prior = class_counts.get(cls, 0) / n_samples
        explanations.append(f"**Class {cls}:**")
        explanations.append(f"- Prior P({cls}) = {prior:.4f}")
        
        # Simplified likelihood calculation
        explanations.append(f"- Likelihood P(X|{cls}) = ∏ᵢ P(xᵢ|{cls}) [tính từ model]")
        explanations.append(f"- Posterior ∝ P({cls}) × P(X|{cls})")
        explanations.append("")
    
    explanations.append(f"**📊 Kết quả dự đoán:**")
    explanations.append(f"- Chọn class có posterior probability cao nhất")
    explanations.append(f"- Áp dụng Laplace smoothing để tránh P=0")
    
    return "\n".join(explanations)

def get_kmeans_detailed_explanation(X, n_clusters=3, random_state=42):
    """
    Trả về giải thích chi tiết từng bước tính toán K-Means
    """
    explanations = []
    
    # Thông tin cơ bản
    n_samples, n_features = X.shape
    
    explanations.append(f"**📊 Dữ liệu đầu vào:**")
    explanations.append(f"- Số samples: {n_samples}")
    explanations.append(f"- Số features: {n_features}")
    explanations.append(f"- Số clusters (K): {n_clusters}")
    explanations.append(f"- Random state: {random_state}")
    explanations.append("")
    
    # Khởi tạo centroids
    np.random.seed(random_state)
    initial_centroids = []
    
    if isinstance(X, pd.DataFrame):
        X_array = X.values
        feature_names = X.columns.tolist()
    else:
        X_array = X
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    # Random initialization của centroids
    for k in range(n_clusters):
        centroid = []
        for j in range(n_features):
            min_val = X_array[:, j].min()
            max_val = X_array[:, j].max()
            random_val = np.random.uniform(min_val, max_val)
            centroid.append(random_val)
        initial_centroids.append(centroid)
    
    explanations.append(f"**🔍 Bước 1: Khởi tạo K centroids ngẫu nhiên**")
    for k, centroid in enumerate(initial_centroids):
        centroid_str = ', '.join([f"{feature_names[j]}={val:.3f}" for j, val in enumerate(centroid)])
        explanations.append(f"- Centroid {k+1}: ({centroid_str})")
    explanations.append("")
    
    # Simulation của vài iterations đầu
    explanations.append(f"**🔍 Bước 2: Gán samples vào clusters (Iteration 1)**")
    
    # Tính khoảng cách từ mỗi sample đến mỗi centroid
    distances = np.zeros((n_samples, n_clusters))
    for i in range(n_samples):
        for k in range(n_clusters):
            # Euclidean distance
            dist = np.sqrt(np.sum((X_array[i] - initial_centroids[k])**2))
            distances[i, k] = dist
    
    # Gán cluster cho mỗi sample
    cluster_assignments = np.argmin(distances, axis=1)
    
    # Hiển thị vài samples đầu
    for i in range(min(5, n_samples)):
        sample_str = ', '.join([f"{val:.3f}" for val in X_array[i]])
        dist_str = ', '.join([f"{dist:.3f}" for dist in distances[i]])
        assigned_cluster = cluster_assignments[i] + 1
        
        explanations.append(f"- Sample {i+1}: ({sample_str})")
        explanations.append(f"  Distances: [{dist_str}]")
        explanations.append(f"  → Assigned to Cluster {assigned_cluster}")
    
    if n_samples > 5:
        explanations.append(f"  ... (và {n_samples-5} samples khác)")
    explanations.append("")
    
    # Cập nhật centroids
    explanations.append(f"**🔍 Bước 3: Cập nhật centroids**")
    new_centroids = []
    
    for k in range(n_clusters):
        cluster_samples = X_array[cluster_assignments == k]
        if len(cluster_samples) > 0:
            new_centroid = np.mean(cluster_samples, axis=0)
            new_centroids.append(new_centroid)
            
            centroid_str = ', '.join([f"{feature_names[j]}={val:.3f}" for j, val in enumerate(new_centroid)])
            explanations.append(f"- Cluster {k+1}: {len(cluster_samples)} samples")
            explanations.append(f"  New centroid: ({centroid_str})")
        else:
            new_centroids.append(initial_centroids[k])
            explanations.append(f"- Cluster {k+1}: 0 samples (giữ nguyên centroid)")
    
    explanations.append("")
    
    # Tính cost function (WCSS)
    wcss = 0
    for i in range(n_samples):
        assigned_cluster = cluster_assignments[i]
        centroid = new_centroids[assigned_cluster]
        wcss += np.sum((X_array[i] - centroid)**2)
    
    explanations.append(f"**🔍 Bước 4: Tính Cost Function (WCSS)**")
    explanations.append(f"- WCSS = Σᵢ ||xᵢ - μₖ||² = {wcss:.4f}")
    explanations.append(f"- **Mục tiêu**: Minimize WCSS qua các iterations")
    explanations.append("")
    
    # Điều kiện hội tụ
    explanations.append(f"**🔍 Bước 5: Kiểm tra hội tụ**")
    explanations.append(f"- So sánh centroids mới vs cũ")
    explanations.append(f"- Dừng khi: centroids không đổi hoặc thay đổi < threshold")
    explanations.append(f"- Hoặc đạt max_iterations")
    explanations.append("")
    
    # Tổng kết
    cluster_sizes = np.bincount(cluster_assignments)
    explanations.append(f"**📊 Tổng kết (sau iteration 1):**")
    for k in range(n_clusters):
        size = cluster_sizes[k] if k < len(cluster_sizes) else 0
        explanations.append(f"- Cluster {k+1}: {size} samples")
    explanations.append(f"- WCSS: {wcss:.4f}")
    
    return "\n".join(explanations)

def get_som_detailed_explanation(X, grid_size=(2, 2), learning_rate=0.1, epochs=100):
    """
    Trả về giải thích chi tiết từng bước tính toán SOM
    """
    explanations = []
    
    # Thông tin cơ bản
    n_samples, n_features = X.shape
    
    explanations.append(f"**📊 Dữ liệu đầu vào:**")
    explanations.append(f"- Số samples: {n_samples}")
    explanations.append(f"- Số features: {n_features}")
    explanations.append(f"- Grid size: {grid_size[0]}×{grid_size[1]} = {grid_size[0]*grid_size[1]} neurons")
    explanations.append(f"- Learning rate α₀: {learning_rate}")
    explanations.append(f"- Epochs: {epochs}")
    explanations.append("")
    
    # Khởi tạo weight matrix
    n_neurons = grid_size[0] * grid_size[1]
    np.random.seed(42)
    weights = np.random.rand(n_neurons, n_features)
    
    explanations.append(f"**🔍 Bước 1: Khởi tạo Weight Matrix**")
    explanations.append(f"- Mỗi neuron có {n_features} weights (tương ứng với số features)")
    explanations.append(f"- Khởi tạo ngẫu nhiên trong [0, 1]")
    explanations.append("")
    
    # Hiển thị weights ban đầu
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_array = X.values
    else:
        feature_names = [f"Feature_{i}" for i in range(n_features)]
        X_array = X
    
    for i in range(min(4, n_neurons)):
        weight_str = ', '.join([f"{feature_names[j]}={weights[i,j]:.3f}" for j in range(n_features)])
        row, col = divmod(i, grid_size[1])
        explanations.append(f"- Neuron ({row},{col}): W = [{weight_str}]")
    
    if n_neurons > 4:
        explanations.append(f"  ... (và {n_neurons-4} neurons khác)")
    explanations.append("")
    
    # Simulation của một epoch
    explanations.append(f"**🔍 Bước 2: Training Process (Epoch 1)**")
    
    # Chọn sample đầu tiên để demo
    sample_idx = 0
    input_vector = X_array[sample_idx]
    sample_str = ', '.join([f"{val:.3f}" for val in input_vector])
    
    explanations.append(f"**Sample {sample_idx+1}: [{sample_str}]**")
    explanations.append("")
    
    # Tính khoảng cách đến tất cả neurons
    distances = []
    for i in range(n_neurons):
        dist = np.sqrt(np.sum((input_vector - weights[i])**2))
        distances.append(dist)
        
        row, col = divmod(i, grid_size[1])
        explanations.append(f"- Distance to Neuron ({row},{col}): {dist:.4f}")
    
    # Tìm BMU (Best Matching Unit)
    bmu_idx = np.argmin(distances)
    bmu_row, bmu_col = divmod(bmu_idx, grid_size[1])
    explanations.append("")
    explanations.append(f"**BMU (Best Matching Unit): Neuron ({bmu_row},{bmu_col})** (distance = {distances[bmu_idx]:.4f})")
    explanations.append("")
    
    # Cập nhật weights
    explanations.append(f"**🔍 Bước 3: Cập nhật Weights**")
    explanations.append(f"- Công thức: W_new = W_old + α(t) × h(t) × (X - W_old)")
    explanations.append(f"- α(t) = learning rate tại thời điểm t")
    explanations.append(f"- h(t) = neighborhood function")
    explanations.append("")
    
    # Cập nhật BMU và neighbors
    for i in range(n_neurons):
        row, col = divmod(i, grid_size[1])
        
        # Tính neighborhood distance
        neighborhood_dist = np.sqrt((row - bmu_row)**2 + (col - bmu_col)**2)
        
        # Neighborhood function (Gaussian)
        sigma = 1.0  # neighborhood radius
        neighborhood_influence = np.exp(-(neighborhood_dist**2) / (2 * sigma**2))
        
        # Weight update
        old_weights = weights[i].copy()
        weights[i] += learning_rate * neighborhood_influence * (input_vector - weights[i])
        
        if i == bmu_idx:
            explanations.append(f"- BMU ({row},{col}): h={neighborhood_influence:.3f}")
            weight_change = weights[i] - old_weights
            change_str = ', '.join([f"{val:+.4f}" for val in weight_change])
            explanations.append(f"  Weight change: [{change_str}]")
        elif neighborhood_influence > 0.1:  # Chỉ hiển thị neighbors có ảnh hưởng đáng kể
            explanations.append(f"- Neighbor ({row},{col}): h={neighborhood_influence:.3f}")
    
    explanations.append("")
    
    # Convergence và learning schedule
    explanations.append(f"**🔍 Bước 4: Learning Schedule**")
    explanations.append(f"- Learning rate decay: α(t) = α₀ × exp(-t/τ)")
    explanations.append(f"- Neighborhood decay: σ(t) = σ₀ × exp(-t/τ)")
    explanations.append(f"- Qua {epochs} epochs, α và σ giảm dần")
    explanations.append("")
    
    # Final mapping
    explanations.append(f"**📊 Kết quả cuối cùng:**")
    explanations.append(f"- Mỗi sample được map vào neuron gần nhất")
    explanations.append(f"- Các samples tương tự sẽ activate cùng neuron hoặc neurons gần nhau")
    explanations.append(f"- Tạo ra topological map 2D của dữ liệu high-dimensional")
    
    return "\n".join(explanations)
