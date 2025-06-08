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
    if not selected_columns or len(selected_columns) != 2:
        raise ValueError("Phải chọn đúng 2 cột: [mã giao dịch, mã sản phẩm].")

    invoice_col, item_col = selected_columns
    df_encoded = pd.crosstab(df_input[invoice_col], df_input[item_col]).T.astype(bool)
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
        majority_class = df[target_attribute_name].mode()[0]
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
        majority_class = df[target_attribute_name].mode()[0]
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
            current_dot.node(node_id, f"{target_attribute_name_for_display} = {current_node.results}", shape="box", style="filled", color="lightgreen")
        else:
            current_dot.node(node_id, str(current_node.attribute))
        if current_parent_id:
            current_dot.edge(current_parent_id, node_id, label=str(edge_label))
        for val, branch_node in current_node.branches.items():
            traverse(branch_node, current_dot, node_id, str(val))
        return current_dot

    if dot is None:
        dot = graphviz.Digraph(comment='Decision Tree')
        dot.attr('node', shape='ellipse', style='filled', color='skyblue')
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
    df_clustered_results["Cluster_KMeans"] = cluster_labels
    
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
