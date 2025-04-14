import pandas as pd  
import numpy as np  
from lifelines import CoxPHFitter  
from lifelines.utils import concordance_index  
import os  
from itertools import combinations  

np.random.seed(42)  

radiomics_dir = r""  
clinical_dir = r""  
deep_dir = r"P"  
output_dir = r""  

os.makedirs(output_dir, exist_ok=True)  

dataset_names = ["HPV", "HPV0", "HPV1", "Test", "Train", "Val"]  

radiomics_files = {name: f"{name.lower() if name not in ['HPV', 'HPV0', 'HPV1'] else name}_processed_mrmr_lasso.csv" for name in dataset_names}  
clinical_files = {name: f"{name}.csv" for name in dataset_names}  
deep_files = {name: f"{name}.csv" for name in dataset_names}  


def load_datasets(files_dict, directory, data_type):  
    datasets = {}  
    for name, filename in files_dict.items():  
        file_path = f"{directory}/{filename}"  
        try:  
            df = pd.read_csv(file_path)  
            print(f"Loaded {name} {data_type} dataset: {df.shape[0]} rows, {df.shape[1]} columns")  
            datasets[name] = df  
        except Exception as e:  
            print(f"Failed to load {name} {data_type} dataset: {e}")  
    return datasets  

def merge_datasets(datasets_dict, id_col="ID"):  
    if not datasets_dict:  
        return None  
    
    dataset_types = list(datasets_dict.keys())  
    
    base_type = dataset_types[0]  
    merged_df = datasets_dict[base_type].copy()  
    
    merged_df[id_col] = merged_df[id_col].astype(str)  
    
    initial_features = [col for col in merged_df.columns if col not in [id_col, "label", "TIME"]]  
    initial_samples = len(merged_df)  
    feature_counts = {base_type: len(initial_features)}  
    
    for data_type in dataset_types[1:]:  
        current_df = datasets_dict[data_type].copy()  
        current_df[id_col] = current_df[id_col].astype(str)  
        
        keep_cols = [col for col in current_df.columns   
                    if col not in [id_col, "label", "TIME"] or col == id_col]  
        
        feature_counts[data_type] = len(keep_cols) - 1  
        
        if len(keep_cols) <= 1:  
            print(f"Warning: {data_type} dataset has no features, skipping merge")  
            continue  
        
        merged_df = pd.merge(merged_df, current_df[keep_cols], on=id_col, how="inner")  
    
    final_samples = len(merged_df)  
    final_features = [col for col in merged_df.columns if col not in [id_col, "label", "TIME"]]  
    
    print(f"  Merge result: {initial_samples} → {final_samples} samples")  
    for data_type, count in feature_counts.items():  
        print(f"  {data_type} features: {count}")  
    print(f"  Total features: {len(final_features)}")  
    
    if "label" not in merged_df.columns or "TIME" not in merged_df.columns:  
        print("  Warning: Merged data missing survival time or event columns")  
    
    return merged_df  

def check_dataset_validity(times, events):  
    event_count = sum(events)  
    non_event_count = len(events) - event_count  
    
    if event_count == 0 or non_event_count == 0:  
        return False, f"Dataset needs both event and non-event samples (events: {event_count}, non-events: {non_event_count})"  
    
    unique_times = len(np.unique(times))  
    if unique_times <= 1:  
        return False, f"Dataset lacks variability in survival times (unique time values: {unique_times})"  
    
    return True, "Dataset is valid"  

def calculate_cindex_with_ci(model, dataset, duration_col, event_col, features, n_bootstraps=1000):  
    actual_times = dataset[duration_col].values  
    actual_events = dataset[event_col].values  
    
    original_risk_scores = model.predict_partial_hazard(dataset[features])  
    predicted_scores = -original_risk_scores  
    
    is_valid, message = check_dataset_validity(actual_times, actual_events)  
    if not is_valid:  
        return None, None, None, message, predicted_scores, original_risk_scores  
    
    try:  
        cindex = concordance_index(actual_times, predicted_scores, actual_events)  
    except ZeroDivisionError:  
        return None, None, None, "No comparable sample pairs", predicted_scores, original_risk_scores  
    except Exception as e:  
        return None, None, None, f"Error calculating C-index: {str(e)}", predicted_scores, original_risk_scores  
    
    cindices = []  
    n_samples = len(dataset)  
    
    n_failures = 0  
    for _ in range(n_bootstraps):  
        indices = np.random.choice(n_samples, n_samples, replace=True)  
        sample_times = actual_times[indices]  
        sample_events = actual_events[indices]  
        sample_scores = predicted_scores.iloc[indices].values if hasattr(predicted_scores, 'iloc') else predicted_scores[indices]  
        
        is_valid, _ = check_dataset_validity(sample_times, sample_events)  
        if not is_valid:  
            n_failures += 1  
            continue  
            
        try:  
            boot_cindex = concordance_index(sample_times, sample_scores, sample_events)  
            cindices.append(boot_cindex)  
        except:  
            n_failures += 1  
            continue  
    
    if n_failures > n_bootstraps/2:  
        warning = f"Warning: {n_failures}/{n_bootstraps} bootstrap samples failed, confidence intervals may be inaccurate"  
    else:  
        warning = None  
        
    if len(cindices) == 0:  
        return cindex, None, None, "Unable to calculate confidence intervals - all bootstrap samples failed", predicted_scores, original_risk_scores  
    
    alpha = 0.05  
    lower_ci = np.percentile(cindices, alpha/2 * 100)  
    upper_ci = np.percentile(cindices, (1 - alpha/2) * 100)  
    
    return cindex, lower_ci, upper_ci, warning, predicted_scores, original_risk_scores  

def get_dataset_stats(dataset, time_col, event_col):  
    n_samples = len(dataset)  
    n_events = sum(dataset[event_col])  
    n_censored = n_samples - n_events  
    median_time = np.median(dataset[time_col])  
    
    return {  
        "Samples": n_samples,  
        "Events": n_events,  
        "Censored": n_censored,  
        "Event Rate": f"{n_events/n_samples:.1%}",  
        "Median Time": f"{median_time:.1f}"  
    }  

def create_model_directories(base_dir, model_name):  
    model_dir = os.path.join(base_dir, model_name.replace('+', '_'))  
    os.makedirs(model_dir, exist_ok=True)  
    
    predictions_dir = os.path.join(model_dir, "predictions")  
    os.makedirs(predictions_dir, exist_ok=True)  
    
    return model_dir, predictions_dir  

def save_predictions(dataset, pred_scores, dataset_name, predictions_dir, id_col="ID", time_col="TIME", event_col="label", risk_scores=None):  
    pred_df = pd.DataFrame({  
        "ID": dataset[id_col],  
        "TIME": dataset[time_col],  
        "label": dataset[event_col],  
        "survival_score": pred_scores.values if hasattr(pred_scores, 'values') else pred_scores  
    })  
    
    if risk_scores is not None:  
        pred_df["risk_score"] = risk_scores.values if hasattr(risk_scores, 'values') else risk_scores  
    
    pred_file = os.path.join(predictions_dir, f"{dataset_name}_predictions.csv")  
    pred_df.to_csv(pred_file, index=False)  
    
    return pred_file  

def train_and_evaluate_model(combined_datasets, feature_types, base_dir=output_dir, id_col="ID", time_col="TIME", event_col="label"):  
    model_name = "+".join(feature_types)  
    print(f"\n================ Model: {model_name} ================")  
    
    model_dir, predictions_dir = create_model_directories(base_dir, model_name)  
    
    if "Train" not in combined_datasets:  
        print(f"Error: {model_name} model cannot be trained - training set doesn't exist")  
        return None, None  
    
    train_df = combined_datasets["Train"]  
    
    all_cols = [col for col in train_df.columns if col not in [id_col, time_col, event_col]]  
    
    feature_type_map = {  
        "Deep": ["risk_score"],  
        "Radiomics": [col for col in all_cols if col.startswith(("logarithm_", "original_"))],  
        "Clinical": [col for col in all_cols if col not in ["risk_score"] and not col.startswith(("logarithm_", "original_"))]  
    }  
    
    current_features = []  
    for ft in feature_types:  
        current_features.extend(feature_type_map[ft])  
    
    if not current_features:  
        print(f"Error: {model_name} model has no available features")  
        return None, None  
    
    print(f"Model using {len(current_features)} features:")  
    for ft in feature_types:  
        print(f"- {ft} features ({len(feature_type_map[ft])}): {', '.join(feature_type_map[ft])}")  
    
    cph = CoxPHFitter()  
    
    try:  
        train_data = train_df.dropna(subset=[time_col, event_col] + current_features)  
        if len(train_data) < len(train_df):  
            print(f"Warning: Removed {len(train_df) - len(train_data)} rows with missing values from training data")  
        
        cph.fit(train_data, duration_col=time_col, event_col=event_col, formula=" + ".join(current_features))  
        print("\nCox model coefficients:")  
        summary_df = cph.summary[["coef", "exp(coef)", "p"]]  
        
        summary_df["Feature Type"] = ""  
        for ft in feature_types:  
            for col in feature_type_map[ft]:  
                if col in summary_df.index:  
                    summary_df.loc[col, "Feature Type"] = ft  
        
        summary_df = summary_df.sort_values("p")  
        print(summary_df)  
        
        train_risk_scores = cph.predict_partial_hazard(train_data[current_features])  
        train_survival_scores = -train_risk_scores  
        save_predictions(train_data, train_survival_scores, "Train", predictions_dir,   
                         id_col, time_col, event_col, train_risk_scores)  
        
    except Exception as e:  
        print(f"Cox model training failed: {e}")  
        return None, None  
    
    print("\nC-index and 95% CI for each dataset:")  
    print("-" * 75)  
    print(f"{'Dataset':<12} {'Samples':<8} {'C-index':<10} {'95% CI':<20} {'Notes'}")  
    print("-" * 75)  
    
    results = {}  
    
    for name, dataset in combined_datasets.items():  
        missing_cols = []  
        for col in [time_col, event_col] + current_features:  
            if col not in dataset.columns:  
                missing_cols.append(col)  
        
        if missing_cols:  
            print(f"{name:<12} {'N/A':<8} {'N/A':<10} {'N/A':<20} Missing columns: {', '.join(missing_cols[:3])}...")  
            continue  
        
        valid_data = dataset.dropna(subset=[time_col, event_col] + current_features)  
        if len(valid_data) < len(dataset):  
            n_missing = len(dataset) - len(valid_data)  
            print(f"  Warning: {name} dataset removed {n_missing} rows with missing values")  
            if len(valid_data) == 0:  
                print(f"{name:<12} {'0':<8} {'N/A':<10} {'N/A':<20} No data after removing missing values")  
                continue  
        
        cindex, lower_ci, upper_ci, message, survival_scores, risk_scores = calculate_cindex_with_ci(  
            cph, valid_data, time_col, event_col, current_features  
        )  
        
        if name != "Train" and survival_scores is not None:  
            save_predictions(valid_data, survival_scores, name, predictions_dir,   
                            id_col, time_col, event_col, risk_scores)  
        
        stats = get_dataset_stats(valid_data, time_col, event_col)  
        
        if cindex is not None:  
            results[name] = {  
                "Samples": len(valid_data),  
                "Events": stats["Events"],  
                "Event Rate": stats["Event Rate"],  
                "C-index": cindex,  
                "Lower": lower_ci if lower_ci is not None else np.nan,  
                "Upper": upper_ci if upper_ci is not None else np.nan,  
                "Notes": message if message else ""  
            }  
            
            ci_str = f"({lower_ci:.3f} - {upper_ci:.3f})" if lower_ci is not None else "N/A"  
            print(f"{name:<12} {len(valid_data):<8} {cindex:.3f}     {ci_str:<20} {message if message else ''}")  
        else:  
            results[name] = {  
                "Samples": len(valid_data),  
                "Events": stats["Events"],  
                "Event Rate": stats["Event Rate"],  
                "C-index": np.nan,  
                "Lower": np.nan,  
                "Upper": np.nan,  
                "Notes": message  
            }  
            print(f"{name:<12} {len(valid_data):<8} {'N/A':<10} {'N/A':<20} {message}")  
    
    print("-" * 75)  
    
    result_df = pd.DataFrame(results).T  
    result_df.index.name = "Dataset"  
    result_file = os.path.join(model_dir, "performance_results.csv")  
    result_df.to_csv(result_file)  
    print(f"\nModel performance results saved to {result_file}")  
    
    coef_file = os.path.join(model_dir, "model_coefficients.csv")  
    summary_df.to_csv(coef_file)  
    print(f"Model coefficients saved to {coef_file}")  
    
    return result_df, summary_df  

def main():  
    print("\n================ Loading Datasets ================")  
    radiomics_datasets = load_datasets(radiomics_files, radiomics_dir, "Radiomics")  
    clinical_datasets = load_datasets(clinical_files, clinical_dir, "Clinical")  
    deep_datasets = load_datasets(deep_files, deep_dir, "Deep")  
    
    feature_types = ["Deep", "Radiomics", "Clinical"]  
    
    all_combinations = []  
    for ft in feature_types:  
        all_combinations.append([ft])  
    for combo in combinations(feature_types, 2):  
        all_combinations.append(list(combo))  
    all_combinations.append(feature_types)  
    
    all_model_results = {}  
    
    for feature_combo in all_combinations:  
        model_name = "+".join(feature_combo)  
        print(f"\n=================================================================")  
        print(f"Building model: {model_name}")  
        print(f"=================================================================")  
        
        current_datasets = {}  
        
        dataset_map = {  
            "Deep": deep_datasets,  
            "Radiomics": radiomics_datasets,  
            "Clinical": clinical_datasets  
        }  
        
        datasets_to_merge = {ft: dataset_map[ft] for ft in feature_combo if ft in dataset_map}  
        
        for name in dataset_names:  
            datasets_present = True  
            for ft in feature_combo:  
                if name not in dataset_map[ft]:  
                    datasets_present = False  
                    break  
            
            if not datasets_present:  
                print(f"{name} dataset incomplete, skipping")  
                continue  
            
            print(f"\nMerging {name} dataset...")  
            current_data_dict = {ft: dataset_map[ft][name] for ft in feature_combo}  
            current_datasets[name] = merge_datasets(current_data_dict)  
        
        results_df, _ = train_and_evaluate_model(current_datasets, feature_combo)  
        
        if results_df is not None:  
            all_model_results[model_name] = results_df  
    
    if all_model_results:  
        print("\n================ Generating Comprehensive Performance Report ================")  
        
        summary_table = pd.DataFrame(index=dataset_names)  
        
        for model_name, results in all_model_results.items():  
            c_indices = []  
            for dataset in dataset_names:  
                if dataset in results.index:  
                    c_index = results.loc[dataset, "C-index"]  
                    lower_ci = results.loc[dataset, "Lower"]  
                    upper_ci = results.loc[dataset, "Upper"]  
                    
                    if not np.isnan(c_index):  
                        if not np.isnan(lower_ci) and not np.isnan(upper_ci):  
                            c_indices.append(f"{c_index:.3f} ({lower_ci:.3f}-{upper_ci:.3f})")  
                        else:  
                            c_indices.append(f"{c_index:.3f}")  
                    else:  
                        c_indices.append("N/A")  
                else:  
                    c_indices.append("N/A")  
            
            summary_table[model_name] = c_indices  
        
        summary_file = os.path.join(output_dir, "cox_models_summary.csv")  
        summary_table.to_csv(summary_file)  
        print(f"Comprehensive performance report saved to {summary_file}")  

if __name__ == "__main__":  
    main()  
