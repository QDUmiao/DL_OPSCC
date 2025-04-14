import os  
import pandas as pd  
import numpy as np  
from scipy import stats  

def calculate_cindex(times, events, predictions):  
    concordant = 0  
    total = 0  
    
    for i in range(len(times)):  
        if events[i] == 1:  
            for j in range(len(times)):  
                if (times[j] > times[i]) or (times[j] == times[i] and events[j] == 0):  
                    total += 1  
                    if predictions[i] > predictions[j]:  
                        concordant += 1  
    
    return concordant / total if total > 0 else 0  

def delong_test(cindex1, n1, cindex2, n2):  
    se1 = np.sqrt((1 - cindex1) / n1)  
    se2 = np.sqrt((1 - cindex2) / n2)  
    
    z = (cindex1 - cindex2) / np.sqrt(se1**2 + se2**2)  
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))  
    
    return p_value  

base_path = r""  

model_folders = ["Clinical", "Deep", "Deep_Clinical", "Deep_Radiomics", "Deep_Radiomics_Clinical", "Radiomics", "Radiomics_Clinical"]  

results = []  

for model in model_folders:  
    model_path = os.path.join(base_path, model)  
    predictions_path = os.path.join(model_path, "predictions")  
    
    if not os.path.exists(predictions_path):  
        print(f"Predictions folder does not exist: {predictions_path}")  
        continue  
    
    hpv0_file = os.path.join(predictions_path, "HPV0_predictions.csv")  
    hpv1_file = os.path.join(predictions_path, "HPV1_predictions.csv")  
    
    if not (os.path.exists(hpv0_file) and os.path.exists(hpv1_file)):  
        print(f"HPV prediction files do not exist: {model}")  
        continue  
    
    hpv0_df = pd.read_csv(hpv0_file)  
    hpv1_df = pd.read_csv(hpv1_file)  
    
    hpv0_times = hpv0_df['TIME'].values  
    hpv0_events = hpv0_df['label'].values  
    hpv0_risk = hpv0_df['risk_score'].values  
    
    hpv1_times = hpv1_df['TIME'].values  
    hpv1_events = hpv1_df['label'].values  
    hpv1_risk = hpv1_df['risk_score'].values  
    
    cindex_hpv0 = calculate_cindex(hpv0_times, hpv0_events, hpv0_risk)  
    cindex_hpv1 = calculate_cindex(hpv1_times, hpv1_events, hpv1_risk)  
    
    n_events_hpv0 = sum(hpv0_events)  
    n_events_hpv1 = sum(hpv1_events)  
    
    p_value = delong_test(cindex_hpv0, n_events_hpv0, cindex_hpv1, n_events_hpv1)  
    
    results.append({  
        "Model": model,  
        "HPV0 C-index": round(cindex_hpv0, 3),  
        "HPV1 C-index": round(cindex_hpv1, 3),  
        "HPV0 Events": n_events_hpv0,  
        "HPV1 Events": n_events_hpv1,   
        "p-value": round(p_value, 4),  
        "Significance": "Significant" if p_value < 0.05 else "Not significant"  
    })  

results_df = pd.DataFrame(results)  
results_df = results_df.sort_values(by="p-value")  

print("\nDeLong test results for HPV0 and HPV1 group C-indices:")  
print(results_df.to_string(index=False))  

results_df.to_csv("HPV_subgroup_analysis.csv", index=False)  
print("\nResults saved to HPV_subgroup_analysis.csv")  
