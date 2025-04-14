import pandas as pd  
import numpy as np  
from lifelines.utils import concordance_index  
from scipy import stats  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.metrics import roc_curve, auc  
import warnings  
warnings.filterwarnings('ignore')  

def calculate_ci_cindex(times, scores, events, n_iterations=1000):  
    c_indices = []  
     
    base_cindex = concordance_index(times, -scores, events)  
    
    n_samples = len(times)  
    for _ in range(n_iterations):  
        indices = np.random.randint(0, n_samples, n_samples)  
        sample_times = times[indices]  
        sample_scores = scores[indices]  
        sample_events = events[indices]  
        
        try:  
            c_index = concordance_index(sample_times, -sample_scores, sample_events)  
            c_indices.append(c_index)  
        except:  
            continue  
    
    ci_lower = np.percentile(c_indices, 2.5)  
    ci_upper = np.percentile(c_indices, 97.5)  
    
    return base_cindex, ci_lower, ci_upper  

def analyze_dataset(file_path, dataset_name):  
    df = pd.read_csv(file_path)  
    
    valid_mask = (df['label'] != -1) & (df['TIME'] > 0)  
    df_valid = df[valid_mask].copy()  
    
    if len(df_valid) == 0:  
        return None  

    times = df_valid['TIME'].values  
    events = df_valid['label'].values  
    scores = df_valid['risk_score'].values  
  
    cindex, ci_lower, ci_upper = calculate_ci_cindex(times, scores, events)  
    
    results = {  
        'dataset': dataset_name,  
        'n_samples': len(df_valid),  
        'n_events': sum(events),  
        'cindex': cindex,  
        'ci_lower': ci_lower,  
        'ci_upper': ci_upper,  
        'median_followup': np.median(times),  
        'data': df_valid  
    }  
    
    
    return results  

def plot_risk_distribution(results, save_path='risk_distribution.png'):  
    plt.figure(figsize=(12, 8))  
    
    for result in results:  
        if result is None:  
            continue  
            
        data = result['data']  
        sns.kdeplot(data=data, x='risk_score', hue='label',   
                   label=result['dataset'], common_norm=False)  
    
    plt.title('Risk Score Distribution by Dataset')  
    plt.xlabel('Risk Score')  
    plt.ylabel('Density')  
    plt.legend()  
    plt.savefig(save_path)  
    plt.close()  

def main():  
    files = {  
    }  
    
    results = []  
    for dataset_name, file_path in files.items():  
        try:  
            result = analyze_dataset(file_path, dataset_name)  
            if result is not None:  
                results.append(result)  
    
    plot_risk_distribution(results)  

    summary_data = []  
    for result in results:  
        if result is not None:  
            summary_data.append({  
                'Dataset': result['dataset'],  
                'N': result['n_samples'],  
                'Events': result['n_events'],  
                'C-index (95% CI)': f"{result['cindex']:.3f} ({result['ci_lower']:.3f}-{result['ci_upper']:.3f})",  
                'Median Follow-up': f"{result['median_followup']:.1f}"  
            })  
    
    summary_df = pd.DataFrame(summary_data)  
    print(summary_df.to_string(index=False))  
    
    summary_df.to_csv('deepsurv_analysis_summary.csv', index=False)  

if __name__ == "__main__":  
    main()  
