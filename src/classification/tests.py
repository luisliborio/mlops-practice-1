# tests.py
import os
import pickle
import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
import config

def load_reports(reports_dir):
    reports = []
    if not os.path.exists(reports_dir):
        print(f"Directory {reports_dir} not found.")
        return []
        
    for filename in os.listdir(reports_dir):
        if filename.endswith(".pkl"):
            filepath = os.path.join(reports_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    reports.append(pickle.load(f))
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return reports

def run_statistical_comparison(reports_dir):
    reports = load_reports(reports_dir)
    if not reports:
        print("No reports found to compare.")
        return

    print(f" Found {len(reports)} reports in '{reports_dir}'\n")

    # ---------------------------------------------------------
    # 1. General Performance Table
    # ---------------------------------------------------------
    summary_data = []
    for r in reports:
        summary_data.append({
            'Tag': r['tag'],
            'Test F1 (W)': r['test_metrics']['f1'],
            'Test Acc': r['test_metrics']['acc'],
            'Features': len(r['features'])
        })
    
    df_summary = pd.DataFrame(summary_data).sort_values(by='Test F1 (W)', ascending=False)
    print(" PERFORMANCE RANKING (Sorted by F1 Score):")
    print(df_summary.to_string(index=False))
    print("-" * 80)

    # ---------------------------------------------------------
    # 2. Champion Details
    # ---------------------------------------------------------
    best_model = max(reports, key=lambda x: x['test_metrics']['f1'])
    
    print(f"\n CHAMPION MODEL DETAILS: [{best_model['tag']}]")
    print("Confusion Matrix (Rows=True, Cols=Pred):")
    print(f"Classes: {best_model.get('class_map', 'N/A')}")
    print(best_model['test_metrics']['conf_matrix'])
    print("-" * 80)

    # ---------------------------------------------------------
    # 3. Statistical Test (McNemar's)
    # ---------------------------------------------------------
    if len(reports) < 2:
        return

    print(f" HYPOTHESIS TEST (McNemar's Test)")
    print(f"   Champion: [{best_model['tag']}]")
    print("-" * 80)
    print(f"{'Challenger':<25} | {'Diff F1':<10} | {'P-Value':<12} | {'Significant?'}")
    
    # Boolean array: True if prediction was correct
    champ_correct = (best_model['y_test_pred'] == best_model['y_test_true'])

    for r in reports:
        if r['tag'] == best_model['tag']:
            continue
            
        chall_correct = (r['y_test_pred'] == r['y_test_true'])
        
        # Build 2x2 Contingency Table for McNemar
        # [Both Correct, Champion Correct/Chall Wrong]
        # [Chall Correct/Champ Wrong, Both Wrong]
        b = np.sum(champ_correct & ~chall_correct) # Champ wins
        c = np.sum(~champ_correct & chall_correct) # Chall wins
        
        # We only need b and c for the calculation
        table = [[0, b], [c, 0]]
        
        try:
            result = mcnemar(table, exact=False, correction=True)
            p_value = result.pvalue
            
            is_significant = "YES" if p_value < 0.05 else "NO"
            diff_f1 = best_model['test_metrics']['f1'] - r['test_metrics']['f1']
            
            print(f"{r['tag']:<25} | {diff_f1:+.4f}     | {p_value:.2e}     | {is_significant}")
        except Exception as e:
            print(f"{r['tag']:<25} | Error: {e}")

if __name__ == "__main__":
    run_statistical_comparison(config.REPORTS_DIR)