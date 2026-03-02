# tests.py
import os
import pickle
import pandas as pd
import numpy as np
from scipy import stats
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
            'Test MAE ($)': r['test_metrics']['mae'],
            'Test RMSE': r['test_metrics']['rmse'],
            'CV MAE (Avg)': r['cv_metrics']['mae_mean'],
            'Features': len(r['features'])
        })
    
    df_summary = pd.DataFrame(summary_data).sort_values(by='Test MAE ($)', ascending=True)
    print(" PERFORMANCE RANKING (Sorted by Test MAE):")
    print(df_summary.to_string(index=False))
    print("-" * 80)

    # ---------------------------------------------------------
    # 2. Statistical Test (Champion vs Challengers)
    # ---------------------------------------------------------
    if len(reports) < 2:
        print(" At least 2 reports are needed for statistical comparison.")
        return

    # Identify the champion (lowest Test MAE)
    best_model = min(reports, key=lambda x: x['test_metrics']['mae'])
    
    print(f"\n  HYPOTHESIS TEST (Wilcoxon Signed-Rank)")
    print(f"   Champion Model: [{best_model['tag']}]")
    print("-" * 80)
    print(f"{'Challenger':<25} | {'Diff MAE':<10} | {'P-Value':<12} | {'Significant?'}")
    
    # Calculate absolute errors for the champion
    errors_champion = np.abs(best_model['y_test_true'] - best_model['y_test_pred'])

    for r in reports:
        if r['tag'] == best_model['tag']:
            continue
            
        # Calculate absolute errors for the challenger
        errors_challenger = np.abs(r['y_test_true'] - r['y_test_pred'])
        
        # Check integrity
        if len(errors_champion) != len(errors_challenger):
            print(f"{r['tag']:<25} | Error: Array length mismatch (check random_state)")
            continue

        # Wilcoxon Test
        # Null Hypothesis: The distribution of differences is symmetric about zero (no significant difference)
        # Alternative 'less': The champion errors are LESS than challenger errors
        try:
            stat, p_value = stats.wilcoxon(errors_champion, errors_challenger, alternative='less')
            
            is_significant = "YES" if p_value < 0.05 else "NO!"
            diff_mae = r['test_metrics']['mae'] - best_model['test_metrics']['mae'] 
            # Note: diff_mae > 0 means Challenger error is higher (Champion is better)
            
            print(f"{r['tag']:<25} | {diff_mae:+.2f}     | {p_value:.2e}     | {is_significant}")
        except Exception as e:
            print(f"{r['tag']:<25} | Error in statistical test: {e}")

if __name__ == "__main__":
    run_statistical_comparison(config.REPORTS_DIR)
