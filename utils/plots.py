import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def standard_plots(train, score_col, method_name, output_path='./outputs'):
    """
    4 standard plots for comparing methods.
    
    Parameters:
    -----------
    train : DataFrame with ['unit_id', 'time_cycles', 'RUL', score_col, 'is_anomaly']
    score_col : score column name (e.g., 'anomaly_score', 'reconstruction_error', 'z_score')
    method_name : str for title
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Score distribution
    axes[0, 0].hist(train[train['is_anomaly']==0][score_col], 
                    bins=50, alpha=0.7, label='Normal', edgecolor='black')
    axes[0, 0].hist(train[train['is_anomaly']==1][score_col], 
                    bins=50, alpha=0.7, label='Anomaly', edgecolor='black')
    axes[0, 0].set_xlabel('Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'{method_name} - Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Temporal evolution (Unit 1)
    unit_1 = train[train['unit_id'] == 1].sort_values('time_cycles')
    axes[0, 1].plot(unit_1['time_cycles'], unit_1[score_col], label='Score', linewidth=1.5)
    anomalies = unit_1[unit_1['is_anomaly'] == 1]
    axes[0, 1].scatter(anomalies['time_cycles'], anomalies[score_col], 
                       color='red', s=50, label='Anomalies', zorder=3)
    axes[0, 1].set_xlabel('Cycles')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Unit 1 - Temporal Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Score vs RUL
    axes[1, 0].scatter(train['RUL'], train[score_col], alpha=0.3, s=10)
    axes[1, 0].set_xlabel('RUL (Remaining Useful Life)')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Score vs RUL')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Anomaly rate by RUL bin
    rul_bins = pd.cut(train['RUL'], bins=10)
    anomaly_rate = train.groupby(rul_bins, observed=True)['is_anomaly'].mean()
    bin_labels = [f"{int(interval.left)}-{int(interval.right)}" 
                  for interval in anomaly_rate.index]
    axes[1, 1].bar(range(len(anomaly_rate)), anomaly_rate.values)
    axes[1, 1].set_xticks(range(len(anomaly_rate)))
    axes[1, 1].set_xticklabels(bin_labels, rotation=45, ha='right')
    axes[1, 1].set_xlabel('RUL Bins')
    axes[1, 1].set_ylabel('Anomaly Rate')
    axes[1, 1].set_title('Anomaly Rate by RUL Range')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/{method_name.lower().replace(" ", "_")}_analysis.png', 
                dpi=150, bbox_inches='tight')
    plt.show()