"""
Interactive Visualization Script for Customer Churn Prediction Results
Displays all charts in separate pop-up windows simultaneously
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results():
    """Load prediction results and evaluation metrics"""

    # Load predictions
    predictions_df = pd.read_csv('data/predictions.csv')

    # Load summary
    with open('data/predictions_summary.json', 'r') as f:
        summary = json.load(f)

    # Load evaluation results
    try:
        with open('logs/evaluation_results.json', 'r') as f:
            evaluation = json.load(f)
    except:
        evaluation = None

    return predictions_df, summary, evaluation


def visualize_churn_distribution(predictions_df, summary):
    """Visualize churn vs retention distribution"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('CHURN vs RETENTION ANALYSIS',
                 fontsize=16, weight='bold', y=0.98)

    # Pie chart
    churn_counts = [summary['predicted_churn'], summary['predicted_retention']]
    labels = [f"Churn\n({churn_counts[0]})", f"Retention\n({churn_counts[1]})"]
    colors = ['#ff6b6b', '#51cf66']

    axes[0].pie(churn_counts, labels=labels, autopct='%1.1f%%', colors=colors,
                startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
    axes[0].set_title('Predicted Churn vs Retention (Pie)',
                      fontsize=13, weight='bold', pad=20)

    # Bar chart
    categories = ['Churn', 'Retention']
    values = churn_counts
    bars = axes[1].bar(categories, values, color=colors,
                       alpha=0.8, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(val)}\n({val/sum(values)*100:.1f}%)',
                     ha='center', va='bottom', fontsize=12, weight='bold')

    axes[1].set_ylabel('Number of Customers', fontsize=12, weight='bold')
    axes[1].set_title('Predicted Churn vs Retention (Bar)',
                      fontsize=13, weight='bold', pad=20)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    # REMOVED plt.show() here to prevent blocking


def visualize_risk_levels(predictions_df):
    """Visualize risk level breakdown"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('RISK LEVEL ANALYSIS', fontsize=16, weight='bold', y=0.98)

    if 'risk_level' in predictions_df.columns:
        risk_counts = predictions_df['risk_level'].value_counts()

        # Ensure all risk levels exist
        risk_order = ['High Risk', 'Medium Risk', 'Low Risk']
        risk_counts = risk_counts.reindex(risk_order, fill_value=0)

        colors_risk = {'High Risk': '#ff6b6b',
                       'Medium Risk': '#ffd93d', 'Low Risk': '#51cf66'}
        bar_colors = [colors_risk.get(risk, '#999')
                      for risk in risk_counts.index]

        # Bar chart
        bars = axes[0].barh(risk_counts.index, risk_counts.values, color=bar_colors,
                            alpha=0.8, edgecolor='black', linewidth=2)

        # Add value labels
        for bar, val in zip(bars, risk_counts.values):
            width = bar.get_width()
            axes[0].text(width, bar.get_y() + bar.get_height()/2.,
                         f' {int(val)} ({val/risk_counts.sum()*100:.1f}%)',
                         ha='left', va='center', fontsize=11, weight='bold')

        axes[0].set_xlabel('Number of Customers', fontsize=12, weight='bold')
        axes[0].set_title('Risk Level Distribution (Horizontal Bar)',
                          fontsize=13, weight='bold', pad=20)
        axes[0].grid(axis='x', alpha=0.3)

        # Pie chart
        axes[1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                    colors=bar_colors, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
        axes[1].set_title('Risk Level Breakdown (Pie)',
                          fontsize=13, weight='bold', pad=20)

    plt.tight_layout()
    # REMOVED plt.show() here to prevent blocking


def visualize_churn_probability(predictions_df):
    """Visualize churn probability distribution"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('CHURN PROBABILITY ANALYSIS',
                 fontsize=16, weight='bold', y=0.98)

    if 'churn_probability' in predictions_df.columns:
        probs = predictions_df['churn_probability']

        # Histogram
        axes[0].hist(probs, bins=50, color='#4ecdc4', alpha=0.7,
                     edgecolor='black', linewidth=1.5)
        axes[0].axvline(probs.mean(), color='red', linestyle='--',
                        linewidth=2.5, label=f'Mean: {probs.mean():.3f}')
        axes[0].axvline(0.5, color='orange', linestyle='--',
                        linewidth=2.5, label='Decision Threshold: 0.5')
        axes[0].set_xlabel('Churn Probability', fontsize=12, weight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, weight='bold')
        axes[0].set_title('Probability Distribution (Histogram)',
                          fontsize=13, weight='bold', pad=20)
        axes[0].legend(fontsize=11, loc='upper right')
        axes[0].grid(alpha=0.3)

        # Box plot by prediction
        churn_probs = probs[predictions_df['predicted_churn'] == 'Will Churn']
        retention_probs = probs[predictions_df['predicted_churn']
                                == 'Will Not Churn']

        box_data = [churn_probs, retention_probs]
        bp = axes[1].boxplot(box_data, labels=['Predicted Churn', 'Predicted Retention'],
                             patch_artist=True, notch=True)

        for patch, color in zip(bp['boxes'], ['#ff6b6b', '#51cf66']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[1].set_ylabel('Churn Probability', fontsize=12, weight='bold')
        axes[1].set_title('Probability by Prediction (Box Plot)',
                          fontsize=13, weight='bold', pad=20)
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].axhline(0.5, color='orange', linestyle='--',
                        linewidth=2, alpha=0.7)

    plt.tight_layout()
    # REMOVED plt.show() here to prevent blocking


def visualize_model_performance(evaluation):
    """Visualize model performance metrics"""

    if evaluation is None:
        print("⚠ Evaluation data not found, skipping model performance visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MODEL PERFORMANCE METRICS',
                 fontsize=16, weight='bold', y=0.995)

    # Extract metrics
    models = list(evaluation.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    colors_models = ['#4ecdc4', '#ff6b6b', '#95e1d3']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        values = [evaluation[model].get(metric, 0) for model in models]
        bars = ax.bar(models, values, color=colors_models[:len(models)],
                      alpha=0.8, edgecolor='black', linewidth=2)

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=11, weight='bold')

        ax.set_ylabel('Score', fontsize=11, weight='bold')
        ax.set_title(f'{metric.upper()}', fontsize=12, weight='bold', pad=15)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)

        # Rotate x labels
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    # REMOVED plt.show() here to prevent blocking


def visualize_summary_stats(summary):
    """Visualize summary statistics"""

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('PREDICTION SUMMARY STATISTICS',
                 fontsize=16, weight='bold', y=0.98)
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    # Key metrics
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')

    metrics_text = f"""
    TOTAL CUSTOMERS ANALYZED: {summary['total_customers']:,}
    
    PREDICTED CHURN: {summary['predicted_churn']} customers ({summary['churn_rate']*100:.2f}%)
    PREDICTED RETENTION: {summary['predicted_retention']} customers ({(1-summary['churn_rate'])*100:.2f}%)
    AT-RISK CUSTOMERS: {summary['at_risk_customers']} customers ({summary['at_risk_rate']*100:.2f}%)
    
    MODEL ACCURACY: {summary.get('prediction_accuracy', 0):.2%}
    """

    ax1.text(0.5, 0.5, metrics_text, fontsize=12, weight='bold',
             ha='center', va='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.8, pad=1.5, linewidth=2))

    # Churn rate gauge
    ax2 = fig.add_subplot(gs[1, 0])
    churn_rate = summary['churn_rate'] * 100
    categories = ['Churn', 'Retention']
    values = [churn_rate, 100-churn_rate]
    colors_gauge = ['#ff6b6b', '#51cf66']

    wedges, texts, autotexts = ax2.pie(values, labels=categories, autopct='%1.1f%%',
                                       colors=colors_gauge, startangle=90,
                                       textprops={'fontsize': 11, 'weight': 'bold'})
    ax2.set_title('Churn Rate', fontsize=12, weight='bold', pad=15)

    # At-risk rate gauge
    ax3 = fig.add_subplot(gs[1, 1])
    at_risk_rate = summary['at_risk_rate'] * 100
    values_risk = [at_risk_rate, 100-at_risk_rate]
    colors_risk_gauge = ['#ffd93d', '#51cf66']

    wedges, texts, autotexts = ax3.pie(values_risk, labels=['At-Risk', 'Safe'],
                                       autopct='%1.1f%%',
                                       colors=colors_risk_gauge, startangle=90,
                                       textprops={'fontsize': 11, 'weight': 'bold'})
    ax3.set_title('At-Risk Rate', fontsize=12, weight='bold', pad=15)

    # Comparison table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    table_data = [
        ['Metric', 'Count', 'Percentage'],
        ['Total Customers', f"{summary['total_customers']:,}", '100%'],
        ['Predicted Churn', f"{summary['predicted_churn']:,}",
            f"{summary['churn_rate']*100:.2f}%"],
        ['Predicted Retention', f"{summary['predicted_retention']:,}",
            f"{(1-summary['churn_rate'])*100:.2f}%"],
        ['At-Risk Customers', f"{summary['at_risk_customers']:,}",
            f"{summary['at_risk_rate']*100:.2f}%"],
    ]

    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.3, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4ecdc4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            table[(i, j)].set_text_props(weight='bold', fontsize=11)

    plt.tight_layout()
    # REMOVED plt.show() here to prevent blocking


def create_dashboard(predictions_df, summary, evaluation):
    """Create a comprehensive interactive dashboard"""

    fig = plt.figure(figsize=(20, 13))
    fig.suptitle('CUSTOMER CHURN PREDICTION - COMPREHENSIVE DASHBOARD',
                 fontsize=18, weight='bold', y=0.995)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # 1. Churn Distribution (pie)
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['#ff6b6b', '#51cf66']
    wedges, texts, autotexts = ax1.pie(
        [summary['predicted_churn'], summary['predicted_retention']],
        labels=['Churn', 'Retention'],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 10, 'weight': 'bold'}
    )
    ax1.set_title('Churn Distribution', fontsize=12, weight='bold', pad=10)

    # 2. Risk Levels
    ax2 = fig.add_subplot(gs[0, 1])
    if 'risk_level' in predictions_df.columns:
        risk_counts = predictions_df['risk_level'].value_counts()
        risk_order = ['High Risk', 'Medium Risk', 'Low Risk']
        risk_counts = risk_counts.reindex(risk_order, fill_value=0)
        colors_risk = {'High Risk': '#ff6b6b',
                       'Medium Risk': '#ffd93d', 'Low Risk': '#51cf66'}
        bar_colors = [colors_risk.get(risk, '#999')
                      for risk in risk_counts.index]
        ax2.barh(risk_counts.index, risk_counts.values,
                 color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Count', fontsize=10, weight='bold')
        ax2.set_title('Risk Level Breakdown',
                      fontsize=12, weight='bold', pad=10)
        ax2.grid(axis='x', alpha=0.3)

    # 3. Probability Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    if 'churn_probability' in predictions_df.columns:
        ax3.hist(predictions_df['churn_probability'], bins=40, color='#4ecdc4',
                 alpha=0.7, edgecolor='black', linewidth=1)
        ax3.axvline(0.5, color='orange', linestyle='--',
                    linewidth=2.5, label='Threshold')
        ax3.set_xlabel('Churn Probability', fontsize=10, weight='bold')
        ax3.set_ylabel('Frequency', fontsize=10, weight='bold')
        ax3.set_title('Probability Distribution',
                      fontsize=12, weight='bold', pad=10)
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)

    # 4-7. Model Performance Metrics
    if evaluation:
        models = list(evaluation.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        colors_models = ['#4ecdc4', '#ff6b6b', '#95e1d3']

        positions = [(1, 0), (1, 1), (1, 2), (2, 0)]

        for idx, (metric, pos) in enumerate(zip(metrics, positions)):
            ax = fig.add_subplot(gs[pos[0], pos[1]])
            values = [evaluation[model].get(metric, 0) for model in models]
            bars = ax.bar(range(len(models)), values, color=colors_models[:len(models)],
                          alpha=0.8, edgecolor='black', linewidth=1.5)

            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9, weight='bold')

            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, fontsize=9, rotation=15)
            ax.set_ylabel('Score', fontsize=10, weight='bold')
            ax.set_title(metric.upper(), fontsize=11, weight='bold', pad=10)
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3)

    # 8. Summary Stats (text)
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')

    summary_text = f"""
    TOTAL: {summary['total_customers']:,} CUSTOMERS  |  CHURN: {summary['predicted_churn']} ({summary['churn_rate']*100:.1f}%)  |  
    RETENTION: {summary['predicted_retention']} ({(1-summary['churn_rate'])*100:.1f}%)  |  
    AT-RISK: {summary['at_risk_customers']}  |  ACCURACY: {summary.get('prediction_accuracy', 0):.1%}
    """

    ax8.text(0.5, 0.5, summary_text, fontsize=11, weight='bold', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6, pad=1.2, linewidth=2))

    plt.tight_layout()
    # REMOVED plt.show() here to prevent blocking


def main():
    """Main visualization pipeline"""

    print("\n" + "="*70)
    print("CUSTOMER CHURN PREDICTION - INTERACTIVE VISUALIZATION")
    print("="*70 + "\n")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    # Load results
    print("📊 Loading prediction results...")
    predictions_df, summary, evaluation = load_results()
    print(f"✓ Loaded {len(predictions_df)} predictions\n")

    # Generate visualizations
    print("📈 Rendering all visualization windows simultaneously...\n")

    print("→ Building: CHURN vs RETENTION ANALYSIS")
    visualize_churn_distribution(predictions_df, summary)

    print("→ Building: RISK LEVEL ANALYSIS")
    visualize_risk_levels(predictions_df)

    print("→ Building: CHURN PROBABILITY ANALYSIS")
    visualize_churn_probability(predictions_df)

    print("→ Building: MODEL PERFORMANCE METRICS")
    visualize_model_performance(evaluation)

    print("→ Building: PREDICTION SUMMARY STATISTICS")
    visualize_summary_stats(summary)

    print("→ Building: COMPREHENSIVE DASHBOARD")
    create_dashboard(predictions_df, summary, evaluation)

    # This single call launches ALL generated figures as individual windows simultaneously
    print("\n🚀 Opening all windows now...")
    plt.show()

    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
