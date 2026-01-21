"""
RFM Customer Segmentation Analysis Script
Generates rfm_results.csv and rfm_summary.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

print("=" * 60)
print("RFM CUSTOMER SEGMENTATION ANALYSIS")
print("=" * 60)

# Load the datasets
print("\n1. Loading data...")
loyal_df = pd.read_csv('../data/loyal_customers_processed.csv')
new_df = pd.read_csv('../data/new_customers_processed.csv')

print(f"   Loyal customers dataset: {loyal_df.shape[0]:,} rows")
print(f"   New customers dataset: {new_df.shape[0]:,} rows")

# Combine datasets
combined_df = pd.concat([loyal_df, new_df], ignore_index=True)

# Convert datetime column
combined_df['ticket_datetime'] = pd.to_datetime(combined_df['ticket_datetime'])
combined_df['date'] = pd.to_datetime(combined_df['date'])

print(f"\n   Combined dataset: {combined_df.shape[0]:,} rows")
print(f"   Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
print(f"   Unique users: {combined_df['user_id'].nunique()}")
print(f"     - Loyal: {combined_df[combined_df['user_type'] == 'loyal']['user_id'].nunique()}")
print(f"     - New: {combined_df[combined_df['user_type'] == 'new']['user_id'].nunique()}")

# Calculate RFM Metrics
print("\n2. Calculating RFM metrics...")

# Define the analysis date (most recent date in the dataset + 1 day)
analysis_date = combined_df['date'].max() + pd.Timedelta(days=1)
print(f"   Analysis date: {analysis_date}")

# Create a user-level summary
# First, get unique ticket information (to avoid counting same ticket multiple times)
ticket_level = combined_df.groupby(['user_id', 'ticket_number', 'user_type']).agg({
    'ticket_datetime': 'first',
    'ticket_amount': 'first',
    'date': 'first'
}).reset_index()

print(f"   Unique tickets: {ticket_level.shape[0]:,}")

# Calculate RFM metrics per customer
rfm = ticket_level.groupby(['user_id', 'user_type']).agg({
    'date': lambda x: (analysis_date - x.max()).days,  # Recency
    'ticket_number': 'count',  # Frequency
    'ticket_amount': 'sum'  # Monetary
}).reset_index()

# Rename columns
rfm.columns = ['user_id', 'user_type', 'recency_days', 'frequency', 'monetary']

print(f"   RFM table created for {rfm.shape[0]} customers")

# RFM Statistics
print("\n   RFM Metrics Summary:")
print(f"     Recency: min={rfm['recency_days'].min()}, max={rfm['recency_days'].max()}, mean={rfm['recency_days'].mean():.1f}")
print(f"     Frequency: min={rfm['frequency'].min()}, max={rfm['frequency'].max()}, mean={rfm['frequency'].mean():.1f}")
print(f"     Monetary: min=${rfm['monetary'].min():.2f}, max=${rfm['monetary'].max():.2f}, mean=${rfm['monetary'].mean():.2f}")

# Create RFM scores using quintiles
print("\n3. Assigning RFM scores...")

# For Recency, lower is better, so we reverse the scoring
rfm['r_score'] = pd.qcut(rfm['recency_days'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(int)
rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)
rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)

# Create combined RFM score
rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)

print("   RFM Scores assigned!")

# Define segment assignment function
def assign_segment(row):
    r, f, m = row['r_score'], row['f_score'], row['m_score']

    # Champion: Recent + Frequent + High spender (R: 4-5, F: 4-5, M: 4-5)
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champion'

    # Loyalist: Frequent buyer, good engagement (R: 3-5, F: 4-5, M: 3-5)
    elif r >= 3 and f >= 4 and m >= 3:
        return 'Loyalist'

    # Newbie: Very recent, first-time buyer (R: 5, F: 1, M: 1-3)
    elif r == 5 and f == 1 and m <= 3:
        return 'Newbie'

    # Emerging: Recent, growing potential (R: 4-5, F: 2-3, M: 2-4)
    elif r >= 4 and f >= 2 and f <= 3 and m >= 2 and m <= 4:
        return 'Emerging'

    # At Risk: Was good, becoming inactive (R: 2-3, F: 3-5, M: 3-5)
    elif r >= 2 and r <= 3 and f >= 3 and m >= 3:
        return 'At Risk'

    # Hibernating: Long time since last visit (R: 1-2, F: 1-3, M: 1-5)
    elif r <= 2 and f <= 3:
        return 'Hibernating'

    # Default cases
    else:
        # Additional logic for edge cases
        if r >= 4:  # Recent but doesn't fit other categories
            return 'Emerging'
        elif r <= 2:  # Not recent
            return 'Hibernating'
        else:
            return 'At Risk'

# Apply segmentation
print("\n4. Assigning customer segments...")
rfm['segment'] = rfm.apply(assign_segment, axis=1)

# Add segment descriptions
segment_descriptions = {
    'Champion': 'High spenders who visit often with long-term loyalty',
    'Loyalist': 'Consistent shoppers with strong engagement',
    'Emerging': 'Newer customers showing promising purchase patterns',
    'Newbie': 'Recent first-time shoppers with potential',
    'At Risk': 'Previously good customers whose visits have dropped off',
    'Hibernating': 'Inactive customers who have not shopped in a long time'
}

rfm['segment_description'] = rfm['segment'].map(segment_descriptions)

segment_order = ['Champion', 'Loyalist', 'Emerging', 'Newbie', 'At Risk', 'Hibernating']

print("\n   Segment Distribution:")
for segment in segment_order:
    count = len(rfm[rfm['segment'] == segment])
    pct = count / len(rfm) * 100
    print(f"     {segment}: {count} ({pct:.1f}%)")

# Save rfm_results.csv
print("\n5. Saving results...")

rfm_results = rfm[['user_id', 'user_type', 'recency_days', 'frequency', 'monetary',
                   'r_score', 'f_score', 'm_score', 'rfm_score', 'segment', 'segment_description']].copy()

# Sort by segment and monetary value
segment_sort_order = {'Champion': 1, 'Loyalist': 2, 'Emerging': 3, 'Newbie': 4, 'At Risk': 5, 'Hibernating': 6}
rfm_results['segment_sort'] = rfm_results['segment'].map(segment_sort_order)
rfm_results = rfm_results.sort_values(['segment_sort', 'monetary'], ascending=[True, False])
rfm_results = rfm_results.drop('segment_sort', axis=1)

rfm_results.to_csv('rfm_results.csv', index=False)
print(f"   Saved: rfm_results.csv ({rfm_results.shape[0]} customers)")

# Create summary statistics
summary_data = []

for segment in segment_order:
    seg_df = rfm[rfm['segment'] == segment]

    if len(seg_df) > 0:
        loyal_count = seg_df[seg_df['user_type'] == 'loyal'].shape[0]
        new_count = seg_df[seg_df['user_type'] == 'new'].shape[0]

        summary_data.append({
            'segment': segment,
            'description': segment_descriptions[segment],
            'customer_count': len(seg_df),
            'percentage': round(len(seg_df) / len(rfm) * 100, 2),
            'loyal_customers': loyal_count,
            'new_customers': new_count,
            'total_revenue': round(seg_df['monetary'].sum(), 2),
            'revenue_share_%': round(seg_df['monetary'].sum() / rfm['monetary'].sum() * 100, 2),
            'avg_recency_days': round(seg_df['recency_days'].mean(), 2),
            'avg_frequency': round(seg_df['frequency'].mean(), 2),
            'avg_monetary': round(seg_df['monetary'].mean(), 2),
            'avg_r_score': round(seg_df['r_score'].mean(), 2),
            'avg_f_score': round(seg_df['f_score'].mean(), 2),
            'avg_m_score': round(seg_df['m_score'].mean(), 2)
        })

rfm_summary = pd.DataFrame(summary_data)
rfm_summary.to_csv('rfm_summary.csv', index=False)
print(f"   Saved: rfm_summary.csv")

# Generate visualizations
print("\n6. Generating visualizations...")

# Set up colors for segments
segment_colors = {
    'Champion': '#2ecc71',      # Green
    'Loyalist': '#3498db',      # Blue
    'Emerging': '#9b59b6',      # Purple
    'Newbie': '#f39c12',        # Orange
    'At Risk': '#e74c3c',       # Red
    'Hibernating': '#95a5a6'    # Gray
}

try:
    # RFM distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(rfm['recency_days'], bins=30, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Days Since Last Purchase')
    axes[0].set_ylabel('Number of Customers')
    axes[0].set_title('Recency Distribution')
    axes[0].axvline(rfm['recency_days'].median(), color='red', linestyle='--', label=f'Median: {rfm["recency_days"].median():.0f}')
    axes[0].legend()

    axes[1].hist(rfm['frequency'], bins=30, color='lightgreen', edgecolor='black')
    axes[1].set_xlabel('Number of Transactions')
    axes[1].set_ylabel('Number of Customers')
    axes[1].set_title('Frequency Distribution')
    axes[1].axvline(rfm['frequency'].median(), color='red', linestyle='--', label=f'Median: {rfm["frequency"].median():.0f}')
    axes[1].legend()

    axes[2].hist(rfm['monetary'], bins=30, color='salmon', edgecolor='black')
    axes[2].set_xlabel('Total Spending ($)')
    axes[2].set_ylabel('Number of Customers')
    axes[2].set_title('Monetary Distribution')
    axes[2].axvline(rfm['monetary'].median(), color='red', linestyle='--', label=f'Median: ${rfm["monetary"].median():.2f}')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('rfm_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: rfm_distributions.png")
except Exception as e:
    print(f"   Warning: Could not save rfm_distributions.png - {e}")

try:
    # Pie chart: Segment distribution
    fig, ax = plt.subplots(figsize=(10, 8))

    segment_data = rfm['segment'].value_counts().reindex([s for s in segment_order if s in rfm['segment'].values])
    if len(segment_data) > 0:
        colors = [segment_colors[seg] for seg in segment_data.index]

        wedges, texts, autotexts = ax.pie(
            segment_data.values,
            labels=segment_data.index,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=[0.02] * len(segment_data)
        )

        ax.set_title('Customer Segment Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('segment_pie_chart.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   Saved: segment_pie_chart.png")
except Exception as e:
    print(f"   Warning: Could not save segment_pie_chart.png - {e}")

try:
    # Bar charts: Segment comparisons
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Filter to only segments that exist
    existing_segments = [s for s in segment_order if s in rfm['segment'].values]

    # Customer Count by Segment
    segment_data = rfm['segment'].value_counts().reindex(existing_segments)
    colors = [segment_colors[seg] for seg in segment_data.index]
    axes[0, 0].bar(segment_data.index, segment_data.values, color=colors, edgecolor='black')
    axes[0, 0].set_title('Customer Count by Segment', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Number of Customers')
    axes[0, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(segment_data.values):
        axes[0, 0].text(i, v + 0.5, str(v), ha='center', fontweight='bold')

    # Average Monetary Value by Segment
    monetary_data = rfm.groupby('segment')['monetary'].mean().reindex(existing_segments)
    colors = [segment_colors[seg] for seg in monetary_data.index]
    axes[0, 1].bar(monetary_data.index, monetary_data.values, color=colors, edgecolor='black')
    axes[0, 1].set_title('Average Spending by Segment', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Average Monetary Value ($)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(monetary_data.values):
        axes[0, 1].text(i, v + 10, f'${v:.0f}', ha='center', fontweight='bold')

    # Average Frequency by Segment
    freq_data = rfm.groupby('segment')['frequency'].mean().reindex(existing_segments)
    colors = [segment_colors[seg] for seg in freq_data.index]
    axes[1, 0].bar(freq_data.index, freq_data.values, color=colors, edgecolor='black')
    axes[1, 0].set_title('Average Transaction Frequency by Segment', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Average Transactions')
    axes[1, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(freq_data.values):
        axes[1, 0].text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')

    # Total Revenue by Segment
    revenue_data = rfm.groupby('segment')['monetary'].sum().reindex(existing_segments)
    colors = [segment_colors[seg] for seg in revenue_data.index]
    axes[1, 1].bar(revenue_data.index, revenue_data.values, color=colors, edgecolor='black')
    axes[1, 1].set_title('Total Revenue Contribution by Segment', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Total Revenue ($)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(revenue_data.values):
        axes[1, 1].text(i, v + 100, f'${v:,.0f}', ha='center', fontweight='bold', fontsize=8)

    plt.tight_layout()
    plt.savefig('segment_bar_charts.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: segment_bar_charts.png")
except Exception as e:
    print(f"   Warning: Could not save segment_bar_charts.png - {e}")

try:
    # Scatter plot: Frequency vs Monetary by Segment
    fig, ax = plt.subplots(figsize=(12, 8))

    existing_segments = [s for s in segment_order if s in rfm['segment'].values]
    for segment in existing_segments:
        segment_df = rfm[rfm['segment'] == segment]
        ax.scatter(
            segment_df['frequency'],
            segment_df['monetary'],
            c=segment_colors[segment],
            label=segment,
            alpha=0.7,
            s=100,
            edgecolor='white'
        )

    ax.set_xlabel('Frequency (Number of Transactions)', fontsize=12)
    ax.set_ylabel('Monetary Value ($)', fontsize=12)
    ax.set_title('Customer Segments: Frequency vs Monetary Value', fontsize=14, fontweight='bold')
    ax.legend(title='Segment', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('frequency_vs_monetary_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: frequency_vs_monetary_scatter.png")
except Exception as e:
    print(f"   Warning: Could not save frequency_vs_monetary_scatter.png - {e}")

try:
    # Heatmap: RFM Score patterns by segment
    existing_segments = [s for s in segment_order if s in rfm['segment'].values]
    rfm_scores_by_segment = rfm.groupby('segment')[['r_score', 'f_score', 'm_score']].mean().reindex(existing_segments)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        rfm_scores_by_segment,
        annot=True,
        cmap='RdYlGn',
        fmt='.2f',
        linewidths=0.5,
        ax=ax,
        vmin=1,
        vmax=5
    )
    ax.set_title('Average RFM Scores by Segment', fontsize=14, fontweight='bold')
    ax.set_xlabel('RFM Component')
    ax.set_ylabel('Segment')

    plt.tight_layout()
    plt.savefig('rfm_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: rfm_heatmap.png")
except Exception as e:
    print(f"   Warning: Could not save rfm_heatmap.png - {e}")

try:
    # Stacked bar: Loyal vs New customers per segment
    existing_segments = [s for s in segment_order if s in rfm['segment'].values]
    user_type_data = rfm.groupby(['segment', 'user_type']).size().unstack(fill_value=0).reindex(existing_segments)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(existing_segments))
    width = 0.6

    loyal_vals = user_type_data.get('loyal', pd.Series([0]*len(existing_segments), index=existing_segments))
    new_vals = user_type_data.get('new', pd.Series([0]*len(existing_segments), index=existing_segments))

    ax.bar(x, loyal_vals, width, label='Loyal', color='#3498db', edgecolor='black')
    ax.bar(x, new_vals, width, bottom=loyal_vals, label='New', color='#f39c12', edgecolor='black')

    ax.set_xlabel('Segment')
    ax.set_ylabel('Number of Customers')
    ax.set_title('Customer Type Distribution by Segment', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(existing_segments, rotation=45)
    ax.legend()

    # Add value labels
    for i in range(len(existing_segments)):
        total = loyal_vals.iloc[i] + new_vals.iloc[i]
        ax.text(i, total + 0.5, str(int(total)), ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('customer_type_by_segment.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: customer_type_by_segment.png")
except Exception as e:
    print(f"   Warning: Could not save customer_type_by_segment.png - {e}")

# Print final summary
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nTotal customers segmented: {len(rfm_results)}")
print(f"\nOutput files generated:")
print(f"  - rfm_results.csv")
print(f"  - rfm_summary.csv")
print(f"\nVisualization files:")
print(f"  - rfm_distributions.png")
print(f"  - segment_pie_chart.png")
print(f"  - segment_bar_charts.png")
print(f"  - frequency_vs_monetary_scatter.png")
print(f"  - rfm_heatmap.png")
print(f"  - customer_type_by_segment.png")

# Display summary table
print("\n" + "=" * 60)
print("SEGMENT SUMMARY")
print("=" * 60)
print(rfm_summary[['segment', 'customer_count', 'percentage', 'total_revenue', 'revenue_share_%']].to_string(index=False))
