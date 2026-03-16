#!/usr/bin/env python3
"""
Generate visualizations for newcomer labels (good first issue, etc) analysis.

Creates visualizations about label adoption and impact.
"""

import os
import sys
import csv
import numpy as np
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import FILTERED_REPO_CSV, COLORS, PLOT_PARAMS, LABEL_VIZ_DIR
from scripts.descriptive_stats.visualization_base import setup_figure, save_figure, style_vertical_bars

def load_label_data():
    """Load label data from filtered repos dataset."""
    data = {
        'has_labels': [],
        'no_labels': [],
        'labels_found': defaultdict(int),
        'label_impact': {
            'with_labels_newcomers': [],
            'without_labels_newcomers': []
        }
    }
    
    with open(FILTERED_REPO_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            has_newcomer_labels = row.get('has_newcomer_labels', '').lower() == 'true'
            found_labels = row.get('found_newcomer_labels', '').strip()
            
            try:
                newcomers = int(row.get('Average number of newcomers per month', 0) or 0)
            except:
                newcomers = 0
            
            repo_name = row.get('full_name', 'unknown')
            
            if has_newcomer_labels:
                data['has_labels'].append({
                    'repo': repo_name,
                    'labels': found_labels,
                    'newcomers': newcomers
                })
                data['label_impact']['with_labels_newcomers'].append(newcomers)
            else:
                data['no_labels'].append({
                    'repo': repo_name,
                    'newcomers': newcomers
                })
                data['label_impact']['without_labels_newcomers'].append(newcomers)
            
            # Count label types
            if found_labels:
                for label in found_labels.split('|'):
                    label = label.strip()
                    if label:
                        data['labels_found'][label] += 1
    
    return data

def plot_label_adoption(data):
    """Plot: Adoption rate of newcomer labels (bar chart)."""
    n_with = len(data['has_labels'])
    n_without = len(data['no_labels'])
    total = n_with + n_without
    
    fig, ax = setup_figure('small')
    
    # Prepare data
    adoption_pct = 100 * n_with / total
    labels_bar = ['With GFI Labels', 'Without GFI Labels']
    values = [adoption_pct, 100 - adoption_pct]
    counts = [n_with, n_without]
    color_list = [COLORS['with_label'], COLORS['without_label']]
    
    # Create bars
    bars = ax.bar(labels_bar, values, color=color_list, 
                  alpha=PLOT_PARAMS['alpha_bar'], edgecolor='black', 
                  linewidth=PLOT_PARAMS['linewidth_edge'], width=PLOT_PARAMS['bar_width'])
    
    # Add labels with counts and percentages
    for bar, count, pct in zip(bars, counts, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=PLOT_PARAMS['fontsize_label'], 
                fontweight='bold')
    
    # Styling
    ax.set_ylabel('Percentage (%)', fontsize=PLOT_PARAMS['fontsize_label'])
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=PLOT_PARAMS['grid_alpha'])
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=PLOT_PARAMS['fontsize_tick'])
    ax.tick_params(axis='x', labelsize=PLOT_PARAMS['fontsize_tick'])
    
    save_figure(fig, os.path.join(str(LABEL_VIZ_DIR), '01_label_adoption_rate.png'))

def print_summary(data):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("NEWCOMER LABELS ANALYSIS SUMMARY")
    print("="*80)
    
    n_with = len(data['has_labels'])
    n_without = len(data['no_labels'])
    total = n_with + n_without
    
    print(f"\nLabel Adoption:")
    print(f"  Repositories with GFI labels: {n_with} ({100*n_with/total:.1f}%)")
    print(f"  Repositories without GFI labels: {n_without} ({100*n_without/total:.1f}%)")
    print(f"  Total repositories: {total}")
    
    print(f"\nLabel Types Found:")
    if data['labels_found']:
        for label, count in sorted(data['labels_found'].items(), key=lambda x: x[1], reverse=True)[:5]:
            pct = 100 * count / n_with
            print(f"  {label}: {count} ({pct:.1f}% of labeled repos)")
    else:
        print("  (No labels found)")
    
    newcomers_with = data['label_impact']['with_labels_newcomers']
    newcomers_without = data['label_impact']['without_labels_newcomers']
    
def main():
    """Generate all label visualizations."""
    print("\n" + "="*80)
    print("NEWCOMER LABELS VISUALIZATION")
    print("="*80)
    
    print("\nLoading label data...")
    data = load_label_data()
    
    print(f"Found {len(data['has_labels'])} repos with GFI labels")
    print(f"Found {len(data['no_labels'])} repos without GFI labels")
    print(f"Found {len(data['labels_found'])} unique label types")
    
    print("\nGenerating visualizations...")
    plot_label_adoption(data)
    
    print_summary(data)
    
    print(f"\n✅ All label visualizations saved to: {LABEL_VIZ_DIR}/")
    print("\nGenerated files:")
    print("  - 01_label_adoption_rate.png")

if __name__ == "__main__":
    main()
