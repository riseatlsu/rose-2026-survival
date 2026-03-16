#!/usr/bin/env python3
"""
Generate publication-quality visualizations for newcomer contribution types.
Shows breakdown of code/docs/config/assets/other contributions.

Generates 3 charts:
1. Stacked bar chart - Top 15 repositories by newcomer count
2. Global distribution - Overall breakdown of contribution types
3. Horizontal percentage bars - Percentage composition for top 15 repos
"""

import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import (
    COMMIT_TYPES_CSV, COLORS, FIGSIZE, PLOT_PARAMS,
    TOP_N_REPOS, COMMIT_TYPES, COMMIT_TYPES_VIZ_DIR
)
from scripts.descriptive_stats.visualization_base import (
    setup_figure, setup_dual_figure, save_figure,
    style_horizontal_bars, style_vertical_bars,
    add_bar_labels, add_hbar_labels
)

def load_data(csv_path):
    """Load and aggregate repository data from commits CSV."""
    repos = {}
    try:
        df = pd.read_csv(csv_path)
        
        # Aggregate by repository
        for _, row in df.iterrows():
            full_name = row.get('full_name', '')
            commit_type = row.get('commit_type', 'other')
            
            # Normalize commit types
            if commit_type not in COMMIT_TYPES:
                commit_type = 'other'
            
            if full_name not in repos:
                repos[full_name] = {
                    'full_name': full_name,
                    'code': 0,
                    'docs': 0,
                    'config': 0,
                    'assets': 0,
                    'other': 0,
                }
            
            repos[full_name][commit_type] += 1
        
        # Convert to list and calculate percentages
        result = []
        for full_name, data in repos.items():
            total = sum(data[ct] for ct in COMMIT_TYPES)
            if total > 0:
                result.append({
                    'full_name': full_name,
                    'total_newcomers': total,
                    'code': data['code'],
                    'docs': data['docs'],
                    'config': data['config'],
                    'assets': data['assets'],
                    'other': data['other'],
                    'pct_code': 100 * data['code'] / total,
                    'pct_docs': 100 * data['docs'] / total,
                    'pct_config': 100 * data['config'] / total,
                    'pct_assets': 100 * data['assets'] / total,
                    'pct_other': 100 * data['other'] / total,
                })
        
        return result
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

def plot_stacked_bar_chart(repos, output_dir, top_n=TOP_N_REPOS):
    """Create stacked bar chart for top repositories."""
    print(f"[1/3] Creating stacked bar chart (top {top_n})...")
    
    top_repos = repos[:top_n]
    repo_names = [r['full_name'] for r in top_repos]
    
    # Prepare data
    code = np.array([r['code'] for r in top_repos])
    docs = np.array([r['docs'] for r in top_repos])
    config = np.array([r['config'] for r in top_repos])
    assets = np.array([r['assets'] for r in top_repos])
    other = np.array([r['other'] for r in top_repos])
    
    # Create figure
    fig, ax = setup_figure('medium')
    
    x = np.arange(len(repo_names))
    width = PLOT_PARAMS['bar_width']
    
    # Stack bars
    ax.bar(x, code, width, label='Code', color=COLORS['code'], alpha=PLOT_PARAMS['alpha_bar'])
    ax.bar(x, docs, width, bottom=code, label='Docs', color=COLORS['docs'], alpha=PLOT_PARAMS['alpha_bar'])
    ax.bar(x, config, width, bottom=code+docs, label='Config', color=COLORS['config'], alpha=PLOT_PARAMS['alpha_bar'])
    ax.bar(x, assets, width, bottom=code+docs+config, label='Assets', color=COLORS['assets'], alpha=PLOT_PARAMS['alpha_bar'])
    ax.bar(x, other, width, bottom=code+docs+config+assets, label='Other', color=COLORS['other'], alpha=PLOT_PARAMS['alpha_bar'])
    
    # Styling
    style_vertical_bars(ax, ylabel='Number of Newcomers', xlabel='Repository')
    ax.set_xticks(x)
    ax.set_xticklabels(repo_names, rotation=45, ha='right', fontsize=PLOT_PARAMS['fontsize_tick'])
    ax.legend(loc='upper right', fontsize=PLOT_PARAMS['fontsize_tick'], framealpha=0.9)
    
    save_figure(fig, os.path.join(output_dir, "01_stacked_bar_top_repos.png"))

def plot_global_bar_chart(repos, output_dir):
    """Create horizontal bar chart of global distribution."""
    print(f"[2/3] Creating global distribution bar chart...")
    
    # Aggregate totals
    total_code = sum(r['code'] for r in repos)
    total_docs = sum(r['docs'] for r in repos)
    total_config = sum(r['config'] for r in repos)
    total_assets = sum(r['assets'] for r in repos)
    total_other = sum(r['other'] for r in repos)
    total = total_code + total_docs + total_config + total_assets + total_other
    
    # Skip if no data
    if total == 0:
        print("⚠️  No data available for global distribution chart")
        return
    
    # Prepare data
    labels = ['Code', 'Docs', 'Config', 'Assets', 'Other']
    counts = [total_code, total_docs, total_config, total_assets, total_other]
    percentages = [100*c/total for c in counts]
    color_list = [COLORS['code'], COLORS['docs'], COLORS['config'], COLORS['assets'], COLORS['other']]
    
    # Create figure
    fig, ax = setup_figure('small')
    
    y = np.arange(len(labels))
    bars = ax.barh(y, percentages, PLOT_PARAMS['bar_width'], color=color_list, 
                   alpha=PLOT_PARAMS['alpha_bar'], edgecolor='black', linewidth=PLOT_PARAMS['linewidth_edge'])
    
    # Add labels
    add_hbar_labels(bars, percentages, counts)
    
    # Styling
    style_horizontal_bars(ax, ylabel='Contribution Type')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=PLOT_PARAMS['fontsize_label'])
    
    save_figure(fig, os.path.join(output_dir, "02_global_distribution_bar.png"))

def plot_horizontal_percentage_bars(repos, output_dir, top_n=15):
    """Create horizontal bars showing percentage composition."""
    print(f"[3/3] Creating percentage composition chart (top {top_n})...")
    
    top_repos = repos[:top_n]
    repo_names = [r['full_name'] for r in top_repos]
    
    # Get percentages
    pct_code = np.array([r['pct_code'] for r in top_repos])
    pct_docs = np.array([r['pct_docs'] for r in top_repos])
    pct_config = np.array([r['pct_config'] for r in top_repos])
    pct_assets = np.array([r['pct_assets'] for r in top_repos])
    pct_other = np.array([r['pct_other'] for r in top_repos])
    
    # Create figure
    fig, ax = setup_figure('large')
    
    y = np.arange(len(repo_names))
    width = PLOT_PARAMS['bar_width']
    
    # Stack horizontal bars
    ax.barh(y, pct_code, width, label='Code', color=COLORS['code'], alpha=PLOT_PARAMS['alpha_bar'])
    ax.barh(y, pct_docs, width, left=pct_code, label='Docs', color=COLORS['docs'], alpha=PLOT_PARAMS['alpha_bar'])
    ax.barh(y, pct_config, width, left=pct_code+pct_docs, label='Config', color=COLORS['config'], alpha=PLOT_PARAMS['alpha_bar'])
    ax.barh(y, pct_assets, width, left=pct_code+pct_docs+pct_config, label='Assets', color=COLORS['assets'], alpha=PLOT_PARAMS['alpha_bar'])
    ax.barh(y, pct_other, width, left=pct_code+pct_docs+pct_config+pct_assets, label='Other', color=COLORS['other'], alpha=PLOT_PARAMS['alpha_bar'])
    
    # Styling
    style_horizontal_bars(ax)
    ax.set_yticks(y)
    ax.set_yticklabels(repo_names, fontsize=PLOT_PARAMS['fontsize_tick'])
    ax.legend(loc='lower right', fontsize=PLOT_PARAMS['fontsize_tick'], framealpha=0.9, ncol=5)
    
    save_figure(fig, os.path.join(output_dir, "03_composition_horizontal_bars.png"))

def main():
    print("=" * 80)
    print("COMMIT TYPE VISUALIZATIONS (PAPER STYLE)")
    print("=" * 80)
    
    # Load data
    repos = load_data(str(COMMIT_TYPES_CSV))
    
    if not repos:
        print("No data found")
        return
    
    # Filter and sort
    repos = [r for r in repos if r['total_newcomers'] > 0]
    repos.sort(key=lambda x: x['total_newcomers'], reverse=True)
    
    print(f"\nTotal repositories: {len(repos)}")
    print(f"Generating visualizations...\n")
    
    # Generate all visualizations
    plot_stacked_bar_chart(repos, str(COMMIT_TYPES_VIZ_DIR), top_n=TOP_N_REPOS)
    plot_global_bar_chart(repos, str(COMMIT_TYPES_VIZ_DIR))
    plot_horizontal_percentage_bars(repos, str(COMMIT_TYPES_VIZ_DIR), top_n=15)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    total_code = sum(r['code'] for r in repos)
    total_docs = sum(r['docs'] for r in repos)
    total_config = sum(r['config'] for r in repos)
    total_assets = sum(r['assets'] for r in repos)
    total_other = sum(r['other'] for r in repos)
    total = total_code + total_docs + total_config + total_assets + total_other
    
    print(f"\nTotal newcomers: {total:,}")
    print(f"Code:    {total_code:6,} ({100*total_code/total:5.1f}%)")
    print(f"Docs:    {total_docs:6,} ({100*total_docs/total:5.1f}%)")
    print(f"Config:  {total_config:6,} ({100*total_config/total:5.1f}%)")
    print(f"Assets:  {total_assets:6,} ({100*total_assets/total:5.1f}%)")
    print(f"Other:   {total_other:6,} ({100*total_other/total:5.1f}%)")
    
    print(f"\nRepository statistics:")
    print(f"  Total: {len(repos)}")
    print(f"  Average: {total/len(repos):.1f} newcomers/repo")
    
    print(f"\n✅ Visualizations saved to: {COMMIT_TYPES_VIZ_DIR}")

if __name__ == "__main__":
    main()
