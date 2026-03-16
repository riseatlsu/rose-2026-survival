#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Elijah Phifer"
__contact__ = "elijah.phifer@lsu.edu"

"""
Plot documentation metrics (README, CONTRIBUTING, CODE_OF_CONDUCT, etc.)
across distributions and owner types.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class DocumentationMetricsVisualizer:
    def __init__(self, filtered_repos_csv, output_folder='plots', plot_prefix='02_plot_docs__00', inflow_csv=None):
        self.filtered_repos_csv = filtered_repos_csv
        self.output_folder = output_folder
        self.plot_prefix = plot_prefix
        self.inflow_csv = inflow_csv
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")
        
        print(f"Loading data from: {filtered_repos_csv}")
        self.df = pd.read_csv(filtered_repos_csv)
        print(f"Loaded {len(self.df)} repositories")
        
        # Filter to only repositories in inflow.csv
        if inflow_csv and os.path.exists(inflow_csv):
            inflow_df = pd.read_csv(inflow_csv)
            allowed_projects = set(inflow_df['project'].values)
            self.df['project_key'] = self.df['Owner'] + '/' + self.df['Name']
            self.df = self.df[self.df['project_key'].isin(allowed_projects)]
            print(f"Filtered to {len(self.df)} repositories from inflow.csv")
        
        # Extract distribution information
        self.df['distros_list'] = self.df['distros_present'].apply(
            lambda x: x.split('|') if pd.notna(x) and x else []
        )
    
    def _categorize_repos_by_distribution(self):
        """
        Categorize repositories as single-distro or multi-distro.
        Returns a dictionary mapping index to category.
        """
        categorized = {}
        for idx, row in self.df.iterrows():
            distros_list = row['distros_list']
            if len(distros_list) == 1:
                categorized[idx] = distros_list[0]
            elif len(distros_list) > 1:
                categorized[idx] = 'multi-distro'
            else:
                categorized[idx] = 'unknown'
        return categorized
    
    def plot_documentation_by_distribution(self):
        """Plot documentation file presence by ROS distribution (single-distro vs multi-distro)."""
        print("\\nGenerating documentation metrics by distribution...")
        
        # Documentation columns to analyze
        doc_columns = {
            'has_readme': 'Has Readme',
            'has_contributing': 'Has Contributing',
            'has_code_of_conduct': 'Has Code of Conduct',
            'has_pr_template': 'Has PR Template',
            'has_issue_template': 'Has Issue Template'
        }
        
        # Categorize repositories
        repo_categories = self._categorize_repos_by_distribution()
        
        # Get unique categories
        all_categories = sorted(set(repo_categories.values()))
        
        # Calculate percentages for each category
        category_stats = {}
        category_counts = {}
        
        for category in all_categories:
            # Filter repos that belong to this category
            category_indices = [idx for idx, cat in repo_categories.items() if cat == category]
            category_df = self.df.loc[category_indices]
            
            stats = {}
            for col, label in doc_columns.items():
                if col in category_df.columns:
                    count = category_df[col].sum()
                    percentage = (count / len(category_df) * 100) if len(category_df) > 0 else 0
                    stats[label] = percentage
            
            category_stats[category] = stats
            category_counts[category] = len(category_df)
        
        print(f"\\nRepository counts per category:")
        for category in sorted(category_counts.keys()):
            print(f"  {category.capitalize():15} - {category_counts[category]:3d} repositories")
        
        # Create grouped horizontal bar chart (sized for single column)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        
        y = np.arange(len(doc_columns))
        
        # Calculate bar height based on number of categories
        num_categories = len(all_categories)
        height = 0.7 / num_categories if num_categories > 0 else 0.15
        
        colors = {
            'foxy': '#FF6B6B',
            'galactic': '#4ECDC4',
            'humble': '#1F618D',  # Darker blue for better colorblind accessibility
            'iron': '#96CEB4',
            'jazzy': '#A9B83E',  # More green (less yellow) for better colorblind accessibility
            'kilted': '#9B59B6',
            'rolling': '#BC6C25',
            'multi-distro': '#E74C3C',
            'unknown': '#95A5A6'
        }
        
        multiplier = 0
        for category in all_categories:
            offset = height * multiplier
            values = [category_stats[category].get(label, 0) for label in doc_columns.values()]
            color = colors.get(category, '#95A5A6')
            
            # Make it clear that single-distro repos show only their distribution
            if category in ['multi-distro', 'unknown']:
                label = f"{category.capitalize()} (n={category_counts[category]})"
            else:
                label = f"{category.capitalize()} only (n={category_counts[category]})"
            
            ax.barh(y + offset, values, height, label=label, color=color, alpha=0.8)
            multiplier += 1
        
        ax.set_ylabel('Documentation Type', fontsize=9)
        ax.set_xlabel('Percentage of Repositories (%)', fontsize=9)
        ax.set_yticks(y + height * (num_categories - 1) / 2)
        ax.set_yticklabels(doc_columns.values(), fontsize=8)
        ax.legend(loc='upper right', fontsize=5.5, framealpha=0.9, handlelength=2, 
                  handletextpad=0.4, borderpad=0.25)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, 105)
        ax.tick_params(axis='x', labelsize=8)
        
        plt.tight_layout()
        
        output_path_png = os.path.join(self.output_folder, f'{self.plot_prefix}_docs_by_distribution.png')
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path_png}")
        
        plt.close()
    
    def plot_documentation_by_owner_type(self):
        """Plot documentation file presence by owner type."""
        print("\\nGenerating documentation metrics by owner type...")
        
        doc_columns = {
            'has_readme': 'Has Readme',
            'has_contributing': 'Has Contributing',
            'has_code_of_conduct': 'Has Code of Conduct',
            'has_pr_template': 'Has PR Template',
            'has_issue_template': 'Has Issue Template'
        }
        
        # Calculate percentages for each owner type
        owner_type_stats = {}
        for owner_type in ['Organization', 'User']:
            owner_df = self.df[self.df['owner_type'] == owner_type]
            
            stats = {}
            for col, label in doc_columns.items():
                if col in owner_df.columns:
                    count = owner_df[col].sum()
                    percentage = (count / len(owner_df) * 100) if len(owner_df) > 0 else 0
                    stats[label] = percentage
            
            owner_type_stats[owner_type] = {
                'stats': stats,
                'count': len(owner_df)
            }
        
        # Create grouped horizontal bar chart (sized for single column)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        
        y = np.arange(len(doc_columns))
        height = 0.35
        
        colors = {
            'Organization': '#3498DB',
            'User': '#E67E22'
        }
        
        for idx, owner_type in enumerate(['Organization', 'User']):
            if owner_type not in owner_type_stats:
                continue
            
            offset = height * idx
            values = [owner_type_stats[owner_type]['stats'].get(label, 0) for label in doc_columns.values()]
            color = colors.get(owner_type, '#95A5A6')
            label = f"{owner_type} (n={owner_type_stats[owner_type]['count']})"
            
            bars = ax.barh(y + offset, values, height, label=label, color=color, alpha=0.8)
            
            # Add count and percentage labels on bars
            owner_df = self.df[self.df['owner_type'] == owner_type]
            for i, (bar, doc_label) in enumerate(zip(bars, doc_columns.keys())):
                count = owner_df[doc_label].sum() if doc_label in owner_df.columns else 0
                percentage = values[i]
                width = bar.get_width()
                ax.text(width + 2, bar.get_y() + bar.get_height()/2,
                       f'{count} ({percentage:.0f}%)',
                       ha='left', va='center', fontsize=6)
        
        ax.set_ylabel('Documentation Type', fontsize=9)
        ax.set_xlabel('Percentage of Repositories (%)', fontsize=9)
        ax.set_yticks(y + height / 2)
        ax.set_yticklabels(doc_columns.values(), fontsize=8)
        ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, 125)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis='x', labelsize=8)
        
        plt.tight_layout()
        
        output_path_png = os.path.join(self.output_folder, f'{self.plot_prefix}_docs_by_owner_type.png')
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path_png}")
        
        plt.close()
    
    def plot_documentation_overall(self):
        """Plot overall documentation file presence statistics."""
        print("\\nGenerating overall documentation metrics...")
        
        doc_columns = {
            'has_readme': 'Has Readme',
            'has_contributing': 'Has Contributing',
            'has_code_of_conduct': 'Has Code of Conduct',
            'has_pr_template': 'Has PR Template',
            'has_issue_template': 'Has Issue Template',
            'has_newcomer_labels': 'Has Labels'
        }
        
        # Calculate overall percentages
        stats = {}
        for col, label in doc_columns.items():
            if col in self.df.columns:
                count = self.df[col].sum()
                percentage = (count / len(self.df) * 100) if len(self.df) > 0 else 0
                stats[label] = {
                    'percentage': percentage,
                    'count': count
                }
        
        # Create horizontal bar chart (sized for single column)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        
        labels = list(stats.keys())
        values = [stats[label]['percentage'] for label in labels]
        counts = [stats[label]['count'] for label in labels]
        
        bars = ax.barh(labels, values, color='goldenrod', alpha=0.8)
        
        # Add count labels on bars
        for idx, (bar, count, value) in enumerate(zip(bars, counts, values)):
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height()/2,
                   f'{count} ({value:.1f}%)',
                   ha='left', va='center', fontsize=7)
        
        ax.set_xlabel('Percentage of Repositories (%)', fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, 110)
        ax.tick_params(axis='both', labelsize=8)
        
        plt.tight_layout()
        
        output_path_png = os.path.join(self.output_folder, f'{self.plot_prefix}_docs_overall.png')
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path_png}")
        
        plt.close()
    
    def generate_plots(self):
        """Generate all documentation metric visualizations."""
        print("\\n" + "="*60)
        print("GENERATING DOCUMENTATION METRICS VISUALIZATIONS")
        print("="*60)
        
        self.plot_documentation_overall()
        # self.plot_documentation_by_distribution()
        self.plot_documentation_by_owner_type()
        
        print("\\n" + "="*60)
        print(f"Plots saved to: {os.path.abspath(self.output_folder)}")
        print("="*60)


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filtered_repos_csv = os.path.join(script_dir, '..', 'out', 'filtered_repo_dataset.csv')
    inflow_csv = os.path.join(script_dir, 'tables', 'inflow.csv')
    output_folder = os.path.join(script_dir, '..', 'figs')
    
    if not os.path.exists(filtered_repos_csv):
        print(f"ERROR: Could not find filtered_repo_dataset.csv at: {filtered_repos_csv}")
        print("Please run the pipeline first to generate the data.")
    else:
        visualizer = DocumentationMetricsVisualizer(filtered_repos_csv, output_folder, inflow_csv=inflow_csv)
        visualizer.generate_plots()
