#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Elijah Phifer"
__contact__ = "elijah.phifer@lsu.edu"

"""
Generate basic descriptive statistics and visualizations for the ROS repository dataset.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import glob
from datetime import datetime, timedelta
from collections import Counter, defaultdict

class BasicStatisticsVisualizer:
    def __init__(self, filtered_repos_csv, output_folder='plots', plot_prefix='03_basic_stats__00', inflow_csv=None):
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
    
    def plot_size_by_owner_type_boxplot(self):
        """Plot box plot of repository size by owner type."""
        print("\nGenerating repository size by owner type box plot...")
        
        owner_types = ['Organization', 'User']
        
        # Collect data
        data = []
        labels = []
        for owner_type in owner_types:
            owner_df = self.df[self.df['owner_type'] == owner_type]
            # Convert to KB
            data.append(owner_df['size'].values / 1024)
            labels.append(f"{owner_type}\n(n={len(owner_df)})")
        
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        
        # Color the boxes
        colors = {'Organization': '#3498DB', 'User': '#E67E22'}
        for patch, owner_type in zip(bp['boxes'], owner_types):
            patch.set_facecolor(colors[owner_type])
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Repository Size (KB)', fontsize=9)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='y', labelsize=8)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_folder, f'{self.plot_prefix}_size_by_owner_type_boxplot.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_repository_age_boxplot_overall(self):
        """Plot box plot of repository age (overall, not split by owner type)."""
        print("\nGenerating repository age box plot (overall)...")
        
        # Filter out NaN values
        valid_ages = self.df['Repository age (months)'].dropna()
        
        if len(valid_ages) == 0:
            print("  No valid data. Skipping plot.")
            return
        
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        
        bp = ax.boxplot([valid_ages.values], patch_artist=True, showfliers=False)
        
        # Color the box
        for patch in bp['boxes']:
            patch.set_facecolor('#3498DB')
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Repository Age (months)', fontsize=9)
        ax.set_xticklabels(['All Repositories'], fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='y', labelsize=8)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_folder, f'{self.plot_prefix}_repository_age_boxplot_overall.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_repository_age_boxplot_by_owner_type(self):
        """Plot box plot of repository age by owner type."""
        print("\nGenerating repository age box plot by owner type...")
        
        owner_types = ['Organization', 'User']
        
        # Collect data for each owner type
        data = []
        labels = []
        for owner_type in owner_types:
            owner_df = self.df[self.df['owner_type'] == owner_type]
            valid_ages = owner_df['Repository age (months)'].dropna()
            if len(valid_ages) > 0:
                data.append(valid_ages.values)
                labels.append(f"{owner_type}\n(n={len(valid_ages)})")
        
        if len(data) == 0:
            print("  No valid data. Skipping plot.")
            return
        
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        
        # Color the boxes
        colors = ['#3498DB', '#E67E22']
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Repository Age (months)', fontsize=9)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='y', labelsize=8)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_folder, f'{self.plot_prefix}_repository_age_boxplot_by_owner_type.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_time_close_pr_boxplot_overall(self):
        """Plot box plot of time to close PRs (overall, not split by owner type)."""
        print("\nGenerating time to close PRs box plot (overall)...")
        
        # Filter out NaN values
        valid_times = self.df['Average time to close a pull request (days)'].dropna()
        
        if len(valid_times) == 0:
            print("  No valid data. Skipping plot.")
            return
        
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        
        bp = ax.boxplot([valid_times.values], patch_artist=True, showfliers=False)
        
        # Color the box
        for patch in bp['boxes']:
            patch.set_facecolor('#3498DB')
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Time to Close PR (days)', fontsize=9)
        ax.set_xticklabels(['All Repositories'], fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='y', labelsize=8)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_folder, f'{self.plot_prefix}_time_close_pr_boxplot_overall.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_time_close_pr_boxplot_by_owner_type(self):
        """Plot box plot of time to close PRs by owner type."""
        print("\nGenerating time to close PRs box plot by owner type...")
        
        owner_types = ['Organization', 'User']
        
        # Collect data for each owner type
        data = []
        labels = []
        for owner_type in owner_types:
            owner_df = self.df[self.df['owner_type'] == owner_type]
            valid_times = owner_df['Average time to close a pull request (days)'].dropna()
            if len(valid_times) > 0:
                data.append(valid_times.values)
                labels.append(f"{owner_type}\n(n={len(valid_times)})")
        
        if len(data) == 0:
            print("  No valid data. Skipping plot.")
            return
        
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        
        # Color the boxes
        colors = ['#3498DB', '#E67E22']
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Time to Close PR (days)', fontsize=9)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='y', labelsize=8)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_folder, f'{self.plot_prefix}_time_close_pr_boxplot_by_owner_type.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_time_close_issue_boxplot_overall(self):
        """Plot box plot of time to close issues (overall, not split by owner type)."""
        print("\nGenerating time to close issues box plot (overall)...")
        
        # Filter out NaN values
        valid_times = self.df['Average time to close an issue (days)'].dropna()
        
        if len(valid_times) == 0:
            print("  No valid data. Skipping plot.")
            return
        
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        
        bp = ax.boxplot([valid_times.values], patch_artist=True, showfliers=False)
        
        # Color the box
        for patch in bp['boxes']:
            patch.set_facecolor('#3498DB')
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Time to Close Issue (days)', fontsize=9)
        ax.set_xticklabels(['All Repositories'], fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='y', labelsize=8)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_folder, f'{self.plot_prefix}_time_close_issue_boxplot_overall.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_time_close_issue_boxplot_by_owner_type(self):
        """Plot box plot of time to close issues by owner type."""
        print("\nGenerating time to close issues box plot by owner type...")
        
        owner_types = ['Organization', 'User']
        
        # Collect data for each owner type
        data = []
        labels = []
        for owner_type in owner_types:
            owner_df = self.df[self.df['owner_type'] == owner_type]
            valid_times = owner_df['Average time to close an issue (days)'].dropna()
            if len(valid_times) > 0:
                data.append(valid_times.values)
                labels.append(f"{owner_type}\n(n={len(valid_times)})")
        
        if len(data) == 0:
            print("  No valid data. Skipping plot.")
            return
        
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        
        # Color the boxes
        colors = ['#3498DB', '#E67E22']
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Time to Close Issue (days)', fontsize=9)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='y', labelsize=8)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_folder, f'{self.plot_prefix}_time_close_issue_boxplot_by_owner_type.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_pr_status_boxplot_overall(self):
        """Plot box plot of PR status (open/merged/closed) overall."""
        print("\nGenerating PR status box plot (overall)...")
        
        data = [
            self.df['Number of pull requests open'].values,
            self.df['Number of pull requests merged'].values,
            self.df['Number of pull requests closed'].values
        ]
        labels = ['Open', 'Merged', 'Closed']
        colors = ['#3498DB', '#27AE60', '#E74C3C']
        
        fig, ax = plt.subplots(figsize=(4, 3.5))
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Number of PRs per Repository', fontsize=9)
        ax.set_xlabel('Pull Request Status', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=8)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_folder, f'{self.plot_prefix}_pr_status_boxplot_overall.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_pr_counts_boxplot_by_owner_type(self):
        """Plot box plot of PR counts by owner type."""
        print("\nGenerating PR counts box plot by owner type...")
        
        owner_types = ['Organization', 'User']
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.5))
        
        colors = {'Organization': '#3498DB', 'User': '#E67E22'}
        
        # Open PRs
        data_open = []
        labels_open = []
        for owner_type in owner_types:
            owner_df = self.df[self.df['owner_type'] == owner_type]
            data_open.append(owner_df['Number of pull requests open'].values)
            labels_open.append(owner_type)
        
        bp1 = ax1.boxplot(data_open, labels=labels_open, patch_artist=True, showfliers=False)
        for patch, owner_type in zip(bp1['boxes'], owner_types):
            patch.set_facecolor(colors[owner_type])
            patch.set_alpha(0.7)
        ax1.set_ylabel('Number of PRs', fontsize=9)
        ax1.set_title('Open PRs', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(labelsize=8)
        
        # Merged PRs
        data_merged = []
        for owner_type in owner_types:
            owner_df = self.df[self.df['owner_type'] == owner_type]
            data_merged.append(owner_df['Number of pull requests merged'].values)
        
        bp2 = ax2.boxplot(data_merged, labels=labels_open, patch_artist=True, showfliers=False)
        for patch, owner_type in zip(bp2['boxes'], owner_types):
            patch.set_facecolor(colors[owner_type])
            patch.set_alpha(0.7)
        ax2.set_ylabel('Number of PRs', fontsize=9)
        ax2.set_title('Merged PRs', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(labelsize=8)
        
        # Closed PRs
        data_closed = []
        for owner_type in owner_types:
            owner_df = self.df[self.df['owner_type'] == owner_type]
            data_closed.append(owner_df['Number of pull requests closed'].values)
        
        bp3 = ax3.boxplot(data_closed, labels=labels_open, patch_artist=True, showfliers=False)
        for patch, owner_type in zip(bp3['boxes'], owner_types):
            patch.set_facecolor(colors[owner_type])
            patch.set_alpha(0.7)
        ax3.set_ylabel('Number of PRs', fontsize=9)
        ax3.set_title('Closed PRs', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.tick_params(labelsize=8)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_folder, f'{self.plot_prefix}_pr_counts_boxplot_by_owner_type.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def load_weekly_commit_data(self, data_folder):
        """Load weekly commit data from JSON files."""
        print("\nLoading weekly commit data from repository folders...")
        
        weekly_data = defaultdict(list)
        repo_count = 0
        
        # Iterate through all repos in the filtered dataset
        for _, row in self.df.iterrows():
            owner = row['Owner']
            name = row['Name']
            repo_folder = f"{owner}__{name}".replace('/', '__')
            
            json_path = os.path.join(data_folder, repo_folder, 'weekly_commit_activity.json')
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if 'data' in data and isinstance(data['data'], list):
                        for week_data in data['data']:
                            if 'week' in week_data and 'total' in week_data:
                                weekly_data[week_data['week']].append(week_data['total'])
                        repo_count += 1
                except Exception as e:
                    print(f"  Error loading {json_path}: {e}")
        
        print(f"  Loaded weekly commit data from {repo_count} repositories")
        
        # Calculate average commits per week across all repos
        weekly_averages = []
        weeks = sorted(weekly_data.keys())
        
        for week in weeks:
            avg_commits = np.mean(weekly_data[week]) if weekly_data[week] else 0
            weekly_averages.append({
                'week': week,
                'avg_commits': avg_commits,
                'repo_count': len(weekly_data[week])
            })
        
        return pd.DataFrame(weekly_averages)
    
    def plot_weekly_commit_patterns(self, data_folder):
        """Plot weekly commit patterns across all repositories (past 6 months)."""
        print("\nGenerating weekly commit patterns plot (past 6 months)...")
        
        weekly_df = self.load_weekly_commit_data(data_folder)
        
        if len(weekly_df) == 0:
            print("  No weekly commit data available. Skipping plot.")
            return
        
        # Filter to past 6 months from March 3, 2026
        cutoff_date = datetime(2026, 3, 3) - timedelta(days=180)
        cutoff_date_str = cutoff_date.strftime('%Y-%m-%d')
        
        # Convert week strings to datetime for filtering
        weekly_df['week_date'] = pd.to_datetime(weekly_df['week'])
        filtered_df = weekly_df[weekly_df['week_date'] >= cutoff_date_str].copy()
        filtered_df = filtered_df.sort_values('week_date')
        
        if len(filtered_df) == 0:
            print(f"  No data in the past 6 months (since {cutoff_date_str}). Skipping plot.")
            return
        
        print(f"  Filtered to {len(filtered_df)} weeks (from {cutoff_date_str} to 2026-03-03)")
        
        fig, ax = plt.subplots(figsize=(7, 3.5))
        
        # Plot the average commits per week
        ax.plot(range(len(filtered_df)), filtered_df['avg_commits'].values, 
                linewidth=2, color='#3498DB', alpha=0.8)
        
        ax.set_xlabel('Week', fontsize=9)
        ax.set_ylabel('Average Commits per Repository', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)
        
        # Add overall average line
        overall_avg = filtered_df['avg_commits'].mean()
        ax.axhline(overall_avg, color='red', linestyle='--', linewidth=1.5, 
                   label=f'6-Month Avg: {overall_avg:.1f}', alpha=0.7)
        ax.legend(loc='upper right', fontsize=7)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_folder, f'{self.plot_prefix}_weekly_commit_patterns_6mo.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_correlation_size_vs_pr_close_time(self):
        """Plot correlation between repo size and PR close time."""
        print("\nGenerating correlation plot: Repo size vs. PR close time...")
        
        # Filter out NaN values
        valid_df = self.df[['size', 'Average time to close a pull request (days)']].dropna()
        
        if len(valid_df) == 0:
            print("  No valid data for correlation plot. Skipping.")
            return
        
        fig, ax = plt.subplots(figsize=(4, 4))
        
        x = valid_df['size'].values / 1024  # Convert to KB
        y = valid_df['Average time to close a pull request (days)'].values
        
        ax.scatter(x, y, alpha=0.5, s=30, color='#3498DB', edgecolors='black', linewidth=0.5)
        
        # Calculate and plot trend line
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.2e}x+{z[1]:.1f}')
            
            # Calculate correlation
            corr = np.corrcoef(x, y)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Repository Size (KB)', fontsize=9)
        ax.set_ylabel('Avg Time to Close PR (days)', fontsize=9)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=7)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_folder, f'{self.plot_prefix}_corr_size_vs_pr_time.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_correlation_contributors_vs_commits(self):
        """Plot correlation between number of contributors and commit frequency."""
        print("\nGenerating correlation plot: Contributors vs. Commit frequency...")
        
        # Filter out NaN values
        valid_df = self.df[['contributors_count', 'Average number of commits per month']].dropna()
        
        if len(valid_df) == 0:
            print("  No valid data for correlation plot. Skipping.")
            return
        
        fig, ax = plt.subplots(figsize=(4, 4))
        
        x = valid_df['contributors_count'].values
        y = valid_df['Average number of commits per month'].values
        
        ax.scatter(x, y, alpha=0.5, s=30, color='#E67E22', edgecolors='black', linewidth=0.5)
        
        # Calculate and plot trend line
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.1f}')
            
            # Calculate correlation
            corr = np.corrcoef(x, y)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Number of Contributors', fontsize=9)
        ax.set_ylabel('Avg Commits per Month', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=7)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_folder, f'{self.plot_prefix}_corr_contributors_vs_commits.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_correlation_stars_vs_pr_activity(self):
        """Plot correlation between stars and PR activity (merged PRs)."""
        print("\nGenerating correlation plot: Stars vs. PR activity...")
        
        # Filter out NaN values
        valid_df = self.df[['Number of stars', 'Number of pull requests merged']].dropna()
        
        if len(valid_df) == 0:
            print("  No valid data for correlation plot. Skipping.")
            return
        
        fig, ax = plt.subplots(figsize=(4, 4))
        
        x = valid_df['Number of stars'].values
        y = valid_df['Number of pull requests merged'].values
        
        ax.scatter(x, y, alpha=0.5, s=30, color='#9B59B6', edgecolors='black', linewidth=0.5)
        
        # Calculate and plot trend line
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.1f}')
            
            # Calculate correlation
            corr = np.corrcoef(x, y)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Number of Stars', fontsize=9)
        ax.set_ylabel('Number of Merged PRs', fontsize=9)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=7)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_folder, f'{self.plot_prefix}_corr_stars_vs_pr_activity.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_correlation_forks_vs_pr_activity(self):
        """Plot correlation between forks and PR activity (merged PRs)."""
        print("\nGenerating correlation plot: Forks vs. PR activity...")
        
        # Filter out NaN values
        valid_df = self.df[['Number of forks', 'Number of pull requests merged']].dropna()
        
        if len(valid_df) == 0:
            print("  No valid data for correlation plot. Skipping.")
            return
        
        fig, ax = plt.subplots(figsize=(4, 4))
        
        x = valid_df['Number of forks'].values
        y = valid_df['Number of pull requests merged'].values
        
        ax.scatter(x, y, alpha=0.5, s=30, color='#27AE60', edgecolors='black', linewidth=0.5)
        
        # Calculate and plot trend line
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.1f}')
            
            # Calculate correlation
            corr = np.corrcoef(x, y)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Number of Forks', fontsize=9)
        ax.set_ylabel('Number of Merged PRs', fontsize=9)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=7)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_folder, f'{self.plot_prefix}_corr_forks_vs_pr_activity.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def generate_plots(self, data_folder=None):
        """Generate basic statistics visualizations."""
        print("\n" + "="*60)
        print("GENERATING BASIC STATISTICS VISUALIZATIONS")
        print("="*60)
        
        # Original plots
        self.plot_size_by_owner_type_boxplot()
        
        # Repository age box plots
        self.plot_repository_age_boxplot_overall()
        self.plot_repository_age_boxplot_by_owner_type()
        
        # Time-based box plots (overall versions)
        self.plot_time_close_pr_boxplot_overall()
        self.plot_time_close_issue_boxplot_overall()
        
        # Time-based box plots (split by owner type)
        self.plot_time_close_pr_boxplot_by_owner_type()
        self.plot_time_close_issue_boxplot_by_owner_type()
        
        # PR distribution box plots (overall and split versions)
        self.plot_pr_status_boxplot_overall()
        self.plot_pr_counts_boxplot_by_owner_type()
        
        # Weekly commit patterns (if data folder provided) - filtered to past 6 months
        if data_folder and os.path.exists(data_folder):
            self.plot_weekly_commit_patterns(data_folder)
        else:
            print("\nNote: Data folder not provided or not found. Skipping weekly commit patterns plot.")
            print("      To enable, provide data_folder parameter with path to ros_robotics_data.")
        
        # Correlation plots
        self.plot_correlation_size_vs_pr_close_time()
        self.plot_correlation_contributors_vs_commits()
        self.plot_correlation_stars_vs_pr_activity()
        self.plot_correlation_forks_vs_pr_activity()
        
        print("\n" + "="*60)
        print(f"Plots saved to: {os.path.abspath(self.output_folder)}")
        print("="*60)
        print("\nNote: 'Average time to RESPOND to pull requests' cannot be calculated")
        print("      from the available data. The JSON files only contain basic PR info")
        print("      (created_at, closed_at, merged_at) without timeline/comment data.")
        print("      To get response times, you would need to collect PR review/comment")
        print("      timestamps from the GitHub API.")
        print("\nTime-based plots are filtered to the past 6 months (Sept 3, 2025 - March 3, 2026).")
        print("All average bar plots have been converted to box plots for better distribution visualization.")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filtered_repos_csv = os.path.join(script_dir, '..', '..', 'out', 'filtered_repo_dataset.csv')
    inflow_csv = os.path.join(script_dir, '..', 'clustering', 'tables', 'inflow.csv')
    output_folder = os.path.join(script_dir, 'plots')
    data_folder = os.path.join(script_dir, '..', 'data', 'ros_robotics_data')
    
    if not os.path.exists(filtered_repos_csv):
        print(f"ERROR: Could not find filtered_repo_dataset.csv at: {filtered_repos_csv}")
        print("Please run the pipeline first to generate the data.")
    else:
        visualizer = BasicStatisticsVisualizer(filtered_repos_csv, output_folder, inflow_csv=inflow_csv)
        visualizer.generate_plots(data_folder=data_folder)
