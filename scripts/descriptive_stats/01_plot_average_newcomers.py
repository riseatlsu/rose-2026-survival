#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Elijah Phifer"
__contact__ = "elijah.phifer@lsu.edu"

"""
Plot average (normalized) newcomer inflow rates.

This provides a more scientifically sound comparison by normalizing for
the number of repositories in each category.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime, timedelta

class AverageInflowVisualizer:
    def __init__(self, csv_path, output_folder='plots', filtered_repos_csv=None, plot_prefix='01_plot_average__00'):
        self.csv_path = csv_path
        self.output_folder = output_folder
        self.filtered_repos_csv = filtered_repos_csv
        self.plot_prefix = plot_prefix
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")
        
        print(f"Loading data from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        
        # Get list of allowed projects from inflow.csv for consistent filtering
        self.allowed_projects = set(self.df['project'].values)
        print(f"Filtering to {len(self.allowed_projects)} repositories from inflow.csv")
        
        # Extract week information and filter to past 6 months
        all_week_columns = [col for col in self.df.columns if col != 'project']
        
        # Filter to past 6 months only
        cutoff_date = datetime(2026, 3, 5) - timedelta(days=180)
        self.week_columns = []
        
        for col in all_week_columns:
            week_str = col.strip('()').replace(' ', '')
            parts = week_str.split(',')
            if len(parts) == 2:
                week, year = int(parts[0]), int(parts[1])
                # Convert ISO week to approximate date
                week_date = datetime.strptime(f'{year}-W{week:02d}-1', '%Y-W%W-%w').date()
                if week_date >= cutoff_date.date():
                    self.week_columns.append(col)
            else:
                self.week_columns.append(col)  # Keep unknown format
        
        # Filter dataframe to only include past 6 months columns
        self.df = self.df[['project'] + self.week_columns]
        
        print(f"Loaded {len(self.df)} repositories with {len(self.week_columns)} weeks of data (past 6 months)")
        
        # Convert week labels
        self.week_labels = []
        for col in self.week_columns:
            week_str = col.strip('()').replace(' ', '')
            parts = week_str.split(',')
            if len(parts) == 2:
                week, year = int(parts[0]), int(parts[1])
                self.week_labels.append(f"{year}-W{week:02d}")
            else:
                self.week_labels.append(col)
        
        print("Loading repository metadata...")
        self.repo_distributions = self._extract_all_distributions()
        self.repo_owner_types = self._extract_owner_types()
        
        # Create monthly aggregated data
        self.monthly_df = self._aggregate_to_monthly()
        self.month_columns = [col for col in self.monthly_df.columns if col != 'project']
        print(f"Aggregated to {len(self.month_columns)} months of data")
    
    def _extract_all_distributions(self):
        """Extract all distributions for each repository."""
        if not self.filtered_repos_csv or not os.path.exists(self.filtered_repos_csv):
            return {row['project']: ['unknown'] for _, row in self.df.iterrows()}
        
        filtered_df = pd.read_csv(self.filtered_repos_csv)
        
        # Filter to only include repositories in inflow.csv
        filtered_df['project_key'] = filtered_df['Owner'] + '/' + filtered_df['Name']
        filtered_df = filtered_df[filtered_df['project_key'].isin(self.allowed_projects)]
        print(f"  Filtered to {len(filtered_df)} repositories from filtered_repos_csv")
        
        distro_mapping = {}
        
        for _, row in filtered_df.iterrows():
            owner = row['Owner']
            name = row['Name']
            project_key = f"{owner}/{name}"
            distros = row.get('distros_present', '')
            
            if distros and '|' in distros:
                distro_mapping[project_key] = distros.split('|')
            elif distros:
                distro_mapping[project_key] = [distros]
            else:
                distro_mapping[project_key] = ['unknown']
        
        repo_distros = {}
        for _, row in self.df.iterrows():
            project = row['project']
            repo_distros[project] = distro_mapping.get(project, ['unknown'])
        
        return repo_distros
    
    def _extract_owner_types(self):
        """Extract owner type for each repository."""
        if not self.filtered_repos_csv or not os.path.exists(self.filtered_repos_csv):
            return {row['project']: 'unknown' for _, row in self.df.iterrows()}
        
        filtered_df = pd.read_csv(self.filtered_repos_csv)
        
        # Filter to only include repositories in inflow.csv
        filtered_df['project_key'] = filtered_df['Owner'] + '/' + filtered_df['Name']
        filtered_df = filtered_df[filtered_df['project_key'].isin(self.allowed_projects)]
        
        owner_type_mapping = {}
        
        for _, row in filtered_df.iterrows():
            owner = row['Owner']
            name = row['Name']
            project_key = f"{owner}/{name}"
            owner_type = row.get('owner_type', 'unknown')
            owner_type_mapping[project_key] = owner_type if owner_type in ['User', 'Organization'] else 'unknown'
        
        repo_owner_types = {}
        for _, row in self.df.iterrows():
            project = row['project']
            repo_owner_types[project] = owner_type_mapping.get(project, 'unknown')
        
        return repo_owner_types
    
    def _aggregate_to_monthly(self):
        """Aggregate weekly data to monthly data."""
        print("Aggregating weekly data to monthly...")
        
        monthly_data = {'project': self.df['project'].values}
        
        # Group weeks by year-month
        month_groups = defaultdict(list)
        for col in self.week_columns:
            week_str = col.strip('()').replace(' ', '')
            parts = week_str.split(',')
            if len(parts) == 2:
                week, year = int(parts[0]), int(parts[1])
                # Approximate month from week number
                month = ((week - 1) // 4) + 1  # Rough approximation: 4 weeks per month
                month = min(month, 12)  # Cap at 12
                month_key = f"{year}-{month:02d}"
                month_groups[month_key].append(col)
        
        # Sum values for each month
        for month_key in sorted(month_groups.keys()):
            week_cols = month_groups[month_key]
            monthly_data[month_key] = self.df[week_cols].sum(axis=1).values
        
        return pd.DataFrame(monthly_data)
    
    def plot_average_overall(self, use_monthly=False):
        """Plot overall average newcomers per repository across all time periods."""
        period_type = "Monthly" if use_monthly else "Weekly"
        print(f"\nGenerating overall average {period_type.lower()} newcomers per repo...")
        
        df_to_use = self.monthly_df if use_monthly else self.df
        period_columns = self.month_columns if use_monthly else self.week_columns
        
        # Calculate mean across all repos for each time period
        period_means = []
        for col in period_columns:
            period_means.append(df_to_use[col].mean())
        
        # Create plot (sized for single column)
        plt.figure(figsize=(4.5, 3.5))
        
        x = range(len(period_means))
        plt.plot(x, period_means, linewidth=2, color='black', marker='o', markersize=3)
        
        period_label = 'Month' if use_monthly else 'Week'
        plt.xlabel(period_label, fontsize=9)
        plt.ylabel(f'Average Newcomers per Repository', fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # Set x-axis ticks
        num_periods = len(period_means)
        tick_step = 4 if use_monthly else 2
        tick_positions = range(0, num_periods, tick_step)
        tick_labels = range(0, num_periods, tick_step)
        plt.xticks(tick_positions, tick_labels, rotation=0, fontsize=8)
        plt.yticks(fontsize=8)
        
        plt.tight_layout()
        
        suffix = "monthly" if use_monthly else "weekly"
        output_path_png = os.path.join(self.output_folder, f'{self.plot_prefix}_avg_overall_{suffix}.png')
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path_png}")
        
        plt.close()
    
    def plot_average_by_owner_type(self, use_monthly=False):
        """Plot average newcomers per repository by owner type."""
        period_type = "monthly" if use_monthly else "weekly"
        print(f"\nGenerating average {period_type} newcomers per repo by owner type...")
        
        df_to_use = self.monthly_df if use_monthly else self.df
        period_columns = self.month_columns if use_monthly else self.week_columns
        
        # Collect data per owner type
        owner_type_weekly_data = defaultdict(list)
        
        for _, row in df_to_use.iterrows():
            project = row['project']
            owner_type = self.repo_owner_types.get(project, 'unknown')
            period_data = row[period_columns].values
            owner_type_weekly_data[owner_type].append(period_data)
        
        # Calculate mean and std for each owner type
        owner_type_stats = {}
        for owner_type, data_list in owner_type_weekly_data.items():
            data_array = np.array(data_list, dtype=np.float64)
            
            # Handle case of single repository
            if len(data_list) == 1:
                means = data_array[0]
                sems = np.zeros_like(means, dtype=np.float64)  # No standard error for single observation
            else:
                means = np.mean(data_array, axis=0)
                # Ensure we get an array back, not a scalar
                stds = np.std(data_array, axis=0, ddof=1, dtype=np.float64)
                if np.isscalar(stds):
                    stds = np.full_like(means, stds, dtype=np.float64)
                sems = stds / np.sqrt(len(data_list))
            
            owner_type_stats[owner_type] = {
                'mean': means,
                'sem': sems,
                'count': len(data_list)
            }
        
        # Define colors and line styles
        owner_type_colors = {
            'Organization': '#3498DB',
            'User': '#E67E22',
            'unknown': '#95A5A6'
        }
        
        owner_type_linestyles = {
            'Organization': '-',
            'User': '--',
            'unknown': ':'
        }
        
        # Create plot (sized for single column, 20% wider and 20% shorter)
        plt.figure(figsize=(4.2, 2.8))
        
        plot_order = ['Organization', 'User', 'unknown']
        for owner_type in plot_order:
            if owner_type not in owner_type_stats:
                continue
            
            stats = owner_type_stats[owner_type]
            color = owner_type_colors.get(owner_type, '#000000')
            linestyle = owner_type_linestyles.get(owner_type, '-')
            label = f"{owner_type} (n={stats['count']})"
            linewidth = 2 if owner_type == 'Organization' else 1.5
            
            x = range(len(stats['mean']))
            plt.plot(x, stats['mean'], linewidth=linewidth, alpha=0.8, color=color, label=label, linestyle=linestyle)
        
        period_label = 'Month' if use_monthly else 'Week'
        plt.xlabel(period_label, fontsize=9)
        plt.ylabel('Average Newcomers per Repository', fontsize=9)
        plt.legend(loc='upper left', fontsize=7, framealpha=0.9, handlelength=2.8, 
                  handletextpad=0.4, borderpad=0.25)
        plt.grid(True, alpha=0.3)
        
        num_periods = len(stats['mean'])
        tick_step = 4 if use_monthly else 2
        tick_positions = range(0, num_periods, tick_step)
        tick_labels = range(0, num_periods, tick_step)
        plt.xticks(tick_positions, tick_labels, rotation=0, fontsize=7)
        plt.yticks(fontsize=8)
        
        plt.tight_layout()
        
        suffix = "monthly" if use_monthly else "weekly"
        output_path_png = os.path.join(self.output_folder, f'{self.plot_prefix}_avg_by_owner_type_{suffix}.png')
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path_png}")
        
        plt.close()
    
    def generate_plots(self):
        """Generate average newcomer visualizations."""
        print("\\n" + "="*60)
        print("GENERATING AVERAGE NEWCOMER RATE VISUALIZATIONS")
        print("="*60)
        
        # Weekly plots (original default behavior)
        self.plot_average_overall()
        self.plot_average_by_owner_type()
        
        # Monthly plots (new)
        self.plot_average_overall(use_monthly=True)
        self.plot_average_by_owner_type(use_monthly=True)
        
        print("\\n" + "="*60)
        print(f"Plots saved to: {os.path.abspath(self.output_folder)}")
        print("="*60)


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, '..', 'clustering', 'tables', 'inflow.csv')
    output_folder = os.path.join(script_dir, 'plots')
    filtered_repos_csv = os.path.join(script_dir, '..', '..', 'out', 'filtered_repo_dataset.csv')
    
    if not os.path.exists(csv_path):
        print(f"ERROR: Could not find inflow.csv at: {csv_path}")
        print("Please run 00_inflow.py first to generate the data.")
    else:
        visualizer = AverageInflowVisualizer(csv_path, output_folder, filtered_repos_csv)
        visualizer.generate_plots()
