#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Elijah Phifer"
__contact__ = "elijah.phifer@lsu.edu"

import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

class InflowVisualizer:
    def __init__(self, csv_path, output_folder='plots', filtered_repos_csv=None, plot_prefix='00_plot_inflow__00'):
        """
        Initialize the visualizer with inflow data.
        
        Args:
            csv_path: Path to the inflow.csv file
            output_folder: Folder to save the plots
            filtered_repos_csv: Path to filtered_repo_dataset.csv for distribution mapping
            plot_prefix: Prefix for plot filenames
        """
        self.csv_path = csv_path
        self.output_folder = output_folder
        self.filtered_repos_csv = filtered_repos_csv
        self.plot_prefix = plot_prefix
        
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")
        
        # Load the data
        print(f"Loading data from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        
        # Extract week information and convert to proper format
        self.week_columns = [col for col in self.df.columns if col != 'project']
        
        print(f"Loaded {len(self.df)} repositories with {len(self.week_columns)} weeks of data")
        
        # Convert week tuples to readable format
        self.week_labels = []
        for col in self.week_columns:
            # Column format is (week, year) as string
            week_str = col.strip('()').replace(' ', '')
            parts = week_str.split(',')
            if len(parts) == 2:
                week, year = int(parts[0]), int(parts[1])
                self.week_labels.append(f"{year}-W{week:02d}")
            else:
                self.week_labels.append(col)
        
        # Extract ROS distribution information
        print("Extracting ROS distribution information...")
        self.repo_distributions = self._extract_distributions()
        self._print_distribution_summary()
        
        # Create monthly aggregated data
        self.monthly_df = self._aggregate_to_monthly()
        self.month_columns = [col for col in self.monthly_df.columns if col != 'project']
        print(f"Aggregated to {len(self.month_columns)} months of data")
    
    def _extract_distributions(self):
        """
        Extract ROS distribution for each repository from filtered_repo_dataset.csv.
        Returns a dictionary mapping project names to distributions.
        """
        if not self.filtered_repos_csv or not os.path.exists(self.filtered_repos_csv):
            print(f"Warning: Filtered repos CSV not found: {self.filtered_repos_csv}")
            print("All repositories will be marked as 'unknown'")
            return {row['project']: 'unknown' for _, row in self.df.iterrows()}
        
        # Load the filtered repos dataset
        print(f"Loading distribution data from: {self.filtered_repos_csv}")
        filtered_df = pd.read_csv(self.filtered_repos_csv)
        
        # Filter to only include repositories in inflow.csv
        allowed_projects = set(self.df['project'].values)
        filtered_df['project_key'] = filtered_df['Owner'] + '/' + filtered_df['Name']
        filtered_df = filtered_df[filtered_df['project_key'].isin(allowed_projects)]
        print(f"  Filtered to {len(filtered_df)} repositories from inflow.csv")
        
        # Create a mapping using Owner/Name format to match inflow.csv project names
        distro_mapping = {}
        for _, row in filtered_df.iterrows():
            owner = row['Owner']
            name = row['Name']
            project_key = f"{owner}/{name}"
            distros = row.get('distros_present', '')
            
            # If multiple distributions, pick the most recent one for single-distro assignment
            if distros and '|' in distros:
                # For repos with multiple distros, pick the most recent one
                distro_list = distros.split('|')
                # Preference order: kilted > jazzy > humble > iron > rolling > galactic > foxy
                distro_priority = ['kilted', 'jazzy', 'humble', 'iron', 'rolling', 'galactic', 'foxy']
                assigned_distro = None
                for d in distro_priority:
                    if d in distro_list:
                        assigned_distro = d
                        break
                distro_mapping[project_key] = assigned_distro if assigned_distro else distro_list[0]
            elif distros:
                distro_mapping[project_key] = distros
            else:
                distro_mapping[project_key] = 'unknown'
        
        # Map each project in inflow.csv to its distribution
        repo_distros = {}
        unmapped_count = 0
        for _, row in self.df.iterrows():
            project = row['project']
            if project in distro_mapping:
                repo_distros[project] = distro_mapping[project]
            else:
                repo_distros[project] = 'unknown'
                unmapped_count += 1
                print(f"  Warning: Could not find distribution for: {project}")
        
        if unmapped_count > 0:
            print(f"  Total unmapped repositories: {unmapped_count}")
        
        return repo_distros
    
    def _extract_all_distributions(self):
        """
        Extract ALL ROS distributions for each repository (for multi-distro analysis).
        Returns a dictionary mapping project names to list of distributions.
        """
        if not self.filtered_repos_csv or not os.path.exists(self.filtered_repos_csv):
            print(f"Warning: Filtered repos CSV not found: {self.filtered_repos_csv}")
            return {row['project']: ['unknown'] for _, row in self.df.iterrows()}
        
        # Load the filtered repos dataset
        filtered_df = pd.read_csv(self.filtered_repos_csv)
        
        # Filter to only include repositories in inflow.csv
        allowed_projects = set(self.df['project'].values)
        filtered_df['project_key'] = filtered_df['Owner'] + '/' + filtered_df['Name']
        filtered_df = filtered_df[filtered_df['project_key'].isin(allowed_projects)]
        
        # Create a mapping using Owner/Name format to match inflow.csv project names
        distro_mapping = {}
        for _, row in filtered_df.iterrows():
            owner = row['Owner']
            name = row['Name']
            project_key = f"{owner}/{name}"
            distros = row.get('distros_present', '')
            
            if distros and '|' in distros:
                # Multiple distributions - return as list
                distro_mapping[project_key] = distros.split('|')
            elif distros:
                # Single distribution
                distro_mapping[project_key] = [distros]
            else:
                distro_mapping[project_key] = ['unknown']
        
        # Map each project in inflow.csv to its distributions
        repo_distros = {}
        for _, row in self.df.iterrows():
            project = row['project']
            if project in distro_mapping:
                repo_distros[project] = distro_mapping[project]
            else:
                repo_distros[project] = ['unknown']
        
        return repo_distros
    
    def _categorize_single_vs_multi_distro(self):
        """
        Categorize repositories as single-distro or multi-distro.
        Returns a dictionary mapping project names to category.
        """
        all_distros = self._extract_all_distributions()
        
        categorized = {}
        for project, distros in all_distros.items():
            if len(distros) == 1:
                categorized[project] = distros[0]
            else:
                categorized[project] = 'multi-distro'
        
        return categorized
    
    def _extract_owner_types(self):
        """
        Extract owner type (User/Organization) for each repository from filtered_repo_dataset.csv.
        Returns a dictionary mapping project names to owner types.
        """
        if not self.filtered_repos_csv or not os.path.exists(self.filtered_repos_csv):
            print(f"Warning: Filtered repos CSV not found: {self.filtered_repos_csv}")
            return {row['project']: 'unknown' for _, row in self.df.iterrows()}
        
        # Load the filtered repos dataset
        filtered_df = pd.read_csv(self.filtered_repos_csv)
        
        # Filter to only include repositories in inflow.csv
        allowed_projects = set(self.df['project'].values)
        filtered_df['project_key'] = filtered_df['Owner'] + '/' + filtered_df['Name']
        filtered_df = filtered_df[filtered_df['project_key'].isin(allowed_projects)]
        
        # Create a mapping using Owner/Name format to match inflow.csv project names
        owner_type_mapping = {}
        for _, row in filtered_df.iterrows():
            owner = row['Owner']
            name = row['Name']
            project_key = f"{owner}/{name}"
            owner_type = row.get('owner_type', 'unknown')
            
            # Normalize owner type
            if owner_type in ['User', 'Organization']:
                owner_type_mapping[project_key] = owner_type
            else:
                owner_type_mapping[project_key] = 'unknown'
        
        # Map each project in inflow.csv to its owner type
        repo_owner_types = {}
        unmapped_count = 0
        for _, row in self.df.iterrows():
            project = row['project']
            if project in owner_type_mapping:
                repo_owner_types[project] = owner_type_mapping[project]
            else:
                repo_owner_types[project] = 'unknown'
                unmapped_count += 1
        
        if unmapped_count > 0:
            print(f"  Could not find owner type for {unmapped_count} repositories")
        
        return repo_owner_types
    
    def _print_distribution_summary(self):
        """
        Print summary of distribution counts.
        """
        distro_counts = defaultdict(int)
        for distro in self.repo_distributions.values():
            distro_counts[distro] += 1
        
        print("\nDistribution Summary:")
        for distro in sorted(distro_counts.keys()):
            print(f"  {distro.capitalize():10} - {distro_counts[distro]:3d} repositories")
    
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
    
    def plot_aggregate_inflow(self, use_monthly=False):
        """
        Plot total inflow across all repositories.
        """
        period_type = "monthly" if use_monthly else "weekly"
        print(f"\nGenerating aggregate {period_type} inflow plot...")
        
        df_to_use = self.monthly_df if use_monthly else self.df
        period_columns = self.month_columns if use_monthly else self.week_columns
        
        # Calculate total inflow per period
        period_totals = df_to_use[period_columns].sum(axis=0)
        
        plt.figure(figsize=(7, 4.5))
        plt.plot(range(len(period_totals)), period_totals.values, linewidth=3, color='black')
        
        period_label = 'Month' if use_monthly else 'Week'
        plt.xlabel(period_label, fontsize=14)
        plt.ylabel('Total Newcomers Across All Repositories', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Show period numbers on x-axis
        num_periods = len(period_totals)
        step = max(1, num_periods // 10)
        tick_positions = range(0, num_periods, step)
        plt.xticks(tick_positions, tick_positions)
        
        plt.tight_layout()
        
        suffix = "monthly" if use_monthly else "weekly"
        output_path_png = os.path.join(self.output_folder, f'{self.plot_prefix}_aggregate_{suffix}.png')
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path_png}")
        
        plt.close()
    
    def plot_all_repositories(self, use_monthly=False):
        """
        Plot all repositories on the same chart.
        """
        period_type = "monthly" if use_monthly else "weekly"
        print(f"\nGenerating {period_type} plot with all {len(self.df)} repositories...")
        
        df_to_use = self.monthly_df if use_monthly else self.df
        period_columns = self.month_columns if use_monthly else self.week_columns
        
        plt.figure(figsize=(7, 5))
        
        # Plot each repository with very thin, semi-transparent lines
        for idx, (_, row) in enumerate(df_to_use.iterrows()):
            period_data = row[period_columns].values
            plt.plot(range(len(period_data)), period_data, 
                    linewidth=0.5, alpha=0.9, color='black')
        
        period_label = 'Month' if use_monthly else 'Week'
        plt.xlabel(period_label, fontsize=14)
        plt.ylabel('Number of Newcomers', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Show period numbers on x-axis
        num_periods = len(period_data)
        step = max(1, num_periods // 10)
        tick_positions = range(0, num_periods, step)
        plt.xticks(tick_positions, tick_positions)
        
        plt.tight_layout()
        
        suffix = "monthly" if use_monthly else "weekly"
        output_path_png = os.path.join(self.output_folder, f'{self.plot_prefix}_all_repositories_{suffix}.png')
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path_png}")
        
        plt.close()
    
    
    def plot_distribution_inflow_multi_category(self, use_monthly=False):
        """
        Plot aggregate inflow by ROS distribution.
        There is a 'multi-distro' category for repos supporting multiple distributions.
        """
        period_type = "monthly" if use_monthly else "weekly"
        print(f"\nGenerating distribution-based {period_type} inflow plot (multi-distro as separate category)...")
        
        df_to_use = self.monthly_df if use_monthly else self.df
        period_columns = self.month_columns if use_monthly else self.week_columns
        
        # Get categorization (single distro name or 'multi-distro')
        repo_categories = self._categorize_single_vs_multi_distro()
        
        # Group repositories by category
        category_data = defaultdict(lambda: [0] * len(period_columns))
        category_repo_counts = defaultdict(int)
        
        for _, row in df_to_use.iterrows():
            project = row['project']
            category = repo_categories.get(project, 'unknown')
            period_data = row[period_columns].values
            
            category_repo_counts[category] += 1
            for i, value in enumerate(period_data):
                category_data[category][i] += value
        
        # Print repo counts per category
        print("\nRepository counts per category:")
        for category in sorted(category_repo_counts.keys()):
            print(f"  {category.capitalize():15} - {category_repo_counts[category]:3d} repositories")
        
        # Define colors and line styles for each category
        category_colors = {
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
        
        category_linestyles = {
            'foxy': '-',
            'galactic': ':',
            'humble': '-.',
            'iron': (0, (3, 1, 1, 1)),  # dash-dot-dot pattern
            'jazzy': '--',
            'kilted': (0, (5, 2)),  # custom dash pattern
            'rolling': (0, (3, 5, 1, 5)),  # dash-dot pattern with spacing
            'multi-distro': '-',  # solid line
            'unknown': (0, (1, 1))  # densely dotted
        }
        
        # Create the plot (sized for single column)
        plt.figure(figsize=(3.5, 2.5))
        
        # Plot multi-distro first so it's in the background
        plot_order = ['multi-distro'] + [c for c in sorted(category_data.keys()) if c not in ['multi-distro', 'unknown']] + ['unknown']
        
        for category in plot_order:
            if category not in category_data:
                continue
            
            weekly_totals = category_data[category]
            color = category_colors.get(category, '#000000')
            linestyle = category_linestyles.get(category, '-')
            
            if category == 'multi-distro':
                label = f"Multi-Distribution (n={category_repo_counts[category]})"
                linewidth = 2
            elif category == 'unknown':
                label = f"Unknown/Other (n={category_repo_counts[category]})"
                linewidth = 1.5
            else:
                label = f"{category.capitalize()} only (n={category_repo_counts[category]})"
                linewidth = 1.5
            
            period_totals = category_data[category]
            plt.plot(range(len(period_totals)), period_totals, 
                    linewidth=linewidth, alpha=0.8, color=color, label=label, linestyle=linestyle)
        
        period_label = 'Month' if use_monthly else 'Week'
        plt.xlabel(period_label, fontsize=13)
        plt.ylabel('Number of Newcomers', fontsize=12)
        plt.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=2, 
                  fontsize=8.5, framealpha=0, handlelength=1, 
                  handletextpad=0.4, borderpad=0)
        plt.grid(True, alpha=0.3)
        
        # Show period numbers on x-axis
        num_periods = len(period_totals)
        tick_step = 4 if use_monthly else 2
        tick_positions = range(0, num_periods, tick_step)
        tick_labels = range(0, num_periods, tick_step)
        plt.xticks(tick_positions, tick_labels, rotation=0, fontsize=7.5)
        plt.yticks(fontsize=10)
        
        plt.tight_layout()
        
        suffix = "monthly" if use_monthly else "weekly"
        output_path_png = os.path.join(self.output_folder, 'newcomer_inflow_distribution.png')
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path_png}")
        
        plt.close()
    
    def plot_owner_type_inflow(self, use_monthly=False):
        """
        Plot aggregate inflow by owner type (User vs Organization).
        """
        period_type = "monthly" if use_monthly else "weekly"
        print(f"\nGenerating owner type {period_type} inflow plot...")
        
        df_to_use = self.monthly_df if use_monthly else self.df
        period_columns = self.month_columns if use_monthly else self.week_columns
        
        # Get owner types for each repo
        repo_owner_types = self._extract_owner_types()
        
        # Group repositories by owner type
        owner_type_data = defaultdict(lambda: [0] * len(period_columns))
        owner_type_repo_counts = defaultdict(int)
        
        for _, row in df_to_use.iterrows():
            project = row['project']
            owner_type = repo_owner_types.get(project, 'unknown')
            period_data = row[period_columns].values
            
            owner_type_repo_counts[owner_type] += 1
            for i, value in enumerate(period_data):
                owner_type_data[owner_type][i] += value
        
        # Print repo counts per owner type
        print("\nRepository counts per owner type:")
        for owner_type in sorted(owner_type_repo_counts.keys()):
            print(f"  {owner_type:15} - {owner_type_repo_counts[owner_type]:3d} repositories")
        
        # Define colors and line styles for each owner type
        owner_type_colors = {
            'Organization': '#3498DB',  # Blue
            'User': '#E67E22',          # Orange
            'unknown': '#95A5A6'        # Gray
        }
        
        owner_type_linestyles = {
            'Organization': '-',
            'User': '--',
            'unknown': ':'
        }
        
        # Create the plot (sized for single column)
        plt.figure(figsize=(3.5, 2.5))
        
        # Plot each owner type
        plot_order = ['Organization', 'User', 'unknown']
        for owner_type in plot_order:
            if owner_type not in owner_type_data:
                continue
            
            period_totals = owner_type_data[owner_type]
            color = owner_type_colors.get(owner_type, '#000000')
            linestyle = owner_type_linestyles.get(owner_type, '-')
            label = f"{owner_type} (n={owner_type_repo_counts[owner_type]})"
            
            if owner_type == 'Organization':
                linewidth = 2
            else:
                linewidth = 1.5
            
            plt.plot(range(len(period_totals)), period_totals, 
                    linewidth=linewidth, alpha=0.8, color=color, label=label, linestyle=linestyle)
        
        period_label = 'Month' if use_monthly else 'Week'
        plt.xlabel(period_label, fontsize=13)
        plt.ylabel('Number of Newcomers', fontsize=12)
        plt.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=2, 
                  fontsize=8.5, framealpha=0, handlelength=1, 
                  handletextpad=0.4, borderpad=0)
        plt.grid(True, alpha=0.3)
        
        # Show period numbers on x-axis
        num_periods = len(period_totals)
        tick_step = 4 if use_monthly else 2
        tick_positions = range(0, num_periods, tick_step)
        tick_labels = range(0, num_periods, tick_step)
        plt.xticks(tick_positions, tick_labels, rotation=0, fontsize=7)
        plt.yticks(fontsize=8)
        
        plt.tight_layout()
        
        suffix = "monthly" if use_monthly else "weekly"
        output_path_png = os.path.join(self.output_folder, 'newcomer_inflow_owner.png')
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path_png}")
        
        plt.close()
    
    def generate_plot(self):
        """
        Generate the visualizations (both weekly and monthly versions).
        """
        print("\n" + "="*60)
        print("GENERATING NEWCOMER INFLOW VISUALIZATIONS")
        print("="*60)
        
        # Weekly plots (original default behavior)
        # self.plot_aggregate_inflow()
        # self.plot_all_repositories()
        self.plot_distribution_inflow_multi_category()
        self.plot_owner_type_inflow()
        
        # Monthly plots (new)
        # self.plot_aggregate_inflow(use_monthly=True)
        # self.plot_all_repositories(use_monthly=True)
        # self.plot_distribution_inflow_multi_category(use_monthly=True)
        # self.plot_owner_type_inflow(use_monthly=True)
        
        print("\n" + "="*60)
        print(f"Plots saved to: {os.path.abspath(self.output_folder)}")
        print("="*60)

if __name__ == '__main__':
    # Use paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'tables', 'inflow.csv')
    output_folder = os.path.join(script_dir, '..', 'figs')
    filtered_repos_csv = os.path.join(script_dir, '..', 'out', 'filtered_repo_dataset.csv')
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"ERROR: Could not find inflow.csv at: {csv_path}")
        print("Please run 00_inflow.py first to generate the data.")
    else:
        visualizer = InflowVisualizer(csv_path, output_folder, filtered_repos_csv)
        visualizer.generate_plot()
