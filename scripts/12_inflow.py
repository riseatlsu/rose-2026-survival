#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ =  "Elijah Phifer"
__contact__ = "elijah.phifer@lsu.edu"

import os
import csv
import json
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
from collections import Counter

class NewcomersInflow():
    def __init__(self, dataset_folder, csv_folder, min_age_months, filtered_repos_csv):
        self.csv_folder = csv_folder
        self.dataset_folder = dataset_folder
        self.min_age_months = min_age_months
        self.filtered_repos_csv = filtered_repos_csv
        
        if not filtered_repos_csv or not os.path.exists(filtered_repos_csv):
            print(f"ERROR: Filtered repos CSV not found: {filtered_repos_csv}")
            return
        
        # Load projects from filtered CSV and check their ages
        projects = self.load_and_filter_projects()
        
        if not projects:
            print("No projects with valid commits found after filtering!")
            return
        
        # Find youngest (latest first commit) and oldest (earliest first commit) repositories
        youngest_repo = None  # Latest first commit
        oldest_repo = None    # Earliest first commit
        
        for project_info in projects:
            # Track youngest (latest first commit) and oldest (earliest first commit)
            if youngest_repo is None or project_info['first_commit_date'] > youngest_repo['first_commit_date']:
                youngest_repo = project_info
            if oldest_repo is None or project_info['first_commit_date'] < oldest_repo['first_commit_date']:
                oldest_repo = project_info
        
        print(f"\nRepository Statistics:")
        if youngest_repo:
            print(f"  Youngest repository: {youngest_repo['full_name']} (first commit: {youngest_repo['first_commit_date']})")
        if oldest_repo:
            print(f"  Oldest repository: {oldest_repo['full_name']} (first commit: {oldest_repo['first_commit_date']})")
        
        # Use the YOUNGEST repo's first commit date as the start point
        if youngest_repo and youngest_repo['first_commit_date']:
            self.start_date = youngest_repo['first_commit_date']
            print(f"\nCounting newcomers from: {self.start_date} (youngest repo's first commit)\n")
        else:
            print("Error: No valid start date found!")
            return
        
        # Track latest commit across all repos
        self.latest_commit_date = None
        
        weekly_series = self.get_weekly_series(projects)
        weekly_min, weekly_max = self.get_number_of_weeks(weekly_series)
        
        # Print commit date range
        print(f"\nCommit Date Statistics:")
        print(f"  Oldest commit in dataset: {weekly_min}")
        print(f"  Newest commit in dataset: {self.latest_commit_date if self.latest_commit_date else weekly_max}")
        
        # Export weekly inflow
        self.export_newcomers_inflow(weekly_series, weekly_min, weekly_max)
        
        # Export monthly inflow
        # self.export_monthly_inflow(weekly_series, weekly_min, weekly_max)
    
    def load_and_filter_projects(self):
        """
        Load repositories from the filtered CSV and filter by age.
        Returns list of project info dicts with folder paths and first commit dates.
        """
        # Determine the data folder path
        if self.dataset_folder.endswith('ros_robotics_data'):
            ros_data_folder = self.dataset_folder
        else:
            ros_data_folder = os.path.join(self.dataset_folder, 'ros_robotics_data')
        
        print(f"Reading filtered repositories from: {self.filtered_repos_csv}")
        print(f"Looking for commit data in: {os.path.abspath(ros_data_folder)}\n")
        
        if not os.path.exists(ros_data_folder):
            print(f"ERROR: Data directory does not exist: {os.path.abspath(ros_data_folder)}")
            return []
        
        # Calculate age cutoff
        cutoff_date = datetime.now().date() - relativedelta(months=self.min_age_months)
        print(f"Filtering: keeping only repositories with first commit before {cutoff_date} ({self.min_age_months} months ago)\n")
        
        projects = []
        repos_in_csv = 0
        repos_with_commits = 0
        repos_old_enough = 0
        
        try:
            with open(self.filtered_repos_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    repos_in_csv += 1
                    
                    # Get owner and repo name
                    owner = row.get('Owner', '').strip()
                    repo_name = row.get('Name', '').strip()
                    
                    if not owner or not repo_name:
                        continue
                    
                    full_name = f"{owner}/{repo_name}"
                    
                    # Build path to commits.json
                    folder_name = f"{owner}__{repo_name}"
                    project_path = os.path.join(ros_data_folder, folder_name)
                    commits_file = os.path.join(project_path, 'commits.json')
                    
                    # Check if commits.json exists
                    if not os.path.isfile(commits_file):
                        print(f"  Warning: {full_name} - no commits.json found, excluding")
                        continue
                    
                    # Get first commit date
                    first_commit_date = self.get_first_commit_date(commits_file)
                    
                    if not first_commit_date:
                        print(f"  Warning: {full_name} - no valid commits found, excluding")
                        continue
                    
                    repos_with_commits += 1
                    
                    # Check age
                    if first_commit_date <= cutoff_date:
                        repos_old_enough += 1
                        projects.append({
                            'full_name': full_name,
                            'folder_path': project_path,
                            'first_commit_date': first_commit_date
                        })
                    else:
                        print(f"  Excluding {full_name} (first commit {first_commit_date}, too young)")
        
        except Exception as e:
            print(f"ERROR reading filtered CSV: {e}")
            return []
        
        print(f"\nFiltering Summary:")
        print(f"  Repositories in filtered CSV: {repos_in_csv}")
        print(f"  Repositories with commit data: {repos_with_commits}")
        print(f"  Repositories old enough (>= {self.min_age_months} months): {repos_old_enough}")
        
        return projects

    def get_first_commit_date(self, commits_file_path):
        """
        Find the first (earliest) commit date in a repository.
        Returns None if no commits found.
        """
        try:
            with open(commits_file_path, 'r', encoding='utf-8') as f:
                commits_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not read {commits_file_path}: {e}")
            return None
        
        # Extract commits from the data field if present
        if isinstance(commits_data, dict) and 'data' in commits_data:
            commits_list = commits_data['data']
        elif isinstance(commits_data, list):
            commits_list = commits_data
        else:
            return None
        
        earliest_date = None
        
        for commit in commits_list:
            # Handle multiple formats:
            # 1. New simplified format: {date: "...", author: "..."}
            # 2. Old nested format: {commit: {author: {date: "..."}}}
            # 3. Alternative format: {author_date: "..."}
            commit_date_str = None
            
            if 'date' in commit:
                commit_date_str = commit['date']
            elif 'author_date' in commit:
                commit_date_str = commit['author_date']
            elif 'commit' in commit and 'author' in commit['commit']:
                commit_date_str = commit['commit']['author'].get('date')
            
            if commit_date_str:
                try:
                    commit_date = datetime.strptime(commit_date_str, '%Y-%m-%dT%H:%M:%SZ').date()
                    
                    if earliest_date is None or commit_date < earliest_date:
                        earliest_date = commit_date
                except Exception:
                    continue
        
        return earliest_date

    def get_weekly_series(self, projects):
        weekly_series = {}

        for project in projects:
            project_weekly_series = self.get_project_weekly_series(project['folder_path'])
            weekly_series[project['full_name']] = project_weekly_series

        return weekly_series

    def get_project_weekly_series(self, folder):
        commits_file_path = os.path.join(folder, 'commits.json')
        
        if not os.path.isfile(commits_file_path):
            return Counter()
        
        try:
            with open(commits_file_path, 'r', encoding='utf-8') as f:
                commits_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not read {commits_file_path}: {e}")
            return Counter()
        
        # Extract commits from the data field if present
        if isinstance(commits_data, dict) and 'data' in commits_data:
            commits_list = commits_data['data']
        elif isinstance(commits_data, list):
            commits_list = commits_data
        else:
            print(f"Warning: Unexpected format in {commits_file_path}")
            return Counter()
        
        newcomers_list = []
        entry_list = []

        for commit in commits_list:
            # Handle multiple formats:
            # 1. New simplified format: {author: "...", date: "..."}
            # 2. Old nested format: {commit: {author: {name: "...", date: "..."}}}
            # 3. Alternative format: {author_name: "...", author_date: "..."}
            newcomer = None
            commit_date = None
            
            if 'author' in commit and 'date' in commit:
                newcomer = commit['author']
                commit_date = commit['date']
            elif 'author_name' in commit and 'author_date' in commit:
                newcomer = commit['author_name']
                commit_date = commit['author_date']
            elif 'commit' in commit and 'author' in commit['commit']:
                newcomer = commit['commit']['author'].get('name')
                commit_date = commit['commit']['author'].get('date')

            if commit_date is not None and newcomer is not None:
                try:
                    commit_date = datetime.strptime(commit_date, '%Y-%m-%dT%H:%M:%SZ').date()
                    
                    # Track the latest commit date
                    if self.latest_commit_date is None or commit_date > self.latest_commit_date:
                        self.latest_commit_date = commit_date
                    
                    # Only count commits from the start date onward
                    if commit_date >= self.start_date:
                        if newcomer not in newcomers_list:
                            newcomers_list.append(newcomer)
                            entry_list.append(commit_date)
                except Exception as e:
                    continue

        ordered_entry_list = Counter(entry_list)

        return ordered_entry_list

    def get_number_of_weeks(self, weekly_series):
        weekly_max = None
        weekly_min = None

        for series in weekly_series.values():
            if series:
                for date in series.keys():
                    if weekly_min is None or date < weekly_min:
                        weekly_min = date
                    if weekly_max is None or date > weekly_max:
                        weekly_max = date

        return weekly_min, weekly_max

    def export_newcomers_inflow(self, weekly_series, weekly_min, weekly_max):
        if weekly_min is None or weekly_max is None:
            print("No commit data found. Creating empty inflow.csv")
            with open(self.csv_folder + '/inflow.csv', 'w', newline='', encoding='utf-8') as inflow_file:
                writer = csv.DictWriter(inflow_file, fieldnames=['project'])
                writer.writeheader()
            return
        
        # Don't include future weeks - cap at today's date
        today = datetime.now().date()
        if weekly_max > today:
            print(f"Capping data collection at today ({today}) instead of latest commit date ({weekly_max})")
            weekly_max = today
        
        fieldnames = []
        step = timedelta(days=1)
        current_date = weekly_min

        while current_date <= weekly_max:
            week = (current_date.isocalendar()[1], current_date.year)
            if not week in fieldnames:
                fieldnames.append(week)
            current_date += step
        
        print(f"Exporting inflow data for {len(fieldnames)} weeks (from week {fieldnames[0]} to week {fieldnames[-1]})...")
        
        with open(self.csv_folder + '/inflow.csv', 'w', newline='', encoding='utf-8') as inflow_file:
            writer = csv.DictWriter(inflow_file, fieldnames=['project'] + fieldnames)
            writer.writeheader()
        
        for project in weekly_series:
            inflow = {}
            inflow['project'] = project

            for week in fieldnames:
                number_of_newcomers = 0
                # Look through all entry dates to find matches for this week
                for entry_date, count in weekly_series[project].items():
                    entry_week = (entry_date.isocalendar()[1], entry_date.year)
                    if entry_week == week:
                        number_of_newcomers += count
                inflow[week] = number_of_newcomers

            with open(self.csv_folder + '/inflow.csv', 'a', newline='', encoding='utf-8') as inflow_file:
                writer = csv.DictWriter(inflow_file, fieldnames=['project'] + fieldnames)
                writer.writerow(inflow)
        
        print(f"Inflow data saved to: {self.csv_folder}/inflow.csv")
    
    # def export_monthly_inflow(self, weekly_series, weekly_min, weekly_max):
    #     """Export monthly aggregated newcomer inflow data."""
    #     if weekly_min is None or weekly_max is None:
    #         print("No commit data found. Creating empty inflow_monthly.csv")
    #         with open(self.csv_folder + '/inflow_monthly.csv', 'w', newline='', encoding='utf-8') as inflow_file:
    #             writer = csv.DictWriter(inflow_file, fieldnames=['project'])
    #             writer.writeheader()
    #         return
        
    #     # Don't include future months - cap at today's date
    #     today = datetime.now().date()
    #     if weekly_max > today:
    #         print(f"Capping monthly data collection at today ({today})")
    #         weekly_max = today
        
    #     # Generate list of months between min and max dates
    #     fieldnames = []
    #     current_date = weekly_min.replace(day=1)  # Start from first day of the month
    #     end_date = weekly_max.replace(day=1)
        
    #     while current_date <= end_date:
    #         month_key = current_date.strftime('%Y-%m')
    #         if month_key not in fieldnames:
    #             fieldnames.append(month_key)
            
    #         # Move to next month
    #         if current_date.month == 12:
    #             current_date = current_date.replace(year=current_date.year + 1, month=1)
    #         else:
    #             current_date = current_date.replace(month=current_date.month + 1)
        
    #     print(f"\nExporting monthly inflow data for {len(fieldnames)} months (from {fieldnames[0]} to {fieldnames[-1]})...")
        
    #     # Create monthly aggregation for each project
    #     with open(self.csv_folder + '/inflow_monthly.csv', 'w', newline='', encoding='utf-8') as inflow_file:
    #         writer = csv.DictWriter(inflow_file, fieldnames=['project'] + fieldnames)
    #         writer.writeheader()
        
    #     for project in weekly_series:
    #         inflow = {}
    #         inflow['project'] = project
            
    #         # Initialize all months to 0
    #         for month in fieldnames:
    #             inflow[month] = 0
            
    #         # Aggregate newcomers by month
    #         for entry_date, count in weekly_series[project].items():
    #             month_key = entry_date.strftime('%Y-%m')
    #             if month_key in fieldnames:
    #                 inflow[month_key] += count
            
    #         with open(self.csv_folder + '/inflow_monthly.csv', 'a', newline='', encoding='utf-8') as inflow_file:
    #             writer = csv.DictWriter(inflow_file, fieldnames=['project'] + fieldnames)
    #             writer.writerow(inflow)
        
    #     print(f"Monthly inflow data saved to: {self.csv_folder}/inflow_monthly.csv")

if __name__ == '__main__':
    # Use paths relative to script location, not current working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Point to scripts/data folder which contains commits.json files (../data from scripts/clustering/)
    dataset_folder = os.path.join(script_dir, 'data')
    # Output to tables subfolder within clustering folder
    csv_folder = os.path.join(script_dir, 'tables')
    # Use the filtered repos dataset (509 repos after exclusion criteria)
    filtered_repos_csv = os.path.join(script_dir, '..', 'out', 'filtered_repo_dataset.csv')
    
    # Minimum repository age in months (repositories must have first commit at least this many months ago)
    min_age_months = 6
    
    # Create output folder if it doesn't exist
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
        print(f"Created output folder: {csv_folder}\n")

    print(f"Filtered repos CSV: {filtered_repos_csv}")
    print(f"Commit data folder: {dataset_folder}")
    print(f"Output folder: {csv_folder}")
    print(f"Minimum repository age: {min_age_months} months\n")
    
    inflow = NewcomersInflow(dataset_folder, csv_folder, min_age_months, filtered_repos_csv)
