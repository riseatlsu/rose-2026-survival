import os
import json
import pandas as pd
from datetime import datetime

# Path to ros_robotics_data folder
data_folder = '../data/ros_robotics_data'

# Reference date: March 3, 2026
ref_date = pd.Timestamp('2026-03-03', tz='UTC')

# Store repository ages
repo_ages = []

# Iterate through all repository folders
for repo_folder in os.listdir(data_folder):
    repo_path = os.path.join(data_folder, repo_folder)
    
    if not os.path.isdir(repo_path):
        continue
    
    # Check for first_commits_by_author.json
    first_commits_file = os.path.join(repo_path, 'first_commits_by_author.json')
    
    first_commit_date = None
    
    # Try to get first commit date from first_commits_by_author.json
    if os.path.exists(first_commits_file):
        try:
            with open(first_commits_file, 'r', encoding='utf-8') as f:
                first_commits = json.load(f)
                if 'data' in first_commits and len(first_commits['data']) > 0:
                    # Get the earliest commit date
                    dates = []
                    for author_data in first_commits['data']:
                        if 'date' in author_data:
                            dates.append(author_data['date'])
                    if dates:
                        first_commit_date = min(dates)
        except Exception as e:
            pass
    
    if first_commit_date:
        try:
            first_commit_dt = pd.Timestamp(first_commit_date)
            age_days = (ref_date - first_commit_dt).days
            age_months = age_days / 30.44
            
            repo_name = repo_folder.replace('__', '/')
            repo_ages.append({
                'repository': repo_name,
                'first_commit_date': first_commit_date,
                'age_days': age_days,
                'age_months': age_months
            })
        except:
            pass

# Create DataFrame
df = pd.DataFrame(repo_ages)

if len(df) > 0:
    print(f'Total repositories with commit data: {len(df)}')
    print(f'\nYoungest repository:')
    youngest = df.loc[df['age_months'].idxmin()]
    print(f"  {youngest['repository']}: {youngest['age_months']:.2f} months ({int(youngest['age_days'])} days)")
    print(f"  First commit: {youngest['first_commit_date']}")
    
    # Find repositories between 5.5 and 6.5 months old
    repos_5_to_7_months = df[
        (df['age_months'] >= 5.5) & (df['age_months'] <= 6.5)
    ].sort_values('age_months')
    
    print(f'\n=== Repositories between 5.5 and 6.5 months old (as of March 3, 2026) ===')
    print(f'Found: {len(repos_5_to_7_months)} repositories')
    
    if len(repos_5_to_7_months) > 0:
        print('\nDetails:')
        for idx, row in repos_5_to_7_months.iterrows():
            print(f"\n  {row['repository']}")
            print(f"    Age: {row['age_months']:.2f} months ({int(row['age_days'])} days)")
            print(f"    First commit: {row['first_commit_date']}")
    
    # Show distribution around that range
    print(f'\n=== Age Distribution ===')
    print(f"Repositories < 6 months: {len(df[df['age_months'] < 6])}")
    print(f"Repositories 6-7 months: {len(df[(df['age_months'] >= 6) & (df['age_months'] < 7)])}")
    print(f"Repositories 7-8 months: {len(df[(df['age_months'] >= 7) & (df['age_months'] < 8)])}")
    print(f"Repositories 8-9 months: {len(df[(df['age_months'] >= 8) & (df['age_months'] < 9)])}")
    print(f"Repositories > 9 months: {len(df[df['age_months'] >= 9])}")
else:
    print('No repository data found')
