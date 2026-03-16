import pandas as pd
from datetime import datetime

# Load inflow data
inflow = pd.read_csv('../clustering/tables/inflow.csv')

# Load repository dataset
repos = pd.read_csv('../../out/filtered_repo_dataset.csv')

# Get inflow projects
inflow_projects = set(inflow['project'])

# Create full_name column
repos['full_name'] = repos['Owner'] + '/' + repos['Name']

# Filter to only repos in inflow
repos_in_inflow = repos[repos['full_name'].isin(inflow_projects)]

# Find minimum age
min_age = repos_in_inflow['Repository age (months)'].min()
max_age = repos_in_inflow['Repository age (months)'].max()
youngest = repos_in_inflow[repos_in_inflow['Repository age (months)'] == min_age]

print(f'\n=== Repository Age Statistics (as of data collection time) ===')
print(f'Youngest repository age: {min_age} months')
print(f'Oldest repository age: {max_age} months')
print(f'Youngest repository: {youngest.iloc[0]["full_name"]}')
print(f'Total repositories in inflow: {len(repos_in_inflow)}')
print(f'Youngest age in years: {min_age/12:.2f} years')
print(f'Youngest age in days (approx): {int(min_age * 30.44)} days')

# Check the first contributor date column
print(f'\n=== Checking first_contributor_date ===')
print(f'First few first_contributor_date values:')
print(repos_in_inflow['first_contributor_date'].head(10))
print(f'\nYoungest repo first contributor date: {youngest.iloc[0]["first_contributor_date"]}')

# Calculate age as of March 5, 2026
current_date = pd.Timestamp('2026-03-05', tz='UTC')
repos_in_inflow['first_contributor_datetime'] = pd.to_datetime(repos_in_inflow['first_contributor_date'])
repos_in_inflow['age_as_of_march_5_2026_days'] = (current_date - repos_in_inflow['first_contributor_datetime']).dt.days
repos_in_inflow['age_as_of_march_5_2026_months'] = repos_in_inflow['age_as_of_march_5_2026_days'] / 30.44

min_age_march = repos_in_inflow['age_as_of_march_5_2026_months'].min()
youngest_march = repos_in_inflow[repos_in_inflow['age_as_of_march_5_2026_months'] == min_age_march]

print(f'\n=== Repository Age as of March 5, 2026 ===')
print(f'Youngest repository age: {min_age_march:.2f} months')
print(f'Youngest repository: {youngest_march.iloc[0]["full_name"]}')
print(f'First contributor date: {youngest_march.iloc[0]["first_contributor_date"]}')
print(f'Age in days: {int(youngest_march.iloc[0]["age_as_of_march_5_2026_days"])} days')

# Check for repositories exactly 6 months old as of March 3, 2026
march_3_date = pd.Timestamp('2026-03-03', tz='UTC')
repos_in_inflow['age_as_of_march_3_2026_days'] = (march_3_date - repos_in_inflow['first_contributor_datetime']).dt.days
repos_in_inflow['age_as_of_march_3_2026_months'] = repos_in_inflow['age_as_of_march_3_2026_days'] / 30.44

# Find repos with age close to 6 months (within a few days)
six_months_in_days = 6 * 30.44
tolerance_days = 3  # Allow 3 days tolerance
repos_around_6_months = repos_in_inflow[
    (repos_in_inflow['age_as_of_march_3_2026_days'] >= six_months_in_days - tolerance_days) &
    (repos_in_inflow['age_as_of_march_3_2026_days'] <= six_months_in_days + tolerance_days)
]

print(f'\n=== Repositories Around 6 Months Old as of March 3, 2026 ===')
print(f'Looking for repos with age between {six_months_in_days - tolerance_days:.0f} and {six_months_in_days + tolerance_days:.0f} days')
print(f'Found {len(repos_around_6_months)} repositories')

if len(repos_around_6_months) > 0:
    print('\nRepositories:')
    for idx, row in repos_around_6_months.iterrows():
        print(f"  - {row['full_name']}: {row['age_as_of_march_3_2026_months']:.2f} months ({int(row['age_as_of_march_3_2026_days'])} days)")
        print(f"    First commit: {row['first_contributor_date']}")
else:
    print('No repositories found exactly at 6 months.')
    
# Show distribution around 6 months
print(f'\n=== Distribution Around 6 Months ===')
repos_5_to_7_months = repos_in_inflow[
    (repos_in_inflow['age_as_of_march_3_2026_months'] >= 5.5) &
    (repos_in_inflow['age_as_of_march_3_2026_months'] <= 6.5)
]
print(f'Repositories between 5.5 and 6.5 months old: {len(repos_5_to_7_months)}')
if len(repos_5_to_7_months) > 0:
    for idx, row in repos_5_to_7_months.iterrows():
        print(f"  - {row['full_name']}: {row['age_as_of_march_3_2026_months']:.2f} months ({int(row['age_as_of_march_3_2026_days'])} days), First commit: {row['first_contributor_date']}")
