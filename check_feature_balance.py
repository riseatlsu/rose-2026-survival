import pandas as pd

df = pd.read_csv('out/survival_dataset_complete.csv')

def to_bin(col_name):
    col = df[col_name]
    if col.dtype == bool:
        return col.astype(int)
    return col.astype(str).str.lower().isin(['true', '1', 'yes']).astype(int)

features = [
    'has_issue_template', 
    'has_pr_template', 
    'has_readme', 
    'has_contributing', 
    'has_code_of_conduct', 
    'has_newcomer_labels'
]

print('Feature Distribution and Death Rates:')
print('=' * 80)
for feat in features:
    if feat in df.columns:
        binary = to_bin(feat)
        total = len(df)
        n_true = binary.sum()
        n_false = total - n_true
        pct_true = 100 * n_true / total
        
        # Death rates
        dead = df['event_dead'].astype(int)
        n_dead_true = dead[binary == 1].sum()
        n_dead_false = dead[binary == 0].sum()
        
        death_rate_true = 100 * dead[binary == 1].mean() if n_true > 0 else 0
        death_rate_false = 100 * dead[binary == 0].mean() if n_false > 0 else 0
        
        print(f'\n{feat:25s}: {n_true:4d}/{total} ({pct_true:5.1f}%)')
        print(f'  TRUE  (n={n_true:3d}): {n_dead_true:3d} dead ({death_rate_true:5.1f}%)')
        print(f'  FALSE (n={n_false:3d}): {n_dead_false:3d} dead ({death_rate_false:5.1f}%)')
        
        # Check for quasi-separation warning
        if n_true < 20 or n_false < 20:
            print(f'  ⚠️  WARNING: Very imbalanced feature (minority class < 20)')
        if n_dead_true < 5 or n_dead_false < 5:
            print(f'  ⚠️  WARNING: Very few events in one class (< 5 deaths)')

print('\n' + '=' * 80)
print(f'Total repositories: {len(df)}')
print(f'Total dead: {df["event_dead"].astype(int).sum()} ({100*df["event_dead"].mean():.1f}%)')
