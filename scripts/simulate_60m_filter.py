"""
Simular aplicação do filtro de 60 meses de idade mínima.
Replica a metodologia de Ait et al. (2022): apenas repos com >= 5 anos de existência.
"""

import pandas as pd
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_FILE = PROJECT_ROOT / "out" / "survival_dataset_complete.csv"

STUDY_END = datetime(2026, 3, 8)
MIN_AGE_MONTHS = 60  # 5 anos
CUTOFF_DATE = datetime(2021, 3, 8)  # 60 meses antes do fim do estudo

def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Dataset original: {len(df)} repositórios")
    
    # Converter datas
    df['created_at_dt'] = pd.to_datetime(df['created_at'], errors='coerce')
    
    # Aplicar filtro
    df_filtered = df[df['created_at_dt'] <= CUTOFF_DATE].copy()
    
    print(f"\n{'='*60}")
    print(f"SIMULAÇÃO: Filtro de ≥{MIN_AGE_MONTHS} meses de idade")
    print(f"  Cutoff: criados antes de {CUTOFF_DATE.date()}")
    print(f"{'='*60}")
    
    print(f"\nRepositórios após filtro: {len(df_filtered)} ({100*len(df_filtered)/len(df):.1f}%)")
    print(f"Repositórios excluídos: {len(df) - len(df_filtered)}")
    
    # Calcular morte
    df_filtered['days_since'] = pd.to_numeric(df_filtered['days_since_last_activity'], errors='coerce')
    df_filtered['dead'] = (df_filtered['days_since'] > 180).astype(int)
    
    n_dead = df_filtered['dead'].sum()
    print(f"\nEstatísticas do dataset filtrado:")
    print(f"  Total: {len(df_filtered)}")
    print(f"  Mortos: {int(n_dead)} ({100*n_dead/len(df_filtered):.1f}%)")
    print(f"  Vivos: {len(df_filtered) - int(n_dead)} ({100*(1-n_dead/len(df_filtered)):.1f}%)")
    
    # Community tiers
    q1 = df_filtered['contributors_count'].quantile(0.25)
    q3 = df_filtered['contributors_count'].quantile(0.75)
    
    def tier(n):
        if n <= q1: return "Tier 1"
        elif n <= q3: return "Tier 2"
        else: return "Tier 3"
    
    df_filtered['tier'] = df_filtered['contributors_count'].apply(tier)
    
    print(f"\nDistribuição por Community Tier (Q1={q1:.0f}, Q3={q3:.0f}):")
    for t in ["Tier 1", "Tier 2", "Tier 3"]:
        sub = df_filtered[df_filtered['tier'] == t]
        dead = sub['dead'].sum()
        print(f"  {t}: {len(sub)} repos ({100*len(sub)/len(df_filtered):.1f}%), "
              f"mortos: {int(dead)} ({100*dead/len(sub):.1f}%)")
    
    # Owner type
    print(f"\nDistribuição por Owner Type:")
    for ot in df_filtered['owner_type'].dropna().unique():
        sub = df_filtered[df_filtered['owner_type'] == ot]
        dead = sub['dead'].sum()
        print(f"  {ot}: {len(sub)} ({100*len(sub)/len(df_filtered):.1f}%), "
              f"mortos: {int(dead)} ({100*dead/len(sub):.1f}%)")
    
    # Anos de criação
    df_filtered['created_year'] = df_filtered['created_at_dt'].dt.year
    print(f"\nDistribuição por ano de criação:")
    year_counts = df_filtered['created_year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {int(year)}: {count} repos")
    
    # Comparação com Ait et al.
    print(f"\n{'='*60}")
    print("COMPARAÇÃO COM AIT ET AL. (2022)")
    print(f"{'='*60}")
    print(f"  Ait et al.: 1,127 repos (NPM/R/WordPress/Laravel)")
    print(f"  Nosso (ROS): {len(df_filtered)} repos")
    print(f"  Ait et al. death rate: ~60-70%")
    print(f"  Nosso death rate: {100*n_dead/len(df_filtered):.1f}%")

if __name__ == "__main__":
    main()
