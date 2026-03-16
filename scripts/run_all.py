#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Scripts to run in order (in scripts/ directory)
SCRIPTS = [
    "00_download_ros_index_json.py",
    "01_build_mapping_from_rosdistro.py",
    "02_join_index_with_rosdistro.py",
    "03_validate_and_stats.py",
    "04_analyze_resolved_packages.py",
    "05_fill_missing_from_index_html.py",
    "06_diagnose_unresolved.py",
    "07_extract_unique_repos.py",
    "08_repo_overlap_table.py",
    # "09_extract_repo_features_and_commits.py",
    "10_build_final_repo_dataset.py",
    "11_apply_exclusion_criteria.py",
]

# Post-processing scripts (in scripts/ directory)
POST_SCRIPTS = [
    "generate_ros_packages_statistics.py",
    "generate_all_commits_spreadsheet.py",
]

def main():
    all_scripts = SCRIPTS + POST_SCRIPTS
    
    print(f"Found {len(SCRIPTS)} main scripts + {len(POST_SCRIPTS)} post-processing scripts")
    print("=" * 60)
    
    failed = []
    successful = []
    
    # Run main scripts
    for script in SCRIPTS:
        script_path = SCRIPT_DIR / script
        print(f"\n[RUN] {script}")
        print("-" * 60)
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=PROJECT_ROOT
            )
            
            if result.returncode == 0:
                print(f"[OK] {script} completed successfully")
                successful.append(script)
            else:
                print(f"[ERROR] {script} exited with code {result.returncode}")
                failed.append((script, f"exit code {result.returncode}"))
                
        except Exception as e:
            print(f"[EXCEPTION] {script}: {e}")
            failed.append((script, str(e)))
    
    # Run post-processing scripts
    print("\n" + "=" * 60)
    print("POST-PROCESSING")
    print("=" * 60)
    
    for script in POST_SCRIPTS:
        script_path = SCRIPT_DIR / script
        
        # Skip if file doesn't exist
        if not script_path.exists():
            print(f"\n[SKIP] {script} (not found)")
            continue
        
        print(f"\n[RUN] {script}")
        print("-" * 60)
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=PROJECT_ROOT
            )
            
            if result.returncode == 0:
                print(f"[OK] {script} completed successfully")
                successful.append(script)
            else:
                print(f"[ERROR] {script} exited with code {result.returncode}")
                failed.append((script, f"exit code {result.returncode}"))
                
        except Exception as e:
            print(f"[EXCEPTION] {script}: {e}")
            failed.append((script, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successful: {len(successful)}/{len(all_scripts)}")
    for s in successful:
        print(f"  ✓ {s}")
    
    if failed:
        print(f"\nFailed: {len(failed)}/{len(all_scripts)}")
        for s, reason in failed:
            print(f"  ✗ {s} ({reason})")
        sys.exit(1)
    else:
        print("\nAll scripts executed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
