#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Elijah Phifer"
__contact__ = "elijah.phifer@lsu.edu"

"""
Run all descriptive statistics plotting scripts in sequence.
"""

import os
import sys
import subprocess

def run_script(script_name):
    """Run a Python script and report status."""
    print(f"\n{'='*70}")
    print(f"Running: {script_name}")
    print('='*70)
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        print(f"ERROR: Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {script_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ {script_name} failed with error: {e}")
        return False

def main():
    """Run all plotting scripts."""
    print("\n" + "="*70)
    print("RUNNING ALL DESCRIPTIVE STATISTICS PLOTTING SCRIPTS")
    print("="*70)
    
    scripts = [
        '00_plot_inflow.py',
        '01_plot_average_newcomers.py',
        '02_plot_documentation_metrics.py',
        '03_basic_statistics.py',
        '04_plot_commit_types.py',
        '05_plot_label_analysis.py',
    ]
    
    results = {}
    for script in scripts:
        results[script] = run_script(script)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for script, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status:12} - {script}")
    
    total = len(results)
    successful = sum(1 for s in results.values() if s)
    
    print("\n" + "="*70)
    print(f"Completed: {successful}/{total} scripts successful")
    print("="*70)
    
    if successful < total:
        sys.exit(1)

if __name__ == '__main__':
    main()
