#!/usr/bin/env python3
"""
generate_all_commits_spreadsheet.py

Generate a comprehensive CSV spreadsheet with all commits from all repositories.
Includes: commit_id, sha, message, author, files_changed, commit_type, and GitHub link.
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote

# =========================
# CONFIG
# =========================
DATA_ROOT = "data/ros_robotics_data"
FILTERED_CSV = "../out/filtered_repo_dataset.csv"
OUTPUT_CSV = "../out/all_commits_spreadsheet.csv"

# Mapping for repos with different owner/name in directories
# Maps full_name from CSV to (directory_owner, directory_repo, actual_full_name)
REPO_NAME_MAPPING = {
    "Fields2Cover/Fields2Cover": ("Fields2Cover", "fields2cover", "Fields2Cover/fields2cover"),
    "FlexBE/flexbe_behavior_engine": ("flexbe", "flexbe_behavior_engine", "flexbe/flexbe_behavior_engine"),
    "OctoMap/octomap": ("octomap", "octomap", "octomap/octomap"),
    "TonyWelte/rosidlcpp": ("Tonywelte", "rosidlcpp", "Tonywelte/rosidlcpp"),
    "autowarefoundation/callback_isolated_executor": ("tier4", "callback_isolated_executor", "tier4/callback_isolated_executor"),
    "enactic/openarm_ros2": ("reazon-research", "openarm_ros2", "reazon-research/openarm_ros2"),
    "frankarobotics/franka_description": ("frankaemika", "franka_description", "frankaemika/franka_description"),
    "frankarobotics/franka_ros2": ("frankaemika", "franka_ros2", "frankaemika/franka_ros2"),
    "frankarobotics/libfranka": ("frankaemika", "libfranka", "frankaemika/libfranka"),
    "fzi-forschungszentrum-informatik/Lanelet2": ("fzi-forschungszentrum-informatik", "lanelet2", "fzi-forschungszentrum-informatik/lanelet2"),
    "moveit/geometric_shapes": ("ros-planning", "geometric_shapes", "ros-planning/geometric_shapes"),
    "moveit/moveit2": ("ros-planning", "moveit2", "ros-planning/moveit2"),
    "moveit/moveit_visual_tools": ("ros-planning", "moveit_visual_tools", "ros-planning/moveit_visual_tools"),
    "moveit/srdfdom": ("ros-planning", "srdfdom", "ros-planning/srdfdom"),
    "moveit/warehouse_ros_sqlite": ("ros-planning", "warehouse_ros_sqlite", "ros-planning/warehouse_ros_sqlite"),
    "realsenseai/librealsense": ("IntelRealSense", "librealsense", "IntelRealSense/librealsense"),
    "realsenseai/realsense-ros": ("IntelRealSense", "realsense-ros", "IntelRealSense/realsense-ros"),
    "ros-navigation/navigation2": ("ros-planning", "navigation2", "ros-planning/navigation2"),
    # Repos with data using different directory names (CSV Name vs GitHub repo name)
    "LOEWE-emergenCITY/ros_babel_fish": ("LOEWE-emergenCITY", "ros2_babel_fish", "LOEWE-emergenCITY/ros_babel_fish"),
    "automatika-robotics/embodied-agents": ("automatika-robotics", "ros-agents", "automatika-robotics/embodied-agents"),
    "automatika-robotics/sugarcoat": ("automatika-robotics", "ros-sugar", "automatika-robotics/sugarcoat"),
    "boschglobal/rokit_ros_bridge": ("boschglobal", "locator_ros_bridge", "boschglobal/rokit_ros_bridge"),
    "SBG-Systems/sbg_ros2_driver": ("SBG-Systems", "sbg_ros2", "SBG-Systems/sbg_ros2_driver"),
}

# =========================
# HELPER FUNCTIONS
# =========================
def load_json(path: str) -> dict:
    """Load JSON file safely."""
    try:
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return {}

def extract_snapshot_data(obj: dict):
    """Extract data from snapshot JSON structure."""
    if isinstance(obj, dict) and "_meta" in obj and "data" in obj:
        return obj["data"]
    return obj

def get_repo_directory(csv_full_name: str) -> tuple:
    """
    Get the actual directory name and owner/repo for a repository.
    Returns (owner, repo_name, actual_full_name) or (None, None, None) if not found.
    """
    # Check if there's a mapping for this repo
    if csv_full_name in REPO_NAME_MAPPING:
        owner, repo, actual_full_name = REPO_NAME_MAPPING[csv_full_name]
        return owner, repo, actual_full_name
    
    # Otherwise use the CSV name as-is
    owner, repo = csv_full_name.split('/')
    return owner, repo, csv_full_name

def main():
    """Main function to generate all commits spreadsheet."""
    print("=" * 80)
    print("GENERATING ALL COMMITS SPREADSHEET")
    print("=" * 80)
    
    # Read filtered dataset
    repos_data = {}
    try:
        with open(FILTERED_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                full_name = row.get('full_name', '').strip()
                if full_name:
                    owner, repo_name = full_name.split('/')
                    repos_data[full_name] = {'owner': owner, 'repo': repo_name}
    except Exception as e:
        print(f"Error reading filtered dataset: {e}")
        return
    
    print(f"Found {len(repos_data)} repositories in filtered dataset\n")
    
    # Collect all commits
    all_commits = []
    commit_id = 1
    repos_processed = 0
    commits_total = 0
    
    for full_name, repo_info in sorted(repos_data.items()):
        owner = repo_info['owner']
        repo_name = repo_info['repo']
        
        # Get the actual directory name (handles mappings)
        actual_owner, actual_repo, actual_full_name = get_repo_directory(full_name)
        repo_dir = os.path.join(DATA_ROOT, f"{actual_owner}__{actual_repo}")
        
        if not os.path.exists(repo_dir):
            print(f"  ⚠ {full_name} - data not found")
            continue
        
        # Load first_commits_by_author (first commit per newcomer)
        first_commits_path = os.path.join(repo_dir, "first_commits_by_author.json")
        first_commits_data = load_json(first_commits_path)
        first_commits = extract_snapshot_data(first_commits_data)
        
        if not isinstance(first_commits, list):
            print(f"  ⚠ {full_name} - no first commits data")
            continue
        
        repos_processed += 1
        
        # Process each first commit
        for commit in first_commits:
            sha = commit.get('sha', '')
            author = commit.get('author', {})
            
            if isinstance(author, dict):
                author_name = author.get('name', 'unknown')
            else:
                author_name = str(author) if author else 'unknown'
            
            message = commit.get('message', '').split('\n')[0]  # First line only
            files_changed = commit.get('files_changed', 0)
            additions = commit.get('additions', 0)
            deletions = commit.get('deletions', 0)
            
            # Get commit type (from V2 classifier)
            commit_type = commit.get('commit_type', 'unknown')
            
            # Extract files and their extensions
            files = commit.get('files', [])
            file_list = []
            file_extensions = []
            
            if isinstance(files, list):
                for file_obj in files:
                    if isinstance(file_obj, dict):
                        filename = file_obj.get('filename', '')
                    else:
                        filename = str(file_obj)
                    
                    # Extract just the filename (basename) to keep size smaller
                    basename = filename.rsplit('/', 1)[-1] if '/' in filename else filename
                    file_list.append(basename)
                    
                    # Extract extension
                    if '.' in basename:
                        ext = basename.rsplit('.', 1)[-1]
                        if ext not in file_extensions:
                            file_extensions.append(ext)
            
            files_str = '|'.join(file_list) if file_list else ''
            extensions_str = '|'.join(file_extensions) if file_extensions else 'none'
            
            # Build GitHub link
            github_url = f"https://github.com/{owner}/{repo_name}/commit/{sha}"
            
            all_commits.append({
                'commit_id': commit_id,
                'full_name': full_name,
                'owner': owner,
                'repo': repo_name,
                'sha': sha,
                'message': message,
                'author': author_name,
                'files_changed': files_changed,
                'additions': additions,
                'deletions': deletions,
                'commit_type': commit_type,
                'files': files_str,
                'file_extensions': extensions_str,
                'github_url': github_url
            })
            
            commit_id += 1
            commits_total += 1
        
        if repos_processed % 50 == 0:
            print(f"  Processed {repos_processed} repositories... ({commits_total} commits)")
    
    print(f"\n✓ Processed {repos_processed} repositories")
    print(f"✓ Collected {commits_total} total commits")
    
    # Write to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    print(f"\nWriting to {OUTPUT_CSV}...")
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'commit_id',
            'full_name',
            'owner',
            'repo',
            'sha',
            'message',
            'author',
            'files_changed',
            'additions',
            'deletions',
            'commit_type',
            'files',
            'file_extensions',
            'github_url'
        ])
        
        for commit in all_commits:
            writer.writerow([
                commit['commit_id'],
                commit['full_name'],
                commit['owner'],
                commit['repo'],
                commit['sha'],
                commit['message'],
                commit['author'],
                commit['files_changed'],
                commit['additions'],
                commit['deletions'],
                commit['commit_type'],
                commit['files'],
                commit['file_extensions'],
                commit['github_url']
            ])
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    from collections import Counter
    
    type_counts = Counter(c['commit_type'] for c in all_commits)
    
    print(f"\nTotal commits: {len(all_commits)}")
    print(f"\nCommit type distribution:")
    for commit_type in sorted(type_counts.keys()):
        count = type_counts[commit_type]
        pct = 100 * count / len(all_commits)
        print(f"  {commit_type:10s}: {count:6d} ({pct:5.1f}%)")
    
    print(f"\n✓ Spreadsheet saved to: {OUTPUT_CSV}")
    print(f"  Total rows: {len(all_commits)}")

if __name__ == "__main__":
    main()
