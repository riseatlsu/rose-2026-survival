# Replication Package
Welcome to the replication package of the paper entitled "The Walking Packages: A Survival Analysis of ROS Repositories"

# Repository Structure
- [**scripts**](scripts): Complete data processing and analysis pipeline
  - **00-11**: Core data collection pipeline (download ROS index, build mappings from rosdistro, extract repository features, apply exclusion criteria)
  - **12-14**: Inflow analysis and documentation metrics
  - **15-17**: Event tables and activity extraction (state machine, comments, reviews)
  - **18-21**: Survival analysis pipeline (dataset preparation, survival models, extended analysis, paper figures)
  - **Utility scripts**: `generate_all_commits_spreadsheet.py`, `generate_ros_packages_statistics.py`
  - [**run_all.py**](scripts/run_all.py): Execute scripts 00-11 sequentially
- [**out**](out): Output directory for all results
  - **survival_analysis**: Survival analysis outputs (Kaplan-Meier plots, Cox model results, feature importance)
  - **events**: Repository event tables (commits, issues, PRs, comments, reviews)
  - **repos**: Per-repository data collected from GitHub (metadata, commits, contributors, issues, community files)
  - Key datasets: `filtered_repo_dataset.csv`,
- [**tables**](scripts/tables): Summarized statistics

# Running the Replication Pipeline
To replicate our data collection and survival analysis:

```bash
pip install -r requirements.txt
echo "GITHUB_TOKEN=your_github_token_here" > .env
python scripts/run_all.py
```

The pipeline analyzes ROS packages across three distributions (ROS 2 Humble, Jazzy, and Kilted) and performs survival analysis following the methodology of Ait et al. (2022). The complete survival analysis dataset is generated at `out/survival_dataset_complete.csv`, with model results in `out/survival_analysis/`.

Individual scripts can be run independently for specific steps of the pipeline. Key scripts include:
- **09_extract_repo_features_and_commits.py**: Extract repository metadata from GitHub (requires GitHub API token)
- **18_prepare_survival_dataset.py**: Prepare survival analysis dataset with event definitions
- **19_survival_analysis.py**: Run Kaplan-Meier, log-rank tests, and Cox proportional hazards models
- **21_paper_figures.py**: Generate publication-ready figures

# Contact
If you have any questions or are interested in contributing to this project, please don't hesitate to contact us:

* Juliana Freitas (jfreit4@lsu.edu)
* Elijah Phifer (ephife3@lsu.edu)
* Felipe Fronchetti (ffronchetti@lsu.edu)
