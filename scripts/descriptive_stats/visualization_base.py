#!/usr/bin/env python3
"""
Visualization utilities and base functions for creating publication-quality charts.
Provides reusable functions for consistent styling across all visualizations.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import PLOT_PARAMS, COLORS, FIGSIZE

def setup_figure(figsize_type='small'):
    """Create a figure with publication-quality settings."""
    figsize = FIGSIZE.get(figsize_type, FIGSIZE['small'])
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def setup_dual_figure(figsize_type='large'):
    """Create a dual subplot figure."""
    figsize = FIGSIZE.get(figsize_type, FIGSIZE['large'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    return fig, ax1, ax2

def save_figure(fig, output_path, dpi=300):
    """Save figure with consistent settings."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(
        output_path,
        dpi=PLOT_PARAMS['dpi'],
        bbox_inches=PLOT_PARAMS['bbox_inches'],
        facecolor=PLOT_PARAMS['facecolor']
    )
    print(f"✅ Saved: {output_path}")
    plt.close(fig)

def style_horizontal_bars(ax, ylabel=None, xlabel='Percentage (%)'):
    """Apply consistent styling to horizontal bar charts."""
    ax.set_xlabel(xlabel, fontsize=PLOT_PARAMS['fontsize_label'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=PLOT_PARAMS['fontsize_label'])
    ax.grid(True, alpha=PLOT_PARAMS['grid_alpha'], axis='x')
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=PLOT_PARAMS['fontsize_tick'])
    ax.tick_params(axis='y', labelsize=PLOT_PARAMS['fontsize_tick'])
    ax.set_xlim(0, 100)

def style_vertical_bars(ax, ylabel=None, xlabel=None, title=None):
    """Apply consistent styling to vertical bar charts."""
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=PLOT_PARAMS['fontsize_label'])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=PLOT_PARAMS['fontsize_label'])
    if title:
        ax.set_title(title, fontsize=PLOT_PARAMS['fontsize_title'], fontweight='bold')
    ax.grid(True, alpha=PLOT_PARAMS['grid_alpha'], axis='y')
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=PLOT_PARAMS['fontsize_tick'])

def add_bar_labels(bars, values, as_percentage=False):
    """Add value labels on top of bars."""
    for bar, value in zip(bars, values):
        height = bar.get_height()
        label_text = f'{value:.1f}%' if as_percentage else f'{int(value):,}'
        plt.gca().text(
            bar.get_x() + bar.get_width()/2., height,
            label_text,
            ha='center', va='bottom',
            fontsize=PLOT_PARAMS['fontsize_label'],
            fontweight='bold'
        )

def add_hbar_labels(bars, values, counts=None, as_percentage=False):
    """Add value labels on horizontal bars."""
    for bar, pct, count in zip(bars, values, counts if counts else values):
        width = bar.get_width()
        if counts and count != pct:
            label_text = f'{count:,}\n({pct:.1f}%)'
        else:
            label_text = f'{pct:.1f}%'
        plt.gca().text(
            width, bar.get_y() + bar.get_height()/2.,
            f' {label_text}',
            ha='left', va='center',
            fontsize=PLOT_PARAMS['fontsize_label'],
            fontweight='bold'
        )

if __name__ == "__main__":
    print("Visualization utilities loaded successfully")
