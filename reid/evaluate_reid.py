"""
evaluate_reid.py
----------------
Evaluation and visualisation for Q1 — Person Re-Identification.

Generates the comparison table (Top-1/5/10 accuracy and runtime)
and the CMC curve plot comparing PCA vs Siamese CNN.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_reid import plot_cmc


def summarise(pca_results: dict, dl_results: dict, save_dir: str = 'outputs'):
    """
    Print a comparison table and save CMC curves to disk.

    Parameters
    ----------
    pca_results : dict
        Output of model_pca.run_pca_reid().
    dl_results : dict
        Output of model_siamese.run_siamese_reid().
    save_dir : str
        Directory to save output plots.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Results table ─────────────────────────────────────────────────────────
    df = pd.DataFrame({
        'Model':       ['PCA', 'Siamese CNN'],
        'Top-1':       [pca_results['top1'],   dl_results['top1']],
        'Top-5':       [pca_results['top5'],   dl_results['top5']],
        'Top-10':      [pca_results['top10'],  dl_results['top10']],
        'Time (s)':    [pca_results['time_s'], dl_results['time_s']],
    })
    print("\nTable 1: Person Re-Identification Results")
    print(df.to_string(index=False))

    # ── CMC curves ────────────────────────────────────────────────────────────
    plot_cmc(
        {'PCA': pca_results['cmc'], 'Siamese CNN': dl_results['cmc']},
        save_path=os.path.join(save_dir, 'q1_cmc_curves.png'),
    )
    print(f"\nCMC curve saved to {save_dir}/q1_cmc_curves.png")
