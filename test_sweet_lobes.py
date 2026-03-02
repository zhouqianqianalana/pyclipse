#!/usr/bin/env python
"""Quick test of the select_and_boost_sweet_lobes functionality."""

import numpy as np
import json
from pyclipse.layer import LobeLayer

# Create a small test layer
print("Creating LobeLayer...")
layer = LobeLayer(
    nx=30, ny=30, nz=15,
    x_len=1000, y_len=1000, z_len=100,
    top_depth=0, dip=0,
    poro_ave=0.2, poro_std=0.02,
    perm_ave=1, perm_std=0.5,
    kzkx=0.1, ntg=0.2
)

# Generate geology and select 5 sweet lobes
print("Generating geology with 5 sweet lobes...")
np.random.seed(42)  # For reproducibility
layer.create_geology(
    dhmin=2, dhmax=3,
    rmin=8, rmax=10,
    asp=1.5, theta0=0, m=100,
    upthinning=True, bouma_factor=0,
    n_sweet=5,
    sweet_amp_min=0.05,
    sweet_amp_max=0.15
)

print("\n" + "="*70)
print("SWEET LOBE METADATA")
print("="*70)

# Print metadata for each selected lobe
for i, meta in enumerate(layer.sweet_metadata):
    print(f"\nSweet Lobe #{i+1}:")
    print(f"  Lobe ID: {meta['lobe_id']}")
    print(f"  Boost Amount: {meta['boost_amount']:.4f}")
    print(f"  Total Cells: {meta['cell_count']}")
    print(f"  Voxel Coords (first 5): {meta['voxel_coords'][:5]}")
    print(f"  Total Voxels Stored: {len(meta['voxel_coords'])}")

print("\n" + "="*70)
print("POROSITY STATISTICS")
print("="*70)
print(f"Min porosity: {layer.poro_mat.min():.4f}")
print(f"Max porosity: {layer.poro_mat.max():.4f}")
print(f"Mean porosity: {layer.poro_mat.mean():.4f}")
print(f"Sweet-boosted cells count: {np.sum(layer.sweet_mask)}")
print(f"Grid shape: {layer.poro_mat.shape}")

print("\n✓ Test passed!")
