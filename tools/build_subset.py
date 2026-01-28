"""Build stratified 10k subset from FLAIR-2 dataset.

This tool creates a stratified 10k subset of aerial images from train+test splits.
Stratification is done by region (Dxxx_YYYY) and zone (Z1_xx) to maintain diversity.
Train:val split is 8:2, and sampling is deterministic with a fixed seed.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple


def discover_samples(root_dir: Path, split: str) -> List[Dict]:
    """Discover all samples in a split.
    
    Args:
        root_dir: Root directory of FLAIR-2 dataset
        split: Split name ("train" or "test")
        
    Returns:
        List of sample dictionaries
    """
    samples = []
    aerial_dir = root_dir / split / "aerial"
    
    if not aerial_dir.exists():
        return samples
    
    for region_dir in sorted(aerial_dir.glob("D*_*")):
        region_name = region_dir.name
        
        for zone_dir in sorted(region_dir.glob("Z*_*")):
            zone_name = zone_dir.name
            img_dir = zone_dir / "img"
            
            if not img_dir.exists():
                continue
            
            for img_path in sorted(img_dir.glob("IMG_*.tif")):
                img_id = img_path.stem.replace("IMG_", "")
                
                # Construct paths
                label_path = root_dir / split / "labels" / region_name / zone_name / "msk" / f"MSK_{img_id}.tif"
                sen_dir = root_dir / split / "sen" / region_name / zone_name / "sen"
                sen_data_path = sen_dir / f"SEN2_sp_{region_name}-{zone_name}_data.npy"
                sen_masks_path = sen_dir / f"SEN2_sp_{region_name}-{zone_name}_masks.npy"
                sen_products_path = sen_dir / f"SEN2_sp_{region_name}-{zone_name}_products.txt"
                
                # Check if paths exist
                if label_path.exists() and sen_data_path.exists():
                    samples.append({
                        'aerial_path': str(img_path),
                        'label_path': str(label_path),
                        'sen_data_path': str(sen_data_path),
                        'sen_masks_path': str(sen_masks_path),
                        'sen_products_path': str(sen_products_path),
                        'region': region_name,
                        'zone': zone_name,
                        'split': split,
                        'img_id': img_id
                    })
    
    return samples


def stratified_sample(
    samples: List[Dict],
    target_size: int,
    seed: int = 42
) -> List[Dict]:
    """Perform stratified sampling by region and zone.
    
    Args:
        samples: List of all samples
        target_size: Target number of samples
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled samples
    """
    np.random.seed(seed)
    
    # Group by region/zone
    strata = defaultdict(list)
    for sample in samples:
        key = f"{sample['region']}/{sample['zone']}"
        strata[key].append(sample)
    
    # Calculate samples per stratum (proportional)
    total_samples = len(samples)
    stratum_sizes = {k: len(v) for k, v in strata.items()}
    
    # Calculate proportional allocation
    sampled = []
    remaining = target_size
    
    for key in sorted(strata.keys()):
        stratum = strata[key]
        stratum_size = len(stratum)
        
        # Proportional allocation
        n_samples = int(target_size * stratum_size / total_samples)
        n_samples = min(n_samples, stratum_size)  # Can't sample more than available
        n_samples = min(n_samples, remaining)  # Can't exceed target
        
        if n_samples > 0:
            # Random sample from stratum
            indices = np.random.choice(len(stratum), size=n_samples, replace=False)
            sampled.extend([stratum[i] for i in indices])
            remaining -= n_samples
    
    # If we haven't reached target, randomly sample remaining
    if remaining > 0:
        # Get all samples not yet selected
        sampled_ids = {(s['region'], s['zone'], s['img_id']) for s in sampled}
        remaining_samples = [
            s for s in samples
            if (s['region'], s['zone'], s['img_id']) not in sampled_ids
        ]
        
        if len(remaining_samples) > 0:
            n_additional = min(remaining, len(remaining_samples))
            indices = np.random.choice(len(remaining_samples), size=n_additional, replace=False)
            sampled.extend([remaining_samples[i] for i in indices])
    
    return sampled


def split_train_val(samples: List[Dict], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Split samples into train and validation sets.
    
    Args:
        samples: List of samples
        val_ratio: Validation ratio
        seed: Random seed
        
    Returns:
        Tuple of (train_samples, val_samples)
    """
    np.random.seed(seed)
    
    # Shuffle
    indices = np.arange(len(samples))
    np.random.shuffle(indices)
    
    # Split
    val_size = int(len(samples) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    
    return train_samples, val_samples


def save_split(samples: List[Dict], output_path: Path):
    """Save split to CSV file.
    
    Args:
        samples: List of samples
        output_path: Output CSV path
    """
    df = pd.DataFrame(samples)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(samples)} samples to {output_path}")


def print_statistics(samples: List[Dict], name: str):
    """Print statistics about samples.
    
    Args:
        samples: List of samples
        name: Name of the split
    """
    print(f"\n{name} Statistics:")
    print(f"  Total samples: {len(samples)}")
    
    # Count by split
    split_counts = defaultdict(int)
    for s in samples:
        split_counts[s['split']] += 1
    
    print(f"  By original split:")
    for split, count in sorted(split_counts.items()):
        print(f"    {split}: {count}")
    
    # Count by region
    region_counts = defaultdict(int)
    for s in samples:
        region_counts[s['region']] += 1
    
    print(f"  By region (top 10):")
    for region, count in sorted(region_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {region}: {count}")
    
    # Count by zone
    zone_counts = defaultdict(int)
    for s in samples:
        zone_counts[s['zone']] += 1
    
    print(f"  By zone (top 10):")
    for zone, count in sorted(zone_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {zone}: {count}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Build stratified 10k subset from FLAIR-2 dataset"
    )
    parser.add_argument(
        '--root-dir',
        type=str,
        required=True,
        help='Root directory of FLAIR-2 dataset (contains train/ and test/)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data_splits',
        help='Output directory for split files'
    )
    parser.add_argument(
        '--subset-size',
        type=int,
        default=10000,
        help='Target subset size'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Validation ratio (default: 0.2 for 8:2 split)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("FLAIR-2 Subset Builder")
    print("=" * 80)
    print(f"Root directory: {root_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target subset size: {args.subset_size}")
    print(f"Train:Val ratio: {1-args.val_ratio}:{args.val_ratio}")
    print(f"Random seed: {args.seed}")
    
    # Discover samples from train and test
    print("\nDiscovering samples...")
    train_samples = discover_samples(root_dir, "train")
    test_samples = discover_samples(root_dir, "test")
    all_samples = train_samples + test_samples
    
    print(f"Found {len(train_samples)} train samples")
    print(f"Found {len(test_samples)} test samples")
    print(f"Total: {len(all_samples)} samples")
    
    # Stratified sampling
    print(f"\nPerforming stratified sampling to {args.subset_size} samples...")
    subset_samples = stratified_sample(all_samples, args.subset_size, seed=args.seed)
    
    print(f"Selected {len(subset_samples)} samples")
    print_statistics(subset_samples, "Subset")
    
    # Split into train and val
    print(f"\nSplitting into train and val ({1-args.val_ratio}:{args.val_ratio})...")
    train_split, val_split = split_train_val(subset_samples, args.val_ratio, seed=args.seed)
    
    print_statistics(train_split, "Train Split")
    print_statistics(val_split, "Val Split")
    
    # Save splits
    print("\nSaving splits...")
    save_split(train_split, output_dir / "subset_10k_train.csv")
    save_split(val_split, output_dir / "subset_10k_val.csv")
    
    # Save metadata
    metadata = {
        'subset_size': len(subset_samples),
        'train_size': len(train_split),
        'val_size': len(val_split),
        'val_ratio': args.val_ratio,
        'seed': args.seed,
        'original_train_samples': len(train_samples),
        'original_test_samples': len(test_samples),
        'original_total_samples': len(all_samples)
    }
    
    metadata_path = output_dir / "subset_10k_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to {metadata_path}")
    print("\n" + "=" * 80)
    print("Subset creation completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
