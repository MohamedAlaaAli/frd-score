"""
Compute and interpret fréchet radiomics distances between two datasets.
"""
import os
from argparse import ArgumentParser
from time import time

import numpy as np
import pandas as pd
import torch

from src.radiomics_utils import convert_radiomic_dfs_to_vectors, compute_and_save_imagefolder_radiomics, compute_and_save_imagefolder_radiomics_parallel, interpret_radiomic_differences, compute_batch_radiomics
from src.utils import frechet_distance

def extract_features_from_batch(image_batch, parallelize=True, to_grayscale=True):
    """
    Compute radiomics feature vectors for a single image batch.

    ```
    Args:
        image_batch: torch.Tensor or numpy.ndarray of shape (B, C, H, W)
        parallelize: bool - whether to use parallel processing in compute_batch_radiomics
        to_grayscale: bool - whether to convert RGB images to grayscale before feature extraction
        
    Returns:
        numpy.ndarray: feature vectors of shape (B, F)
    """

    # Convert torch batch to numpy
    if isinstance(image_batch, torch.Tensor):
        batch_np = image_batch.detach().cpu().numpy()
    else:
        batch_np = image_batch

    # Optionally convert to grayscale
    if to_grayscale and batch_np.shape[1] == 3:
        batch_np = 0.299 * batch_np[:, 0] + 0.587 * batch_np[:, 1] + 0.114 * batch_np[:, 2]
        batch_np = batch_np[:, np.newaxis, :, :]  # restore channel dimension

    # Compute radiomics
    radiomics_df = compute_batch_radiomics(batch_np, parallelize=parallelize)

    # Convert dataframe to feature vectors
    features, _ = convert_radiomic_dfs_to_vectors(radiomics_df, radiomics_df, match_sample_count=True)

    return features

# def get_distance_fn(image_batch, parallelize=True, to_grayscale=True):

#     batch_features = extract_features_from_batch(image_batch, parallelize, to_grayscale)

#     def distance_fn(image):

#         try:
#             import torch
#         except ImportError:
#             print("Warning: torch not available. Please install torch for batch processing.")
#             torch = None
        
#         batch1_np = None

#         # Convert torch batches to numpy and ensure they're grayscale
#         if torch is not None and isinstance(image, torch.Tensor):
#             batch1_np = image.detach().cpu().numpy()
#         else:
#             batch1_np = image

#         # Convert to grayscale if needed (assume RGB input)
#         if to_grayscale and batch1_np.shape[1] == 3:  # RGB
#             # Convert RGB to grayscale using standard weights
#             batch1_np = 0.299 * batch1_np[:, 0] + 0.587 * batch1_np[:, 1] + 0.114 * batch1_np[:, 2]
#             batch1_np = batch1_np[:, np.newaxis, :, :]  # Add channel dimension back
        
#         # Compute radiomics
#         radiomics_df = compute_batch_radiomics(batch1_np, parallelize=parallelize)

#         # Convert dataframe to feature vectors
#         features, _ = convert_radiomic_dfs_to_vectors(radiomics_df, radiomics_df, match_sample_count=True)

#         fd = frechet_distance(batch_features, features, means_only = True)
#         frd = np.abs(fd)

#         return frd
    
#     return distance_fn


def get_distance_fn(image_batch, parallelize=True, to_grayscale=False):
    """
    Factory function to create a callable for FRD between a single RGB image and a precomputed reference batch.
    
    Args:
        image_batch: torch.Tensor or np.ndarray of shape (B, C, H, W) - reference batch (e.g., 12 RGB images).
        parallelize: bool - Use parallel processing for radiomics.
        to_grayscale: bool - Convert RGB to grayscale for radiomics (recommended).
    
    Returns:
        distance_fn: Callable that takes a single image (shape (C, H, W) or (1, C, H, W)) and returns FRD float.
    """
    # Compute radiomics for reference batch
    batch_radiomics_df = compute_batch_radiomics(
        image_batch.detach().cpu().numpy() if isinstance(image_batch, torch.Tensor) else image_batch,
        parallelize=parallelize
    )
    
    # Convert to features and get feature names
    batch_features, _, feature_names = convert_radiomic_dfs_to_vectors(
        batch_radiomics_df, batch_radiomics_df, match_sample_count=True, return_feature_names=True, normalize=False
    )
    
    # Remove NaN/inf features from batch_features
    valid_feature_mask = ~np.isnan(batch_features).any(axis=0) & ~np.isinf(batch_features).any(axis=0)
    batch_features = batch_features[:, valid_feature_mask]
    valid_feature_names = feature_names[valid_feature_mask]
    
    # Normalize batch_features
    mean = np.mean(batch_features, axis=0)
    std = np.std(batch_features, axis=0) + 1e-6  # Avoid division by zero
    batch_features = (batch_features - mean) / std

    def distance_fn(image):
        # Convert to numpy if tensor
        if isinstance(image, torch.Tensor):
            batch1_np = image.detach().cpu().numpy()
        else:
            batch1_np = np.asarray(image)
        
        # Ensure batch dimension: (1, C, H, W)
        if batch1_np.ndim == 3:  # (C, H, W)
            batch1_np = batch1_np[np.newaxis, :, :, :]  # Add batch dim
        
        # Convert to grayscale if needed (for RGB input)
        if to_grayscale and batch1_np.shape[1] == 3:  # RGB
            batch1_np = 0.299 * batch1_np[:, 0] + 0.587 * batch1_np[:, 1] + 0.114 * batch1_np[:, 2]
            batch1_np = batch1_np[:, np.newaxis, :, :]  # (1, 1, H, W)
        
        # Compute radiomics for the single image
        radiomics_df = compute_batch_radiomics(batch1_np, parallelize=parallelize)
        
        # Align features to reference batch's valid feature names
        # Select only the valid feature names from radiomics_df
        radiomics_df = radiomics_df[['img_fname'] + [col for col in radiomics_df.columns if col in valid_feature_names]]
        
        # Convert to features
        features, _ = convert_radiomic_dfs_to_vectors(
            radiomics_df, radiomics_df, match_sample_count=True, normalize=False
        )
        
        # Ensure features have the same order as valid_feature_names
        feature_indices = [radiomics_df.columns.get_loc(col) - 1 for col in valid_feature_names]  # -1 for img_fname
        features = features[:, feature_indices]
        
        # Normalize features using reference batch stats
        features = (features - mean) / std
        
        # Compute FRD (squared L2 distance on means)
        fd = frechet_distance(batch_features, features, means_only=True)
        frd = np.abs(fd)
        
        return frd
    
    return distance_fn

def main(
        image_batch1,
        image_batch2,
        force_compute_fresh = False,
        interpret = False,
        parallelize = True,
        comp=True,
        means_only=False
):
    """
    Compute FRD between two torch image batches.
    
    Args:
        image_batch1: torch.Tensor of shape (B, C, H, W) - first batch of images
        image_batch2: torch.Tensor of shape (B, C, H, W) - second batch of images
        force_compute_fresh: bool - not used in batch mode
        interpret: bool - whether to generate interpretability plots
        parallelize: bool - whether to use parallel processing
        comp: bool - whether to compute the actual FRD value
        
    Returns:
        float: FRD value if comp=True, else None
    """
    
    # Convert torch batches to numpy and ensure they're grayscale
    if isinstance(image_batch1, torch.Tensor):
        batch1_np = image_batch1.detach().cpu().numpy()
    else:
        batch1_np = image_batch1
        
    if isinstance(image_batch2, torch.Tensor):
        batch2_np = image_batch2.detach().cpu().numpy()
    else:
        batch2_np = image_batch2
    
    # Convert to grayscale if needed (assume RGB input)
    if batch1_np.shape[1] == 3:  # RGB
        # Convert RGB to grayscale using standard weights
        batch1_np = 0.299 * batch1_np[:, 0] + 0.587 * batch1_np[:, 1] + 0.114 * batch1_np[:, 2]
        batch1_np = batch1_np[:, np.newaxis, :, :]  # Add channel dimension back
        
    if batch2_np.shape[1] == 3:  # RGB
        batch2_np = 0.299 * batch2_np[:, 0] + 0.587 * batch2_np[:, 1] + 0.114 * batch2_np[:, 2]
        batch2_np = batch2_np[:, np.newaxis, :, :]  # Add channel dimension back

    # Compute radiomics for both batches
    print("Computing radiomics for batch 1...")
    radiomics_df1 = compute_batch_radiomics(batch1_np, parallelize=parallelize)
    print("Computed radiomics for batch 1.")
    
    print("Computing radiomics for batch 2...")
    radiomics_df2 = compute_batch_radiomics(batch2_np, parallelize=parallelize)
    print("Computed radiomics for batch 2.")

    if comp:
        feats1, feats2 = convert_radiomic_dfs_to_vectors(radiomics_df1, 
                                                             radiomics_df2,
                                                             match_sample_count=False, # needed for distance measures
                                                             ) 
        
        print(feats1.shape, feats2.shape)

        # Frechet distance
        fd = frechet_distance(feats1, feats2, means_only = means_only)
        frd = np.abs(fd)

        print("FRD = {}".format(frd))

        if interpret:
            # For interpretation, we need to save temporary CSV files
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp1:
                radiomics_df1.to_csv(tmp1.name, index=False)
                radiomics_path1 = tmp1.name
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp2:
                radiomics_df2.to_csv(tmp2.name, index=False)
                radiomics_path2 = tmp2.name
                
            run_tsne = True
            interpret_radiomic_differences(radiomics_path1, radiomics_path2, run_tsne=run_tsne)
            
            # Clean up temporary files
            os.unlink(radiomics_path1)
            os.unlink(radiomics_path2)

        return frd

def main_folders(
        image_folder1,
        image_folder2,
        force_compute_fresh = False,
        interpret = False,
        parallelize = True,
        comp=True
):
    """
    Original main function that works with image folders (for backward compatibility).
    """
    radiomics_fname = 'radiomics.csv'

    radiomics_path1 = os.path.join(image_folder1, radiomics_fname)
    radiomics_path2 = os.path.join(image_folder2, radiomics_fname)

    # if needed, compute radiomics for the images
    if force_compute_fresh or not os.path.exists(radiomics_path1):
        print("No radiomics found computed for image folder 1 at {}, computing now.".format(radiomics_path1))
        if parallelize:
            compute_and_save_imagefolder_radiomics_parallel(image_folder1, radiomics_fname=radiomics_fname)
        else:
            compute_and_save_imagefolder_radiomics(image_folder1, radiomics_fname=radiomics_fname)
        print("Computed radiomics for image folder 1.")
    else:
        print("Radiomics already computed for image folder 1 at {}.".format(radiomics_path1))

    if force_compute_fresh or not os.path.exists(radiomics_path2):
        print("No radiomics found computed for image folder 2 at {}, computing now.".format(radiomics_path2))
        if parallelize:
            compute_and_save_imagefolder_radiomics_parallel(image_folder2, radiomics_fname=radiomics_fname)
        else:
            compute_and_save_imagefolder_radiomics(image_folder2, radiomics_fname=radiomics_fname)
        print("Computed radiomics for image folder 2.")
    else:
        print("Radiomics already computed for image folder 2 at {}.".format(radiomics_path2))
    if comp:
        # load radiomics
        radiomics_df1 = pd.read_csv(radiomics_path1)
        radiomics_df2 = pd.read_csv(radiomics_path2)

        feats1, feats2 = convert_radiomic_dfs_to_vectors(radiomics_df1, 
                                                             radiomics_df2,
                                                             match_sample_count=True, # needed for distance measures
                                                             ) 
        # Frechet distance
        fd = frechet_distance(feats1, feats2)
        frd = np.abs(fd)

        print("FRD = {}".format(frd))

        if interpret:
            run_tsne = True
            interpret_radiomic_differences(radiomics_path1, radiomics_path2, run_tsne=run_tsne)

        return frd

if __name__ == "__main__":
    tstart = time()
    parser = ArgumentParser()

    parser.add_argument('--image_folder1', type=str, required=True)
    parser.add_argument('--image_folder2', type=str, required=True)
    parser.add_argument('--force_compute_fresh', action='store_true', help='re-compute all radiomics fresh')
    parser.add_argument('--interpret', action='store_true', help='interpret the features underlying Fréchet Radiomics Distance')
    parser.add_argument('--comp', type=bool, required=False, default=True)
    args = parser.parse_args()

    main_folders(
        args.image_folder1,
        args.image_folder2,
        force_compute_fresh=args.force_compute_fresh,
        interpret=args.interpret,
        comp=args.comp
        )

    tend = time()
    print("compute time (sec): {}".format(tend - tstart))