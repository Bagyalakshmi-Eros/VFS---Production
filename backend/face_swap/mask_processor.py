"""
Module for improving occlusion mask quality in face swapping applications.
Focuses on making white regions whiter, black regions blacker,
removing outliers, and providing temporal consistency.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def improve_occlusion_mask(mask, threshold=0.5, clean_outliers=True):
    """
    Improve occlusion mask by:
    1. Sharpening the contrast (white more white, black more black)
    2. Applying thresholding for decisive areas
    3. Cleaning outliers (small black regions in white areas and vice versa)
    
    Args:
        mask: Original occlusion mask (single channel, 0-1 range)
        threshold: Threshold value for binary mask creation
        clean_outliers: Whether to remove small outliers
        
    Returns:
        Improved mask
    """
    try:
        # Ensure mask is in correct format
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        # Step 1: Apply contrast enhancement 
        # Normalize to full range first
        enhanced_mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)
        
        # Apply gamma correction to enhance contrast
        gamma = 0.5  # < 1 makes bright regions brighter, dark regions darker
        enhanced_mask = np.power(enhanced_mask, gamma)
        
        # Step 2: Apply thresholding to make a more decisive mask
        _, binary_mask = cv2.threshold(enhanced_mask, threshold, 1.0, cv2.THRESH_BINARY)
        
        if clean_outliers:
            # Step 3: Remove small outliers
            
            # First identify connected components
            binary_uint8 = (binary_mask * 255).astype(np.uint8)
            
            # Find white regions (value == 1, non-occluded)
            white_components = cv2.connectedComponentsWithStats(binary_uint8, connectivity=8)
            white_labels, white_stats = white_components[1], white_components[2]
            
            # Find black regions (value == 0, occluded) by inverting the mask
            inverted = (1 - binary_uint8).astype(np.uint8)
            black_components = cv2.connectedComponentsWithStats(inverted, connectivity=8)
            black_labels, black_stats = black_components[1], black_components[2]
            
            # Create clean mask
            clean_mask = binary_mask.copy()
            
            # Remove small white regions (area < threshold)
            min_white_area = binary_mask.shape[0] * binary_mask.shape[1] * 0.01  # 1% of mask area
            for i in range(1, len(white_stats)):  # Skip background (label 0)
                if white_stats[i, cv2.CC_STAT_AREA] < min_white_area:
                    clean_mask[white_labels == i] = 0
            
            # Remove small black regions (area < threshold)
            min_black_area = binary_mask.shape[0] * binary_mask.shape[1] * 0.01  # 1% of mask area
            for i in range(1, len(black_stats)):  # Skip background (label 0)
                if black_stats[i, cv2.CC_STAT_AREA] < min_black_area:
                    clean_mask[black_labels == i] = 1
            
            # Apply morphological operations to clean up remaining noise
            kernel = np.ones((5, 5), np.uint8)
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
            
            return clean_mask
        else:
            return binary_mask
            
    except Exception as e:
        logger.error(f"Error in improve_occlusion_mask: {str(e)}")
        return mask  # Return original mask on error

def refine_occlusion_mask_with_temporal(face_id, current_mask, history=None, history_size=5):
    """
    Apply temporal consistency to occlusion masks across frames
    
    Args:
        face_id: Identifier for the face
        current_mask: Current frame's mask
        history: Dictionary of mask history for each face
        history_size: Maximum history size to maintain
        
    Returns:
        Temporally refined mask and updated history
    """
    try:
        if history is None:
            history = {}
        
        # Ensure we have history for this face
        if face_id not in history:
            history[face_id] = []
        
        # Add current mask to history
        history[face_id].append(current_mask.copy())
        
        # Keep history limited to specified size
        if len(history[face_id]) > history_size:
            history[face_id].pop(0)
        
        # If we only have one frame, return current mask
        if len(history[face_id]) == 1:
            return current_mask, history
        
        # Apply temporal smoothing
        weights = np.exp(np.linspace(0, 2, len(history[face_id])))  # Exponential weights
        weights = weights / np.sum(weights)  # Normalize
        
        # Initialize output mask
        output_mask = np.zeros_like(current_mask)
        
        # Apply weighted average of historical masks
        for i, mask in enumerate(history[face_id]):
            output_mask += mask * weights[i]
        
        # Re-threshold for crisp edges
        _, output_mask = cv2.threshold(output_mask, 0.5, 1.0, cv2.THRESH_BINARY)
        
        return output_mask, history
        
    except Exception as e:
        logger.error(f"Error in refine_occlusion_mask_with_temporal: {str(e)}")
        return current_mask, history  # Return original mask on error

def detect_mask_quality(mask):
    """
    Analyze mask quality metrics for debugging
    
    Args:
        mask: Occlusion mask to analyze
        
    Returns:
        Dictionary of quality metrics
    """
    try:
        # Ensure mask is single channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            
        # Calculate basic statistics
        mean_value = np.mean(mask)
        min_value = np.min(mask)
        max_value = np.max(mask)
        std_value = np.std(mask)
        
        # Calculate histogram
        hist = cv2.calcHist([mask], [0], None, [256], [0, 1])
        hist_peaks = np.argmax(hist)
        
        # Calculate edge amounts (measure of sharpness)
        edges = cv2.Canny((mask * 255).astype(np.uint8), 100, 200)
        edge_percent = np.count_nonzero(edges) / (mask.shape[0] * mask.shape[1])
        
        # Return quality metrics
        return {
            "mean": mean_value,
            "min": min_value,
            "max": max_value,
            "std": std_value,
            "hist_peaks": hist_peaks,
            "edge_percent": edge_percent
        }
        
    except Exception as e:
        logger.error(f"Error in detect_mask_quality: {str(e)}")
        return {}  # Return empty dict on error