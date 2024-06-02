import numpy as np
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
from scipy.ndimage.measurements import label, find_objects
from scipy.stats import pearsonr

import sys
sys.path.extend(['./','../','../Models/','../Data_Loader/','../Custom_Functions/'])

def dc(result, reference):
    """
    Dice Coefficient (DC), a statistical measure for evaluating the similarity between two sets.
    It is often used to compare the pixel-wise agreement between a predicted segmentation and its corresponding ground truth.

    Args:
    result (numpy.ndarray): Predicted binary segmentation.
    reference (numpy.ndarray): Ground truth binary segmentation.

    Returns:
    float: Dice coefficient score.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    
    intersection = np.count_nonzero(result & reference)
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)
    
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    return dc

def jc(result, reference):
    """
    Jaccard Coefficient (JC), also known as the Intersection over Union (IoU).
    This metric measures the similarity and diversity of sample sets, commonly used in the evaluation of segmentation tasks.

    Args:
    result (numpy.ndarray): Predicted binary segmentation.
    reference (numpy.ndarray): Ground truth binary segmentation.

    Returns:
    float: Jaccard coefficient score.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    
    intersection = np.count_nonzero(result & reference)
    union = np.count_nonzero(result | reference)
    
    try:
        jc = float(intersection) / float(union)
    except ZeroDivisionError:
        jc = 0.0
    return jc

def precision(result, reference):
    """
    Precision metric, which evaluates the accuracy of the positive predictions.
    It is the ratio of true positive predictions to the total predicted positives.

    Args:
    result (numpy.ndarray): Predicted binary segmentation.
    reference (numpy.ndarray): Ground truth binary segmentation.

    Returns:
    float: Precision score.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
        
    tp = np.count_nonzero(result & reference)
    fp = np.count_nonzero(result & ~reference)
    
    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0
    return precision
    
def recall(result, reference):
    """
    Recall metric, also known as sensitivity or true positive rate.
    It measures the ability of a model to find all the relevant cases (true positives) within a dataset.

    Args:
    result (numpy.ndarray): Predicted binary segmentation.
    reference (numpy.ndarray): Ground truth binary segmentation.

    Returns:
    float: Recall score.
    """ 
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
        
    tp = np.count_nonzero(result & reference)
    fn = np.count_nonzero(~result & reference)
    
    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0
    return recall
    
def sensitivity(result, reference):
    """
    Sensitivity, identical to recall, measures the proportion of actual positives that are correctly identified.
    It is particularly useful in medical imaging to evaluate how well a model detects conditions.

    Args:
    result (numpy.ndarray): Predicted binary segmentation.
    reference (numpy.ndarray): Ground truth binary segmentation.

    Returns:
    float: Sensitivity score.
    """
    return recall(result, reference)

def specificity(result, reference):
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
       
    tn = np.count_nonzero(~result & ~reference)
    fp = np.count_nonzero(result & ~reference)
    
    try:
        specificity = tn / float(tn + fp)
    except ZeroDivisionError:
        specificity = 0.0
    return specificity

def asd(result, reference, voxelspacing=None, connectivity=1):
    """
    Average Surface Distance (ASD), a metric used to quantify the average distance between the surfaces of two binary objects.
    It is useful in assessing the spatial accuracy of segmentation models.

    Args:
    result (numpy.ndarray): Predicted binary segmentation.
    reference (numpy.ndarray): Ground truth binary segmentation.
    voxelspacing (tuple, optional): Voxel spacing in each dimension.
    connectivity (int, optional): Connectivity defining neighborhood for surface extraction.

    Returns:
    float: Average surface distance.
    """
    sds = __surface_distances(result, reference, voxelspacing, connectivity)
    asd = sds.mean()
    return asd

def assd(result, reference, voxelspacing=None, connectivity=1):
    """
    Average Symmetric Surface Distance (ASSD), an extension of ASD that calculates the mean distance from each surface point of one segmentation to the nearest point on the surface of another segmentation and vice versa.
    It provides a symmetric measure of the surface distance between two segmentations.

    Args:
    result (numpy.ndarray): Predicted binary segmentation.
    reference (numpy.ndarray): Ground truth binary segmentation.
    voxelspacing (tuple, optional): Voxel spacing in each dimension.
    connectivity (int, optional): Connectivity defining neighborhood for surface extraction.

    Returns:
    float: Average symmetric surface distance.
    """
    assd = (asd(result, reference, voxelspacing, connectivity)
            + asd(reference, result, voxelspacing, connectivity)) / 2
    return assd

def hd(result, reference, voxelspacing=None, connectivity=1):
    """
    Hausdorff Distance (HD), a measure of the distance between the surfaces of two objects.
    It is often used in medical imaging to evaluate the spatial distance between two segmentations.

    Args:
    result (numpy.ndarray): Predicted binary segmentation.
    reference (numpy.ndarray): Ground truth binary segmentation.
    voxelspacing (tuple, optional): Voxel spacing in each dimension.
    connectivity (int, optional): Connectivity defining neighborhood for surface extraction.

    Returns:
    float: Hausdorff distance.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd

def ravd(result, reference):
    """
    Relative Absolute Volume Difference (RAVD), a measure of the relative difference in volume between two segmentations.
    It is calculated as the absolute difference in volume between the two segmentations divided by the volume of the reference segmentation.

    Args:
    result (numpy.ndarray): Predicted binary segmentation.
    reference (numpy.ndarray): Ground truth binary segmentation.

    Returns:
    float: Relative absolute volume difference.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
        
    vol1 = np.count_nonzero(result)
    vol2 = np.count_nonzero(reference)

    if 0 == vol2:
        raise RuntimeError("The second supplied array does not contain any binary object.")
    return (vol1 - vol2) / float(vol2)
    
def obj_asd(result, reference, voxelspacing=None, connectivity=1):
    """
    Average Surface Distance (ASD) between objects, a metric used to quantify the average distance between the surfaces of two binary objects.
    It is useful in assessing the spatial accuracy of segmentation models.

    Args:
    result (numpy.ndarray): Predicted binary segmentation.
    reference (numpy.ndarray): Ground truth binary segmentation.
    voxelspacing (tuple, optional): Voxel spacing in each dimension.
    connectivity (int, optional): Connectivity defining neighborhood for surface extraction.

    Returns:
    float: Average surface distance.
    """
    sds = list()
    labelmap1, labelmap2, _a, _b, mapping = __distinct_binary_object_correspondences(result, reference, connectivity)
    for lid2, lid1 in mapping.items():
        object1 = labelmap1 == lid1
        object2 = labelmap2 == lid2
        sds.extend(__surface_distances(object1, object2, voxelspacing, connectivity))
    asd = np.mean(sds)
    return asd

def obj_assd(result, reference, voxelspacing=None, connectivity=1):
    """
    Average Symmetric Surface Distance (ASSD) between objects, an extension of ASD that calculates the mean distance from each surface point of one segmentation to the nearest point on the surface of another segmentation and vice versa.
    It provides a symmetric measure of the surface distance between two segmentations.

    Args:
    result (numpy.ndarray): Predicted binary segmentation.
    reference (numpy.ndarray): Ground truth binary segmentation.
    voxelspacing (tuple, optional): Voxel spacing in each dimension.
    connectivity (int, optional): Connectivity defining neighborhood for surface extraction.

    Returns:
    float: Average symmetric surface distance.
    """
    assd = (obj_asd(result, reference, voxelspacing, connectivity)
            + obj_asd(reference, result, voxelspacing, connectivity)) / 2
    return assd
        
def obj_fpr(result, reference, connectivity=1):
    """
    False Positive Rate (FPR) of distinct binary object detection.
    It measures the proportion of false positive predictions compared to the total predicted positives.

    Args:
    result (numpy.ndarray): Predicted binary segmentation.
    reference (numpy.ndarray): Ground truth binary segmentation.
    connectivity (int, optional): Connectivity defining neighborhood for surface extraction.

    Returns:
    float: False positive rate.
    """
    _, _, _, n_obj_reference, mapping = __distinct_binary_object_correspondences(reference, result, connectivity)
    return (n_obj_reference - len(mapping)) / float(n_obj_reference)
    
def obj_tpr(result, reference, connectivity=1):
    """
    True Positive Rate (TPR) of distinct binary object detection.
    It measures the proportion of true positive predictions compared to the total predicted positives.

    Args:
    result (numpy.ndarray): Predicted binary segmentation.
    reference (numpy.ndarray): Ground truth binary segmentation.
    connectivity (int, optional): Connectivity defining neighborhood for surface extraction.

    Returns:
    float: True positive rate.
    """
    _, _, n_obj_result, _, mapping = __distinct_binary_object_correspondences(reference, result, connectivity)
    return len(mapping) / float(n_obj_result)

def __distinct_binary_object_correspondences(reference, result, connectivity=1):
    """
    Determines all distinct (where connectivity is defined by the connectivity parameter
    passed to scipy's `generate_binary_structure`) binary objects in both of the input
    parameters and returns a 1to1 mapping from the labelled objects in reference to the
    corresponding (whereas a one-voxel overlap suffices for correspondence) objects in
    result.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # label distinct binary objects  
    labelmap1, n_obj_result = label(result, footprint)
    labelmap2, n_obj_reference = label(reference, footprint)
    
    # find all overlaps from labelmap2 to labelmap1; collect one-to-one relationships and store all one-two-many for later processing
    slicers = find_objects(labelmap2)
    mapping = dict() 
    used_labels = set()
    one_to_many = list()
    for l1id, slicer in enumerate(slicers, 1):
        bobj = (l1id) == labelmap2[slicer] 
        l2ids = np.unique(labelmap1[slicer][bobj]) 
        l2ids = l2ids[0 != l2ids]
        if 1 == len(l2ids): 
            l2id = l2ids[0]
            if not l2id in used_labels:
                mapping[l1id] = l2id
                used_labels.add(l2id)
        elif 1 < len(l2ids):
            one_to_many.append((l1id, set(l2ids)))
            
    # process one-to-many mappings, always choosing the one with the least labelmap2 correspondences first
    while True:
        one_to_many = [(l1id, l2ids - used_labels) for l1id, l2ids in one_to_many]
        one_to_many = [x for x in one_to_many if x[1]]
        if 0 == len(one_to_many):
            break
        one_to_many = sorted(one_to_many, key=lambda x: len(x[1]))
        l2id = one_to_many[0][1].pop()
        mapping[one_to_many[0][0]] = l2id
        used_labels.add(l2id)
        one_to_many = one_to_many[1:]
        
    return labelmap1, labelmap2, n_obj_result, n_obj_reference, mapping
    
def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
            
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    if 0 == np.count_nonzero(result):
        raise RuntimeError("The first supplied array does not contain any binary object.")
    if 0 == np.count_nonzero(reference):
        raise RuntimeError("The second supplied array does not contain any binary object.")
        
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    
    return sds