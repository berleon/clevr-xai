"""
Code to evaluate explanation methods (aka heatmaps/XAI methods) w.r.t. ground truths, as in:
Towards Ground Truth Evaluation of Visual Explanations, Osman et al. 2020
https://arxiv.org/pdf/2003.07258.pdf
(code by Leila Arras, Fraunhofer HHI - Berlin, Germany)
"""
from typing import Dict, List
import os, json
import numpy as np


def pool_heatmap (heatmap: np.ndarray, pooling_type: str) -> np.ndarray:
    """
    Pool the relevance along the channel axis, according to the pooling technique specified by pooling_type.
    """
    C, H, W = heatmap.shape

    if pooling_type=="sum,abs":
        pooled_heatmap = np.abs(np.sum(heatmap, axis=0))

    elif pooling_type=="sum,pos":
        pooled_heatmap = np.sum(heatmap, axis=0) ; pooled_heatmap = np.where(pooled_heatmap>0.0, pooled_heatmap, 0.0)

    elif pooling_type=="max-norm":
        pooled_heatmap = np.amax(np.abs(heatmap), axis=0)

    elif pooling_type=="l1-norm":
        pooled_heatmap = np.linalg.norm(heatmap, ord=1, axis=0)

    elif pooling_type=="l2-norm":
        pooled_heatmap = np.linalg.norm(heatmap, ord=2, axis=0)

    elif pooling_type=="l2-norm,sq":
        pooled_heatmap = (np.linalg.norm(heatmap, ord=2, axis=0))**2

    assert pooled_heatmap.shape == (H, W) and np.all(pooled_heatmap>=0.0)
    return pooled_heatmap


def evaluate_single (heatmap: np.ndarray, ground_truth: np.ndarray, pooling_type: str) -> Dict[str, np.float64]:
    """
    Given an image's relevance heatmap and a corresponding ground truth boolean ndarray of the same vertical and horizontal dimensions, return both:
     - the ratio of relevance falling into the ground truth area w.r.t. the total amount of relevance ("relevance mass accuracy" metric)
     - the ratio of pixels within the N highest relevant pixels (where N is the size of the ground truth area) that effectively belong to the ground truth area
       ("relevance rank accuracy" metric)
    Both ratios are calculated after having pooled the relevance across the channel axis, according to the pooling technique defined by the pooling_type argument.
    Args:
    - heatmap (np.ndarray):         of shape (C, H, W), with dtype float
    - ground_truth (np.ndarray):    of shape (H, W), with dtype bool
    - pooling_type (str):           specifies how to pool the relevance across the channels, i.e. defines a mapping function f: R^C -> R^+
                                    that maps a real-valued vector of dimension C to a positive number (see details of each pooling_type in the function pool_heatmap)
    Returns:
    A dict wich keys=["mass", "rank"] and resp. values:
    - relevance_mass_accuracy (np.float64):     relevance mass accuracy, float in the range [0.0, 1.0], the higher the better.
    - relevance_rank_accuracy (np.float64):     relevance rank accuracy, float in the range [0.0, 1.0], the higher the better.
    """
    C, H, W = heatmap.shape # C relevance values per pixel coordinate (C=number of channels), for an image with vertical and horizontal dimensions HxW
    assert ground_truth.shape == (H, W)

    heatmap = heatmap.astype(dtype=np.float64) # cast heatmap to float64 precision (better for computing relevance accuracy statistics)

    # step 1: pool the relevance across the channel dimension to get one positive relevance value per pixel coordinate
    pooled_heatmap = pool_heatmap (heatmap, pooling_type)

    # step 2: compute the ratio of relevance mass within ground truth w.r.t the total relevance
    relevance_within_ground_truth = np.sum(pooled_heatmap * np.where(ground_truth, 1.0, 0.0).astype(dtype=np.float64) )
    relevance_total               = np.sum(pooled_heatmap)
    relevance_mass_accuracy       = 1.0 * relevance_within_ground_truth/relevance_total ; assert (0.0<=relevance_mass_accuracy) and (relevance_mass_accuracy<=1.0)

    # step 3: order the pixel coordinates in decreasing order of their relevance, then count the number N_gt of pixels within the N highest relevant pixels that fall
    # into the ground truth area, where N is the total number of pixels of the ground truth area, then compute the ratio N_gt/N
    pixels_sorted_by_relevance = np.argsort(np.ravel(pooled_heatmap))[::-1] ; assert pixels_sorted_by_relevance.shape == (H*W,) # sorted pixel indices over flattened array
    gt_flat = np.ravel(ground_truth)                                        ; assert gt_flat.shape == (H*W,) # flattened ground truth array
    N    = np.sum(gt_flat)
    N_gt = np.sum(gt_flat[pixels_sorted_by_relevance[:int(N)]])
    relevance_rank_accuracy = 1.0 * N_gt/N ; assert (0.0<=relevance_rank_accuracy) and (relevance_rank_accuracy<=1.0)

    return {"mass": relevance_mass_accuracy, "rank": relevance_rank_accuracy} # dict of relevance accuracies, with key=evaluation metric


def evaluate (heatmap_DIR: str, ground_truth_DIR: str, output_DIR: str, idx_list: List[int], output_name: str = "", evaluation_metric: str = "rank"):
    """
    Given a set of relevance heatmaps obtained via an explanation method (e.g. via LRP or Integrated Gradients), and a set of ground truth masks,
    both previously saved to disk as numpy ndarrays (resp. in the directories heatmap_DIR and ground_truth_DIR, and with filenames <data point idx>+".npy"),
    compute various relevance accuracy metric statistics over the subset of data points specified by their indices via the argument idx_list.
    The statistics over this subset of data points, as well as the relevance accuracies for each data point, will be saved to disk in the directory output_DIR as JSON files.
    Additionally, two text files will be written to the directory output_DIR, they contain the summary statistics for different types of pooling ordered by their mean value
    in decreasing order, or printed in a pre-defined fixed order, and formatted as tables.
    The relevance accuracy metric used for evaluating the heatmaps is specified via the evaluation_metric argument, it can be either "mass" or "rank".
    """

    accuracy = {} # key=idx of data point (as string), value=dict containing one relevance accuracy per pooling type (with pooling_type as key, and relevance accuracy as value)

    # iterate over data points
    for idx in idx_list:

        heatmap      = np.load(os.path.join(heatmap_DIR,      str(idx) + ".npy"))
        ground_truth = np.load(os.path.join(ground_truth_DIR, str(idx) + ".npy"))

        accuracy_single = {} # key=pooling_type, value= relevance accuracy
        for pooling_type in ["sum,abs", "sum,pos", "max-norm", "l1-norm", "l2-norm", "l2-norm,sq"]:
            accuracy_single[pooling_type] = evaluate_single (heatmap, ground_truth, pooling_type) [evaluation_metric]

        accuracy[str(idx)] = accuracy_single

    # compute statistics
    accuracy_statistics = {} # key=pooling_type, value=dict containing the relevance statistics (with the statistic's name as key, and statistic's value as value)
    for pooling_type in ["sum,abs", "sum,pos", "max-norm", "l1-norm", "l2-norm", "l2-norm,sq"]:
        accuracy_statistics[pooling_type] = {}
        values = np.asarray( [ accuracy[str(idx)][pooling_type] for idx in idx_list ] ) ; assert values.shape==(len(idx_list),)
        accuracy_statistics[pooling_type]["mean"]     = np.mean(values)
        accuracy_statistics[pooling_type]["std"]      = np.std(values)
        accuracy_statistics[pooling_type]["min"]      = np.amin(values)
        accuracy_statistics[pooling_type]["max"]      = np.amax(values)
        accuracy_statistics[pooling_type]["median"]   = np.percentile(values, q=50, interpolation='linear')
        accuracy_statistics[pooling_type]["perc-80"]  = np.percentile(values, q=80, interpolation='linear')
        accuracy_statistics[pooling_type]["perc-20"]  = np.percentile(values, q=20, interpolation='linear')

    # write statistics over all pooling_types to text file in decreasing order of the mean, and formatted as a table
    list_to_sort = [(k, v["mean"]) for (k, v) in accuracy_statistics.items()]
    sorted_keys  = [ k for (k, _) in sorted(list_to_sort, key=lambda tup: tup[1], reverse=True)]

    col_width = 17
    with open(os.path.join(output_DIR,  output_name + "_ORDERED.txt"), "w") as stat_file:
        titles = ["pooling_type", "mean", "std", "min", "max", "median", "perc-80", "perc-20"]
        stat_file.write(''.join([(title).ljust(col_width) for title in titles]) + '\n')
        for k in sorted_keys:
            row = [k] + ["{:4.2f}".format(accuracy_statistics[k][stat_name]) for stat_name in titles[1:]]
            stat_file.write(''.join([value.ljust(col_width) for value in row]) + '\n')
        stat_file.write("\n\nRelevance accuracy metric:            " + evaluation_metric)
        stat_file.write("\n\nStatistics computed over data points: " + str(len(idx_list)))

    # write statistics over all pooling_types to text file in fixed order, and formatted as a table
    col_width = 17
    with open(os.path.join(output_DIR,  output_name + "_FIXED.txt"), "w") as stat2_file:
        titles = ["pooling_type", "mean", "std", "median"]
        stat2_file.write(''.join([(title).ljust(col_width) for title in titles]) + '\n')
        for k in ["max-norm", "l2-norm,sq", "l2-norm", "l1-norm", "sum,abs", "sum,pos"]: # fixed order DEFINED HERE !
            row = [k] + ["{:4.2f}".format(accuracy_statistics[k][stat_name]) for stat_name in titles[1:]]
            row[2] = "("+row[2]+")" # put parenthesis around std
            stat2_file.write(''.join([value.ljust(col_width) for value in row]) + '\n')
        stat2_file.write("\n\nRelevance accuracy metric:            " + evaluation_metric)
        stat2_file.write("\n\nStatistics computed over data points: " + str(len(idx_list)))

    # save accuracy statistics and accuracies for each data point as JSON files
    json.dump(accuracy,            open(os.path.join(output_DIR, output_name + '_datapoint'),  "w"), indent=4)
    json.dump(accuracy_statistics, open(os.path.join(output_DIR, output_name + '_statistic'),  "w"), indent=4)

