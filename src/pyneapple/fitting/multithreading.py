"""
Multithreading module for parallel processing of data.
Takes functions or partials and a zipped list of arguments to process in parallel.

Methods:
    multithreader(func, arg_list, n_pools) -> list

    IDEAL related methods:
    sort_interpolated_array(results, array) -> np.ndarray
    sort_fit_array(results, array) -> np.ndarray
"""

from __future__ import annotations
import numpy as np

from typing import Callable
from functools import partial
from multiprocessing import Pool


def multithreader(
    func: Callable | partial,
    arg_list: zip | tuple,  # tuple for @JJ segmentation wise?
    n_pools: int | None = None,
) -> list:
    """Handles multithreading for different Functions.

    Will take a complete partial function and a zipped list io arguments containing
    indexes for array positions and process them ether sequential or parallel. The
    results of each call are returned as a list of tuples.

    Args:
        func (Callable | partial): Function to process with only the indices
            and the array missing.
        arg_list (zip | tuple): Data to process zip(position, array).
        n_pools (int | None): Number of threads to use for multiprocessing. If ether
            0 or None is given no threads will be used.
    Returns:
        results (list): List of tuples of indexes and processed data.
    """

    def starmap_handler(
        function: Callable, arguments_list: zip, number_pools: int
    ) -> list:
        """Handles starmap multithreading for different Functions.

        Will tak ea a partial function and a zipped list to run it in parallel threats.

        Args:
            function (Callable): Function to process returning tuple of index and data.
            arguments_list (zip): Data to process zip(position, array).
            number_pools (int): Number of threads to use for multiprocessing.

        Returns:
            results_list (list): List of processed data.
        """
        if number_pools != 0:
            with Pool(number_pools) as pool:
                results_list = pool.starmap(function, arguments_list)
        return results_list

    results = list()
    # Perform multithreading accordingly
    if n_pools:
        results = starmap_handler(func, arg_list, n_pools)
    else:
        for element in arg_list:
            results.append(func(*element))
    return results


# IDEAL related functions


def sort_interpolated_array(results: list, array: np.ndarray) -> np.ndarray:
    """Sort results interpolated image planes to array of desired shape.

    Args:
        results (list): List of tuples containing the index and the processed data.
        array (np.ndarray): New array to cast into.

    Returns:
        array (np.ndarray): With sorted results.
    """
    for element in results:
        array[:, :, tuple(element[0])] = element[1]  # python 3.9 support
    return array


def sort_fit_array(results: list, array: np.ndarray) -> np.ndarray:
    """Sort results from fitted pixels to array of desired shape.

    Args:
        results (list): List of tuples containing the index and the processed data.
        array (np.ndarray): New array to cast into.

    Returns:
        array (np.ndarray): With sorted results.
    """
    for element in results:
        array[element[0]] = element[1]
    return array
