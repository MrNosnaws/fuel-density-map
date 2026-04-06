
import numpy as np
import analysis
import main

def get_multi_match_raw_data(video_path_array, start_time_array=None, end_time_array=None):
    if start_time_array is None or len(start_time_array) != len(video_path_array):
        start_time_array = [0] * len(video_path_array)

    if end_time_array is None or len(end_time_array) != len(video_path_array):
        end_time_array = [None] * len(video_path_array)

    results = []

    for video_path, start_time, end_time in zip(video_path_array, start_time_array, end_time_array):
        data = analysis.analyze_video_total(video_path, start_time, end_time)
        results.append(data)

    return results

def get_multi_match_combined_data(video_path_array, start_time_array=None, end_time_array=None):
    all_maps = get_multi_match_raw_data(video_path_array, start_time_array, end_time_array)

    if not all_maps:
        return np.array([])

    # Ensure all maps are same shape
    base_shape = all_maps[0].shape

    for m in all_maps:
        if m.shape != base_shape:
            raise ValueError(f"Shape mismatch: {m.shape} vs {base_shape}")

    combined = np.zeros(base_shape, dtype=np.uint32)

    for m in all_maps:
        combined += m

    return combined

def get_multi_match_color_data(video_path_array, start_time_array=None, end_time_array=None, average_display_color=(255, 255, 0), pct_from_average_to_max=0.5):
    combined_data = get_multi_match_combined_data(video_path_array, start_time_array, end_time_array)
    max_value = np.max(combined_data)
    average_of_non_zero_values = np.mean(combined_data[combined_data > 0]) if np.any(combined_data > 0) else 0
    average_of_non_zero_values = average_of_non_zero_values + pct_from_average_to_max * (max_value - average_of_non_zero_values)
    color_data = main.create_color_array(combined_data, max_value, average_of_non_zero_values, average_display_color)
    return color_data