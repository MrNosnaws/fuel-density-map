import os

import cv2
import numpy as np
import analysis
import multi_match as mm
import numpy as np
from PIL import Image

vid_path = "videos/citrus_bread_wr_blackboxed.mp4"
start_time = 0  # in seconds
end_time = None   # in seconds
pct_from_average_to_max = 0.5 # how much of the way from the average to the max value should the average color be, as a percentage
average_display_color = (255, 0, 255) # magenta, easy to see

def raw_data_analysis():
    # Example usage
    output_dir = "output"

    
    
    print("Analyzing video...")
    vid_totals = analysis.get_total_yellow_pixels_from_video(vid_path, start_time, end_time)
    # Save the raw results to a file

    print("Saving results...")
    raw_data_file_name = vid_path.split("/")[-1].split(".")[0] + "_raw_data.txt"
    with open(os.path.join(output_dir, raw_data_file_name), "w") as f:
        for row in vid_totals:
            f.write(" ".join(str(int(x)) for x in row) + "\n")
    print("Results saved.")

def process_raw_data():
    data_path = os.path.join("output", vid_path.split("/")[-1].split(".")[0] + "_raw_data.txt")
    processed_image_path = "images"

    print("Processing raw data...")
    # Read the raw data from the file
    raw_data = np.loadtxt(data_path, dtype=np.int32)
    
    processed_folder_name = vid_path.split("/")[-1].split(".")[0] + "_processed"
    processed_folder_path = os.path.join(processed_image_path, processed_folder_name)
    os.makedirs(processed_folder_path, exist_ok=True)

    max_value = int(raw_data.max())
    non_zero_values = raw_data[raw_data > 0]
    actual_average = float(non_zero_values.mean()) if non_zero_values.size else 0.0
    average_of_non_zero_values = actual_average + pct_from_average_to_max * (max_value - actual_average)
    print(f"Max value: {max_value}")
    print(f"Actual average of non-zero values: {actual_average}")
    print(f"Used Average of non-zero values: {average_of_non_zero_values}")

    # turn the raw data into an array of colors where the color is determined by the value of the raw data at that point, 
    # with 0 being black, 
    # max_value being white,
    #  average_of_non_zero_values being average_display_color
    # and values between 0 and average_of_non_zero_values being a gradient from black to average_display_color,
    # and values between average_of_non_zero_values and max_value being a gradient from average_display_color to white

    color_array = create_color_array(raw_data, max_value, average_of_non_zero_values, average_display_color)

    print("Creating image...")
    img = Image.fromarray(color_array, mode="RGB")
    img.save(os.path.join(processed_folder_path, "processed_image.png"))
    print("Image saved.")

    print("Creating transparent image...")
    alpha_channel = np.where(np.any(color_array != 0, axis=2), 128, 0).astype(np.uint8)
    transparent_image = np.dstack((color_array, alpha_channel))
    img = Image.fromarray(transparent_image, mode="RGBA")
    img.save(os.path.join(processed_folder_path, "processed_image_transparent.png"))
    print("Transparent image saved.")

# def get_color(value, max_value, average_of_non_zero_values, average_display_color):
#     if value == 0:
#         return (0, 0, 0)  # black

#     if max_value == 0:
#         return average_display_color

#     if value >= max_value:
#         return (255, 255, 255)  # white

#     # Avoid division by zero
#     if average_of_non_zero_values == 0:
#         return average_display_color

#     if value <= average_of_non_zero_values:
#         # gradient from black to average_display_color
#         ratio = value / average_of_non_zero_values
#         return tuple(int(ratio * average_display_color[i]) for i in range(3))
#     else:
#         # gradient from average_display_color to white
#         denom = (max_value - average_of_non_zero_values)
#         if denom == 0:
#             return average_display_color

#         ratio = (value - average_of_non_zero_values) / denom
#         return tuple(
#             int(average_display_color[i] + ratio * (255 - average_display_color[i]))
#             for i in range(3)
#         )       

def create_color_array(raw_data, max_value, average_of_non_zero_values, average_display_color):
    color_array = np.zeros((*raw_data.shape, 3), dtype=np.uint8)

    if max_value <= 0:
        return color_array

    average_color = np.array(average_display_color, dtype=np.float32)
    non_zero_mask = raw_data > 0

    if average_of_non_zero_values > 0:
        lower_mask = non_zero_mask & (raw_data <= average_of_non_zero_values)
        if np.any(lower_mask):
            lower_ratio = (raw_data[lower_mask] / average_of_non_zero_values).astype(np.float32)
            color_array[lower_mask] = np.clip(lower_ratio[:, None] * average_color, 0, 255).astype(np.uint8)

    upper_mask = non_zero_mask & (raw_data > average_of_non_zero_values)
    upper_range = max_value - average_of_non_zero_values
    if np.any(upper_mask):
        if upper_range <= 0:
            color_array[upper_mask] = 255
        else:
            upper_ratio = ((raw_data[upper_mask] - average_of_non_zero_values) / upper_range).astype(np.float32)
            color_array[upper_mask] = np.clip(
                average_color + upper_ratio[:, None] * (255 - average_color),
                0,
                255,
            ).astype(np.uint8)

    color_array[raw_data >= max_value] = 255
    # print(f"Color array created with shape: {color_array.shape} and pixel range: {color_array.min()} to {color_array.max()}")
    return color_array
 
def process_into_video_progression():
    frame_counts = analysis.analyze_video_progression(vid_path, start_time, end_time)
    output_dir = "videos/progression/"
    output_name = os.path.join(output_dir, vid_path.split("/")[-1].split(".")[0] + "_processed_video.mp4")

    if (not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    print("Processing video progression...")
    # convert the frame counts into color arrays
    frame_color_arrays = []
    for frame_count in frame_counts:
        print(f"Turning Frame into Images {len(frame_color_arrays)}/{len(frame_counts)}", end="\r")
        max_value = int(frame_count.max())
        non_zero_values = frame_count[frame_count > 0]
        actual_average = float(non_zero_values.mean()) if non_zero_values.size else 0.0
        average_of_non_zero_values = actual_average + pct_from_average_to_max * (max_value - actual_average)

        color_array = create_color_array(frame_count, max_value, average_of_non_zero_values, average_display_color)
        frame_color_arrays.append(color_array)
    print("Frame color arrays created. Average pixel value: " + str(np.mean(frame_color_arrays)) + ". Range: " + str(np.min(frame_color_arrays)) + " to " + str(np.max(frame_color_arrays)))
    # turn the color arrays and turn them into a video
    video_array = np.array(frame_color_arrays)
    array_to_video(video_array, output_name, fps=30)

    # # get the frames now with the black parts transparent and everything else half transparent
    # transparent_frame_color_arrays = []
    # for color_array in frame_color_arrays:
    #     transparent_color_array = []
    #     for row in color_array:
    #         transparent_color_row = []
    #         for color in row:
    #             if color == (0, 0, 0):
    #                 transparent_color_row.append((0, 0, 0, 0)) # transparent
    #             else:
    #                 transparent_color_row.append(color + (128,)) # half transparent
    #         transparent_color_array.append(transparent_color_row)
    #     transparent_frame_color_arrays.append(transparent_color_array)

    # transparent_video_array = np.array(transparent_frame_color_arrays)
    # array_to_video(transparent_video_array, output_name.replace(".mp4", "_transparent.mp4"), fps=30)

def array_to_video(video_array, output_path, fps=30):
    video_array = np.ascontiguousarray(video_array)
    print("VIDEO ARRAY SHAPE:", video_array.shape)

    num_frames, height, width, channels = video_array.shape
    
    if num_frames == 0:
        raise ValueError("video_array has 0 frames — nothing to write")

    assert channels == 3, "Must have 3 color channels (RGB)"
    
    if video_array.dtype != np.uint8:
        video_array = video_array.astype(np.uint8)

    print("Pixel range:", video_array.min(), video_array.max())

    # Try H.264 first
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print("avc1 failed, falling back to XVID...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = output_path.replace(".mp4", ".avi")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            raise RuntimeError("All codecs failed")

    for i in range(num_frames):
        print(f"Writing frame {i+1}/{num_frames}", end="\r")
        frame_rgb = video_array[i]

        # Sanity check
        if frame_rgb.shape != (height, width, 3):
            raise ValueError(f"Frame shape mismatch: {frame_rgb.shape}")

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # print(f"Frame {i+1} pixel average: {frame_rgb.mean():.2f}", end="\r")
        out.write(frame_bgr)

    out.release()
    print("Saved to:", output_path)

def process_multi_match():
    name = "ns_finals"
    video_paths = [
        "videos/north_star_finals/finals_1.mp4",
        "videos/north_star_finals/finals_2.mp4"
    ]
    start_times = [0, 0]
    end_times = [None, None]

    combined_color_data = mm.get_multi_match_color_data(video_paths, start_times, end_times, average_display_color, pct_from_average_to_max=0.25)
    output_path = os.path.join("images/" + name, "_multi_match_image.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = Image.fromarray(combined_color_data, mode="RGB")
    img.save(output_path)
    print("Multi-match image saved to:", output_path)

if __name__ == "__main__":
    # raw_data_analysis()
    # process_raw_data()
    # process_into_video_progression()
    process_multi_match()