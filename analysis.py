import cv2
import numpy as np

def get_total_yellow_pixels_from_video(vid_path, start_time=0, end_time=None):
    cap = cv2.VideoCapture(vid_path)

    fps = cap.get(cv2.CAP_PROP_FPS)  # frames per second

    if end_time is None:
        end_time = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    print(f"Video FPS: {fps}")
    print(f"Start Frame: {start_frame}")
    print(f"End Frame: {end_frame}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_frame_totals = np.zeros((frame_height, frame_width), dtype=np.uint32)

    # go frame by frame
    while frame_count < end_frame:
        print(f"Processing frame {frame_count}/{end_frame}", end="\r")
        ret, frame = cap.read()
        if not ret:
            break

        frame_mask = analyze_frame(frame)
        video_frame_totals += frame_mask

        frame_count += 1
    cap.release()
    return video_frame_totals

def analyze_frame(frame):
    # OpenCV frames are BGR. Match the original RGB thresholds without converting.
    return (
        (frame[:, :, 2] > 200) &
        (frame[:, :, 1] > 200) &
        (frame[:, :, 0] < 100)
    ).astype(np.uint32)

def analyze_video_progression(vid_path, start_time=0, end_time=None):
    cap = cv2.VideoCapture(vid_path)

    fps = cap.get(cv2.CAP_PROP_FPS)  # frames per second

    if end_time is None:
        end_time = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    print(f"Video FPS: {fps}")
    print(f"Start Frame: {start_frame}")
    print(f"End Frame: {end_frame}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame

    total_frames = end_frame - start_frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_counts = np.zeros((total_frames, frame_height, frame_width), dtype=np.uint32)

    # go frame by frame
    while frame_count < end_frame:
        print(f"Processing frame {frame_count}/{end_frame}", end="\r")
        ret, frame = cap.read()
        if not ret:
            break

        # get the frame's data as a 2D array where its 1 if the frame is mostly yellow and 0 otherwise
        frame_mask = analyze_frame(frame)

        if (frame_count == start_frame):
            frame_counts[0] = frame_mask
        else:
            frame_counts[frame_count - start_frame] = frame_counts[frame_count - start_frame - 1] + frame_mask
        
        frame_count += 1
    print(f"Total Frames:" + str(len(frame_counts)))
    cap.release()
    return frame_counts

def analyze_video_total(vid_path, start_time=0, end_time=None):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if end_time is None:
        end_time = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_map = np.zeros((frame_height, frame_width), dtype=np.uint32)

    frame_count = start_frame

    while frame_count < end_frame:
        print(f"Processing frame {frame_count}/{end_frame}", end="\r")

        ret, frame = cap.read()
        if not ret:
            break

        total_map += analyze_frame(frame)
        frame_count += 1

    cap.release()
    print()  # newline after progress
    return total_map
