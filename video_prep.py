import cv2
import yt_dlp
import os

def download_video(url, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': f'{output_path}.%(ext)s',

        # 🔑 Force MP4 output
        'merge_output_format': 'mp4',

        # 🔑 Ensure proper conversion if needed
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def prepare_video(
    input_path,
    output_path,
    start_time=0,
    end_time=None,
    cutoff_percent=0.2
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Convert time → frames
    start_frame = max(0, int(start_time * fps))
    end_frame = min(total_frames, int(end_time * fps) if end_time else total_frames)

    if start_frame >= end_frame:
        raise ValueError("Invalid time range.")

    # Seek directly to start frame (big performance win)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cutoff_height = int(height * cutoff_percent)

    # Process only needed frames
    frames_to_process = end_frame - start_frame

    for i in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break

        # Fast slice blackout (already optimal)
        frame[:cutoff_height, :] = 0

        out.write(frame)

        # Lightweight progress update (every 30 frames)
        if i % 30 == 0:
            print(f"Processing frame {i}/{frames_to_process}", end="\r")

    cap.release()
    out.release()