import cv2

def read_video(video_path):
    cap=cv2.VideoCapture(video_path)
    frames=[]
    while True:
        ret, frame=cap.read() # Ret -> True (Video is still on), False -> Video Ended
        if not ret:
            break
        frames.append(frame)
        del frame
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    out=cv2.VideoWriter(output_video_path,fourcc,25.0,(1920,1080))
    for frame in output_video_frames:
        out.write(frame)
        del frame
    out.release()
