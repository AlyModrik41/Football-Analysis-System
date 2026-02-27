import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from main import read_video, Tracker, TeamAssigner, run_full_pipeline  # your main.py functions
from export_csv_stats import export_player_statistics,export_team_statistics

st.set_page_config(page_title="Football Analysis System", layout="wide")
st.title("Football Analysis System")

uploaded_file = st.file_uploader("Upload Football Video", type=["mp4", "avi"])

if uploaded_file:
    # Save uploaded video to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.info("Processing video, this may take a while...")
    
    # --- Step 1: Read frames ---
    frames = read_video(video_path)

    # --- Step 2: Run your pipeline ---
    # This function should return:
    # tracks (players, ball, referee)
    # stats (possession, passes, etc.)
    # annotated_frames (with radar, camera movement, overlays)
    tracks, frames = run_full_pipeline(frames)

    st.success("Analysis complete!")

    # --- Step 3: Display stats ---
    stats_df=export_team_statistics(tracks)
    stats_melted = stats_df.drop(columns=['total_passes']).melt(var_name='metric', value_name='value')
    st.bar_chart(stats_melted.set_index('metric'))

    # --- Step 4: Display annotated video ---
    st.subheader("Annotated Video")
    video_writer_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(video_writer_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for f in frames:
        out.write(f)
    out.release()

    # Show video
    video_file = open(video_writer_path, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)