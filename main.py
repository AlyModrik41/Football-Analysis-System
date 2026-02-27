import sys
from utils import read_video
from trackers import Tracker
from utils import get_center_of_bbox, get_bbox_width
from team_assigner import TeamAssigner
import os
import pickle
import numpy as np
from stable_tracker import StableTracker
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator
from radar import Radar
from export_csv_stats import export_player_statistics, export_team_statistics
from stable_tracker import StableTracker

def main():
    video_frames = read_video(r'input_videos/D35bd9041_1 (27).mp4')

    tracker = Tracker(
        player_model_path='models/football-player-detection.pt',
        ball_model_path='models/football-ball-detection.pt',
        pitch_model_path='models/football-pitch-detection.pt'  # optional
    )

    stub_path = 'stubs/tracks_stubs.pkl'
    CHUNK_SIZE = 150
    all_tracks = {'players': [], 'referee': [], 'ball': [], 'pitch': []}

    if os.path.exists(stub_path):
        print('Loading existing tracks stub')
        with open(stub_path, 'rb') as f:
            all_tracks = pickle.load(f)
    else:
        for i in range(0, len(video_frames), CHUNK_SIZE):
            chunk_frames = video_frames[i:i + CHUNK_SIZE]
            is_last_chunk = (i + CHUNK_SIZE) >= len(video_frames)

            chunk_tracks = tracker.get_object_tracks(
                chunk_frames,
                read_from_stubs=is_last_chunk,
                stub_path=stub_path if is_last_chunk else None
            )

            all_tracks['players'] += chunk_tracks['players']
            all_tracks['referee'] += chunk_tracks['referee']
            all_tracks['ball'] += chunk_tracks['ball']
            all_tracks['pitch'] += chunk_tracks['pitch']

            del chunk_frames, chunk_tracks
    
    tracks = all_tracks
    
    stable_tracker=StableTracker(max_distance=80, max_lost=60)
    for frame_num in range(len(tracks['players'])):
        tracks['players'][frame_num]=stable_tracker.update(
            tracks['players'][frame_num]
        )
    all_ids=set()
    for frame in tracks['players']:
        for track_id in frame.keys():
            all_ids.add(track_id)
    print(f"Unique Player IDs after stable Tracker:")

    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'],max_gap=3,max_jump=250)

    tracker.add_position_to_tracks(tracks)
    # Add in main.py right after loading/generating tracks

    camera_movement_estimator=CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame=camera_movement_estimator.get_camera_movement(video_frames,
                                                                            read_from_stub=True,
                                                                            stub_path='stubs/camera_movement_stub.pkl')

    
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    view_transformer=ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    speed_and_distance_estimator=SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
   
    for frame_num, track in enumerate(tracks['players'][:10]):
        for track_id, track_info in track.items():
            pos = track_info.get('position_transformed', None)
            speed = track_info.get('speed', None)
            

    # Team assignment
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track['bbox'], player_id
            )
            # if player_id == 15:
            #     tracks['players'][frame_num][player_id]['team']=2
            #     tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[2]
            # else:
            tracks['players'][frame_num][player_id]['team']=team
            tracks['players'][frame_num][player_id]['team_color']=team_assigner.team_colors[team]
    # Ball assignment
    player_assigner = PlayerBallAssigner()
    team_ball_control=[]
    neutral_ball_control=[]
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox', None)
        if ball_bbox is None:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
            continue
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(0)
   
    team_ball_control=np.array(team_ball_control)

    radar=Radar()

    # Draw and save directly — no RAM spike
    tracker.draw_annotations(video_frames, tracks, 'output_videos/output_video_2.avi',
                             team_ball_control,camera_movement_per_frame,radar=radar)

    export_player_statistics(tracks)
    export_team_statistics(tracks)

    return tracks


if __name__ == '__main__':
    main()
