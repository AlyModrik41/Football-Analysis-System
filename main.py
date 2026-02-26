<<<<<<< HEAD
from utils import read_video
from trackers import Tracker
from utils import get_center_of_bbox, get_bbox_width
from team_assigner import TeamAssigner
import os
import pickle
import numpy as np
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator

def main():
    video_frames = read_video(r'input_videos\A1606b0e6_0 (10).mp4')

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
    
    tracker.add_position_to_tracks(tracks)

    camera_movement_estimator=CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame=camera_movement_estimator.get_camera_movement(video_frames,
                                                                            read_from_stub=True,
                                                                            stub_path='stubs/camera_movement_stub.pkl')

    
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

    tracks = all_tracks
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'],max_gap=3,max_jump=250)

    # Team assignment
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track['bbox'], player_id
            )
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

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

    # Draw and save directly — no RAM spike
    tracker.draw_annotations(video_frames, tracks, 'output_videos/output_video.avi',
                             team_ball_control,camera_movement_per_frame)



if __name__ == '__main__':
=======
from utils import read_video,save_video
from trackers import Tracker
from utils import get_center_of_bbox, get_bbox_width
from team_assigner import TeamAssigner
import cv2
import torch
import os
import pickle
from sklearn.cluster import KMeans
import numpy as np
from player_ball_assigner import PlayerBallAssigner

def main():
    video_frames=read_video(r'input_videos\A1606b0e6_0 (10).mp4')

    tracker=Tracker('models/best (1).pt')

    stub_path = 'stubs/tracks_stubs.pkl'

    CHUNK_SIZE = 150
    all_tracks = {
    'players': [],
    'referee': [],
    'ball': []
    }

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

            del chunk_frames, chunk_tracks

    tracks = all_tracks

    # Interpolate Ball Positions 
    tracks['ball']=tracker.interpolate_ball_positions(tracks['ball'])

    #save cropped video of a player

    # for track_id,player in tracks['players'][0].items():
    #     bbox=player['bbox']
    #     frame=video_frames[0]

    #     # crop bbox from frame
    #     cropped_image=frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

    #     # save the cropped image

    #     cv2.imwrite(f"output_videos/cropped_image.jpg",cropped_image)
    #     break
    
    # Assign Player Team with temporal consistency
    # Assign Player Team with temporal consistency
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    player_assigner=PlayerBallAssigner()
    team_ball_control=[]
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox=tracks['ball'][frame_num].get(1,{}).get('bbox',None)
        if ball_bbox is None:
            continue
        assigned_player=player_assigner.assign_ball_to_player(player_track,ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball']=True

    # Draw Output
    output_video_frames=tracker.draw_annotations(video_frames,tracks)

    save_video(output_video_frames,'output_videos/output_video_framies.avi')


if __name__=='__main__':
>>>>>>> f84cac4dd0eb7e1285a7b195b49bd14ef20a4c10
    main()