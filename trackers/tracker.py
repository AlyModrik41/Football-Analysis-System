from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import numpy as np
import pandas as pd
sys.path.append('../')
from utils import get_bbox_width, get_center_of_bbox,get_foot_position

class Tracker:
    def __init__(self, player_model_path, ball_model_path, pitch_model_path=None):
        self.player_model = YOLO(player_model_path, task='detect')
        self.ball_model = YOLO(ball_model_path, task='detect')
        self.player_model.to('cuda')
        self.ball_model.to('cuda')
        
        # Pitch model is optional
        if pitch_model_path and os.path.exists(pitch_model_path):
            self.pitch_model = YOLO(pitch_model_path, task='detect')
            self.pitch_model.to('cuda')
        else:
            self.pitch_model = None
            
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num,track in enumerate(object_tracks):
                for track_id,track_info in track.items():
                    bbox=track_info['bbox']
                    if object == 'ball':
                        position=get_center_of_bbox(bbox)
                    else :
                        position=get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position']=position

    def interpolate_ball_positions(self, ball_positions, max_gap=3, max_jump=250):
        raw_positions = [x.get(1, {}).get('bbox', [np.nan]*4) for x in ball_positions]
        
        cleaned = []
        last_valid_center = None
        consecutive_rejected = 0
        
        for bbox in raw_positions:
            if any(np.isnan(v) for v in bbox):
                cleaned.append([np.nan]*4)
                consecutive_rejected = 0
                continue
                
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            
            if last_valid_center is not None:
                dist = np.sqrt((cx - last_valid_center[0])**2 + (cy - last_valid_center[1])**2)
                if dist > max_jump:
                    consecutive_rejected += 1
                    # After 5 consecutive rejections, reset anchor — ball genuinely moved far
                    if consecutive_rejected >= 5:
                        last_valid_center = (cx, cy)
                        cleaned.append(bbox)
                        consecutive_rejected = 0
                    else:
                        cleaned.append([np.nan]*4)
                    continue
            
            consecutive_rejected = 0
            last_valid_center = (cx, cy)
            cleaned.append(bbox)
        
        df = pd.DataFrame(cleaned, columns=['x1','y1','x2','y2'])
        df_interp = df.interpolate(limit=max_gap, limit_direction='both')
        df_interp = df_interp.bfill(limit=max_gap).ffill(limit=max_gap)
        df_smooth = df_interp.rolling(window=2, center=True, min_periods=1).mean()

        return [
            {1: {'bbox': x.tolist()}} if not any(np.isnan(v) for v in x) else {}
            for x in df_smooth.to_numpy()
        ]

    def detect_frames(self, frames):
        player_detections = []
        ball_detections = []
        pitch_detections = []

        for i in range(0, len(frames), 1):
            player_detections += self.player_model.predict(
                frames[i:i+1], conf=0.3, imgsz=1920, device=0, verbose=False
            )
            ball_detections += self.ball_model.predict(
                frames[i:i+1], conf=0.3, imgsz=1280, device=0, verbose=False
            )
            if self.pitch_model:
                pitch_detections += self.pitch_model.predict(
                    frames[i:i+1], conf=0.5, imgsz=640, device=0, verbose=False
                )

        return player_detections, ball_detections, pitch_detections

    def get_object_tracks(self, frames, read_from_stubs=False, stub_path=None):
        if read_from_stubs and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        player_detections, ball_detections, pitch_detections = self.detect_frames(frames)

        tracks = {'players': [], 'referee': [], 'ball': [], 'pitch': []}

        for frame_num, detection in enumerate(player_detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper to player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['referee'].append({})
            tracks['ball'].append({})
            tracks['pitch'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv.get('player', -1):
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}
                if cls_id == cls_names_inv.get('referee', -1):
                    tracks['referee'][frame_num][track_id] = {'bbox': bbox}

            # Ball detection
            ball_detection = ball_detections[frame_num]
            ball_supervision = sv.Detections.from_ultralytics(ball_detection)
            ball_cls_names = ball_detection.names
            ball_cls_names_inv = {v: k for k, v in ball_cls_names.items()}

            best_ball = None
            best_conf = -1

            for frame_detection in ball_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                conf = frame_detection[2]

                if cls_id == ball_cls_names_inv.get('ball', -1):
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    area = width * height
                    aspect_ratio = width / (height + 1e-6)
                    # print(f"Ball: area={width*height:.0f}, conf={conf:.2f}")

                    if area < 150 or area > 900:
                        continue
                    if not (0.4 < aspect_ratio < 1.6):
                        continue
                    if conf<0.5:
                        continue
                    if conf > best_conf:
                        best_conf = conf
                        best_ball = bbox

            if best_ball is not None:
                tracks['ball'][frame_num][1] = {'bbox': best_ball}

            # Pitch detection
            if pitch_detections and frame_num < len(pitch_detections):
                pitch_det = pitch_detections[frame_num]
                pitch_supervision = sv.Detections.from_ultralytics(pitch_det)
                for pd_det in pitch_supervision:
                    bbox = pd_det[0].tolist()
                    tracks['pitch'][frame_num][len(tracks['pitch'][frame_num])] = {'bbox': bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, center=(x_center, y2),
                    axes=(int(width), int(0.35*width)),
                    angle=0, startAngle=-45, endAngle=235,
                    color=color, thickness=2, lineType=cv2.LINE_4)

        rectangle_width = 30
        rectangle_height = 15
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)), color, -1)
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            cv2.putText(frame, f"{track_id}",
                        (int(x1_text-9), int(y1_rect+9)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 2)
        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay=frame.copy()
        cv2.rectangle(overlay,(1350,820),(1900,990),(255,255,255),-1,)
        alpha=0.4
        cv2.addWeighted(overlay, alpha, frame, 1-alpha,0,frame)

        team_ball_control_till_frame=team_ball_control[:frame_num+1]

        team_1_num_frames=team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        neutral_num_frames=team_ball_control_till_frame[team_ball_control_till_frame==0].shape[0]
        team_2_num_frames=team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        total=team_1_num_frames+team_2_num_frames+neutral_num_frames
        total=max(total,1)

        team_1=team_1_num_frames/total
        team_2=team_2_num_frames/total
        neutral=neutral_num_frames/total


        cv2.putText(frame,f"Team 1 Ball Control {team_1*100:.2f}%",(1400,850),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        cv2.putText(frame,f"Team 2 Ball Control {team_2*100:.2f}%",(1400,900),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        cv2.putText(frame,f"Neutral Ball Control {neutral*100:.2f}%",(1400,950),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

        return frame


    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_points = np.array([[x, y], [x-7, y-11], [x+7, y-11]])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    def draw_annotations(self, video_frames, tracks, output_path,team_ball_control,camera_movement_per_frame):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = video_frames[0].shape[:2]
        out = cv2.VideoWriter(output_path, fourcc, 25.0, (width, height))

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referee_dict = tracks['referee'][frame_num]

            for track_id, player in player_dict.items():
                color = player.get('team_color', (0, 0, 255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)
                if player.get('has_ball') == True:
                    frame = self.draw_triangle(frame, player['bbox'], (255, 0, 0))

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255))

            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0, 255, 0))
            
            frame=self.draw_team_ball_control(frame,frame_num,team_ball_control)


            if camera_movement_per_frame is not None:
                overlay=frame.copy()
                cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
                alpha=0.6
                cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)
                x_movement,y_movement=camera_movement_per_frame[frame_num]
                frame=cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
                frame=cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            out.write(frame)
            del frame

        out.release()