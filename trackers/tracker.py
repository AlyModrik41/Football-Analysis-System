from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import numpy as np
import pandas as pd
sys.path.append('../')
from utils import get_bbox_width, get_center_of_bbox

class Tracker:
    def __init__(self,model_path):
        self.model=YOLO(model_path,task='detect')
        self.model.to('cuda')
        self.tracker=sv.ByteTrack()

    
    def interpolate_ball_positions(self, ball_positions, max_gap=8):
        ball_positions = [x.get(1, {}).get('bbox', [np.nan]*4) for x in ball_positions]
        df = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        # Only fill genuine short gaps, not long absences
        df_interp = df.interpolate(limit=max_gap, limit_direction='both')
        df_interp = df_interp.bfill(limit=max_gap).ffill(limit=max_gap)

        # Smooth with rolling window to reduce jitter
        df_smooth = df_interp.rolling(window=3, center=True, min_periods=1).mean()

        ball_positions = [
            {1: {'bbox': x.tolist()}} if not any(np.isnan(v) for v in x) else {}
            for x in df_smooth.to_numpy()
        ]
        return ball_positions


    
    def detect_frames(self,frames):
        batch_size=1
        detections=[]
        for i in range(0,len(frames),batch_size):
            detections_batch=self.model.predict(frames[i:i+batch_size],conf=0.25,imgsz=1920,device=0)
            detections+=detections_batch
            if i==len(frames)-1:
                print('Model Classes:',detections_batch[0].names)
            del detections_batch

        return detections

    def get_object_tracks(self, frames,read_from_stubs=False,stub_path=None):


        if read_from_stubs and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks=pickle.load(f)
            return tracks

        detections=self.detect_frames(frames)

        tracks={
            'players':[],
            'referee':[],
            'ball':[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names=detection.names
            cls_names_inv={v:k for k,v in cls_names.items()}

            # Convert to Supervision Detections
            detection_supervision=sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object

            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id]=='goalkeeper':
                    detection_supervision.class_id[object_ind]=cls_names_inv['player']


            detection_with_tracks=self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['referee'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox=frame_detection[0].tolist()
                cls_id=frame_detection[3]
                track_id=frame_detection[4]

                if cls_id== cls_names_inv['player']:
                    tracks['players'][frame_num][track_id]={'bbox':bbox}
                if cls_id==cls_names_inv['referee']:
                    tracks['referee'][frame_num][track_id]={'bbox':bbox}

            best_ball=None
            best_conf=-1
            for frame_detection in detection_supervision:
                bbox=frame_detection[0].tolist()
                cls_id=frame_detection[3]
                conf=frame_detection[2]

                if cls_id== cls_names_inv.get('ball',-1):
                    width=bbox[2]-bbox[0]
                    height=bbox[3]-bbox[1]
                    area=width*height
                    aspect_ratio=width/(height+ 1e-6)
                    print(f"Ball candidate: area={width*height:.0f}, conf={conf:.2f}, ratio={width/height:.2f}")

                    if area<70 or area > 200:
                        continue
                    if not (0.5<aspect_ratio<1.5):
                        continue
                    if conf>best_conf:
                        best_conf=conf
                        best_ball=bbox
            ball_count = sum(1 for d in detection_supervision if cls_names[d[3]] == 'ball')
            if best_ball is not None:
                tracks['ball'][frame_num][1]={'bbox':best_ball}
            if frame_num % 30 == 0:
                print(f"Frame {frame_num}: {ball_count} ball(s) detected")

        # For Saving Tracks

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2=int(bbox[3])
        x_center, _=get_center_of_bbox(bbox)
        width=get_bbox_width(bbox)
        
        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width),int(0.35*width)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        rectangle_width=30
        rectangle_height=15
        x1_rect=x_center - rectangle_width//2
        x2_rect=x_center + rectangle_width//2
        y1_rect= (y2- rectangle_height//2)+ 15
        y2_rect=(y2 + rectangle_height//2)+15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect)),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          -1)
            x1_text=x1_rect+12
            if track_id>99:
                x1_text-=10
            
            cv2.putText(frame,
                        f"{track_id}",
                        (int(x1_text-9),int(y1_rect+9)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.43,
                        (0,0,0),
                        2)

        
        return frame

    def draw_triangle(self,frame,bbox,color):
        y=int(bbox[1])
        x,_=get_center_of_bbox(bbox)
        
        triangle_points=np.array([
            [x,y],
            [x-7,y-11],
            [x+7,y-11]
        ])
        cv2.drawContours(frame,
                         [triangle_points],
                         0,color,cv2.FILLED
                         )
        cv2.drawContours(frame,[triangle_points],
                         0,(0,0,0),2)
        return frame


    def draw_annotations(self,video_frames,tracks):
        output_video_frames=[]
        for frame_num, frame in enumerate(video_frames):
            frame=frame.copy()

            player_dict=tracks['players'][frame_num]
            ball_dict=tracks['ball'][frame_num]
            referee_dict=tracks['referee'][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color=player.get('team_color',(0,0,255))
                frame=self.draw_ellipse(frame, player['bbox'],color,track_id)

                if player.get('has_ball')==True:
                    frame=self.draw_triangle(frame,player['bbox'],(255,0,0))

            # Draw Referee
            for _,referee in referee_dict.items():
                frame=self.draw_ellipse(frame,referee['bbox'],(0,255,255))
            
            # Draw Ball
            for _,ball in ball_dict.items():
                frame=self.draw_triangle(frame,ball['bbox'],(0,255,0))

            
            output_video_frames.append(frame)
            del frame
            
        return output_video_frames




            

