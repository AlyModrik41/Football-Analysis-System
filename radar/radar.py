import cv2
import numpy as np

class Radar:
    def __init__(self):

        # Minimap dimensions
        self.width=400
        self.height=240
        self.margin=20

        # Pitch Dimensions (Real-World)
        self.court_width=75
        self.court_length=80

        self.x_start=760
        self.y_start=840
    
    def transform_to_radar(self, position):

        x=position[0]/self.court_length
        y=position[1]/self.court_width

        radar_x=int(self.margin+x*(self.width-2*self.margin))
        radar_y=int(self.margin+y*(self.height-2*self.margin))

        return radar_x,radar_y

    def draw_pitch_markings(self, radar):
        h,w=radar.shape[:2]
        m=self.margin

        cv2.rectangle(radar,(m,m),(w-m,h-m),(255,255,255),2)

        cv2.line(radar,(w//2,m),(w//2,h-m),(255,255,255),1)

        cv2.circle(radar,(w//2,h//2),25,(255,255,255),1)
        cv2.circle(radar,(w//2,h//2),2,(255,255,255),-1)

        return radar
    
    def draw_radar(self, frame, tracks, frame_num, team_colors):
        radar=np.zeros((self.height,self.width,3),dtype=np.uint8)
        radar[:]=(40,100,40)

        radar=self.draw_pitch_markings(radar)

        for track_id, player in tracks['players'][frame_num].items():
            position=player.get('position_transformed',None)
            if position is None:
                continue

            radar_pos=self.transform_to_radar(position)
            alpha=0.6

            team=player.get('team',0)
            color=player.get('team_color',(200,200,200))
            color=tuple(map(int, color))

            overlay=frame.copy()
            cv2.circle(radar, radar_pos, 5, color, -1)
            # cv2.circle(radar, radar_pos, 5, (0,0,0), 1)
            cv2.addWeighted(overlay,alpha,frame,0.4,0,frame)

            if player.get('has_ball'):
                cv2.circle(radar, radar_pos, 8, (255,255,0),2)
        
        ball=tracks['ball'][frame_num].get(1,{})
        ball_pos=ball.get('position_transformed',None)
        if ball_pos is not None:
            radar_pos=self.transform_to_radar(ball_pos)
            cv2.circle(radar, radar_pos, 4, (0,255,0),-1)
            cv2.circle(radar, radar_pos, 4, (0,0,0), 1)
        
        for _,referee in tracks['referee'][frame_num].items():
            position=referee.get('position_transformed',None)
            if position is None:
                continue
            radar_pos=self.transform_to_radar(position)
            cv2.circle(radar, radar_pos, 5, (0,255,255),-1)

        overlay=frame.copy()
        cv2.rectangle(overlay,
                      (self.x_start-5, self.y_start-5),
                      (self.x_start+self.width+5,self.y_start+self.height+5),
                      (0,0,0),-1)
        cv2.addWeighted(overlay,0.6,frame,0.4,0,frame)

        frame[self.y_start:self.y_start+self.height,
              self.x_start:self.x_start+self.width]=radar
        
        return frame
    
            