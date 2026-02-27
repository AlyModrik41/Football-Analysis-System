import numpy as np

class StableTracker:
    def __init__(self, max_distance=80, max_lost=25):
        self.stable_tracks = {}
        self.next_stable_id = 1
        self.max_distance = max_distance
        self.max_lost = max_lost
        self.frame_num = 0

    def update(self, player_tracks_frame):
        self.frame_num += 1
        new_frame = {}

        for orig_id, track_info in player_tracks_frame.items():
            bbox = track_info['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            pos = (cx, cy)

            best_stable_id = None
            best_dist = self.max_distance

            for stable_id, stable_info in self.stable_tracks.items():
                # Skip tracks lost for too long
                if self.frame_num - stable_info['last_seen'] > self.max_lost:
                    continue
                dist = np.sqrt(
                    (pos[0] - stable_info['last_pos'][0])**2 +
                    (pos[1] - stable_info['last_pos'][1])**2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_stable_id = stable_id

            if best_stable_id is None:
                best_stable_id = self.next_stable_id
                self.next_stable_id += 1

            self.stable_tracks[best_stable_id] = {
                'last_pos': pos,
                'last_seen': self.frame_num
            }

            new_frame[best_stable_id] = track_info

        return new_frame