from utils import measure_distance

class SpeedAndDistanceEstimator:

    def __init__(self):
        self.frame_window = 10
        self.frame_rate = 25
        # Pixels per meter at center of frame — calibrate this
        self.pixels_per_meter = 10  # tune this value

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object in ['ball', 'referee', 'pitch']:
                continue

            number_of_frames = len(object_tracks)

            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id in object_tracks[frame_num]:
                    if track_id not in object_tracks[last_frame]:
                        continue

                    # Try position_transformed first, fall back to position_adjusted
                    start_pos = object_tracks[frame_num][track_id].get(
                        'position_transformed',
                        object_tracks[frame_num][track_id].get('position_adjusted', None)
                    )
                    end_pos = object_tracks[last_frame][track_id].get(
                        'position_transformed',
                        object_tracks[last_frame][track_id].get('position_adjusted', None)
                    )

                    if start_pos is None or end_pos is None:
                        continue

                    distance = measure_distance(start_pos, end_pos)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate

                    if time_elapsed == 0:
                        continue

                    # If using pixel positions, convert to meters
                    using_transformed = object_tracks[frame_num][track_id].get(
                        'position_transformed', None
                    ) is not None
                    
                    if not using_transformed:
                        distance = distance / self.pixels_per_meter

                    speed_kmh = (distance / time_elapsed) * 3.6

                    # Clamp unrealistic speeds
                    if speed_kmh > 42:
                        continue

                    if object not in total_distance:
                        total_distance[object] = {}
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0

                    total_distance[object][track_id] += distance

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_kmh
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]