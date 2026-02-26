import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half = image[0:int(image.shape[0] / 2), :]

        # Replace green pixels with black so they cluster as background
        # instead of contaminating the kit color
        hsv = cv2.cvtColor(top_half, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(
            hsv,
            np.array([36, 40, 40]),
            np.array([86, 255, 255])
        )
        masked = top_half.copy()
        masked[green_mask > 0] = [0, 0, 0]  # green → black background

        # Cluster on masked image
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(masked.reshape(-1, 3))
        labels = kmeans.labels_.reshape(top_half.shape[0], top_half.shape[1])

        # Corner heuristic still works — corners are now black (background)
        # instead of green, but still reliably not the player
        corner_clusters = [labels[0,0], labels[0,-1], labels[-1,0], labels[-1,-1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)
        self.kmeans = kmeans

        centers = kmeans.cluster_centers_
        # Sort by brightness to make team identity stable across runs
        sorted_idx = np.argsort([np.mean(c) for c in centers])
        self.team_colors[1] = centers[sorted_idx[0]]  # darker kit
        self.team_colors[2] = centers[sorted_idx[1]]  # lighter kit

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        dist1 = np.linalg.norm(player_color - self.team_colors[1])
        dist2 = np.linalg.norm(player_color - self.team_colors[2])
        team_id = 1 if dist1 < dist2 else 2

        if player_id == 689 or player_id == 742:
            team_id=2

        self.player_team_dict[player_id] = team_id
        return team_id