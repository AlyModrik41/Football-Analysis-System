import pandas as pd
import os

def export_player_statistics(tracks,output_dir=r'C:\Users\Ali\football project\Statistics'):
    os.makedirs(output_dir,exist_ok=True)

    player_stats={}

    for frame_players in tracks['players']:
        for player_id, data in frame_players.items():
            
            if player_id not in player_stats:
                player_stats[player_id]={
                    'team':data.get('team',0),
                    'total_distance':0,
                    'speed_sum':0,
                    'speed_count':0,
                    'ball_touches':0
                }
            
            if 'distance' in data:
                player_stats[player_id]['distance_covered']+=data['distance']

            if 'speed' in data:
                player_stats[player_id]['speed_sum']+=data['speed']
                player_stats[player_id]['speed_sum']+=1
            if data.get('has_ball',False):
                player_stats[player_id]['ball_touches']+=1

    rows=[]
    for player_id, stats in player_stats.items():
        avg_speed=(
            stats['speed_sum']/stats['speed_count']
            if stats['speed_count']>0 else 0
        )

        rows.append({
            'player_id':player_id,
            'team':stats['team'],
            'total_distance_covered (m)':round(stats['distance_covered'],2),
            'avg_speed_km_per_hr': round(avg_speed,2),
            'ball_touches':stats['ball_touches']
        })

    df=pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir,'player_stats.csv'),index=False)
    print('Player Statistics Exported Successfully !!')

def export_team_statistics(tracks,output_dir=r'C:\Users\Ali\football project\Statistics'):
    os.makedirs(output_dir,exist_ok=True)

    team_stats={
        0:{'possession_frames':0},
        1:{'possession_frames':0},
        2:{'possession_frames':0}
    }

    total_frames=len(tracks['players'])

    for frame_players in tracks['players']:
        possession_assigned=False
        for data in frame_players.values():
            if data.get('has_ball',False):
                team_id=data.get('team',0)
                team_stats[team_id]+=1
                possession_assigned=True
                break

        if not possession_assigned:
            team_stats[0]+=1

    rows=[]
    for team_id,frames in team_stats.items():
        possession_percent=(frames/total_frames)*100 if total_frames>0 else 0

        team_id= (
            'Team 1' if team_id == 1 else
            'Team 2' if team_id == 2 else
            'Neutral'
        )
        
        rows.append({
            'team':team_id,
            'possession_percent':round(possession_percent,2)
        })

    df=pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir,'team_stats.csv'),index=False)

    print('Team Statistics Exported Successfully !!')
    return df


