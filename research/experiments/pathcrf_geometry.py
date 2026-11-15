"""Geometry-only 5 Hz input adapter for the public PathCRF network.

Physical features follow the official utility's 25 Hz Savitzky-Golay filters.
No ball, events, possession/status fields or reference boundaries are read.
"""
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
from player_node_data import metrica_nodes


def prepare_geometry(game):
    if game not in ('Sample_Game_1','Sample_Game_2'):
        raise ValueError('Initial geometry adapter supports Metrica CSV matches only')
    path = Path('evaluation/reports/feature_cache')/('pathcrf_geometry_'+game+'.npz')
    if path.exists():
        with np.load(path,allow_pickle=False) as data:
            return {key:data[key].copy() for key in data.files}
    xy, teams, period = metrica_nodes(game,return_raw=True)
    physical = np.full(xy.shape[:-1]+(6,),np.nan,np.float32)
    physical[...,:2] = xy
    for half in np.unique(period):
        for player in range(len(teams)):
            idx = np.flatnonzero((period == half) & np.isfinite(xy[:,player]).all(axis=1))
            if len(idx) < 17:
                continue
            coords = xy[idx,player].copy()
            speed = np.linalg.norm(np.diff(coords,axis=0),axis=1)*25.
            outlier = np.r_[False,speed > 12.]
            if outlier.any() and (~outlier).any():
                for axis in range(2):
                    coords[:,axis] = np.interp(np.arange(len(idx)),np.flatnonzero(~outlier),coords[~outlier,axis])
            velocity = savgol_filter(np.diff(coords,axis=0)*25.,15,2,axis=0)
            acceleration = savgol_filter(np.diff(velocity,axis=0)*25.,9,2,axis=0)
            physical[idx,player,:2] = coords
            physical[idx[1:],player,2:4] = velocity
            physical[idx[1:],player,4] = np.linalg.norm(velocity,axis=1)
            physical[idx[1:-1],player,5] = np.linalg.norm(acceleration,axis=1)
            physical[idx[0],player,2:5] = physical[idx[1],player,2:5]
            physical[idx[[0,-1]],player,5] = 0.
    frames = np.arange(0,len(xy),5)
    result = dict(physical=physical[frames],teams=teams,period=period[frames],
                  frames=frames,n_frames=np.array(len(xy)))
    np.savez_compressed(path,**result)
    return result


def phases(data):
    physical, teams, period = data['physical'], data['teams'], data['period']
    valid = np.isfinite(physical).all(axis=2)
    changes = np.r_[True,(valid[1:] != valid[:-1]).any(axis=1) | (period[1:] != period[:-1])]
    boundaries = np.r_[np.flatnonzero(changes),len(physical)]
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        if b-a < 10:
            continue
        players = [np.flatnonzero(valid[a] & (teams == team)) for team in (0,1)]
        if any(len(ids) < 7 or len(ids) > 11 for ids in players):
            continue
        keepers = [ids[np.argmax(np.abs(physical[a:b,ids,0].mean(axis=0)-52.5))] for ids in players]
        left = 0 if physical[a:b,keepers[0],0].mean() < physical[a:b,keepers[1],0].mean() else 1
        input_data = np.zeros((b-a,26,8),np.float32)
        available = np.zeros(26,bool)
        home_nodes = np.zeros(26,bool)
        for side,team in enumerate((left,1-left)):
            ids = players[team]
            slots = np.arange(side*11,side*11+len(ids))
            input_data[:,slots,0] = -1. if side == 0 else 1.
            input_data[:,slots,1] = ids == keepers[team]
            input_data[:,slots,2:] = physical[a:b,ids]
            available[slots] = True
            home_nodes[slots] = team == 0
        input_data[:,22:,2:4] = [[0.,34.],[105.,34.],[52.5,0.],[52.5,68.]]
        available[22:] = True
        yield a,b,input_data,available,home_nodes
