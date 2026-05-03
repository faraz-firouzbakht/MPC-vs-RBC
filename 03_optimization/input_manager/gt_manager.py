import os
import pandas as pd
import sys

from utils import load_chunks


class GroundTruthManager:
    """
    Loads fine-resolution ground truth time series for multiple buildings, and provides resampled versions on demand.    
    """

    def __init__(self, config: dict):
        self.buildings = config['optimization']['buildings']
        self.mpc_freqs = config['optimization']['mpc_update_freq']
        self.gt_freq = config['optimization']['gt_freq']  
        
        # --- FIX 1: Dynamically get the path from config (default to 1min if not provided) ---
        self.gt_path = config['optimization'].get('gt_path', '01_data/prosumption_data/1min')

        self.start_time = pd.Timestamp(config['optimization']['start_time'])
        self.end_time = pd.Timestamp(config['optimization']['end_time'])

        self.mpc_horizon = config['optimization']['mpc_horizon']

        self._dfs = {}
        self._cache = {}  


        self._load_gt()  
        self._validate_freq()


    def _load_gt(self):        
        for b in self.buildings:
            num_pv_modules, orientation = self.map_building_to_pv_num_orientation(b)

            # --- FIX 2: Use the dynamic path from the config instead of hardcoding '1min' ---
            path = f'{self.gt_path}/prosumption_{b}_num_pv_modules_{num_pv_modules}_pv_{orientation}_hp_1.0.csv'

            df = load_chunks(path, self.start_time, self.end_time + pd.Timedelta(hours=self.mpc_horizon - 1), filter_col='index', parse_dates=['index'], usecols=['index', 'P_TOT'])
            df['P_TOT'] = df['P_TOT'] / 1000.0  # Convert from W to kW

            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"GT file for '{b}' must have a DatetimeIndex")
            
            self._dfs[b] = df.sort_index()

        

    def _validate_freq(self):
        freqs = (self._dfs[self.buildings[0]].index.to_series().diff().dropna().unique())
        if len(freqs) != 1:
            raise ValueError(f"GT for '{self.buildings[0]}' has irregular timestamps.")
        
        # --- FIX 3: Calculate in seconds instead of minutes to prevent Division-by-Zero ---
        base_freq_sec = int(freqs[0].total_seconds())

        for mpc_freq in self.mpc_freqs:
            # mpc_freq is in minutes, so we multiply by 60 to compare in seconds
            if (mpc_freq * 60) % base_freq_sec != 0:
                raise ValueError(
                    f"mpc_freq={mpc_freq}min is not a multiple of base GT freq={base_freq_sec}sec"
                )
            
    def get_gt(self, building):
        if building not in self._dfs:
            raise KeyError(f"Unknown building '{building}'")

        key = (building, self.gt_freq)

        if key not in self._cache:
            df_fine = self._dfs[building]
            
            # --- FIX 4: Check if gt_freq is an integer (minutes) or a string (like '10S') ---
            if isinstance(self.gt_freq, int) or str(self.gt_freq).isdigit():
                rule = f"{self.gt_freq}min"
            else:
                rule = str(self.gt_freq)

            df_rs = (df_fine
                    .resample(rule)   
                    .mean()
                    )
            self._cache[key] = df_rs

        return self._cache[key]
        
        
    def map_building_to_pv_num_orientation(self, b): 
        mapper = {
            'SFH3': (26, 'SOUTH'),
            'SFH4': (30, 'SOUTH'),
            'SFH9': (36, 'SOUTH'),
            'SFH10': (25, 'SOUTH'),
            'SFH12': (21, 'SOUTH'),
            'SFH14': (28, 'SOUTH'),
            'SFH16': (25, 'EAST'),
            'SFH18': (16, 'EAST'),
            'SFH19': (38, 'EAST'),
            'SFH22': (21, 'EAST'),
            'SFH27': (20, 'WEST'),
            'SFH28': (27, 'WEST'),
            'SFH29': (19, 'WEST'),
            'SFH30': (18, 'WEST'),
            'SFH32': (30, 'WEST'),
            'SFH36': (20, 'WEST'),

        }

        if b not in mapper:
            raise ValueError(f"Building '{b}' not found in GT mapping. Available buildings: {list(mapper.keys())}")
        
        num_pv_modules, orientation = mapper[b]
        return num_pv_modules, orientation