# Contains the optimizer that uses perfect forecasts. Deterministic MPC.

from .mpc_det import MpcDetOptimizer
import pandas as pd
from pathlib import Path
from utils import map_building_to_pv_num_orientation

class IdealOptimizer(MpcDetOptimizer):

    def __init__(self, *args, gt_path=None, **kwargs):
        self.super_fc_long = None  # Store the ground truth to decrease number of file access
        self._custom_gt_path = gt_path
        super().__init__(*args, **kwargs)

    def _prepare_forecast(self, forecast: pd.DataFrame) -> pd.DataFrame:
        ''' Instead of using the provided forecast => use ground truth as forecast. '''

        if self.super_fc_long is None:
            # Load the full gt
            num_pv_modules, orientation = map_building_to_pv_num_orientation(self.b)


            project_root = Path(__file__).resolve().parents[3]
            print(f"\n🚀 === PROJECT ROOT IS: {project_root} === 🚀\n")

            target_folder = self._custom_gt_path if self._custom_gt_path else f'01_data/prosumption_data/{self.mpc_freq}min'
            
            data_dir = project_root / target_folder
            path = data_dir / f'prosumption_{self.b}_num_pv_modules_{num_pv_modules}_pv_{orientation}_hp_1.0.csv'
            
            df = pd.read_csv(path, parse_dates=['index'], index_col='index', usecols=['index', 'P_TOT'])
            df.index.name = 'timestamp'
            self.super_fc_long = df
        
        self.super_fc = self.super_fc_long.copy()
        self.super_fc = self.super_fc.loc[forecast.index.get_level_values('timestamp')]
        self.super_fc['P_TOT'] = self.super_fc['P_TOT'] / 1000.0  # Convert from W to kW

        self.super_fc.rename(columns={'P_TOT': 'expected_value'}, inplace=True)
        return self.super_fc