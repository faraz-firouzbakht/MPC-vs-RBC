from .mpc_det import MpcDetOptimizer
import pandas as pd
from pathlib import Path
from utils import map_building_to_pv_num_orientation

class MpcRuleProportionalOptimizer(MpcDetOptimizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_soe(self, t_now, decision, gt):
        ''' 
        Standalone Proportional Rule:
        Follows const-grid behavior normally, but linearly reduces battery action 
        (Soft Clipping) when SoC is within a 10% buffer zone near physical limits.
        '''

        pg_mpc = decision.get('pg', 0.0)
        

        pb_desired = gt - pg_mpc

        # ==========================================
        # -- Proportional (Soft SoC Limits) --
        # ==========================================
        

        buffer_zone = 0.1 * (self.cap_max - self.cap_min)
        upper_threshold = self.cap_max - buffer_zone  
        lower_threshold = self.cap_min + buffer_zone  

        if pb_desired < 0.0:  
            if self.soe_now > upper_threshold:

                factor = (self.cap_max - self.soe_now) / buffer_zone

                factor = max(0.0, factor) 
                pb_desired = pb_desired * factor
                
        elif pb_desired > 0.0:  
            if self.soe_now < lower_threshold:

                factor = (self.soe_now - self.cap_min) / buffer_zone

                factor = max(0.0, factor)
                pb_desired = pb_desired * factor


        
        # Ensure pb is within hardware limits
        pb_desired = max(self.pb_min, min(self.pb_max, pb_desired))

        if pb_desired < 0.0:  # CHARGING
            alpha = self.eta_ch
            available_cap = self.cap_max - self.soe_now
            pb_energy_min = - available_cap / (alpha * self.gt_inc)
            pb = max(pb_desired, pb_energy_min)

        else:  # DISCHARGING
            alpha = 1/self.eta_dis
            available_cap = self.soe_now - self.cap_min
            pb_energy_max = available_cap / (alpha * self.gt_inc)
            pb = min(pb_desired, pb_energy_max)

        pg_actual = gt - pb
        soe_new = self.soe_now - pb * alpha * self.gt_inc


        if round(soe_new, 5) > self.cap_max or round(soe_new, 5) < self.cap_min:
            raise ValueError(f"State of charge out of bounds: {soe_new} kWh.")

        if soe_new > self.cap_max:
            soe_new = self.cap_max
        if soe_new < self.cap_min:
            soe_new = self.cap_min


        self.results_realization[t_now] = {
            'timestamp': t_now,
            'action': pb,
            'pb': pb,
            'pg': pg_actual,
            'gt': gt,
            'soe_now': self.soe_now,
            'soe_new': soe_new,
            'pb_mpc': decision.get('pb'),
            'pg_mpc': pg_mpc
        }   

        self.soe_now = soe_new
        return soe_new


class IdealRuleProportionalOptimizer(MpcRuleProportionalOptimizer):
    def __init__(self, *args, **kwargs):
        self.super_fc_long = None
        super().__init__(*args, **kwargs)

    def _prepare_forecast(self, forecast: pd.DataFrame) -> pd.DataFrame:
        if self.super_fc_long is None:
            num_pv_modules, orientation = map_building_to_pv_num_orientation(self.b)
            path = Path(f'01_data/prosumption_data/{self.mpc_freq}min/prosumption_{self.b}_num_pv_modules_{num_pv_modules}_pv_{orientation}_hp_1.0.csv')
            df = pd.read_csv(path, parse_dates=['index'], index_col='index', usecols=['index', 'P_TOT'])
            df.index.name = 'timestamp'
            self.super_fc_long = df
        
        self.super_fc = self.super_fc_long.copy()
        self.super_fc = self.super_fc.loc[forecast.index.get_level_values('timestamp')]
        self.super_fc['P_TOT'] = self.super_fc['P_TOT'] / 1000.0
        self.super_fc.rename(columns={'P_TOT': 'expected_value'}, inplace=True)
        return self.super_fc