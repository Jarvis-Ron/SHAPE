"""Metric names and source/target information boundary helpers."""
import time
from SHAPE.stage3.tasks import FIXED_VARIANT

FIVE = {
    'M2_od_cpc': 'OD CPC',
    'M3_period_space_jsd_mean': 'Spatiotemporal JSD',
    'A_radius_of_gyration_distribution_jsd': 'RoG JSD',
    'A_jump_distance_spectrum_jsd': 'Jump-length JSD',
    'A_visited_poi_category_share_jsd': 'POI-category JSD',
}



def announce(message):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), message, flush=True)


def config_id(config):
    if config != FIXED_VARIANT:
        raise ValueError('Only the frozen SHAPE variant is supported')
    return 'SHAPE'
