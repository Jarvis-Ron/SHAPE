"""Fixed SHAPE contract and source-behavior sampling tasks.

Private X1/HRA labels remain in the landing kernel for checkpoint compatibility;
they are internal identifiers, not additional model choices.
"""
import json
from pathlib import Path

FIXED_VARIANT = {'weighted_rank': True, 'lambda': 0, 'kappa': 0, 'cost': True}
DEFAULT_CONTRACT = Path(__file__).parents[1] / 'shape_contract.json'


def load_contract(path=DEFAULT_CONTRACT):
    """Reject alternate mechanisms; this release packages one selected model."""
    contract = json.loads(Path(path).read_text(encoding='utf-8'))
    if contract != json.loads(DEFAULT_CONTRACT.read_text(encoding='utf-8')):
        raise ValueError('SHAPE configuration differs from the frozen release contract')
    return contract


def make_x1_hra_tasks(payloads, contract):
    """One independent sampling lane per seed, with source-day behavior fixed."""
    if not {'source_train_real', 'source_test_real'} <= set(payloads):
        raise ValueError('SHAPE needs observed source train and test code payloads')
    return [dict(
        analysis_id='X1_HRA_map_rescaled_attraction', arm='X1',
        scheme='HRA_map_rescaled_attraction', code_source='NYC_real',
        home_strategy='T2_context_quota', memory_mode='rolling_Paris_past',
        attraction_mode='source_role_target_map_rescaled', scale_mode='rank',
        split='test', generator_seed=None, sampling_seed=int(seed),
        payload_key='source_test_real', warmup_payload_key='source_train_real',
    ) for seed in contract['sampling']['landing_seeds']]
