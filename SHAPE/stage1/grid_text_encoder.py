"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from SHAPE.gridspec import StaticGrid


class CityGridEmbedding(nn.Module):
    """Encode sparse city grids as bounded Transformer tiles with halos.

    Input: 20-dimensional static features and explicit raw grid identifiers.
    Output: an ID-keyed, L2-normalized 64-dimensional embedding table.
    Features are standardized and combined with coordinate positional encodings
    before the Transformer and projection layers.
    """

    def __init__(self, d_feat=20, d_model=128, nhead=8, num_layers=6,
                 dim_feedforward=512, dropout=0.1, d_proj=64,
                 max_grid_size=64, num_grids=None,
                 num_poi_categories=None):
        super().__init__()
        self.d_feat = d_feat
        self.d_model = d_model
        self.max_grid_size = max_grid_size
        self.num_grids = num_grids
        self.num_poi_categories = num_poi_categories

        self.input_proj = nn.Linear(d_feat, d_model)
        self.pos_encoder = GridPositionalEncoding(d_model, max_grid_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_proj),
        )
        # Optional 13-D POI profile classification head used as an auxiliary
        # semantic constraint during cross-city contrastive training.
        # Typical usage: loss = L_contrastive + w * CE(proj(h), poi_profile).
        self.semantic_head = None
        if self.num_poi_categories is not None:
            self.semantic_head = nn.Linear(d_proj, self.num_poi_categories)
        self._init_weights()

        # Feature statistics (will be updated during training)
        self.register_buffer('feat_mean', torch.zeros(d_feat))
        self.register_buffer('feat_std', torch.ones(d_feat))

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def update_feat_stats(self, features: torch.Tensor):
        with torch.no_grad():
            self.feat_mean = features.mean(dim=0)
            self.feat_std = features.std(dim=0).clamp(min=1e-6)

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.feat_mean) / self.feat_std

    def encode_city(self, features: torch.Tensor, grid_ids: torch.Tensor,
                    positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.normalize_features(features)
        x = self.input_proj(x)
        pe = self.pos_encoder(grid_ids, positions=positions)
        x = x + pe
        x = self.transformer(x)
        x = self.norm(x)
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=-1)
        return x

    def forward_with_semantic(self, features: torch.Tensor,
                              grid_ids: torch.Tensor,
                              positions: Optional[torch.Tensor] = None):
        """Return (embedding, semantic_logits) if semantic_head is present."""
        emb = self.encode_city(features, grid_ids, positions=positions)
        if self.semantic_head is None:
            raise ValueError("semantic_head not configured; rebuild model with num_poi_categories.")
        logits = self.semantic_head(emb)
        return emb, logits

    def forward(self, features: torch.Tensor, grid_ids: torch.Tensor,
                positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encode_city(features, grid_ids, positions=positions)

    @torch.no_grad()
    def encode_static_grid(
            self, static_grid: StaticGrid, device: str = 'cuda',
            tile_size: int = 32, halo: int = 4,
            anchor_ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Encode emitted cells in bounded tiles and return an ID-keyed table."""
        self.eval()
        anchors = static_grid.grid_ids if anchor_ids is None else np.unique(
            np.asarray(anchor_ids, dtype=np.int64)
        )
        anchors = anchors[static_grid.contains(anchors)]
        output = np.zeros((len(anchors), self.projection[-1].out_features), dtype=np.float32)
        output_index = {int(grid_id): index for index, grid_id in enumerate(anchors)}
        assigned = np.zeros(len(anchors), dtype=bool)
        for tile in static_grid.tiles(tile_size=tile_size, halo=halo,
                                      anchor_ids=anchors):
            features = torch.from_numpy(
                static_grid.features[tile.context_indices]
            ).float().to(device)
            grid_ids = torch.from_numpy(tile.context_ids).long().to(device)
            positions = torch.from_numpy(np.column_stack((
                tile.relative_rows, tile.relative_cols,
            ))).long().to(device)
            context_embeddings = self.encode_city(features, grid_ids, positions)
            context_lookup = {
                int(grid_id): index for index, grid_id in enumerate(tile.context_ids)
            }
            for grid_id in tile.core_ids:
                out_index = output_index[int(grid_id)]
                output[out_index] = context_embeddings[
                    context_lookup[int(grid_id)]
                ].detach().cpu().numpy()
                assigned[out_index] = True
        if not assigned.all():
            raise RuntimeError(
                "Tiled encoding missed %d anchor cells" % int((~assigned).sum())
            )
        return output, anchors

    @torch.no_grad()
    def encode_all_grids(self, features: np.ndarray, grid_ids: np.ndarray,
                         num_grids: Optional[int] = None, batch_size: int = 256,
                         device: str = 'cuda') -> np.ndarray:
        """Legacy fixture encoder; production extraction uses encode_static_grid."""
        self.eval()
        limit = len(features) if num_grids is None else min(int(num_grids), len(features))
        feat_tensor = torch.from_numpy(features[:limit]).float().to(device)
        gid_tensor = torch.from_numpy(grid_ids[:limit]).long().to(device)
        return self.encode_city(feat_tensor, gid_tensor).cpu().numpy()


class GridPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_grid_size: int = 64):
        super().__init__()
        self.d_model = d_model
        self.max_grid_size = max_grid_size
        pe_row = torch.zeros(max_grid_size, d_model)
        pe_col = torch.zeros(max_grid_size, d_model)
        position = torch.arange(max_grid_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe_row[:, 0::2] = torch.sin(position * div_term)
        pe_row[:, 1::2] = torch.cos(position * div_term)
        pe_col[:, 0::2] = torch.sin(position * div_term)
        pe_col[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe_row', pe_row)
        self.register_buffer('pe_col', pe_col)

    def forward(self, grid_ids: torch.Tensor,
                positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        if positions is None:
            # Backwards-compatible only for small synthetic/legacy fixtures.
            # Released datasets always provide stride-aware tile positions.
            row = grid_ids // self.max_grid_size
            col = grid_ids % self.max_grid_size
        else:
            if positions.shape[-1] != 2:
                raise ValueError("positions must have shape [..., 2]")
            row, col = positions[..., 0], positions[..., 1]
        if torch.any(row < 0) or torch.any(col < 0) \
                or torch.any(row >= self.max_grid_size) \
                or torch.any(col >= self.max_grid_size):
            raise ValueError("relative tile position exceeds positional table")
        row = row.long()
        col = col.long()
        return self.pe_row[row] + self.pe_col[col]


def load_grid_features(
        csv_path: str, city: Optional[str] = None,
        grid_spec_path: Optional[str] = None,
        verify_release: bool = False,
        return_static_grid: bool = False):
    """Read features in CSV row order for fixtures or through StaticGrid.

    Production callers pass ``city`` plus ``grid_spec_path`` and receive the
    release-verified, sorted sparse table.  The CSV-only fallback is retained
    for small self-contained unit fixtures, never for released training.
    """
    if city is not None and grid_spec_path is not None:
        grid = StaticGrid.load(
            csv_path, city=city, grid_spec_path=grid_spec_path,
            verify_release=verify_release,
        )
        if return_static_grid:
            return grid
        return grid.features, grid.grid_ids, grid.centers
    import csv
    with open(csv_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    N = len(rows)
    features = np.zeros((N, 20), dtype=np.float32)
    grid_ids = np.zeros(N, dtype=np.int64)
    centers = np.zeros((N, 2), dtype=np.float64)
    for i, row in enumerate(rows):
        gid = int(row['grid_id'])
        grid_ids[i] = gid
        for j in range(13):
            features[i, j] = float(row.get('poi_prop_' + str(j), 0.0))
        features[i, 13] = float(row.get('poi_density_level', 0.0))
        cnames = ['total_stations', 'metro_stations', 'bus_stations',
                   'total_frequency', 'metro_frequency', 'bus_frequency']
        for j, cn in enumerate(cnames):
            features[i, 14 + j] = float(row.get(cn, 0.0))
        lat_center = (float(row['grid_lat_max']) + float(row['grid_lat_min'])) / 2.0
        lng_center = (float(row['grid_lng_min']) + float(row['grid_lng_max'])) / 2.0
        centers[i] = [lat_center, lng_center]
    print(f'  [load_grid_features] Loaded {N} grids from {csv_path}')
    if return_static_grid:
        raise ValueError("return_static_grid requires city and grid_spec_path")
    return features, grid_ids, centers
