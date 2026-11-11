"""Build target home weights from public residential population, without check-ins.

The optional rasterio dependency is needed only to rebuild the released weights.
WGS84 population counts are apportioned by spherical rectangle-overlap area;
they are never interpolated as intensities or rounded to individual residents.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import urllib.request

import numpy as np

from SHAPE.gridspec import GridSpec, sha256_file

SCHEMA = 'target_public_population_quota_v1'
DATASET_URL = 'https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php'
DATASET_DOI = '10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE'
RASTER_URL = ('https://jrc-ghsl.s3.eu-central-1.amazonaws.com/ghs-pop/r2023a/'
              '4326/3asec/{year}/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss_V1_0.tif')


def population_probabilities(quota, vocabulary=None, *, city=None, grid_sha256=None):
    """Reject legacy check-in quotas and validate the public population boundary."""
    fields = {'schema', 'city', 'grid_ids', 'population_counts', 'provenance'}
    if set(quota) != fields or quota.get('schema') != SCHEMA:
        raise ValueError('A public residential-population quota is required; rebuild legacy home quotas')
    provenance = quota['provenance']
    if not isinstance(provenance, dict) or provenance.get('target_checkins_used') is not False \
            or provenance.get('source_kind') != 'public_residential_population' \
            or not provenance.get('dataset') or not isinstance(provenance.get('epoch'), int):
        raise ValueError('Population quota lacks public-data provenance')
    original_ids = np.asarray(quota['grid_ids'])
    ids = np.asarray(quota['grid_ids'], dtype=np.int64)
    counts = np.asarray(quota['population_counts'], dtype=np.float64)
    if ids.ndim != 1 or not len(ids) or not np.array_equal(original_ids, ids) \
            or np.any(ids < 0) or np.any(np.diff(ids) <= 0) or counts.shape != ids.shape \
            or not np.isfinite(counts).all() or np.any(counts < 0) \
            or not np.isfinite(counts.sum()) or counts.sum() <= 0:
        raise ValueError('Invalid public population counts or grid IDs')
    if vocabulary is not None and not np.array_equal(ids, vocabulary):
        raise ValueError('Population quota is not aligned with the target vocabulary')
    if city is not None and quota['city'].lower() != city.lower():
        raise ValueError('Population quota city differs from the target city')
    if grid_sha256 is not None and provenance.get('grid_sha256') != grid_sha256:
        raise ValueError('Population quota was built for a different static grid')
    return counts / counts.sum()


def rectangle_population(counts, transform, bounds):
    """Integrate north-up WGS84 raster counts into [south,north,west,east] boxes.

    Within each population pixel, mass is uniform per spherical surface area.
    Shared latitude bands are integrated together for the SHAPE row geometry.
    The sum over a non-overlapping partition equals the enclosing rectangle.
    """
    a, b, c, d, e, f = tuple(transform)[:6]
    if b != 0 or d != 0 or a <= 0 or e >= 0:
        raise ValueError('Population raster must be north-up, without rotation')
    counts = np.asarray(counts)
    bounds = np.asarray(bounds, dtype=np.float64)
    if counts.ndim != 2 or bounds.ndim != 2 or bounds.shape[1] != 4 \
            or not np.isfinite(counts).all() or np.any(counts < 0) \
            or not np.isfinite(bounds).all():
        raise ValueError('Invalid raster counts or rectangle bounds')
    height, width = counts.shape
    if np.any(bounds[:, 0] >= bounds[:, 1]) or np.any(bounds[:, 2] >= bounds[:, 3]) \
            or np.any(bounds[:, 0] < f + e*height - 1e-9) \
            or np.any(bounds[:, 1] > f + 1e-9) \
            or np.any(bounds[:, 2] < c - 1e-9) \
            or np.any(bounds[:, 3] > c + a*width + 1e-9):
        raise ValueError('Population raster does not cover the requested grid')
    result = np.zeros(len(bounds), dtype=np.float64)
    bands, inverse = np.unique(bounds[:, :2], axis=0, return_inverse=True)
    for index, (south, north) in enumerate(bands):
        lo = max(0, int(math.floor((f-north)/(-e))))
        hi = min(height, int(math.ceil((f-south)/(-e))))
        pixel_north = f + np.arange(lo, hi)*e
        pixel_south = pixel_north + e
        overlap_north = np.minimum(north, pixel_north)
        overlap_south = np.maximum(south, pixel_south)
        numerator = np.maximum(0., np.sin(np.radians(overlap_north))
                               - np.sin(np.radians(overlap_south)))
        denominator = np.sin(np.radians(pixel_north)) - np.sin(np.radians(pixel_south))
        weights = numerator / denominator
        columns = np.einsum('i,ij->j', weights, counts[lo:hi], dtype=np.float64)
        prefix = np.r_[0., np.cumsum(columns, dtype=np.float64)]
        selected = np.flatnonzero(inverse == index)

        def integral(longitude):
            x = np.clip((longitude-c)/a, 0, width)
            left = np.floor(x).astype(np.int64)
            return prefix[left] + (x-left)*columns[np.minimum(left, width-1)]

        result[selected] = np.maximum(0., integral(bounds[selected, 3])
                                      - integral(bounds[selected, 2]))
    return result


def crop_raster(url, spec, destination):
    """Read only a city window from the official cloud-optimized global raster."""
    import rasterio
    from rasterio.windows import Window, from_bounds

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(destination.name+'.partial')
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=40) as response:
        identity = dict(source_etag=response.headers.get('ETag', ''),
                        source_last_modified=response.headers.get('Last-Modified', ''),
                        source_bytes=response.headers.get('Content-Length', ''))
    box = spec.bbox
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif', GDAL_HTTP_TIMEOUT='40',
                      GDAL_HTTP_MAX_RETRY='2', GDAL_CACHEMAX=512*1024*1024):
        with rasterio.open(url) as source:
            if source.crs != rasterio.crs.CRS.from_epsg(4326) or source.count != 1:
                raise ValueError('Expected a single-band EPSG:4326 population-count raster')
            w = from_bounds(box['lng_min'], box['lat_min'], box['lng_max'],
                            box['lat_max'], source.transform)
            left, top = math.floor(w.col_off), math.floor(w.row_off)
            right, bottom = math.ceil(w.col_off+w.width), math.ceil(w.row_off+w.height)
            window = Window(left, top, right-left, bottom-top)
            if left < 0 or top < 0 or right > source.width or bottom > source.height:
                raise ValueError('Population raster does not cover the city extent')
            profile = source.profile.copy()
            profile.update(width=int(window.width), height=int(window.height),
                           transform=source.window_transform(window), compress='deflate',
                           tiled=True, blockxsize=512, blockysize=512, BIGTIFF='IF_SAFER')
            with rasterio.open(temporary, 'w', **profile) as target:
                target.update_tags(source_url=url, **identity)
                for row in range(0, int(window.height), 512):
                    rows = min(512, int(window.height)-row)
                    block = source.read(1, window=Window(left, top+row, window.width, rows))
                    target.write(block, 1, window=Window(0, row, window.width, rows))
                    print(f'{spec.city}: raster rows {row+rows}/{int(window.height)}', flush=True)
    temporary.replace(destination)
    return destination


def build_quota(city, grid_csv, grid_spec, raster_path, *, epoch=2010, source_url=None):
    import rasterio

    spec = GridSpec.load(city, grid_spec)
    with Path(grid_csv).open(newline='') as handle:
        ids = np.asarray(sorted(int(row['grid_id']) for row in csv.DictReader(handle)), dtype=np.int64)
    if len(np.unique(ids)) != len(ids) or not spec.valid_nominal_ids(ids).all():
        raise ValueError('Static grid IDs are invalid')
    with rasterio.open(raster_path) as source:
        if source.crs != rasterio.crs.CRS.from_epsg(4326) or source.count != 1:
            raise ValueError('Expected a single-band EPSG:4326 population-count raster')
        data = source.read(1, masked=True)
        masked = int(np.ma.count_masked(data))
        counts = data.filled(0)
        transform = source.transform
        source_identity = {key: source.tags().get(key) for key in
                           ('source_etag', 'source_last_modified', 'source_bytes')}
        grid_counts = rectangle_population(counts, transform, spec.bounds_of(ids))
        box = spec.bbox
        bbox_population = float(rectangle_population(counts, transform, [[box['lat_min'],
            box['lat_max'], box['lng_min'], box['lng_max']]])[0])
    retained = float(grid_counts.sum())
    if bbox_population <= 0 or retained <= 0 or retained > bbox_population*(1+1e-9):
        raise ValueError('Population conservation or coverage check failed')
    provenance = dict(
        source_kind='public_residential_population', dataset='GHS-POP R2023A',
        epoch=int(epoch), doi=DATASET_DOI, dataset_url=DATASET_URL,
        raster_url=source_url or RASTER_URL.format(year=epoch), license='CC-BY-4.0',
        attribution='European Commission, Joint Research Centre (JRC); Schiavina, Freire, Carioli and MacManus (2023)',
        source_resolution='3 arcseconds (approximately 100 m at the equator)',
        population_unit='estimated residents per cell; fractional values retained',
        method='spherical_rectangle_overlap_area_weighting_uniform_within_each_input_pixel',
        support='existing_emitted_static_grid_vocabulary; excluded mass not reassigned',
        raster_crop_sha256=sha256_file(raster_path),
        grid_sha256=sha256_file(grid_csv), grid_spec_sha256=sha256_file(grid_spec),
        bbox=box, bbox_population=bbox_population, retained_population=retained,
        excluded_population=max(0., bbox_population-retained),
        retained_population_fraction=retained/bbox_population,
        positive_population_grids=int(np.count_nonzero(grid_counts)),
        masked_input_pixels=masked, target_checkins_used=False,
        target_test_used=False, target_user_ids_used=False,
        **source_identity,
    )
    quota = dict(schema=SCHEMA, city=city.lower(), grid_ids=ids.tolist(),
                 population_counts=grid_counts.tolist(), provenance=provenance)
    population_probabilities(quota, ids, city=city, grid_sha256=sha256_file(grid_csv))
    return quota


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--city', choices=['nyc', 'ca', 'paris'], required=True)
    parser.add_argument('--grid', required=True)
    parser.add_argument('--grid-spec', default='dataset/grid_spec.json')
    parser.add_argument('--year', type=int, choices=range(1975, 2031, 5), default=2010)
    parser.add_argument('--cache', required=True, help='Directory for the downloaded city crop')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    spec = GridSpec.load(args.city, args.grid_spec)
    url = RASTER_URL.format(year=args.year)
    crop = Path(args.cache)/f'ghs_pop_{args.year}_{args.city}.tif'
    if crop.exists():
        import rasterio
        with rasterio.open(crop) as source:
            if source.tags().get('source_url') != url:
                raise ValueError('Cached crop source differs from the requested population product')
    else:
        crop_raster(url, spec, crop)
    quota = build_quota(args.city, args.grid, args.grid_spec, crop,
                        epoch=args.year, source_url=url)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(quota, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(quota['provenance']), flush=True)


if __name__ == '__main__':
    main()
