"""Prepare target reference trajectories for evaluation only."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from SHAPE.gridspec import StaticGrid
from SHAPE.stage2.prepare_city_event_data import CITY_CONFIG, _load_city_records, load_city_paths
from SHAPE.stage2.sparse_event_data import map_records_to_grid
from SHAPE.stage3.prepare_codes import assign_day_splits
from SHAPE.stage3.reference import prepare_ca_reference, REFERENCE_SCHEMA


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--city', required=True, choices=('nyc', 'ca', 'paris'))
    parser.add_argument('--grid-csv', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--raw-checkins', nargs='+')
    parser.add_argument('--config', help='Same city-path JSON accepted by prepare-events')
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        raise FileExistsError('Reference already exists; test data must not be overwritten')
    if args.config:
        load_city_paths(args.config)
    if args.city == 'ca':
        arrays, receipt = prepare_ca_reference(
            args.raw_checkins or CITY_CONFIG['ca']['checkin_files'],
            args.grid_csv, args.output,
        )
    else:
        if args.raw_checkins:
            if len(args.raw_checkins) != 1:
                parser.error('NYC/Paris accept one checkin file; use --config for Paris POI metadata')
            CITY_CONFIG[args.city]['checkin_file'] = args.raw_checkins[0]
        static = StaticGrid.load(args.grid_csv, city=args.city)
        records, mapping = map_records_to_grid(_load_city_records(args.city, static), static)
        users = defaultdict(list)
        for row in records:
            users[str(row['user_id'])].append(row)
        days = []
        for user in sorted(users):
            rows = sorted(users[user], key=lambda row: row['local_dt'])
            if len(rows) < 3:
                continue
            splits = assign_day_splits(user, [row['date'] for row in rows], seed=42)
            by_date = defaultdict(list)
            for row in rows:
                by_date[row['date']].append(row)
            days.extend((user, date, splits[date], by_date[date]) for date in sorted(by_date))
        if not days:
            raise ValueError('No target reference days')
        lengths = np.asarray([len(day[3]) for day in days], dtype=np.int64)
        receipt = dict(schema=REFERENCE_SCHEMA, city=args.city, seed=42, mapping=mapping,
                       purpose='evaluation_only', active_grid_filter_used=False)
        arrays = dict(schema=np.asarray(REFERENCE_SCHEMA), source_city=np.asarray(args.city),
            metadata_json=np.asarray(json.dumps(receipt)), day_offsets=np.r_[0,np.cumsum(lengths)],
            day_user_ids=np.asarray([day[0] for day in days]),
            day_dates=np.asarray([day[1] for day in days]), day_splits=np.asarray([day[2] for day in days]),
            event_user_ids=np.repeat([day[0] for day in days],lengths),
            event_splits=np.repeat([day[2] for day in days],lengths),
            grid_id=np.asarray([row['_grid_id'] for day in days for row in day[3]],dtype=np.int64),
            minute=np.asarray([row['minute'] for day in days for row in day[3]],dtype=np.int16))
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output,**arrays)
    print(json.dumps(receipt,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
