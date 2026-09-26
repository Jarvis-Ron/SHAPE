# Included data

This release includes the actual processed sequences needed by the four experiments, static map features, and raw-POI/transport provenance. Generation and evaluation do not require the entire worldwide Foursquare dataset.

| Direction | Source users / check-ins / days | Target reference users / check-ins / days |
|---|---:|---:|
| NYC → CA | 1,083 / 227,427 / 93,862 | 4,314 / 241,596 / 80,532 |
| NYC → Paris | 1,083 / 227,427 / 93,862 | 3,267 / 85,792 / 52,505 |
| CA → NYC | 1,070 / 172,573 / 53,816 | 1,083 / 227,427 / 93,862 |
| CA → Paris | 1,070 / 172,573 / 53,816 | 3,267 / 85,792 / 52,505 |

These counts include all splits before evaluation eligibility and the two-event-day restriction. CA's historical source cohort is smaller than its target cohort. The NYC source archives contain the same check-ins but different upstream encodings; they are not interchangeable. Native generation eligibility leaves 1,082 NYC users, 9,400 test days and 22,772 test events per seed, or 1,070 CA users, 5,385 test days and 17,466 test events per seed.

## Files

- `dataset/checkin_NY_TY2014/dataset_TSMC2014_NYC.txt`: original NYC source file (227,428 records; the processed archive contains 227,427).
- `dataset/Gowalla-CA/gowalla_{train,val,test}.csv`: supplied California input partitions. Subsequent model user/day splits are recorded in the processed archives.
- `dataset/prepared/ca_to_{nyc,paris}/`: historical source-event features and train/validation/test user membership for optional prepared-input training.
- `inputs/*/source_codes.npz`: frozen source codes and schedules.
- `inputs/*/target_reference.npz`: processed target check-ins and splits, including the used Paris subset of global Foursquare.
- `dataset/{CA,NYC,PARIS}_grid_merged_features.csv`: 105,640, 7,376, and 709 emitted static grids, respectively. Grid IDs are sparse identifiers, not row numbers.

The raw NYC and CA files support full fresh-training commands. Paris is a target in all four experiments, so its fixed processed reference is sufficient. To independently rebuild Paris preprocessing, obtain the original worldwide check-in and POI files and set their paths in `configs/city_paths.example.json`; bounding boxes are in `dataset/grid_spec.json`. Credits and data rights are described in `THIRD_PARTY_DATA.md`.
