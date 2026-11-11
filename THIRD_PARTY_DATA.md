# Third-party data

Names and links here identify original dataset providers, not authors of the anonymous SHAPE submission. Data retain their original rights and attribution; inclusion does not grant additional rights or place third-party data under a software license.

## Foursquare (NYC and Paris)

Source: [Foursquare dataset collection](https://sites.google.com/site/yangdingqi/home/foursquare-dataset).

NYC uses the NYC portion of TSMC2014. Paris uses the city subset of global TIST2015. Original provider readmes are included in the dataset directory. Follow the provider's use and citation instructions. Provider references:

- Dingqi Yang, Daqing Zhang, Vincent W. Zheng, and Zhiyong Yu. *Modeling User Activity Preference by Leveraging User Spatial Temporal Characteristics in LBSNs*. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2015.
- Dingqi Yang, Daqing Zhang, and Bingqing Qu. *Participatory Cultural Mapping Based on Collective Behavior Data in Location-Based Social Networks*. ACM Transactions on Intelligent Systems and Technology, 2015.

## Gowalla (California)

Source: [Stanford SNAP Gowalla collection](https://snap.stanford.edu/data/loc-gowalla.html).

California uses the supplied regional partitions. Provider reference: Eunjoon Cho, Seth A. Myers, and Jure Leskovec. *Friendship and Mobility: User Movement in Location-Based Social Networks*. ACM KDD, 2011.

## Static features

The processed grids include POI and transport attributes, with numerical preprocessing receipts in their provenance files. Do not infer an unrestricted upstream data license from their inclusion.

## Public residential population

The files in `dataset/population_quotas/` are derived from **GHS-POP R2023A,
epoch 2010**, using the WGS84 3-arcsecond population-count raster. They are
spatial aggregates of estimated residential population, not check-in counts.
Source: [European Commission, Joint Research Centre (JRC)](https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php).
The [JRC-managed AWS listing](https://registry.opendata.aws/ghs-pop/) states
CC BY 4.0. These derived data retain that attribution and license; see each
file's provenance for the source URL, version, crop checksum and modification.

Schiavina M., Freire S., Carioli A., MacManus K. (2023).
*GHS-POP R2023A – GHS population grid multitemporal (1975–2030).* European
Commission, Joint Research Centre. DOI:
[10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE](https://doi.org/10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE).
