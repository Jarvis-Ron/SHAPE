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

## Public residential population (Home quotas)

`dataset/home_quotas/*.json` contains precomputed Home-allocation quotas derived
from the **GHS-POP R2023A 2010 residential population layer**, as confirmed by
the repository maintainer. Stage 3 uses these public population statistics as
a static prior. They are not inferred from target-city check-in trajectories.

Source: [European Commission Joint Research Centre, GHS-POP R2023A](https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php).
Dataset citation: Schiavina, M., Freire, S., Carioli, A., and MacManus, K. (2023).
*GHS-POP R2023A — GHS population grid multitemporal (1975–2030).* European
Commission, Joint Research Centre (JRC).
[DOI: 10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE](https://doi.org/10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE).
R2023A identifies the product release; 2010 identifies the population epoch.
Follow the provider's data-use and attribution terms.
