import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'chicago-tracts',
  title: 'Chicago census tracts with demographics, health and nature',
  description:
    '791 2020 census tracts inside Chicago (exact shared edges) with CDC/ATSDR SVI 2022, ACS income, CDC PLACES health measures, LODES jobs and 2023 iNaturalist observation counts.',
  license:
    'Public domain (US Census Bureau, CDC/ATSDR); nature counts from iNaturalist observations (CC0 / CC BY / CC BY-NC)',
  attribution:
    'US Census Bureau (TIGER/Cartographic Boundary, ACS), CDC/ATSDR SVI 2022, CDC PLACES, LEHD LODES, City of Chicago / CPD',
  sourceUrl: 'https://svi.cdc.gov/dataDownloads/data-download.html',
  approxBytes: 900535,
  bbox: [-87.94, 41.644, -87.524, 42.023]
} satisfies DatasetInfo;
