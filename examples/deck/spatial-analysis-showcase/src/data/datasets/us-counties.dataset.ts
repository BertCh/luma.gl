import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'us-counties',
  title: 'US counties: demographics, vulnerability and health',
  description:
    'Contiguous US counties (3,109, Census 1:20m boundaries with exact shared edges) with ACS population and race counts, CDC social vulnerability themes, CDC PLACES health prevalences, USDA rural-urban codes, median household income and unemployment 2000-2023.',
  license: 'US Government works (Census, CDC/ATSDR, CDC PLACES, USDA ERS): public domain',
  attribution:
    'US Census Bureau cartographic boundaries 2022; CDC/ATSDR Social Vulnerability Index 2022; CDC PLACES 2024 release; USDA Economic Research Service',
  sourceUrl:
    'https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html',
  approxBytes: 3550000,
  bbox: [-124.7258, 24.4981, -66.9499, 49.3844]
} satisfies DatasetInfo;
