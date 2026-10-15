// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Dataset descriptors of the S2 lens data files (`computation-lenses/data/s2-*.json`). */
export const S2_DATASETS = {
  heleneRain: {
    id: 's2-helene-rain',
    licence: 'Public domain (NOAA NCEI GHCN-Daily, US Government work)',
    source:
      'ghcn-stations: NOAA NCEI GHCN-Daily via AWS Open Data (noaa-ghcn-pds), quality-flagged values removed'
  },
  roads: {
    id: 's2-chicago-north-roads',
    licence: 'ODbL 1.0',
    source: 'chicago-roads: \u00a9 OpenStreetMap contributors'
  },
  areas: {
    id: 's2-chicago-north-areas',
    licence: 'City of Chicago Data Portal Terms of Use',
    source:
      'chicago-community-areas: 77 official Chicago community areas; feature i has areaId i+1. Binary polygons are GeoArrow-style: polygon p (a polygon part) owns rings polygonRingOffsets[p]..[p+1], ring r owns vertices ringOffsets[r]..[r+1]; partFeature maps part -> feature index. Topology-preserving simplification (~4 m), shared edges identical.'
  },
  tracts: {
    id: 's2-chicago-north-tracts',
    licence:
      'Public domain: US Census Bureau cartographic boundaries and ACS, CDC/ATSDR SVI 2022, CDC PLACES; natureObs/birdObs counts derive from iNaturalist observations (CC0 / CC BY 4.0 / CC BY-NC 4.0, iNaturalist contributors)',
    source:
      'chicago-tracts: {"boundaries":"US Census Bureau cartographic boundary file cb_2022_17_tract_500k, public domain","svi":"CDC/ATSDR SVI 2022, public domain","places":"CDC PLACES, public domain","acs":"US Census Bureau ACS, public domain","nature":"iNaturalist observations, CC0 / CC BY 4.0 / CC BY-NC 4.0 per observation (iNaturalist contributors)"}'
  },
  parks: {
    id: 's2-chicago-north-parks',
    licence: 'ODbL 1.0',
    source: 'chicago-parks: OpenStreetMap contributors via the Overpass API, downloaded 2026-10-07'
  },
  facilities: {
    id: 's2-chicago-north-facilities',
    licence:
      'City of Chicago Data Portal Terms of Use (libraries x8fc-8rcq, CPS hexd-c4gn, fire stations 28km-gtjn); hospitals from Overture Maps (CDLA-Permissive-2.0)',
    source:
      'chicago-facilities: Public facilities in Chicago for location-allocation / coverage: hospitals (Overture places, de-duplicated within 300 m, approximate), Chicago Public Library branches, CPS schools (SY2024-25 locations), CFD fire stations. names.csv rows align with the columns.'
  },
  counties: {
    id: 's2-midwest-counties',
    licence:
      'Public domain: US Census Bureau cb_2022 boundaries, CDC/ATSDR SVI 2022, USDA ERS / BLS LAUS unemployment; traffic deaths: NHTSA FARS 2017-2023',
    source:
      'us-counties: 437 counties of IL, IN, OH, MI and WI (Census cb_2022_us_county_20m; USDA ERS / BLS LAUS unemployment rates 2008-2023, percent; NHTSA FARS traffic deaths 2017-2023; personYears is seven times the 2020 census population); queen contiguity computed from the full source geometry'
  },
  nature: {
    id: 's2-chicago-north-nature',
    licence: 'CC0 / CC BY 4.0 / CC BY-NC 4.0 per observation; attribute iNaturalist contributors',
    source: 'iNaturalist observations 2023, Chicago North Side (showcase chicago-nature)'
  },
  aisTracks: {
    id: 's3-ny-harbor-ais',
    licence: 'CC0 1.0',
    source:
      'AIS tracks of 12 moving vessels in Upper New York Bay, 2024-06-12 14:00-16:00 UTC (showcase public/data/ais-vessels, NOAA / USCG Nationwide AIS 2024); zones are NOAA ENC anchorage areas (s3-ny-harbor-zones, public domain)'
  }
} as const;
