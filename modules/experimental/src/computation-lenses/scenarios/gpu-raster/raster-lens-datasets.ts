// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Dataset descriptors for raster and terrain lens registrations. Plain data: registering a
// scenario must never load the data files or GPU code.

/** Dixie Fire window near Greenville, California: 64 × 64 DEM at 120 m (UTM 10N). */
export const T_DIXIE_DEM = {
  id: 't-dixie-dem',
  licence: 'Public domain (USGS 3DEP); terrain tiles by Mapzen/Tilezen on AWS Open Data',
  source: 'AWS Terrain Tiles (USGS 3DEP), Dixie Fire window near Greenville CA, 120 m block means'
} as const;

/** Sentinel-2 NIR and SWIR before (2021-07-13) and after (2021-09-21) the Dixie Fire. */
export const T_DIXIE_BANDS = {
  id: 't-dixie-bands',
  licence: 'Contains modified Copernicus Sentinel data 2021 (free and open, Copernicus licence)',
  source: 'Copernicus Sentinel-2 L2A B08/B12, Dixie Fire window, 120 m block means'
} as const;

/** ESA WorldCover 2021 land-cover classes on the same window. */
export const T_DIXIE_LANDCOVER = {
  id: 't-dixie-landcover',
  licence: 'ESA WorldCover 2021 v200 (c) ESA, CC BY 4.0',
  source: 'ESA WorldCover 2021, Dixie Fire window, 120 m block mode'
} as const;

/** Final 2021 Dixie Fire perimeter clipped to the window (UTM 10N metres). */
export const T_DIXIE_PERIMETER = {
  id: 't-dixie-perimeter',
  licence: 'Public domain (US Government work, NIFC)',
  source: 'NIFC final 2021 Dixie Fire perimeter, clipped to the Greenville window'
} as const;

/** NOAA GFS 10 m wind over the central US, 2024-09-26 12Z, 0.25° lon/lat cells. */
export const T_GFS_WIND = {
  id: 't-gfs-wind',
  licence: 'Public domain (NOAA)',
  source: 'NOAA GFS 0.25° 10 m wind, 2024-09-26T12Z, 64 × 48 crop (8-bit packaged)'
} as const;

/** Raw Terrarium RGB of a 32 × 32 block of the Dixie Fire DEM at 20 m. */
export const T_DIXIE_TERRARIUM = {
  id: 't-dixie-terrarium',
  licence: 'Public domain (USGS 3DEP); terrain tiles by Mapzen/Tilezen on AWS Open Data',
  source: 'AWS Terrain Tiles Terrarium RGB, 32 × 32 block near Greenville CA, 20 m'
} as const;
