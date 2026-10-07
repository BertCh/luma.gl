# chicago-taxi-od
Source: City of Chicago Data Portal, Taxi Trips (2013-2023), https://data.cityofchicago.org/resource/wrvz-psew (filtered to 2023).
Licence: City of Chicago Terms of Use (open public data). Attribution: City of Chicago.
Steps: `fetch.py RAW` (Socrata group-by aggregation per month), then `build.py RAW community-areas.geojson OUT`.
Community-area centroids come from the City's Boundaries - Community Areas (igwz-8jzy), cached in raw/chicago-community-areas.
