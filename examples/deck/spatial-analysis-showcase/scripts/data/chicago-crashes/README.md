# chicago-crashes
Source: City of Chicago Data Portal, Traffic Crashes - Crashes, https://data.cityofchicago.org/resource/85ca-t3if (2023). Open data under the portal's Terms of Use (https://www.chicago.gov/city/en/narr/foia/data_disclaimer.html).
Run `fetch.py RAW_DIR`, build chicago-roads first (writes a cache pickle), then `build.py crashes2023.csv roads.pkl <roads out dir> OUT`.
