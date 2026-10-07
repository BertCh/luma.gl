// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {fetchText, getDataFileUrl, parseCsv} from '../../data/loaders';

/** Continent-level regions used to group OpenFlights airports. */
export const CONTINENT_NAMES = [
  'Europe',
  'Asia',
  'North America',
  'South America',
  'Africa',
  'Oceania'
] as const;

export type ContinentName = (typeof CONTINENT_NAMES)[number];

/**
 * Categorical colors, one per continent, in `CONTINENT_NAMES` order. The same list feeds the layer
 * palette and the legend. Chosen to stay legible on light and dark basemaps.
 */
export const CONTINENT_COLORS: readonly (readonly [number, number, number, number])[] = [
  [86, 156, 255, 255],
  [255, 140, 70, 255],
  [92, 212, 140, 255],
  [236, 200, 60, 255],
  [200, 120, 255, 255],
  [255, 100, 150, 255]
];

const COUNTRIES_BY_CONTINENT: Record<ContinentName, string> = {
  Europe:
    'Albania|Austria|Belarus|Belgium|Bosnia and Herzegovina|Bulgaria|Croatia|Cyprus|Czech Republic|Denmark|Estonia|Faroe Islands|Finland|France|Germany|Gibraltar|Greece|Guernsey|Hungary|Iceland|Ireland|Isle of Man|Italy|Jersey|Latvia|Lithuania|Luxembourg|Macedonia|Malta|Moldova|Montenegro|Netherlands|Norway|Poland|Portugal|Romania|Russia|Serbia|Slovakia|Slovenia|Spain|Sweden|Switzerland|Turkey|Ukraine|United Kingdom',
  Asia: 'Afghanistan|Armenia|Azerbaijan|Bahrain|Bangladesh|Bhutan|Brunei|Burma|Cambodia|China|Georgia|Hong Kong|India|Indonesia|Iran|Iraq|Israel|Japan|Jordan|Kazakhstan|Kuwait|Kyrgyzstan|Laos|Lebanon|Macau|Malaysia|Maldives|Mongolia|Nepal|North Korea|Oman|Pakistan|Philippines|Qatar|Saudi Arabia|Singapore|South Korea|Sri Lanka|Taiwan|Tajikistan|Thailand|East Timor|Turkmenistan|United Arab Emirates|Uzbekistan|Vietnam|Yemen|Christmas Island|Cocos (Keeling) Islands',
  'North America':
    'Anguilla|Antigua and Barbuda|Aruba|Bahamas|Barbados|Belize|Bermuda|British Virgin Islands|Canada|Cayman Islands|Costa Rica|Cuba|Dominica|Dominican Republic|El Salvador|Greenland|Grenada|Guadeloupe|Guatemala|Haiti|Honduras|Jamaica|Martinique|Mexico|Netherlands Antilles|Nicaragua|Panama|Puerto Rico|Saint Kitts and Nevis|Saint Lucia|Saint Pierre and Miquelon|Saint Vincent and the Grenadines|Trinidad and Tobago|Turks and Caicos Islands|United States|Virgin Islands',
  'South America':
    'Argentina|Bolivia|Brazil|Chile|Colombia|Ecuador|Falkland Islands|French Guiana|Guyana|Paraguay|Peru|Suriname|Uruguay|Venezuela',
  Africa:
    "Algeria|Angola|Benin|Botswana|Burkina Faso|Burundi|Cameroon|Cape Verde|Central African Republic|Chad|Comoros|Congo (Brazzaville)|Congo (Kinshasa)|Cote d'Ivoire|Djibouti|Egypt|Equatorial Guinea|Eritrea|Ethiopia|Gabon|Gambia|Ghana|Guinea|Guinea-Bissau|Kenya|Lesotho|Liberia|Libya|Madagascar|Malawi|Mali|Mauritania|Mauritius|Mayotte|Morocco|Mozambique|Namibia|Niger|Nigeria|Reunion|Rwanda|Sao Tome and Principe|Senegal|Seychelles|Sierra Leone|Somalia|South Africa|South Sudan|Sudan|Swaziland|Tanzania|Togo|Tunisia|Uganda|Western Sahara|Zambia|Zimbabwe",
  Oceania:
    'American Samoa|Australia|Cook Islands|Fiji|French Polynesia|Guam|Kiribati|Marshall Islands|Micronesia|Nauru|New Caledonia|New Zealand|Niue|Norfolk Island|Northern Mariana Islands|Palau|Papua New Guinea|Samoa|Solomon Islands|Tonga|Tuvalu|Vanuatu|Wallis and Futuna'
};

const CONTINENT_BY_COUNTRY = new Map<string, number>();
CONTINENT_NAMES.forEach((name, index) => {
  for (const country of COUNTRIES_BY_CONTINENT[name].split('|')) {
    CONTINENT_BY_COUNTRY.set(country, index);
  }
});

/** Index into {@link CONTINENT_NAMES} for an OpenFlights country name (Europe when unknown). */
export function getContinentIndex(country: string): number {
  return CONTINENT_BY_COUNTRY.get(country) ?? 0;
}

/** One row of a dataset's `airports.csv`: `index,iata,name,city,country`. */
export type AirportRecord = {iata: string; name: string; city: string; country: string};

/** Fetches and parses the `airports.csv` table that ships beside the flight datasets. */
export async function loadAirportTable(
  datasetId: string,
  signal?: AbortSignal
): Promise<AirportRecord[]> {
  const table = parseCsv(await fetchText(getDataFileUrl(datasetId, 'airports.csv'), signal));
  const column = (name: string) => table.header.indexOf(name);
  const [iata, name, city, country] = ['iata', 'name', 'city', 'country'].map(column);
  return table.rows.map(row => ({
    iata: row[iata],
    name: row[name],
    city: row[city],
    country: row[country]
  }));
}
