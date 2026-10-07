// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** A numeric county column offered by the statistics scenes. */
export type CountyVariable = {
  /** Column name in the `us-counties` dataset. */
  id: string;
  label: string;
  /** Unit for legends and readouts. */
  unit: string;
  /** What higher values mean, for story text. */
  note: string;
};

/** County variables used by the classification and similarity scenes. */
export const COUNTY_VARIABLES: readonly CountyVariable[] = [
  {
    id: 'popDensity',
    label: 'Population density',
    unit: 'people per km²',
    note: 'extremely right-skewed: a few dense counties dominate'
  },
  {id: 'population', label: 'Population', unit: 'residents', note: 'extremely right-skewed'},
  {
    id: 'medianHouseholdIncome',
    label: 'Median household income',
    unit: 'US dollars',
    note: 'moderately skewed'
  },
  {
    id: 'places_diabetes_ageAdj',
    label: 'Diabetes prevalence',
    unit: '% of adults',
    note: 'age-adjusted, CDC PLACES'
  },
  {
    id: 'places_obesity_ageAdj',
    label: 'Obesity prevalence',
    unit: '% of adults',
    note: 'age-adjusted, CDC PLACES'
  },
  {
    id: 'poverty150',
    label: 'Below 150% of poverty line',
    unit: '% of persons',
    note: 'ACS 2018-2022'
  },
  {id: 'age65plus', label: 'Aged 65 and over', unit: '% of persons', note: 'ACS 2018-2022'},
  {
    id: 'minorityShare',
    label: 'Racial or ethnic minority',
    unit: '% of persons',
    note: 'all but non-Hispanic white alone'
  },
  {
    id: 'noInternet',
    label: 'Households without internet',
    unit: '% of households',
    note: 'ACS 2018-2022'
  },
  {id: 'mobileHomes', label: 'Mobile homes', unit: '% of housing units', note: 'ACS 2018-2022'},
  {id: 'unemploymentRate2023', label: 'Unemployment rate 2023', unit: '%', note: 'BLS LAUS'}
];

/** Looks up a variable by id. */
export function getCountyVariable(id: string): CountyVariable {
  return COUNTY_VARIABLES.find(variable => variable.id === id) ?? COUNTY_VARIABLES[0];
}

/** One attribute of the county similarity search. */
export type SimilarityAttribute = {
  id: string;
  label: string;
  unit: string;
  /** `'log10'` for heavy-tailed counts. */
  transform?: 'log10';
};

/** The twelve county attributes `GPUSimilarLocations` compares, in matrix column order. */
export const SIMILARITY_ATTRIBUTES: readonly SimilarityAttribute[] = [
  {id: 'medianHouseholdIncome', label: 'Median household income', unit: 'US dollars'},
  {id: 'poverty150', label: 'Poverty (below 150% of the line)', unit: '% of persons'},
  {id: 'popDensity', label: 'Population density (log)', unit: 'people per km²', transform: 'log10'},
  {id: 'age65plus', label: 'Aged 65 and over', unit: '% of persons'},
  {id: 'age17under', label: 'Aged 17 and under', unit: '% of persons'},
  {id: 'minorityShare', label: 'Racial or ethnic minority', unit: '% of persons'},
  {id: 'noHighSchool', label: 'No high school diploma', unit: '% of adults'},
  {id: 'noInternet', label: 'No internet subscription', unit: '% of households'},
  {id: 'mobileHomes', label: 'Mobile homes', unit: '% of housing units'},
  {id: 'multiUnit', label: 'Large apartment buildings', unit: '% of housing units'},
  {id: 'places_diabetes_ageAdj', label: 'Diabetes prevalence', unit: '% of adults'},
  {id: 'places_obesity_ageAdj', label: 'Obesity prevalence', unit: '% of adults'}
];

/** Reference counties the story can choose without a click: `[name, state]`. */
export const REFERENCE_PRESETS = [
  {
    value: 'loudoun',
    label: 'Loudoun County, VA (affluent DC suburb)',
    name: 'Loudoun',
    state: 'VA'
  },
  {value: 'cook', label: 'Cook County, IL (Chicago)', name: 'Cook', state: 'IL'},
  {
    value: 'mcdowell',
    label: 'McDowell County, WV (poorest in the data)',
    name: 'McDowell',
    state: 'WV'
  },
  {value: 'boulder', label: 'Boulder County, CO (college town)', name: 'Boulder', state: 'CO'},
  {value: 'maricopa', label: 'Maricopa County, AZ (Phoenix)', name: 'Maricopa', state: 'AZ'},
  {value: 'robeson', label: 'Robeson County, NC', name: 'Robeson', state: 'NC'},
  {value: 'story', label: 'Story County, IA (Ames)', name: 'Story', state: 'IA'}
] as const;
