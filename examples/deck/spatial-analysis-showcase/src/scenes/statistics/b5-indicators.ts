// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** One vulnerability indicator of a Chicago tract. */
export type TractIndicator = {
  id: string;
  label: string;
  /** What the unit is. */
  unit: string;
  /** Social vulnerability theme in the CDC/ATSDR SVI. */
  theme: string;
  /** 1 when higher is worse, -1 when higher is protective. */
  direction: 1 | -1;
};

/** The ten indicators combined by the vulnerability index, in the column order of the matrix. */
export const TRACT_INDICATORS: readonly TractIndicator[] = [
  {
    id: 'poverty150Pct',
    label: 'Poverty (below 150% of the line)',
    unit: '% of persons',
    theme: 'Socioeconomic status',
    direction: 1
  },
  {
    id: 'unemployedPct',
    label: 'Unemployment',
    unit: '% of civilians 16+',
    theme: 'Socioeconomic status',
    direction: 1
  },
  {
    id: 'noHighSchoolPct',
    label: 'No high school diploma',
    unit: '% of residents',
    theme: 'Socioeconomic status',
    direction: 1
  },
  {
    id: 'uninsuredPct',
    label: 'No health insurance',
    unit: '% of civilians',
    theme: 'Socioeconomic status',
    direction: 1
  },
  {
    id: 'housingBurdenPct',
    label: 'Housing cost burden',
    unit: '% of households',
    theme: 'Socioeconomic status',
    direction: 1
  },
  {
    id: 'perCapitaIncome',
    label: 'Per-capita income',
    unit: 'US dollars',
    theme: 'Socioeconomic status',
    direction: -1
  },
  {
    id: 'age65Pct',
    label: 'Aged 65 and over',
    unit: '% of persons',
    theme: 'Household characteristics',
    direction: 1
  },
  {
    id: 'disabilityPct',
    label: 'Disability',
    unit: '% of civilians',
    theme: 'Household characteristics',
    direction: 1
  },
  {
    id: 'limitedEnglishPct',
    label: 'Limited English',
    unit: '% of residents',
    theme: 'Household characteristics',
    direction: 1
  },
  {
    id: 'noVehiclePct',
    label: 'No vehicle',
    unit: '% of households',
    theme: 'Housing and transportation',
    direction: 1
  }
];

/** Variables whose inequality across tracts is measured within community areas. */
export const INEQUALITY_VARIABLES = [
  {id: 'perCapitaIncome', label: 'Per-capita income', unit: 'US dollars'},
  {id: 'medianHouseholdIncome', label: 'Median household income', unit: 'US dollars'},
  {id: 'natureObs2023', label: 'Nature observations 2023', unit: 'observations'},
  {id: 'jobsWac2021', label: 'Jobs located in the tract (2021)', unit: 'jobs'},
  {id: 'population', label: 'Population', unit: 'residents'}
] as const;

/** Per-community-area inequality measures. */
export const INEQUALITY_INDICES = [
  {
    id: 'gini',
    label: 'Gini coefficient',
    help: 'Area between the Lorenz curve and equality; 0 equal, 1 one tract holds everything.'
  },
  {id: 'theilT', label: 'Theil T', help: 'Sensitive to the top of the distribution.'},
  {
    id: 'theilL',
    label: 'Theil L (mean log deviation)',
    help: 'Sensitive to the bottom; undefined if a tract has zero.'
  },
  {
    id: 'atkinson',
    label: 'Atkinson (epsilon)',
    help: 'Inequality aversion epsilon: higher weighs the poorest tracts more.'
  },
  {
    id: 'hoover',
    label: 'Hoover (Robin Hood) index',
    help: 'Share of the total that would move to equalize.'
  },
  {
    id: 'palma',
    label: 'Palma ratio',
    help: 'Share held by the top tail over the share held by the bottom tail.'
  },
  {id: 'mean', label: 'Mean value', help: 'Population-weighted mean of the variable.'}
] as const;
