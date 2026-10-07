// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Tract variable moved by the change-of-support scene. */
export type CosVariableId =
  | 'population'
  | 'observations'
  | 'jobs'
  | 'introducedShare'
  | 'poverty'
  | 'noVehicle'
  | 'age65'
  | 'diabetes'
  | 'income'
  | 'dominantGroup';

/** Target zone system. */
export type CosTarget = 'hexagon' | 'square' | 'triangle' | 'community';

/** How the transferred value is computed from the weights. */
export type CosRule = 'conserving' | 'area-mean' | 'naive';

/** What the map shows. */
export type CosView = 'source' | 'target' | 'surface';

/** Static description of one variable. Data is read from `chicago-tracts` by column name. */
export type CosVariableMeta = {
  id: CosVariableId;
  label: string;
  help: string;
  /** `count`: a total; `ratio`: numerator / denominator times `scale`; `category`: a class label. */
  kind: 'count' | 'ratio' | 'category';
  /** What a count or rate measures, used in legend titles. */
  unit: string;
};

/** The variables, in panel order. */
export const COS_VARIABLES: readonly CosVariableMeta[] = [
  {
    id: 'population',
    label: 'Residents (count)',
    help: 'Census residents per tract: an extensive total.',
    kind: 'count',
    unit: 'residents'
  },
  {
    id: 'observations',
    label: 'Nature observations 2023 (count)',
    help: 'iNaturalist observations of wild life located in the tract. An extensive count, extremely skewed: parks and the lakefront hold thousands, most residential tracts a handful.',
    kind: 'count',
    unit: 'nature observations'
  },
  {
    id: 'jobs',
    label: 'Jobs by workplace (count)',
    help: 'LEHD LODES 2021 jobs at workplaces in the tract: extremely concentrated downtown.',
    kind: 'count',
    unit: 'jobs'
  },
  {
    id: 'introducedShare',
    label: 'Introduced share of observations (rate)',
    help: 'Ratio of two extensive transfers: observations of non-native taxa over all observations. Tracts with no observation use the citywide share, 15.7%.',
    kind: 'ratio',
    unit: '% of observations of introduced taxa'
  },
  {
    id: 'poverty',
    label: 'Poverty, below 150% of the line (rate)',
    help: 'SVI 2022 persons below 150% of poverty over residents.',
    kind: 'ratio',
    unit: '% of residents below 150% of poverty'
  },
  {
    id: 'noVehicle',
    label: 'Households without a vehicle (rate)',
    help: 'Households without a vehicle over households.',
    kind: 'ratio',
    unit: '% of households without a vehicle'
  },
  {
    id: 'age65',
    label: 'Residents aged 65 or older (rate)',
    help: 'Residents aged 65+ over residents.',
    kind: 'ratio',
    unit: '% of residents aged 65+'
  },
  {
    id: 'diabetes',
    label: 'Diabetes prevalence (rate)',
    help: 'CDC PLACES model-based prevalence among adults, weighted by residents. Three tracts with no estimate use the citywide mean.',
    kind: 'ratio',
    unit: '% of adults with diabetes'
  },
  {
    id: 'income',
    label: 'Per-capita income (rate)',
    help: 'ACS per-capita income: total income (income × residents) over residents. Three tracts with no estimate use the citywide mean.',
    kind: 'ratio',
    unit: 'US$ per capita'
  },
  {
    id: 'dominantGroup',
    label: 'Largest race/ethnicity group (category)',
    help: 'The largest of five groups in each tract. A categorical source variable: the target shows the group with the largest overlap area.',
    kind: 'category',
    unit: 'group'
  }
];

/** The five race/ethnicity groups of the categorical variable. */
export const COS_GROUPS = [
  {id: 'nhWhite', label: 'White (non-Hispanic)', color: [86, 180, 233, 255]},
  {id: 'nhBlack', label: 'Black (non-Hispanic)', color: [213, 94, 0, 255]},
  {id: 'hispanic', label: 'Hispanic or Latino', color: [0, 158, 115, 255]},
  {id: 'nhAsian', label: 'Asian (non-Hispanic)', color: [240, 228, 66, 255]},
  {id: 'nhOther', label: 'Other or multiple (non-Hispanic)', color: [204, 121, 167, 255]}
] as const;

/** Looks up a variable's metadata. */
export function getCosVariable(id: CosVariableId): CosVariableMeta {
  return COS_VARIABLES.find(variable => variable.id === id) ?? COS_VARIABLES[0];
}

/** Legend title and unit of the value map for a state. */
export function getCosLegendText(state: {
  variable: CosVariableId;
  view: CosView;
  rule: CosRule;
  target: CosTarget;
}): {title: string; unit: string} {
  const meta = getCosVariable(state.variable);
  const view = state.view;
  if (meta.kind === 'category') {
    return {title: 'Largest group', unit: ''};
  }
  if (meta.kind === 'ratio') {
    const where =
      view === 'source' ? 'tract' : view === 'surface' ? 'smooth surface' : 'target zone';
    if (view === 'target' && state.rule === 'area-mean') {
      return {title: `Area-weighted mean of tract rates (${where})`, unit: meta.unit};
    }
    if (view === 'target' && state.rule === 'naive') {
      return {title: 'Rates summed as if they were counts (wrong)', unit: meta.unit};
    }
    return {title: `Value per ${where}`, unit: meta.unit};
  }
  if (view === 'source') return {title: `Tract total: ${meta.unit}`, unit: meta.unit};
  if (view === 'surface') {
    return {title: `Pycnophylactic density: ${meta.unit} per km²`, unit: 'per km²'};
  }
  if (state.rule === 'area-mean') {
    return {title: 'Area-weighted mean of tract densities', unit: `${meta.unit} per km²`};
  }
  if (state.rule === 'naive') {
    return {title: 'Densities summed as if they were counts (wrong)', unit: `${meta.unit} per km²`};
  }
  return {
    title: `${meta.unit.charAt(0).toUpperCase()}${meta.unit.slice(1)} per ${state.target === 'community' ? 'community area' : 'cell'}`,
    unit: meta.unit
  };
}
