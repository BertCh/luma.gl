// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Value set drawn as dots. */
export type DotValueSet = 'race' | 'poverty' | 'vehicle';

/** One category of a dot value set: a tract column and a color. */
export type DotCategory = {
  /** Tract column holding the count. */
  column: string;
  label: string;
  color: readonly [number, number, number, number];
};

/** The value sets of the dot-density scene. At most five categories (compile-time). */
export const DOT_VALUE_SETS: Record<
  DotValueSet,
  {label: string; unit: string; categories: readonly DotCategory[]; help: string}
> = {
  race: {
    label: 'Race and ethnicity',
    unit: 'residents',
    help: 'Five groups from the 2020 Census (CDC SVI 2022 file). Every dot is residents of one group.',
    categories: [
      {column: 'nhWhite', label: 'White (non-Hispanic)', color: [86, 180, 233, 255]},
      {column: 'nhBlack', label: 'Black (non-Hispanic)', color: [213, 94, 0, 255]},
      {column: 'hispanic', label: 'Hispanic or Latino', color: [0, 158, 115, 255]},
      {column: 'nhAsian', label: 'Asian (non-Hispanic)', color: [240, 228, 66, 255]},
      {column: 'nhOther', label: 'Other or multiple (non-Hispanic)', color: [204, 121, 167, 255]}
    ]
  },
  poverty: {
    label: 'Poverty status',
    unit: 'residents',
    help: 'Residents below and above 150% of the poverty line (ACS 2018–2022).',
    categories: [
      {column: 'poverty150', label: 'Below 150% of the poverty line', color: [230, 97, 1, 255]},
      {column: 'aboveP150', label: 'At or above 150%', color: [94, 140, 202, 255]}
    ]
  },
  vehicle: {
    label: 'Household vehicle access',
    unit: 'households',
    help: 'Households with and without a vehicle (ACS 2018–2022).',
    categories: [
      {column: 'noVehicle', label: 'No vehicle', color: [213, 94, 0, 255]},
      {column: 'withVehicle', label: 'At least one vehicle', color: [0, 114, 178, 255]}
    ]
  }
};
