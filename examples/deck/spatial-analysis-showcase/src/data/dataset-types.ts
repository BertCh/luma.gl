// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Catalog metadata of one showcase dataset. Each `src/data/datasets/<id>.dataset.ts` default-exports
 * one of these; the `#/data` page and every scene's attribution drawer read it. The bytes themselves
 * are described by `public/data/<id>/manifest.json` (see `src/data/README.md`).
 */
export type DatasetInfo = {
  /** Kebab-case id; equals the folder name under `public/data/` and the dataset file name. */
  id: string;
  /** Human-readable title. */
  title: string;
  /** One or two sentences describing the data. */
  description: string;
  /** Licence name or terms of use. */
  license: string;
  /** Attribution line shown next to every scene that uses the dataset. */
  attribution: string;
  /** Page the data was obtained from. */
  sourceUrl: string;
  /** Approximate download size in bytes. */
  approxBytes: number;
  /** `[west, south, east, north]` in degrees. */
  bbox: [number, number, number, number];
};
