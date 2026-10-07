import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'poopdeck-animals',
  title: "Marsh harrier, Montagu's harrier and spoonbill migration (poopdeck animals archive)",
  description:
    "101 animal-years of GPS tracks (42 tagged birds from Flanders, the Netherlands and the Belgium-Netherlands border) with a fix about every 2 hours, folded onto one calendar year. Western marsh harriers, Montagu's harriers and Eurasian spoonbills, from the Low Countries to the Sahel and the Atlantic coast of France and Iberia.",
  license: 'CC0 1.0 Universal (all five source datasets, checked in the GBIF dataset API)',
  attribution:
    'INBO LifeWatch Belgium via GBIF: MH_WATERLAND (doi:10.5281/zenodo.10053583), H_GRONINGEN (doi:10.5281/zenodo.10053658), MH_ANTWERPEN (doi:10.5281/zenodo.10054153), BOP_RODENT (doi:10.5281/zenodo.17310324), SPOONBILL_VLAANDEREN (doi:10.5281/zenodo.23103735). Folded tracks from the poopdeck.gl animals archive.',
  sourceUrl: 'https://www.gbif.org/dataset/66e0553e-75f6-49de-b614-22efd9fbf6e9',
  approxBytes: 2_720_000,
  bbox: [-17.29, 6.76, 15.78, 58.28]
} satisfies DatasetInfo;
