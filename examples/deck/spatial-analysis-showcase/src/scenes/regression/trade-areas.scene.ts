// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene, type LegendSpec} from '../scene';
import {REGION_PALETTE} from './b6-colors';
import type {TradeAreasOptions} from './trade-areas.compute';

const FACILITY_NOUN: Record<TradeAreasOptions['facility'], string> = {
  grocery: 'grocery stores',
  clinic: 'clinics and doctors’ offices',
  hospital: 'hospitals',
  library: 'public libraries',
  school: 'CPS schools'
};

/** Floating catchment accessibility and Huff trade areas for Chicago census tracts. */
export default defineScene<TradeAreasOptions>({
  id: 'trade-areas',
  title: 'Chicago facility access per 1,000 residents',
  chapter: 'regression',
  order: 4,
  summary:
    'Floating catchment and Huff models combine Chicago tracts with mapped facilities to estimate access per 1,000 demand units and modal trade areas; straight-line centroids and site counts approximate travel and capacity.',
  contributors: ['GPUCatchmentAccessibility', 'GPUHuffTradeAreas', 'GPUNeighborSearch'],
  datasets: [
    {id: 'chicago-tracts', role: 'demand: population counts per tract'},
    {id: 'chicago-places', role: 'grocery stores and clinics (Overture Maps)'},
    {id: 'chicago-facilities', role: 'hospitals, libraries and CPS schools'}
  ],
  initialView: {longitude: -87.68, latitude: 41.84, zoom: 9.9},
  basemap: ground('night'),
  furniture: {
    title: {
      title: 'Chicago facility access',
      subtitle: 'Floating catchments and Huff trade areas'
    },
    scaleBar: {units: 'metric'},
    credit: 'US Census Bureau ACS; Overture Maps Foundation; City of Chicago',
    caveat: 'Straight-line centroid distances and facility counts approximate travel and capacity.'
  },

  options: [
    {
      kind: 'select',
      id: 'facility',
      label: 'Facility (supply)',
      group: 'Supply and demand',
      apply: 'param',
      default: 'grocery',
      help: 'Which facilities supply the service. The choice is a mask over the facility rows, so the same graph is re-encoded.',
      options: [
        {value: 'grocery', label: 'Grocery stores (Overture)', help: 'About 1,250 open stores.'},
        {
          value: 'clinic',
          label: 'Clinics and doctors (Overture)',
          help: 'Over 12,000 listings; use a smaller bandwidth.'
        },
        {
          value: 'hospital',
          label: 'Hospitals (approximate)',
          help: '53 hospitals; try a bandwidth of several kilometres.'
        },
        {value: 'library', label: 'Public libraries', help: '82 Chicago Public Library branches.'},
        {value: 'school', label: 'CPS schools', help: 'About 650 schools, charters included.'}
      ]
    },
    {
      kind: 'select',
      id: 'demand',
      label: 'Demand (who needs it)',
      group: 'Supply and demand',
      apply: 'param',
      default: 'population',
      help: 'The count at each tract centroid that competes for supply. The ratio of supply to this demand is the accessibility.',
      options: [
        {value: 'population', label: 'All residents'},
        {
          value: 'noVehicle',
          label: 'Households without a vehicle',
          help: 'Who cannot drive to a store.'
        },
        {value: 'seniors', label: 'Residents aged 65+'},
        {value: 'children', label: 'Residents under 18'},
        {value: 'uninsured', label: 'Uninsured residents'}
      ]
    },
    {
      kind: 'select',
      id: 'supply',
      label: 'Facility size (supply and attractiveness)',
      group: 'Supply and demand',
      apply: 'param',
      default: 'uniform',
      help: 'Overture and the city list sites, not floor area or beds. Uniform counts every site as one. The proxies below are honest stand-ins, not measurements.',
      options: [
        {value: 'uniform', label: 'One per site'},
        {
          value: 'confidence',
          label: 'Overture confidence',
          help: 'The listing’s match confidence (0.5 to 1): a reliability weight, not a size.'
        },
        {
          value: 'agglomeration',
          label: 'Listings within 150 m',
          help: 'Counts same-kind listings nearby: malls, medical campuses and school complexes weigh more.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'decay',
      label: 'Distance decay',
      group: 'Catchment',
      apply: 'param',
      default: 'binary',
      help: 'How access falls with distance inside the catchment. The weights are written by GPUNeighborSearch, so changing them is a parameter write.',
      options: [
        {
          value: 'binary',
          label: 'Binary (inside = 1)',
          help: 'The classic two-step floating catchment.'
        },
        {
          value: 'gaussian',
          label: 'Gaussian',
          help: 'Smooth fall-off with the bandwidth as its width.'
        },
        {
          value: 'triangular',
          label: 'Linear (triangular)',
          help: '1 at the facility, 0 at the bandwidth.'
        },
        {value: 'bisquare', label: 'Bisquare'},
        {
          value: 'inverse',
          label: 'Inverse distance power',
          help: 'w = max(d, 100 m) to the minus power; the Huff-style decay.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'bandwidth',
      label: 'Catchment radius',
      group: 'Catchment',
      apply: 'param',
      min: 400,
      max: 8000,
      step: 200,
      default: 1600,
      unit: 'm',
      help: 'Distance from a tract centroid within which facilities are reachable (1,600 m is about a mile, a 20-minute walk). Also the Huff cut-off.'
    },
    {
      kind: 'slider',
      id: 'power',
      label: 'Inverse-distance power',
      group: 'Catchment',
      apply: 'param',
      min: 0.5,
      max: 4,
      step: 0.25,
      default: 2,
      disabledWhen: state => state.decay !== 'inverse',
      help: 'Exponent of the distance decay when the decay is inverse distance.'
    },
    {
      kind: 'slider',
      id: 'alpha',
      label: 'Huff attractiveness exponent α',
      group: 'Huff model',
      apply: 'param',
      min: 0.25,
      max: 4,
      step: 0.25,
      default: 1,
      format: value => value.toFixed(2),
      help: 'p_ij is proportional to A_j^α times the decay. Above 1 the biggest facilities win disproportionately; below 1 choice spreads out. It only matters when sites differ in size.'
    },
    {
      kind: 'select',
      id: 'map',
      label: 'Color tracts by',
      group: 'Display',
      apply: 'param',
      default: 'access2sfca',
      help: 'The tract fill. Accessibility is facility-equivalents per resident, shown per 1,000.',
      options: [
        {value: 'access2sfca', label: '2SFCA accessibility'},
        {value: 'access3sfca', label: '3SFCA accessibility'},
        {value: 'reachable', label: 'Facilities within reach (count)'},
        {value: 'huffProbability', label: 'Huff probability of the modal facility'},
        {value: 'tradeArea', label: 'Huff trade area (modal facility)'}
      ]
    },
    {
      kind: 'select',
      id: 'facilityView',
      label: 'Facility discs show',
      group: 'Display',
      apply: 'param',
      default: 'expectedDemand',
      help: 'Disc area and color of each facility.',
      options: [
        {value: 'expectedDemand', label: 'Huff expected demand'},
        {value: 'ratio', label: '2SFCA supply-to-demand ratio R_j'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showSpokes',
      label: 'Trade-area spokes',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draws a line from every tract to its most probable facility, faded by the Huff probability.'
    },
    {
      kind: 'toggle',
      id: 'showFacilities',
      label: 'Facilities',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws the selected facilities with demand within reach.'
    }
  ],

  readouts: [
    {id: 'demand', label: 'Demand'},
    {id: 'facilities', label: 'Facilities'},
    {
      id: 'meanAccess',
      label: 'Demand-weighted mean access',
      help: 'Facility-equivalents per 1,000 of the chosen demand, averaged over demand.'
    },
    {
      id: 'unreached',
      label: 'Reaching no facility',
      help: 'Demand in tracts whose catchment holds no facility.'
    },
    {
      id: 'equity',
      label: 'Equity gap (3SFCA)',
      help: 'Demand-weighted mean access in high-poverty tracts (150% poverty rate of 30% or more) versus low-poverty tracts (under 10%).'
    },
    {
      id: 'conservation',
      label: 'Supply conservation',
      help: 'Σ demand × access divided by total supply. Equal to 1 for 2SFCA when the same decay is used in both directions (every unit of supply is allocated exactly once).'
    },
    {
      id: 'captured',
      label: 'Huff demand captured',
      help: 'Expected demand summed over facilities, out of the demand that reaches any facility.'
    },
    {id: 'overflow', label: 'Weights capacity'}
  ],

  legends: state => {
    const entries: LegendSpec[] = [];
    if (state.map === 'access2sfca' || state.map === 'access3sfca') {
      entries.push({
        kind: 'ramp',
        id: 'access',
        title: `${state.map === 'access2sfca' ? '2SFCA' : '3SFCA'} accessibility to ${FACILITY_NOUN[state.facility]}`,
        ramp: 'ylgnbu',
        extent: 'gpu',
        unit: 'facility-equivalents per 1,000',
        labels: ['poor access', '95th percentile'],
        format: value => (1000 * value).toFixed(2)
      });
    } else if (state.map === 'huffProbability') {
      entries.push({
        kind: 'ramp',
        title: 'Huff probability of the modal facility',
        ramp: 'ylgnbu',
        extent: [0, 1],
        labels: ['0: choice is split', '1: one clear choice']
      });
    } else if (state.map === 'reachable') {
      entries.push({
        kind: 'ramp',
        title: `${FACILITY_NOUN[state.facility]} within reach`,
        ramp: 'cividis',
        extent: [0, 12],
        labels: ['0', '12 or more']
      });
    } else {
      entries.push({
        kind: 'categories',
        title: 'Huff trade area',
        entries: REGION_PALETTE.slice(0, 5).map((color, index) => ({
          color,
          label: `Trade-area color ${index + 1}`
        })),
        note: 'Tracts with the same modal facility share a color; touching areas never repeat.'
      });
    }
    if (state.showFacilities) {
      entries.push({
        kind: 'ramp',
        id: 'expected',
        title:
          state.facilityView === 'expectedDemand'
            ? 'Facilities (disc area = Huff expected demand)'
            : 'Facilities (disc area = supply per demand R_j)',
        ramp: 'inferno',
        extent: [0, 1],
        labels: ['little', 'much']
      });
    }
    return entries;
  },

  snippet: state => `import {
  GPUNeighborSearch, GPUCatchmentAccessibility, GPUHuffTradeAreas, getGPUNeighborSearchParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';

// Weights of both directions: demand rows list facilities in reach, facility rows list demand.
// Decay (${state.decay}) and radius (${state.bandwidth} m) live in a parameter buffer.
graph.add(new GPUNeighborSearch({mode: 'radius', positions: facilities, queryPositions: demand,
  mask: facilityMask, parameters, gridSize: [64, 64], weights: demandWeights, overflow}));
graph.add(new GPUNeighborSearch({mode: 'radius', positions: demand, queryPositions: facilities,
  queryMask: facilityMask, parameters, gridSize: [64, 64], weights: facilityWeights, overflow}));

// Two- and three-step floating catchment accessibility (access.two_step_fca)
graph.add(new GPUCatchmentAccessibility({
  method: '2sfca', supply, demand, facilityWeights, demandWeights,
  accessibility, ratios, reachableFacilities
}));
graph.add(new GPUCatchmentAccessibility({method: '3sfca', supply, demand, facilityWeights, demandWeights, accessibility: access3}));

// Huff trade areas: p_ij = A_j^alpha w_ij / sum_k A_k^alpha w_ik
graph.add(new GPUHuffTradeAreas({
  attractiveness: supply, demandWeights, facilityWeights, demand,
  parameters: alphaParameter,           // [${state.alpha}]: a one-float per-frame buffer
  tradeArea, tradeAreaProbability, expectedDemand
}));

parameters.write(getGPUNeighborSearchParameterValues({
  bounds, radius: ${state.bandwidth}, weightKind: '${state.decay === 'binary' ? 'binary' : state.decay === 'inverse' ? 'inverseDistance' : 'kernel'}'${state.decay === 'inverse' ? `, power: ${state.power}, distanceFloor: 100` : state.decay === 'binary' ? '' : `, kernel: '${state.decay}'`}
}));`,

  about: {
    what: '`GPUNeighborSearch` finds the facilities within reach of every tract and the tracts within reach of every facility. `GPUCatchmentAccessibility` computes two-step and three-step floating catchment accessibility: each facility’s supply is divided by the demand that can reach it (step 1), then each tract sums the ratios of the facilities it can reach (step 2). `GPUHuffTradeAreas` computes the probability that a tract uses each facility and the modal trade area.',
    why: 'Counting stores within a radius ignores competition: a tract with one store next to a dense, store-less district has less access than it seems. Floating catchments conserve supply, so under-served areas are real shortfalls, and the Huff model shows which store each neighbourhood actually patronises.',
    howToRead:
      'On the accessibility maps, dark tracts have little supply for their demand. 3SFCA additionally discounts facilities that many closer facilities compete for. The facility discs show where the demand ends up (Huff) or how thinly each is spread (R_j). The equity readout compares high- and low-poverty tracts.'
  },

  create: async ctx => (await import('./trade-areas.compute')).createTradeAreas(ctx),

  story: [
    {
      id: 'question',
      title: 'Which Chicago neighbourhoods can walk to a grocery store?',
      headline: 'Some tracts reach no grocery stores',
      textAlternative:
        'Chicago tracts are shaded by the number of grocery stores within 1,600 meters, including tracts with none.',
      body: 'Food access is a classic planning question: who lives far from a supermarket? A first answer counts the stores within reach. The map shows how many of about 1,250 open **grocery stores** (Overture Maps; **Facility (supply)** below) fall within **1,600 m of each census tract’s centroid** (**Catchment radius**), a roughly 20-minute walk.\n\nDark tracts have none. But counting stores treats a lone store the same whether it serves 500 residents or 50,000. The next steps fix that.',
      options: {map: 'reachable', facility: 'grocery', bandwidth: 1600, decay: 'binary'},
      camera: {longitude: -87.68, latitude: 41.84, zoom: 9.9},
      controls: ['facility', 'bandwidth'],
      readouts: ['facilities', 'unreached']
    },
    {
      id: 'two-step',
      title: 'Two steps: supply per demand, then access',
      headline: 'Supply allocation exposes demand-adjusted shortages',
      textAlternative:
        'A sequential tract map shows two-step floating-catchment access as facility-equivalents per 1,000 residents.',
      body: '**`GPUCatchmentAccessibility`** with the **2SFCA** method (Luo and Wang 2003; **Color tracts by** *2SFCA accessibility*). Step 1: for each store, divide its supply by the **demand of all tracts that can reach it**: `R_j = S_j / Σ P_i`. Step 2: each tract adds up the ratios of the stores it can reach: `A_i = Σ R_j`. Accessibility is therefore stores per resident, and total supply is conserved. The legend is per 1,000 residents.\n\nTwo **`GPUNeighborSearch`** runs write the weights in both directions on the GPU. See **Supply conservation**: 1.0 means every store’s supply was allocated exactly once.',
      options: {map: 'access2sfca'},
      controls: ['map'],
      readouts: ['conservation', 'meanAccess']
    },
    {
      id: 'decay',
      title: 'Distance should matter inside the catchment',
      headline: 'Distance decay lowers peripheral facility access',
      textAlternative:
        'Tract accessibility declines where reachable grocery stores lie near the edge of the Gaussian catchment.',
      body: 'A store at the edge of the radius counts as much as one next door. **Distance decay** is now *Gaussian*: access falls smoothly with distance, and the demand sharing each store also decays. Try *Linear* too, or *Inverse distance power* and then its **Inverse-distance power** exponent. Moving the **Catchment radius** slider rewrites the search parameters and re-runs the same compiled graph: nothing is rebuilt.',
      options: {map: 'access2sfca', decay: 'gaussian'},
      controls: ['decay', 'power', 'bandwidth'],
      readouts: ['meanAccess']
    },
    {
      id: 'three-step',
      title: 'Three steps: stores compete with each other',
      headline: 'Competition redistributes access among nearby stores',
      textAlternative:
        'Three-step floating-catchment colors show access after each tract distributes demand among competing nearby stores.',
      body: '2SFCA lets every store in the catchment count fully, however many nearer stores the tract also has. **3SFCA** (Wan et al. 2012) adds a **Huff-style selection weight**: a tract divides its demand among the stores it can reach in proportion to their closeness, so close stores take most of it. **Color tracts by** is now *3SFCA accessibility*. Dense cores of stores share demand more fairly, and tracts that only reach one far-away store score lower. The equity readout compares high- and low-poverty tracts.',
      options: {map: 'access3sfca', decay: 'gaussian'},
      controls: ['map', 'decay'],
      readouts: ['equity']
    },
    {
      id: 'demand',
      title: 'Who is demand? Households without a car',
      headline: 'Car-free demand shifts low-access tracts',
      textAlternative:
        'Accessibility is recalculated and mapped using households without vehicles rather than total population as demand.',
      body: 'Demand need not be everyone. **Demand** is now *Households without a vehicle*: each store’s supply is divided only among the households who must walk or take transit. The pattern shifts toward the South and West sides, where car-free households are concentrated. The **Equity gap (3SFCA)** readout reports the demand-weighted access in tracts with at least 30 percent poverty against those under 10 percent. Try seniors, children and uninsured residents with *Clinics and doctors (Overture)* as the **Facility (supply)**.',
      options: {demand: 'noVehicle'},
      controls: ['demand', 'facility'],
      readouts: ['equity']
    },
    {
      id: 'huff',
      title: 'Where will people actually shop? Huff trade areas',
      headline: 'Huff probabilities partition tracts by likely facility',
      textAlternative:
        'Categorical tract colors and spokes identify each tract’s highest-probability grocery store and its expected demand.',
      body: '**`GPUHuffTradeAreas`** gives each tract the probability of choosing every reachable store: `p_ij = A_j^α · w_ij / Σ_k A_k^α · w_ik`. The tract fill is now the **modal store** (its most probable choice) and the spokes link each tract to it, so the map divides into **trade areas**. The discs (**Facility discs show**) show each store’s **expected demand**: the sum of the demand it is expected to capture.\n\nThe exponent **α** only matters when stores differ in size: set **Facility size (supply and attractiveness)** to *Listings within 150 m* and raise **Huff attractiveness exponent α** to let agglomerated stores win more of their neighbours.',
      options: {map: 'tradeArea', showSpokes: true, decay: 'inverse'},
      controls: ['map', 'showSpokes', 'facilityView', 'supply', 'alpha'],
      readouts: ['captured']
    },
    {
      id: 'try',
      title: 'Try other services, and know the limits',
      headline: 'Hospital catchments leave fewer tracts unreached',
      textAlternative:
        'Chicago tracts are shaded by three-step hospital access within a five-kilometer straight-line catchment.',
      body: 'Switch **Facility (supply)** to *Hospitals (approximate)* (about 53) and raise the **Catchment radius** to 5,000 m or more: hospital access is a regional question. Try libraries or CPS schools with *Residents under 18* as **Demand (who needs it)**.\n\n**Limits:** distances are straight lines between tract centroids, not walking or driving time (network catchments are in the Networks chapter); supply is a count of sites because open data has no floor area or bed counts; demand sits at one point per tract; and Overture’s coverage varies by neighbourhood, so a gap may be a data gap. Reference: `access` `two_step_fca`, Luo and Wang (2003), Wan et al. (2012), CARTO Huff.',
      options: {
        facility: 'hospital',
        bandwidth: 5000,
        decay: 'gaussian',
        map: 'access3sfca',
        showSpokes: false
      },
      controls: ['facility', 'bandwidth', 'demand'],
      readouts: ['unreached']
    }
  ]
});
