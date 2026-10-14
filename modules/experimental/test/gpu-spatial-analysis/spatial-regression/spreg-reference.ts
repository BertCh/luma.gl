// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Reference values from PySAL (spreg 1.9.1, esda 2.10, libpysal 4.15; float64) for the seeded
 * scenes of the spatial regression specs. Each `[statistic, p-value]` pair is spreg's own output:
 * - `lattice`: `createDiagnosticsScene(8, 11, 0.6, true)` through `spreg.OLS(y, X, w, spat_diag=True, moran=True)`.
 * - `nearestNeighbor`: `createNearestNeighborScene(80, 4, 7, 0.6)`, same call.
 * - `twoStage`: `createLagScene(12, 21, 0.5, true)` through `spreg.GM_Lag(y, X, w, w_lags=1, spat_diag=True)`.
 * The weights are built in TypeScript, passed to spreg as `libpysal.weights.W` with these exact
 * (float32-rounded) values, so both sides see identical inputs.
 */
export const SPREG_DIAGNOSTICS_REFERENCE = {
  lattice: {
    lmLag: [2.1781171429819812, 0.13998562738452128],
    lmError: [10.110328451778033, 0.0014744068475693495],
    robustLmLag: [3.880200012617102, 0.048858736427866846],
    robustLmError: [11.812411321413155, 0.0005883718118910642],
    lmSarma: [13.990528464395135, 0.0009162106685826607],
    moran: [0.30469178193233465, 3.4236534610027003, 0.0006178535529915366], // [I, z, p]
    sigmaSquared: 0.40664551912699204
  },
  nearestNeighbor: {
    lmLag: [3.303642877429317, 0.06912642274886084],
    lmError: [24.549263237941155, 7.243403257387193e-7],
    robustLmLag: [0.5922562789502267, 0.44154795356517484],
    robustLmError: [21.83787663946206, 2.966852049155504e-6],
    lmSarma: [25.141519516891382, 3.4720694501400972e-6],
    moran: [0.3554255772781454, 5.272691153564993, 1.3443769414986321e-7], // [I, z, p]
    sigmaSquared: 0.4458628104578354
  }
};

export const SPREG_TWO_STAGE_REFERENCE = {
  betas: [1.23824369342492, 1.969538194948548, -1.0142239608093329, 0.49540927749656305],
  standardErrors: [
    1.008114889586518, 0.03275027576602071, 0.07707459295869029, 0.03061655007352109
  ],
  z: [1.228276366330419, 60.13806445538382, -13.15899211239334, 16.18109409149337],
  pValues: [0.2193432397568228, 0.0, 1.5108785307455502e-39, 6.8565136650646785e-59],
  sigma2: 1.1888040616856232,
  pseudoRSquared: 0.9679751600832437,
  anselinKelejian: [1.9327826389521514, 0.16445433146956856]
};

/**
 * Directed (asymmetric pattern), row-standardized kNN k = 3 scenes from
 * `createNearestNeighborScene(count, 3, seed, 0.6, false)`, run through spreg 1.9.1 (libpysal 4.15.0)
 * with the weights passed untransformed (`W(neighbors, weights)`, transform 'O'):
 * - `diagnostics`: `spreg.OLS(y, X, w, spat_diag=True, moran=True)`; spreg's LM tests use
 *   `T = tr(W'W + WW)` from the sparse matrix, the same definition as the GPU (no deviation).
 * - `twoStage`: `spreg.GM_Lag(y, X, w, w_lags=1, spat_diag=True)`.
 * Scripts: units/pin-knn-spreg.py with units/dump-knn-scene.ts.
 */
export const SPREG_ASYMMETRIC_KNN_REFERENCE = {
  scene_60_13: {
    diagnostics: {
      lmLag: [4.5578249253324845, 0.03276866432898341],
      lmError: [3.0416369472371168, 0.08115404343037846],
      robustLmLag: [5.678514781890249, 0.017173899587959715],
      robustLmError: [4.162326803794881, 0.041332587135519794],
      lmSarma: [8.720151729127366, 0.012777418259503803],
      moran: [0.173863780314526, 2.005663556867764, 0.044892155894202694], // [I, z, p]
      sigmaSquared: 0.3771058430968138
    },
    twoStage: {
      betas: [2.401078692231522, 1.9573081784360493, -0.9686376747123155, -0.053487993500805686],
      standardErrors: [
        0.44257452427508526, 0.024982592575443576, 0.06974088347321429, 0.021582269834742672
      ],
      z: [5.42525283434325, 78.34687983343925, -13.889093835244354, -2.4783303104987535],
      sigma2: 0.3472710093189902,
      pseudoRSquared: 0.9903578570498213,
      anselinKelejian: [3.376942602308577, 0.06611459459741005]
    }
  },
  scene_90_29: {
    diagnostics: {
      lmLag: [0.7320308360333511, 0.3922257065160192],
      lmError: [34.51788316629333, 4.223530809698318e-9],
      robustLmLag: [0.05930084857315993, 0.8076044095313839],
      robustLmError: [33.84515317883314, 5.967758123188103e-9],
      lmSarma: [34.57718401486649, 3.1021292685077353e-8],
      moran: [0.47274757783875176, 6.105616642045846, 1.0240453535771756e-9], // [I, z, p]
      sigmaSquared: 0.4929339362195826
    },
    twoStage: {
      betas: [0.7689941341493522, 1.9944263552075263, -1.0315178811634595, -0.003626528559294684],
      standardErrors: [
        0.5345731230346307, 0.025705061346544295, 0.06536970953695867, 0.02582668991136801
      ],
      z: [1.4385200097303346, 77.5888580198098, -15.779753168097846, -0.1404178612025079],
      sigma2: 0.4943665785123574,
      pseudoRSquared: 0.9861658303259061,
      anselinKelejian: [28.15597425580343, 1.1192166022341022e-7]
    }
  }
};

/**
 * Instrument order 2 (`spreg.GM_Lag(y, X, w, w_lags=2, spat_diag=True)`, spreg 1.9.1, instruments
 * `[1, X, WX, W^2 X]`) on the same scenes as `twoStage` and the asymmetric kNN scenes.
 */
export const SPREG_TWO_STAGE_ORDER_TWO_REFERENCE = {
  lattice: {
    betas: [1.578911232993505, 1.970847045363378, -1.0153607343728748, 0.4846791386034681],
    standardErrors: [
      0.9744240368023686, 0.03272684588219068, 0.07704982710915466, 0.029515166865008467
    ],
    z: [1.6203533301321238, 60.22111181926869, -13.177975505829979, 16.42135857880162],
    sigma2: 1.1881878661049017,
    pseudoRSquared: 0.9679906425497444,
    anselinKelejian: [1.8890255909019582, 0.16931186628353218]
  },
  scene_60_13: {
    betas: [2.405537823720465, 1.9573063843293799, -0.9685075239218222, -0.053737984159109686],
    standardErrors: [
      0.4423540937299509, 0.024983645948526218, 0.06974253365710009, 0.021567768391027056
    ],
    z: [5.4380367624426285, 78.34350472152929, -13.886899043324673, -2.491587594267127],
    sigma2: 0.34730031386071225,
    pseudoRSquared: 0.9903570442899385,
    anselinKelejian: [3.3822319439334834, 0.06590275261420858]
  },
  scene_90_29: {
    betas: [0.7149671671942315, 1.9951180619611293, -1.0324144556956796, -0.0007798351166385942],
    standardErrors: [
      0.5330653526191322, 0.025672288139991263, 0.06529206662421597, 0.02574569975470366
    ],
    z: [1.341237361763156, 77.71485155829231, -15.812249620427401, -0.030289917309243873],
    sigma2: 0.4932231360127057,
    pseudoRSquared: 0.9861978275385085,
    anselinKelejian: [27.94383675914465, 1.2488810966002538e-7]
  }
};

/**
 * Masked-fit diagnostics from spreg 1.9.1 (libpysal 4.15.0, numpy 2.5.4, scipy 1.18.1), produced by
 * `fix/work/C/U1/pin-diagnostics-spreg.py` on scenes dumped by `dump-diagnostics-scenes.ts`, through
 * `spreg.OLS(y[keep], X[keep], w=W_keep, spat_diag=True, moran=True)` on the fitted rows `keep`:
 * - `lattice`: `createDiagnosticsScene(8, 11, 0.6, true)` with `createDiagnosticsExclusionMask(64, 7, 3)`.
 * - `knnDirected`: `createNearestNeighborScene(60, 3, 13, 0.6, false)` with
 *   `createDiagnosticsExclusionMask(60, 6, 2)` (the restricted W has one island).
 * - `original`: `W_keep` keeps the original values of the surviving links (`transform = 'o'`, what
 *   `restandardizeRows: false` means). libpysal 4.15 `w_subset` returns binary weights, so this W is
 *   built by hand with the same renumbering.
 * - `restandardized`: `w_subset(w, keep)` with `transform = 'r'` (`restandardizeRows: true`); it equals
 *   the hand-built restricted W transformed to `'r'` because each input row holds equal weights.
 * Each test is `[statistic, p-value]`; `moran` is `[I, z, p]`; `weightsSum` is S0 of `W_keep`.
 */
export const SPREG_MASKED_DIAGNOSTICS_REFERENCE = {
  lattice: {
    excluded: 9,
    original: {
      lmLag: [4.271709645163579, 0.03875192242833354],
      lmError: [9.351622079645233, 0.0022278820188490135],
      robustLmLag: [5.666957461328777, 0.017287414472131717],
      robustLmError: [10.746869895810432, 0.0010445584290466041],
      lmSarma: [15.01857954097401, 0.0005479701351038491],
      moran: [0.3475617230115019, 3.27790733013456, 0.0010457972262293632],
      sigmaSquared: 0.43692792418976717,
      weightsSum: 46.00000047683716,
      islands: 0
    },
    restandardized: {
      lmLag: [0.29410784908107246, 0.5876000698670006],
      lmError: [7.2167083352964045, 0.007222803844701725],
      robustLmLag: [0.8749433518149689, 0.34959040528942653],
      robustLmError: [7.797543838030302, 0.005231730208900379],
      lmSarma: [8.091651687111373, 0.01749525034836215],
      moran: [0.31264511480349166, 2.8918613689437356, 0.0038296689027284806],
      sigmaSquared: 0.43692792418976717,
      weightsSum: 55.0,
      islands: 0
    }
  },
  knnDirected: {
    excluded: 10,
    original: {
      lmLag: [1.089607942592391, 0.2965580924296086],
      lmError: [1.4992311315362408, 0.22078970279352816],
      robustLmLag: [1.3378683773843072, 0.2474102342254813],
      robustLmError: [1.7474915663281565, 0.18619238841365487],
      lmSarma: [2.8370995089205477, 0.24206481588277912],
      moran: [0.146931713030652, 1.496382489416781, 0.13455401001169798],
      sigmaSquared: 0.3529113342172203,
      weightsSum: 41.6666679084301,
      islands: 1
    },
    restandardized: {
      lmLag: [4.6832083159784155, 0.030458811722952787],
      lmError: [0.9497110986310322, 0.32979284556782207],
      robustLmLag: [5.265481750640849, 0.021752401450863212],
      robustLmError: [1.5319845332934667, 0.21581484587341462],
      lmSarma: [6.215192849271881, 0.044708286034778146],
      moran: [0.12061252760514764, 1.2101219311063578, 0.22623210889747136],
      sigmaSquared: 0.3529113342172203,
      weightsSum: 48.99999999999999,
      islands: 1
    }
  }
};

/**
 * 2SLS diagnostics reference: numpy 2.5.4 float64 definitions (first-stage F and partial R^2 of the excluded
 * instruments in the regression of W y on [1, X, W X (, W2 X)] against [1, X]; Belsley condition number of the
 * raw Gram of [1, P Z]; correlation condition number and largest VIF of the centered Z'PZ; correlation
 * condition number of the centered instruments; ||W||_inf) and the spreg 1.9.1 `GM_Lag(w_lags = order)` rho
 * used for the admissibility flag, generated by `fix/work/C/U2/pin-two-stage-diagnostics.py` on scenes dumped by
 * `fix/work/C/U2/dump-error-scenes.ts`: `lag_12_21` = `createLagScene(12, 21, 0.5, true)`, `lag_binary_10_5` =
 * `createLagScene(10, 5, 0.1, false)`, `lag_binary_10_5_high` = `createLagScene(10, 5, 0.255, false)`, `weak_80_7`
 * = `createWeakInstrumentScene(80, 7)`, `knn_60_13` / `knn_90_29` = `createNearestNeighborScene(60, 3, 13, 0.6,
 * false)` / `(90, 3, 29, 0.6, false)`.
 */
export type SpregTwoStageDiagnosticsFixture = {
  rho: number;
  rhoOutsideAdmissibleRange: boolean;
  firstStageF: number;
  firstStagePartialRSquared: number;
  conditionNumber: number;
  correlationConditionNumber: number;
  maximumVarianceInflationFactor: number;
  instrumentCorrelationConditionNumber: number;
};

export const SPREG_TWO_STAGE_DIAGNOSTICS_REFERENCE: Record<
  string,
  {
    weightsInfinityNorm: number;
    order1: SpregTwoStageDiagnosticsFixture;
    order2: SpregTwoStageDiagnosticsFixture;
  }
> = {
  lag_12_21: {
    weightsInfinityNorm: 1.0000000298023224,
    order1: {
      rho: 0.49540927749656305,
      rhoOutsideAdmissibleRange: false,
      firstStageF: 591.8012412875278,
      firstStagePartialRSquared: 0.8949041742841943,
      conditionNumber: 26.525308035899737,
      correlationConditionNumber: 1.275947208123687,
      maximumVarianceInflationFactor: 1.0148649193048154,
      instrumentCorrelationConditionNumber: 1.329286259471158
    },
    order2: {
      rho: 0.4846791386034681,
      rhoOutsideAdmissibleRange: false,
      firstStageF: 877.6093010654695,
      firstStagePartialRSquared: 0.9624393807685239,
      conditionNumber: 25.59745444133365,
      correlationConditionNumber: 1.264979580730944,
      maximumVarianceInflationFactor: 1.0138218350184842,
      instrumentCorrelationConditionNumber: 6.363406788311573
    }
  },
  lag_binary_10_5: {
    weightsInfinityNorm: 4.0,
    order1: {
      rho: 0.10796091887325598,
      rhoOutsideAdmissibleRange: false,
      firstStageF: 288.0515550642979,
      firstStagePartialRSquared: 0.858442020955921,
      conditionNumber: 12.111579545885077,
      correlationConditionNumber: 1.557202310164205,
      maximumVarianceInflationFactor: 1.0498304677015227,
      instrumentCorrelationConditionNumber: 1.6834480701905912
    },
    order2: {
      rho: 0.10342834326923622,
      rhoOutsideAdmissibleRange: false,
      firstStageF: 2321.2849648786437,
      firstStagePartialRSquared: 0.9900833212776574,
      conditionNumber: 11.375492270983457,
      correlationConditionNumber: 1.5111140374718581,
      maximumVarianceInflationFactor: 1.0432050176783831,
      instrumentCorrelationConditionNumber: 6.171175261215236
    }
  },
  lag_binary_10_5_high: {
    weightsInfinityNorm: 4.0,
    order1: {
      rho: 0.25517786327841296,
      rhoOutsideAdmissibleRange: true,
      firstStageF: 13.375945260082604,
      firstStagePartialRSquared: 0.21972464169445002,
      conditionNumber: 11.54743146702975,
      correlationConditionNumber: 2.188700634550681,
      maximumVarianceInflationFactor: 1.1613816550024556,
      instrumentCorrelationConditionNumber: 1.6834480701905912
    },
    order2: {
      rho: 0.2549530428959996,
      rhoOutsideAdmissibleRange: true,
      firstStageF: 39.771895112932896,
      firstStagePartialRSquared: 0.6310805957463377,
      conditionNumber: 7.8490947021293,
      correlationConditionNumber: 1.599695484120214,
      maximumVarianceInflationFactor: 1.0561885859912015,
      instrumentCorrelationConditionNumber: 6.171175261215236
    }
  },
  weak_80_7: {
    weightsInfinityNorm: 1.0000000298023224,
    order1: {
      rho: 1.0548292079458053,
      rhoOutsideAdmissibleRange: true,
      firstStageF: 0.9987069718925647,
      firstStagePartialRSquared: 0.02594131207112251,
      conditionNumber: 5.856216467704843,
      correlationConditionNumber: 6.337730906911635,
      maximumVarianceInflationFactor: 2.114072220760831,
      instrumentCorrelationConditionNumber: 2.107038402847996
    },
    order2: {
      rho: 0.6710238518830325,
      rhoOutsideAdmissibleRange: false,
      firstStageF: 0.8768974182090956,
      firstStagePartialRSquared: 0.045846296920810374,
      conditionNumber: 5.033850614265729,
      correlationConditionNumber: 4.36233432511586,
      maximumVarianceInflationFactor: 1.6426437950165873,
      instrumentCorrelationConditionNumber: 13.54249537871137
    }
  },
  knn_60_13: {
    weightsInfinityNorm: 1.0000000298023224,
    order1: {
      rho: -0.053487993500805686,
      rhoOutsideAdmissibleRange: false,
      firstStageF: 2524.8665220202056,
      firstStagePartialRSquared: 0.9892256853540636,
      conditionNumber: 12.881894755217582,
      correlationConditionNumber: 1.5254390337816628,
      maximumVarianceInflationFactor: 1.0451783899136238,
      instrumentCorrelationConditionNumber: 2.0342395553421126
    },
    order2: {
      rho: -0.053737984159109686,
      rhoOutsideAdmissibleRange: false,
      firstStageF: 1402.3430262519105,
      firstStagePartialRSquared: 0.9906399651917739,
      conditionNumber: 12.873772839375091,
      correlationConditionNumber: 1.5251662854174828,
      maximumVarianceInflationFactor: 1.045139657431078,
      instrumentCorrelationConditionNumber: 11.171268184782251
    }
  },
  knn_90_29: {
    weightsInfinityNorm: 1.0000000298023224,
    order1: {
      rho: -0.003626528559294684,
      rhoOutsideAdmissibleRange: false,
      firstStageF: 1319.7824040858077,
      firstStagePartialRSquared: 0.9688023570791692,
      conditionNumber: 15.828336919713141,
      correlationConditionNumber: 1.8348664972730269,
      maximumVarianceInflationFactor: 1.0922717455225817,
      instrumentCorrelationConditionNumber: 2.9876108707246627
    },
    order2: {
      rho: -0.0007798351166385942,
      rhoOutsideAdmissibleRange: false,
      firstStageF: 737.9976669880799,
      firstStagePartialRSquared: 0.9726523047084031,
      conditionNumber: 15.799660177218941,
      correlationConditionNumber: 1.8332990370650426,
      maximumVarianceInflationFactor: 1.0920140595734025,
      instrumentCorrelationConditionNumber: 12.084380083112954
    }
  }
};

/**
 * Belsley condition number (`np.linalg.cond` of the unit-column `[1, P Z]` design, numpy 2.5.4
 * float64) and correlation condition number of `createIllConditionedLagScene(12, 21, 0.5)`,
 * generated by `fix/work/C/U2/pin-ill-conditioned.py` on the scene dumped by
 * `fix/work/C/U2/dump-ill-conditioned-scene.ts`.
 */
export const TWO_STAGE_ILL_CONDITIONED_REFERENCE = {
  order1: {conditionNumber: 8076.73090388829, correlationConditionNumber: 255.1851871628051},
  order2: {conditionNumber: 8075.5773450251345, correlationConditionNumber: 254.93555994843797}
};
