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
