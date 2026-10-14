// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Reference values for `GPUOrdinaryLeastSquares` from PySAL spreg 1.9.1 (libpysal 4.15.0, numpy
 * 2.5.4, scipy 1.18.1; float64), produced by
 * `fix/work/C/U1/pin-ols-spreg.py` on the scenes dumped by `dump-ols-scenes.ts`:
 * - `heteroskedastic`: `createHeteroskedasticOrdinaryLeastSquaresScene(41, 300)`
 * - `collinearModerate`: `createCollinearOrdinaryLeastSquaresScene(17, 200, 0.3)`
 * - `collinearSevere`: `createCollinearOrdinaryLeastSquaresScene(19, 200, 0.03)`
 * - `plain`: `createOrdinaryLeastSquaresScene(2, 400, 2, [3, -1], 0.5)`
 * through `spreg.OLS(y, X)`. Fields:
 * - `mulColli`: `reg.mulColli` (Belsley condition number); `aic`, `schwarz`, `logll`:
 *   `spreg.diagnostics.akaike`, `schwarz`, `log_likelihood`; `jarqueBera`: `reg.jarque_bera`.
 * - `breuschPagan`, `koenkerBassett`: `[statistic, p-value, df]` of `spreg.diagnostics.breusch_pagan`
 *   and `koenker_bassett` with `z = reg.x` (the `[1, X]` auxiliary design the GPU uses; spreg's
 *   default is the squared design). spreg 1.9.1 tests `z == None`, which raises for an array, so the
 *   script runs spreg's own function source with `z is None`.
 * - `maximumVif`: largest of `spreg.diagnostics.vif(reg)` over the predictor columns (the centered
 *   `1 / (1 - R_j^2)`); `correlationConditionNumber`: max/min eigenvalue of `np.corrcoef(X.T)`.
 * - `studentT`: `scipy.stats.t.sf` and `2 * t.sf(|t|)`.
 */
export const SPREG_ORDINARY_LEAST_SQUARES_REFERENCE = {
  heteroskedastic: {
    n: 300,
    betas: [4.22944812433343, 3.0664536897587755, -1.103263035214468],
    stdErrors: [4.111072661278836, 0.06218339360616816, 0.06363294274685896],
    tStats: [1.028794300857181, 49.313064339650396, -17.33792258521514],
    pValues: [0.3044134516516162, 4.498710333065824e-145, 5.247572358505158e-47],
    sig2: 9.798931289546466,
    sig2n: 9.700941976651002,
    utu: 2910.2825929953005,
    r2: 0.9009498522047156,
    ar2: 0.9002828478424578,
    logll: -766.5150087297995,
    aic: 1539.030017459599,
    schwarz: 1550.1413648835678,
    mulColli: 48.31672166939461,
    jarqueBera: [2676.7452397114184, 0.0],
    breuschPagan: [162.77198953128328, 4.5134804854191485e-36, 2],
    koenkerBassett: [20.804234965653123, 3.036811092570579e-5, 2],
    correlationConditionNumber: 1.0379435563458912,
    maximumVif: 1.0003467706551505,
    vifs: [1.00034677065515, 1.0003467706551505]
  },
  collinearModerate: {
    n: 200,
    betas: [1.057828748830616, 0.47114087611953437, 0.27906686462847574, -2.009570325139805],
    stdErrors: [0.34834315316943126, 0.06794594531241026, 0.06816052835725206, 0.01702050261216708],
    tStats: [3.0367433354319346, 6.934054327351907, 4.094259116739725, -118.06762531814229],
    pValues: [
      0.002716736265991636, 5.784396909868541e-11, 6.189240934606553e-5, 4.650091459620045e-184
    ],
    sig2: 0.0805810022084104,
    sig2n: 0.07896938216424218,
    utu: 15.793876432848437,
    r2: 0.9924616444347566,
    ar2: 0.992346261441411,
    logll: -29.918199722928136,
    aic: 67.83639944585627,
    schwarz: 81.02966891204842,
    mulColli: 432.9936835153085,
    jarqueBera: [3.1991723959075653, 0.201980080474896],
    breuschPagan: [3.2074425504125683, 0.3607340507357026, 3],
    koenkerBassett: [3.13769166064462, 0.3708791658692303, 3],
    correlationConditionNumber: 364.01396012838404,
    maximumVif: 91.43247544902258,
    vifs: [91.43247544902258, 91.41531802077313, 1.0017955409943429]
  },
  collinearSevere: {
    n: 200,
    betas: [0.9877711891131185, -0.005453975405544043, 0.7567838914692402, -2.0096930400836754],
    stdErrors: [0.29251673026373404, 0.6396854618254487, 0.6394548948430229, 0.016088020725440143],
    tStats: [3.376802373739583, -0.008526026822589055, 1.1834828344773558, -124.91860088828257],
    pValues: [
      0.0008843285439593937, 0.9932059690193931, 0.23805125971629498, 8.489304965230265e-189
    ],
    sig2: 0.06397914218440295,
    sig2n: 0.06269955934071489,
    utu: 12.539911868142978,
    r2: 0.9936896889944756,
    ar2: 0.9935931026015339,
    logll: -6.847620698261295,
    aic: 21.69524139652259,
    schwarz: 34.88851086271474,
    mulColli: 4545.982172918511,
    jarqueBera: [2.5260634235340085, 0.28279537150477674],
    breuschPagan: [1.8021414486939706, 0.6144690434362641, 3],
    koenkerBassett: [2.454590006188839, 0.4835536100069243, 3],
    correlationConditionNumber: 45470.55396290199,
    maximumVif: 11365.771184757783,
    vifs: [11365.63203916301, 11365.771184757783, 1.0022003123985481]
  },
  plain: {
    n: 400,
    betas: [2.3934733981004683, 2.9910419909905723, -0.9992091968725845],
    stdErrors: [0.32273541780967524, 0.004902097401769359, 0.004922986118615289],
    tStats: [7.416209272426234, 610.1555611504145, -202.96811179179937],
    pValues: [7.362633891566572e-13, 0.0, 0.0],
    sig2: 0.08011546990365624,
    sig2n: 0.07951460387937882,
    utu: 31.80584155175153,
    r2: 0.9990381970878578,
    ar2: 0.9990333517331367,
    logll: -61.21249776266018,
    aic: 128.42499552532036,
    schwarz: 140.3993891666443,
    mulColli: 48.4016006572458,
    jarqueBera: [0.7536486758519761, 0.6860365739194909],
    breuschPagan: [1.9495361123547692, 0.3772798511463734, 2],
    koenkerBassett: [2.0439295665267325, 0.35988714482969025, 2],
    correlationConditionNumber: 1.00912471185187,
    maximumVif: 1.0000206268772838,
    vifs: [1.0000206268772822, 1.0000206268772838]
  },
  studentT: [
    {t: 0.0, df: 5, sf: 0.5, twoSided: 1.0},
    {t: 1.0, df: 5, sf: 0.18160873382456136, twoSided: 0.3632174676491227},
    {t: -1.0, df: 5, sf: 0.8183912661754387, twoSided: 0.3632174676491227},
    {t: 2.5, df: 10, sf: 0.01572342211830441, twoSided: 0.03144684423660882},
    {t: -3.7, df: 296, sf: 0.9998714999400092, twoSided: 0.00025700011998146954},
    {t: 12.0, df: 296, sf: 1.3325193691086993e-27, twoSided: 2.6650387382173985e-27},
    {t: 30.0, df: 40, sf: 2.4254538302951023e-29, twoSided: 4.8509076605902047e-29},
    {t: 0.3, df: 1, sf: 0.4072264209222577, twoSided: 0.8144528418445154},
    {t: 5.0, df: 2, sf: 0.018874775675311862, twoSided: 0.037749551350623724},
    {t: 1.96, df: 1000, sf: 0.025136592477874344, twoSided: 0.05027318495574869},
    {t: 40.0, df: 3, sf: 1.7190340394579253e-5, twoSided: 3.4380680789158506e-5}
  ]
};
