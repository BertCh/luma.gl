# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""SciPy reference for the complete scalar GPUChangeDetection output contract.

Run <reference-venv>/bin/python -I this-file. Decode the full 16x256x256 Sentinel
NDVI stack, select the fixed lattice rows 0::8 / columns 0::4 (2048 cells), and
compare every scalar for all 16 dates. Zero bytes are NaN; decoded inputs are
rounded to float32 before float64 references. Cell-major layout, a cell mask,
first/last or selected two-slice comparisons, Welch split at ordinal slice 8.
Time is the ORIGINAL ordinal index, not days.bin and not the compressed index
of valid observations; this matches the public API. No claim of slope/year.

Sen slope: scipy.stats.theilslopes. Welch: scipy.stats.ttest_ind(equal_var=False)
with after minus before and scipy's df; fewer than 2 samples per group or zero
standard error -> NaN. Difference/log-ratio/percent are NumPy ufuncs.
Mann-Kendall is a composite reference: scipy.stats.kendalltau(variant='b',
method='asymptotic') supplies tau and uncorrected two-sided p. Recover integer
S using the tau-b denominator; norm.isf(p/2) supplies the uncorrected Z, then
apply the published continuity correction Z *= (|S|-1)/|S|. This is not SciPy's
uncorrected Kendall p-value presented as Mann-Kendall. All ties -> S=Z=0,p=1;
fewer than two valid values -> S=0,Z=p=NaN. Significance p<alpha, 1 up / 2 down.
Definition: Hirsch, Slack & Smith (1982), Water Resources Research 18, 107-121,
https://doi.org/10.1029/WR018i001p00107. Also SciPy kendalltau documentation:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html
Profile S: atol=1e-6*output scale, rtol=1e-4, integer outputs and NaN masks exact.
"""
import hashlib
import json
import math
import platform
import warnings
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import kendalltau, norm, theilslopes, ttest_ind

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data/ndvi-timeseries'


def encode(value):
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
        return float(value)
    return value


def make_case(name, slices, mask, before, after, split=8, alpha=.05, epsilon=1e-6):
    slices = np.asarray(slices, dtype=np.float32).astype(float)
    cell_count, slice_count = slices.shape
    alpha, epsilon = float(np.float32(alpha)), float(np.float32(epsilon))
    names = ['difference', 'logRatio', 'percentChange', 'tStatistic', 'tDegreesOfFreedom',
             'tPValue', 'senSlope', 'mannKendallZ', 'mannKendallP']
    expected = {name: np.full(cell_count, np.nan) for name in names}
    for output_name in ['mannKendallS', 'significance', 'welchSignificance']:
        expected[output_name] = np.zeros(cell_count, dtype=int)
    for cell, values in enumerate(slices):
        if not mask[cell]:
            continue
        first, last = values[before], values[after]
        expected['difference'][cell] = last - first
        if first > -epsilon and last > -epsilon:
            expected['logRatio'][cell] = np.log((last + epsilon) / (first + epsilon))
        if first != 0:
            expected['percentChange'][cell] = 100 * (last - first) / abs(first)
        valid = np.flatnonzero(np.isfinite(values))
        observed = values[valid]
        if len(valid) >= 2:
            expected['senSlope'][cell] = theilslopes(observed, valid).slope
            _, multiplicities = np.unique(observed, return_counts=True)
            pairs = len(valid) * (len(valid) - 1) / 2
            tied_pairs = np.sum(multiplicities * (multiplicities - 1) / 2)
            if tied_pairs == pairs:
                score, statistic, probability = 0, 0., 1.
            else:
                result = kendalltau(valid, observed, variant='b', method='exact' if len(valid) == 2 else 'asymptotic')
                score = int(round(result.statistic * np.sqrt(pairs * (pairs - tied_pairs))))
                statistic = 0. if score == 0 else np.sign(score) * norm.isf(result.pvalue / 2) * (abs(score) - 1) / abs(score)
                probability = 2 * norm.sf(abs(statistic))
            expected['mannKendallS'][cell] = score
            expected['mannKendallZ'][cell] = statistic
            expected['mannKendallP'][cell] = probability
            expected['significance'][cell] = (1 if score > 0 else 2) if probability < alpha else 0
        first_group = values[:split][np.isfinite(values[:split])]
        last_group = values[split:][np.isfinite(values[split:])]
        if len(first_group) >= 2 and len(last_group) >= 2 and (np.var(first_group) + np.var(last_group)) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                result = ttest_ind(last_group, first_group, equal_var=False)
            expected['tStatistic'][cell] = result.statistic
            expected['tDegreesOfFreedom'][cell] = result.df
            expected['tPValue'][cell] = result.pvalue
            expected['welchSignificance'][cell] = (1 if result.statistic > 0 else 2) if result.pvalue < alpha else 0
    return dict(name=name, slices=slices.ravel(), cellCount=cell_count, sliceCount=slice_count,
                mask=mask, settings=dict(beforeSlice=before, afterSlice=after,
                splitSlice=split, alpha=alpha, epsilon=epsilon), expected=expected)


# Round 8: scene temporal controls and multiband documented output modes.
source = HERE / 'fixtures/parity-wp4-change-detection.json'
parent = json.loads(source.read_text())
raw = np.asarray(parent['cases'][1]['slices'], dtype=float).reshape(12,16)
cases=[]
for split in [1,4,8,12,15]:
    for alpha in [.01,.1]:
        cases.append(make_case(f'scene-window-split{split}-alpha{alpha}',raw,np.ones(12,dtype=int),2,13,split,alpha,.01))
# Round 9: exact zero standard error must stay invalid for non-power-of-two counts.
# All references still use SciPy on float32-rounded inputs. The shifted ramp checks
# that centering preserves nonconstant Welch statistics at a large common offset.
for split in [4,8,12]:
    regression=np.stack([np.full(16,.1),np.ones(16),
        np.r_[np.ones(split),np.full(16-split,2.)],1000+np.arange(16)*.125])
    cases.append(make_case(f'constant-and-offset-split{split}',regression,np.ones(4,dtype=int),0,15,split))
# Scene dateWindow preserves ordinal positions but clears outside dates; minimumValid masks cells.
for low,high,minimum in [(2,13,4),(4,11,8),(7,9,4)]:
    window=raw.copy();window[:,:low]=np.nan;window[:,high+1:]=np.nan
    mask=(np.sum(np.isfinite(window),axis=1)>=minimum).astype(int)
    cases.append(make_case(f'date-window-{low}-{high}-minimum-{minimum}',window,mask,low,high,8,.05,.01))
# Larger Welch df without monotone p underflow; full original ordinal time axis.
long = np.stack([np.sin(np.arange(64)*.5)+index*.002*np.arange(64) for index in range(6)])
cases.append(make_case('long-series-df',long,np.ones(6,dtype=int),0,63,32))
vector_cases=[]
for band_count in [2,3,16]:
    values=(np.arange(7*3*band_count).reshape(7,3,band_count)%13+1).astype(float)
    values[0,2]=values[0,0]; values[1,2,0]=np.nan
    values[2,0,0]=0; values[3,0,0]=-1
    mask=np.ones(7,dtype=int);mask[-1]=0
    before=values[:,0];after=values[:,2];difference=after-before
    difference[mask==0]=np.nan
    with np.errstate(invalid='ignore',divide='ignore'):
        log_ratio=np.where((before>-.01)&(after>-.01),np.log((after+.01)/(before+.01)),np.nan)
        percent=np.where(before!=0,100*(after-before)/abs(before),np.nan)
    log_ratio[mask==0]=np.nan;percent[mask==0]=np.nan
    magnitude=np.linalg.norm(difference,axis=1)
    if band_count==2:
        direction=np.arctan2(difference[:,1],difference[:,0]);direction[magnitude==0]=0
    else:
        direction=np.sum((difference>0).astype(np.uint64)*(2**np.arange(band_count,dtype=np.uint64)),axis=1)
        direction+=np.sum((difference<0).astype(np.uint64)*(2**(16+np.arange(band_count,dtype=np.uint64))),axis=1)
        direction[~np.isfinite(magnitude)]=2**32-1
    vector_cases.append(dict(name=f'multiband-{band_count}',bandCount=band_count,cellCount=7,sliceCount=3,
      slices=values.ravel(),mask=mask,settings=dict(beforeSlice=0,afterSlice=2,epsilon=float(np.float32(.01))),
      expected=dict(difference=difference.ravel(),logRatio=log_ratio.ravel(),percentChange=percent.ravel(),changeMagnitude=magnitude,changeDirection=direction)))
fixture=dict(reference=parent['reference']+'; numpy.linalg.norm/arctan2/sign',versions=parent['versions'],
 source=dict(path=str(source.relative_to(ROOT)),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),cases=cases,vectorCases=vector_cases)
output=HERE/'fixtures/parity-wp4-change-modes.json';output.write_text(json.dumps(encode(fixture),separators=(',',':'),allow_nan=False)+'\n')
assert output.stat().st_size<1_500_000
print(output.name,output.stat().st_size)
