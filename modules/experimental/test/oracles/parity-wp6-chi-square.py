# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""SciPy regression for the GPU quadrat chi-square tails, including maximum df.

Input: deterministic integer events per unit square, all at its exact f32 center.
No masks, weights, RNG or spatial correction. scipy.stats.chisquare and chi2
supply the independent statistic and lower/upper/two-sided tails; numpy supplies
sample variance. Full bin arrays are retained. Cases straddle the old df=100
switch, exercise both gamma branches and the 65536-quadrat public limit.
"""
import json
import platform
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import chi2, chisquare

cases = []
for width, height, pattern in [(10,10,[1,3,4,5,7]), (17,6,[1,3,4,5,7]), (12,12,[2,4,5,6,8]), (12,12,[0,3,4,5,8]), (32,32,[1,3,4,5,7]), (256,256,[1,3,4,5,7])]:
    counts = np.resize(pattern, width * height)
    statistic = chisquare(counts).statistic
    lower, upper = chi2.cdf(statistic, len(counts)-1), chi2.sf(statistic, len(counts)-1)
    cases.append({'name': f'{width}x{height}-{pattern[0]}', 'quadratGrid':[width,height], 'counts': counts.tolist(), 'statistics':[len(counts), float(counts.mean()), float(counts.var(ddof=1)), float(counts.var(ddof=1)/counts.mean()), float(statistic), len(counts)-1], 'tails': {'lesser':lower,'greater':upper,'two-sided':min(1,2*min(lower,upper))}})
output=Path(__file__).parent/'fixtures/parity-wp6-chi-square.json'
output.write_text(json.dumps({'reference':'scipy.stats.chisquare/chi2; numpy.var(ddof=1)', 'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}, 'conventions':__doc__,'sources':'synthetic integer counts; no RNG','cases':cases},separators=(',',':'),allow_nan=False)+'\n')
assert output.stat().st_size<=1500000
print(output.name,output.stat().st_size)
