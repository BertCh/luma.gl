# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Round 8 showcase option references. Run reference venv Python -I.

Reuses only the prior independent pandas/NumPy/sklearn reference generators,
never TS/WGSL or local CPU oracles. All floats round to f32 before reference.
Categorical compile-time options are enumerated; continuous sliders use named
representatives, not an assertion of exhaustive real-valued parameter coverage.
"""
import importlib.util
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent

def load_module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

reductions = load_module('parity-wp6-dataframe-reductions')
columns = load_module('parity-wp6-column-analysis')
output = HERE / 'fixtures'
# Full original tract rows, with the four-column topology of the nature/calendar scenes.
keys = reductions.load('chicago-tracts', 'communityArea.bin', 'u1')
income = reductions.load('chicago-tracts', 'perCapitaIncome.bin', '<f4')
population = reductions.load('chicago-tracts', 'population.bin', '<f4')
rows = len(keys)
mask = np.where(np.arange(rows) % 7, 1, 0)
group_cases = []
for omitted_mask in [False, True]:
    scene = reductions.group_case(f'showcase-key-only-mask-omitted={omitted_mask}', keys, [],
        np.ones(rows, int) if omitted_mask else mask, 80, dense=80)
    scene['omitMask'] = omitted_mask
    group_cases.append(scene)
for count in [1, 4]:
    group_cases.append(reductions.group_case(f'showcase-{count}-columns', keys,
        [income, population, np.arange(rows) % 24, np.arange(rows) % 9][:count], mask, 80, dense=80))
# Exact per-column statistic sets in direct scene calls and fleet-dwell helper.
statistic_fields = {'count':['counts'], 'mean':['means'], 'sum':['sumValues'], 'minimum':['minimums'],
    'maximum':['maximums'], 'median':['medians'], 'standardDeviation':['standardDeviations'],
    'percentiles':['percentiles'], 'mode':['modes'], 'uniqueCount':['uniqueCounts']}
configurations = [
    ('nature', [['mean','median','standardDeviation','percentiles','mode'],['sum','mean'],['mode','uniqueCount'],['uniqueCount']], False, True),
    ('introduced', [['mean']], False, False),
    ('osm', [['minimum','maximum','mean','median']], False, False),
    ('quake-depth', [['mean','standardDeviation','percentiles']], False, True),
    ('calendar', [['count','mean','median','standardDeviation','percentiles','mode'],['uniqueCount'],['mean'],['mean']], False, True),
    ('outage-peak', [['maximum','sum']], True, False),
    ('outage-now', [['sum']], False, False),
    ('transit', [['sum'],['mean','median'],['mean']], True, False),
    ('zone-dwell', [['sum','mean','maximum']], False, False)]
for name, statistics, omit_mask, both_variances in configurations:
    count=129
    case=reductions.group_case('showcase-exact-'+name,keys[:count],
        [income[:count],population[:count],np.arange(count)%24,np.arange(count)%9][:len(statistics)],
        np.ones(count,int) if omit_mask else mask[:count],80,dense=80)
    case['statistics']=statistics;case['omitMask']=omit_mask
    case['variances']=['sample','population'] if both_variances else ['sample']
    for variance in ['sample','population']:
        for column_index, requested in enumerate(statistics):
            fields=[field for statistic in requested for field in statistic_fields[statistic]]
            if name=='zone-dwell': fields.append('sums')
            case['expected']['columns'][variance][column_index]={field:case['expected']['columns'][variance][column_index][field] for field in fields}
    if name in ['nature','quake-depth','calendar']:
        fraction_indices=[3] if name=='calendar' else [1,3,5]
        previous_count=len(case['fractions'])
        case['fractions']=case['fractions'][fraction_indices]
        for variance in ['sample','population']:
            for result in case['expected']['columns'][variance]:
                if 'percentiles' in result:
                    result['percentiles']=np.asarray(result['percentiles']).reshape(80,previous_count)[:,fraction_indices].ravel()
    group_cases.append(case)
reductions.write(output, 'round8-groups', {'cases': group_cases}, 'pandas.DataFrame.groupby.agg; scipy.stats.skew/kurtosis; numpy.quantile/rint')

# Every profile resolution x precision exposed in choropleth-classes; auto domains.
# Four mixed columns, including a category, match its topology. Keep small cases
# so full HLL registers (not register summaries) fit the fixture size limit.
values = np.arange(97, dtype=float)
profile_columns = [{'values': values}, {'values': (values % 17) * 2},
    {'values': (values % 10).astype(int), 'kind': 'category', 'categoryCount': 10},
    {'values': np.log1p(values).astype(np.float32)}]
profiles = []
for bins in [12, 24, 48, 96]:
    for precision in [6, 8, 10, 12, 14]:
        profiles.append(columns.profile_case(f'showcase-bins-{bins}-p-{precision}', profile_columns,
            np.where(np.arange(97) % 7, 1, 0), bins=bins, precision=precision, top_count=10))
profiles.append(columns.profile_case('flood-statistics-only', [{'values': [0, .5, 1, 2, 0, .25]}], [1]*6, bins=1, precision=4, top_count=1))
profiles[-1]['statisticsOnly'] = True
profiles.append(columns.profile_case('documented-p16', [{'values': values}], np.ones(97, int), bins=12, precision=16))
domain_scene=columns.profile_case('documented-live-domains',[{'values':[0,1,2,4,8]}],[1]*5,bins=4,precision=4)
domain_scene['parameterRuns']=[]
for label, parameters, resolved in [('auto-min',[np.nan,4],[0,4]),('auto-max',[2,np.nan],[2,8]),
    ('inverted',[8,0],[8,0]),('infinite',[np.inf,8],[np.inf,8])]:
    run=columns.profile_case(label,[{'values':[0,1,2,4,8],'domain':resolved}],[1]*5,bins=4,precision=4)
    domain_scene['parameterRuns'].append({'name':label,'domains':[parameters],'expected':run['expected']})
profiles.append(domain_scene)
columns.write(output, 'round8-profiles' , profiles, 'pandas.Series.describe; numpy.histogram; published Flajolet HLL + Appleby fmix32 composite')

# Full ten-column matrix and both income direction settings. Every effective
# scaler/aggregation pair emitted by the four scene nodes (PCA forces z-score).
indicator_names = ['poverty150Pct','unemployedPct','noHighSchoolPct','uninsuredPct','housingBurdenPct',
                   'perCapitaIncome','age65Pct','disabilityPct','limitedEnglishPct','noVehiclePct']
derived = {}
for name, numerator, denominator in [('noHighSchoolPct','noHighSchool','population'),
    ('housingBurdenPct','housingBurdened','households'),('limitedEnglishPct','limitedEnglish','population')]:
    top = columns.load('chicago-tracts', numerator).astype(float)
    bottom = columns.load('chicago-tracts', denominator).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        derived[name] = np.where((bottom > 0) & np.isfinite(top), 100 * top / bottom, np.nan).astype(np.float32)
indicators = np.column_stack([derived[name] if name in derived else columns.load('chicago-tracts', name) for name in indicator_names])
for scaler in ['min-max', 'z-score', 'rank']:
    cases = []
    for protective in [False, True]:
        directions = [1]*10
        directions[5] = -1 if protective else 1
        for aggregation in ['weighted-sum', 'weighted-geometric-mean', 'principal-component']:
            if aggregation == 'weighted-geometric-mean' and scaler == 'z-score': continue
            if aggregation == 'principal-component' and scaler != 'z-score': continue
            cases.append(columns.composite_case(f'showcase-ten-{scaler}-{aggregation}-protective={protective}',
                indicators, np.ones(len(indicators), int), scaler, aggregation, [1]*10, directions,
                principal=aggregation == 'principal-component'))
    # Separate files by scaler keep complete arrays below 1.5 MB.
    columns.write(output, f'round8-composite-{scaler}', cases, 'sklearn StandardScaler/MinMaxScaler; scipy.stats.rankdata/gmean; scipy.linalg.eigh; numpy.average')

histograms=[]
for name, bins in [('binary',2),('order',8),('watershed',8)]:
    values = np.r_[np.arange(bins), np.arange(bins), 0xffffffff].astype(np.uint32)
    histograms.append(reductions.histogram_case('showcase-uint-domain-'+name, values, np.ones(len(values),int), bins, domain=[0,bins],scalar_format='uint32'))
for bins in [2,5,8]:
    values = np.r_[np.arange(bins), np.arange(bins), bins, 0xffffffff].astype(np.uint32)
    histograms.append(reductions.histogram_case(f'showcase-uint-edges-{bins}', values,np.ones(len(values),int),bins,edges=list(range(bins+1)),scalar_format='uint32'))
values=np.array([0,.1,.25,.5,.75,.9,1,1.0001,np.nan,np.inf],np.float32)
histograms.append(reductions.histogram_case('showcase-wind-ratio',values,np.ones(len(values),int),20,domain=[0,float(np.float32(1.0001))]))
empty_histogram=reductions.histogram_case('documented-all-invalid-auto',[np.nan,np.inf,-np.inf],[1,1,1],8,domain=[-1,1])
empty_histogram['domain']='auto';empty_histogram['resolvedDomain']=[np.nan,np.nan]
empty_histogram['layouts']=['auto-domain','gpu-parameters']
histograms.append(empty_histogram)
reductions.write(output,'round8-histograms' ,{'cases':histograms},'numpy.histogram')

# Reuse one compiled topology; change origin/width, then return to the initial
# settings. All 3..14-day widths in both migration scenes are included.
times=np.arange(257,dtype=float)*43200
cells=np.arange(257)%3
values=(np.arange(257)%19-9).astype(float)
temporal=[]
for bucket_count in [64,128]:
 for days in range(3,15):
    temporal.append(reductions.temporal_case(f'showcase-migration-{bucket_count}-buckets-{days}-days',cells,times,values,np.ones(257,int),3,bucket_count,0,days*86400,'float'))
for width in [365.2425,365.2425/2,365.2425/4]:
    temporal.append(reductions.temporal_case(f'showcase-osm-width-{width}',cells,times/86400,values,np.ones(257,int),3,80,0,float(np.float32(width)),'float'))
for name, buckets, width, masked, ones, days in [
    ('emerging-weeks',52,7*86400,True,True,False),
    ('space-time-fortnight',26,14*86400,False,True,False),
    ('quake-month',60,365.25/12,False,False,True)]:
    case=reductions.temporal_case('showcase-'+name,cells,times/86400 if days else times,
        np.ones(257) if ones else values,np.where(np.arange(257)%7,1,0) if masked else np.ones(257,int),
        3,buckets,0,float(np.float32(width)),'float')
    case['omitMask']=not masked
    temporal.append(case)
for case in temporal:
    if 'omitMask' not in case: case['omitMask']=True
signed_zero=reductions.temporal_case('documented-signed-zero',[0,0,0],[0,1,2],[-0.,0.,-0.],[1,1,1],1,1,0,4,'float')
# IEEE-754 ordered extrema have -0 < +0; pandas supplies timestamp selections.
signed_zero['expected']['minimums']=[-0.];signed_zero['expected']['maximums']=[0.]
temporal.append(signed_zero)
reductions.write(output,'round8-temporal'  ,{'cases':temporal},'pandas.cut(IntervalIndex); pandas.groupby.agg/idxmin/idxmax')

# All 15 exposed correlogram band counts x both band definitions x both nulls.
correlogram = load_module('parity-wp6-correlogram')
previous = json.loads((output / 'parity-wp6-correlogram.json').read_text())['cases'][0]
coordinates = np.asarray(previous['positions']).reshape(-1,2)[::24]
observations = np.asarray(previous['values'])[::24]
correlograms = [correlogram.makeCase(f'showcase-bands-{count}', coordinates, observations,
    previous['bounds'], 262144, bandCount=count, screen=True) for count in range(4,33,2)]
reductions.SOURCES.update(correlogram.SOURCES)
# Documented polygon clipping mode, independently selected with Shapely.
import shapely
from shapely.geometry import Polygon
polygon=Polygon([(0,0),(96,0),(96,48),(48,48),(48,96),(0,96)])
polygon_points=(2+92*(coordinates-coordinates.min(axis=0))/np.ptp(coordinates,axis=0)).astype(np.float32)
selection=shapely.covers(polygon,shapely.points(polygon_points)).astype(int)
polygon_case=correlogram.makeCase('documented-polygon',polygon_points,observations,[0,0,96,96],24,bandCount=8,mask=selection,screen=True)
polygon_case['mask']=np.ones(len(selection),int)
polygon_case['window']={'positions':np.array(polygon.exterior.coords)[:-1].ravel(),'ringOffsets':[0,6],'polygonOffsets':[0,1]}
correlograms.append(polygon_case)
reductions.write(output,'round8-correlogram',{'cases':correlograms},'esda.Moran; scipy.pdist/kurtosis/find_peaks')
# NumPy operates on f64 ranks; adapt probabilities to the public f32 rank
# convention. This is observable at p=.1,n=6 (h=.5, ties-to-even -> rank zero).
quantiles=[]
for name, values in [('rank-rounding', [0,10,20,30,40,50]),
    ('extreme-interpolation', [-np.finfo(np.float32).max,np.finfo(np.float32).max]),
    ('near-maximum', [np.finfo(np.float32).max/2,np.finfo(np.float32).max])]:
    case=columns.quantile_case(name,values,[1]*len(values),[.1,.9])
    probabilities=np.array([0,.01,.05,.1,.2,.3,.5,.7,.9,.99,1],np.float32)
    ranks=(np.float32(len(values)-1)*probabilities).astype(np.float32).astype(float)
    adjusted=ranks/(len(values)-1)
    case['probabilities']=probabilities
    case['expected']['quantiles']={method:np.quantile(np.asarray(values,float),adjusted,method=method)
        for method in ['lower','higher','nearest','linear','midpoint']}
    quantiles.append(case)
quartiles=columns.quantile_case('showcase-three-quartiles',columns.load('us-counties','medianHouseholdIncome'),np.where(np.arange(3109)%7,1,0),[.1,.9])
quartiles['probabilities']=quartiles['probabilities'][2:5]
quartiles['expected']['quantiles']={method:values[2:5] for method,values in quartiles['expected']['quantiles'].items()}
quantiles.append(quartiles)
columns.write(output,'round8-quantiles',quantiles,'numpy.quantile; probability adjusted to documented f32 rank h/(n-1); numpy.partition')

pca_scalers=[]
for scaler in ['min-max','rank']:
    for protective in [False,True]:
        directions=[1]*10;directions[5]=-1 if protective else 1
        pca_scalers.append(columns.composite_case(f'showcase-pca-selector-{scaler}-protective={protective}',indicators,
            np.ones(len(indicators),int),scaler,'principal-component',[1]*10,directions,principal=True))
columns.write(output,'round8-composite-pca-selectors',pca_scalers,'scipy.linalg.eigh of standardized covariance; PCA always standardizes regardless of scaler selector')
