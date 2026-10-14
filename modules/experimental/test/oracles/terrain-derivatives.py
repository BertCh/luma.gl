#!/usr/bin/env python3
"""Oracle generator for the luma.gl terrain derivative and ruggedness contributors.

Reference implementation: the real GDAL `gdaldem` library code (GDALDEMProcessing, the C entry point
that the `gdaldem` command line tool calls), taken from the GDAL build bundled in the `rasterio`
wheel. Nothing is reimplemented: slope, aspect, hillshade, TPI, TRI (Riley and Wilson) and
roughness values are exactly what `gdaldem <mode> <args>` writes. The `gdal` python bindings and
the `gdaldem` binary do not install in this environment (no libgdal headers, `gdal-config`), and
`richdem` has no wheel for Python 3.14, so the bundled libgdal is called through ctypes.

GDAL source references (https://github.com/OSGeo/gdal, apps/gdaldem_lib.cpp):
  Horn slope        GDALSlopeHornAlg        dx = ((a+2d+g)-(c+2f+i))/(8 xres), dy = ((g+2h+i)-(a+2b+c))/(8 yres),
                                            slope = atan(sqrt(dx^2+dy^2)/scale)  (degrees, or 100*tan for -p)
  aspect            GDALAspectAlg           aspect = atan2(dy, -dx), degrees clockwise from north of the downslope
                                            direction; flat cells are nodata (-9999) unless -zero_for_flat
  hillshade         GDALHillshadeAlg        Lambertian shade (sin(alt) - cos(alt) z sqrt(dx^2+dy^2) sin(aspect-az))
                                            / sqrt(1 + z^2 (dx^2+dy^2)); Byte output 1 + 254 * max(shade, 0)
  TPI               GDALTPIAlg              center - mean of the 8 neighbours
  TRI               GDALTRIAlg (Riley)      sqrt(sum (neighbour - center)^2)    (default since GDAL 3.3)
                    GDALTRIAlg (Wilson)     mean(|neighbour - center|)          (-alg Wilson)
  roughness         GDALRoughnessAlg        max(window) - min(window)
  edges             GDALGeneric3x3Processing  outer ring and windows with a nodata sample are nodata; with
                                            -compute_edges rows/columns are linearly extrapolated (2a - b) and
                                            nodata neighbours are replaced by the center.
Documentation: https://gdal.org/programs/gdaldem.html

Usage (writes fixtures/terrain-derivatives.json and fixtures/terrain-ruggedness.json):
  python3 -I terrain-derivatives.py
Needs numpy and rasterio (versions are recorded in the fixtures).
"""
import ctypes
import glob
import json
import os
import sys
import tempfile
from importlib.metadata import version

import numpy as np
import rasterio
from rasterio.transform import Affine

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DIR = os.path.join(HERE, 'fixtures')
GENERATOR = 'modules/experimental/test/oracles/terrain-derivatives.py'
WIDTH, HEIGHT = 17, 13
EPS32 = 2.0 ** -24  # float32 unit roundoff


def load_gdal():
    pattern = os.path.join(os.path.dirname(rasterio.__file__), '.dylibs', 'libgdal*')
    candidates = glob.glob(pattern) + glob.glob(
        os.path.join(os.path.dirname(rasterio.__file__), '..', 'rasterio.libs', 'libgdal*'))
    if not candidates:
        sys.exit('bundled libgdal not found next to rasterio')
    lib = ctypes.CDLL(candidates[0])
    lib.GDALOpen.restype = ctypes.c_void_p
    lib.GDALOpen.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.GDALDEMProcessingOptionsNew.restype = ctypes.c_void_p
    lib.GDALDEMProcessingOptionsNew.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_void_p]
    lib.GDALDEMProcessingOptionsFree.argtypes = [ctypes.c_void_p]
    lib.GDALDEMProcessing.restype = ctypes.c_void_p
    lib.GDALDEMProcessing.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_char_p,
                                      ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.GDALClose.argtypes = [ctypes.c_void_p]
    return lib


GDAL = load_gdal()


def gdaldem(directory, elevation, cell_x, cell_y, mode, args):
    """Runs `gdaldem <mode> in.tif out.tif <args>` and returns the output band as float64 (NaN = nodata)."""
    source = os.path.join(directory, 'in.tif')
    target = os.path.join(directory, 'out.tif')
    with rasterio.open(source, 'w', driver='GTiff', width=WIDTH, height=HEIGHT, count=1,
                       dtype='float64', nodata=float('nan'),
                       transform=Affine(cell_x, 0, 0, 0, -cell_y, 0)) as dataset:
        dataset.write(elevation, 1)
    argv = (ctypes.c_char_p * (len(args) + 1))(*[a.encode() for a in args], None)
    options = GDAL.GDALDEMProcessingOptionsNew(argv, None)
    source_handle = GDAL.GDALOpen(source.encode(), 0)
    usage_error = ctypes.c_int(0)
    handle = GDAL.GDALDEMProcessing(target.encode(), source_handle, mode.encode(), None, options,
                                    ctypes.byref(usage_error))
    if not handle or usage_error.value:
        sys.exit(f'gdaldem {mode} {args} failed')
    GDAL.GDALClose(handle)
    GDAL.GDALClose(source_handle)
    GDAL.GDALDEMProcessingOptionsFree(options)
    with rasterio.open(target) as dataset:
        band = dataset.read(1).astype('float64')
        nodata = dataset.nodata
        dtype = dataset.dtypes[0]
    nodata_mask = band == nodata
    if dtype == 'uint8':
        # Hillshade Byte output: 0 nodata, 1 + 254 * shade otherwise.
        band = (band - 1.0) / 254.0
    band[nodata_mask] = np.nan
    return band


def make_cases():
    rng = np.random.default_rng(20261014)
    column, row = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))

    def surface(amplitude):
        noise = rng.uniform(-0.5, 0.5, (HEIGHT, WIDTH))
        z = 200 + amplitude * np.sin(column * 0.45) * np.cos(row * 0.35) + 2.5 * column - 1.5 * row + noise
        return np.round(z, 4)

    analytic = surface(30.0)
    nonsquare = surface(12.0)
    hole = surface(30.0)
    hole[6:8, 7:9] = np.nan  # 2x2 nodata hole
    hole[7:12, 11:16] = 250.0  # 5x5 plateau: 3x3 flat interior, aspect undefined
    return [
        ('analytic-17x13', 'Analytic sinusoid plus tilt plus seeded noise, square 10 m cells', analytic, 10.0, 10.0),
        ('nonsquare-17x13', 'Same family, non-square 30 m (x) by 12 m (y) cells', nonsquare, 30.0, 12.0),
        ('nodata-hole-flat-17x13', 'NaN 2x2 hole and a 5x5 flat plateau, square 10 m cells', hole, 10.0, 10.0),
    ]


def nullable(array):
    return [None if np.isnan(v) else float(f'{v:.9g}') for v in array.ravel()]


def derivative_tolerance(elevation, cell_x, cell_y):
    """Derives per-output tolerances for a float32 kernel against the float64 reference."""
    finite = elevation[np.isfinite(elevation)]
    magnitude = float(np.max(np.abs(finite)))
    # Sobel sum of 8 float32 inputs each carrying |z| * 2^-24 rounding, then / (8 * smallest cell).
    gradient_error = 8 * magnitude * EPS32 / (8 * min(cell_x, cell_y))
    slope_degrees = float(np.degrees(gradient_error) * 4)  # d atan(g) <= dg; x4 for f32 trig and sqrt
    hillshade = 1.0 / 254.0 + 4e-5  # GDAL Byte quantisation dominates the f32 error
    return {
        'slopeDegrees': {'absolute': max(slope_degrees, 1e-4),
                         'derivation': f'GDAL f32 output ulp (<=1e-5 deg) + kernel: 8*|z|max({magnitude:.0f})*2^-24/(8*cellMin) = {gradient_error:.2e} slope-units, x4 for f32 trig'},
        'slopePercent': {'absolute': max(float(gradient_error * 100 * 4), 1e-3),
                         'derivation': '100 * gradient error x4 (percent = 100 tan(slope), derivative of tan stays below 4 for slope < 60 deg)'},
        'aspectDegrees': {'absolute': 0.05, 'minimumGradient': 0.05,
                          'derivation': 'angle error ~ gradient error / |gradient| <= 1e-5/0.05 rad = 0.012 deg; compared (circularly) only where |gradient| >= 0.05'},
        'hillshade': {'absolute': hillshade,
                      'derivation': 'GDAL writes Byte 1 + 254*shade, one level = 1/254 = 3.94e-3; f32 kernel error < 4e-5'},
    }


def build_derivatives(directory, cases):
    fixture_cases = []
    for name, description, elevation, cell_x, cell_y in cases:
        tolerance = derivative_tolerance(elevation, cell_x, cell_y)
        variants = []

        def add(variant, mode, args, ours, tolerance_key, output_name):
            output = gdaldem(directory, elevation, cell_x, cell_y, mode, args)
            variants.append({'name': variant, 'gdaldem': ' '.join(['gdaldem', mode, 'in.tif', 'out.tif'] + args),
                             'output': output_name, 'ours': ours, 'toleranceKey': tolerance_key,
                             'values': nullable(output)})

        add('slope-degrees', 'slope', [], {'slopeUnits': 'degrees', 'zFactor': 1}, 'slopeDegrees', 'slope')
        add('slope-percent', 'slope', ['-p'], {'slopeUnits': 'percent', 'zFactor': 1}, 'slopePercent', 'slope')
        add('slope-scale-2', 'slope', ['-s', '2'], {'slopeUnits': 'degrees', 'zFactor': 0.5}, 'slopeDegrees', 'slope')
        add('aspect', 'aspect', [], {'zFactor': 1, 'cellSizeOverride': [1.0, 1.0]}, 'aspectDegrees', 'aspect')
        add('hillshade-default', 'hillshade', [], {'zFactor': 1, 'azimuthDegrees': 315, 'altitudeDegrees': 45},
            'hillshade', 'hillshade')
        add('hillshade-z2-az225-alt30', 'hillshade', ['-z', '2', '-az', '225', '-alt', '30'],
            {'zFactor': 2, 'azimuthDegrees': 225, 'altitudeDegrees': 30}, 'hillshade', 'hillshade')
        fixture_cases.append({'name': name, 'description': description, 'width': WIDTH, 'height': HEIGHT,
                              'cellSize': [cell_x, cell_y], 'elevation': nullable(elevation),
                              'tolerances': tolerance, 'variants': variants})
    return {
        'generator': GENERATOR,
        'reference': reference_metadata(),
        'edgePolicy': 'gdaldem default: outer ring and any window containing a nodata (NaN) sample is nodata '
                      '(ours: borderMode "nodata"). -compute_edges is not mapped: our borderMode "clamp" replicates '
                      'the edge sample, GDAL extrapolates 2a-b, so edge values differ by design.',
        'nodataPolicy': 'null = nodata (GDAL -9999 / Byte 0 / NaN input). Aspect on flat cells is nodata in GDAL '
                        'and -1 in ours.',
        'conventionMap': {
            'cellSize': 'geotransform pixel size (x, |y|) <-> settings[0], settings[1], rowDirection "south"',
            'slope': 'degrees atan(|grad|/scale) <-> slopeUnits "degrees"; -p percent <-> slopeUnits "percent"; '
                     'GDAL -s scale divides the gradient, so ours zFactor = 1/scale',
            'aspect': 'degrees clockwise from north of the downslope direction, [0,360) <-> atan2(-east,-north); '
                      'flat: GDAL nodata, ours -1. GDAL QUIRK: GDALAspectAlg does not divide the Horn sums by xres/yres, '
                      'so on non-square cells its aspect is the angle of the unscaled gradient, not the physical one '
                      '(verified by hand on nonsquare-17x13). The aspect variant therefore records '
                      'ours.cellSizeOverride [1,1] to reproduce GDAL; ours keeps the physical aspect for real cell sizes.',
            'hillshade': 'GDAL -z z-factor <-> zFactor, -az <-> azimuthDegrees, -alt <-> altitudeDegrees; '
                         'GDAL Byte (1 + 254 shade) decoded to [0,1]; shade <= 0 decodes to 0 in both',
            'algorithm': 'GDAL default -alg Horn <-> ours 3x3 Sobel / 8 (identical kernel)',
        },
        'cases': fixture_cases,
    }


def build_ruggedness(directory, cases):
    fixture_cases = []
    for name, description, elevation, cell_x, cell_y in cases[:1] + cases[2:]:
        finite = elevation[np.isfinite(elevation)]
        magnitude = float(np.max(np.abs(finite)))
        tolerance = {
            'absolute': max(16 * magnitude * EPS32, 1e-4),
            'derivation': f'sums of up to 9 float32 inputs with |z|max {magnitude:.0f}: 16*|z|max*2^-24 = '
                          f'{16 * magnitude * EPS32:.2e} m (TRI Riley sqrt is first order in the same error); '
                          'GDAL output is float32 (<= 1.5e-5 m at these magnitudes)',
        }
        variants = []
        for edge, edge_args in (('nodata', []), ('extrapolate', ['-compute_edges'])):
            for variant, mode, args in (('tpi', 'TPI', []), ('tri-riley', 'TRI', ['-alg', 'Riley']),
                                        ('tri-wilson', 'TRI', ['-alg', 'Wilson']), ('roughness', 'roughness', [])):
                output = gdaldem(directory, elevation, cell_x, cell_y, mode, args + edge_args)
                variants.append({'name': f'{variant}-{edge}', 'gdaldem': ' '.join(['gdaldem', mode, 'in.tif', 'out.tif'] + args + edge_args),
                                 'output': variant, 'edgeMode': edge, 'values': nullable(output)})
        fixture_cases.append({'name': name, 'description': description, 'width': WIDTH, 'height': HEIGHT,
                              'cellSize': [cell_x, cell_y], 'elevation': nullable(elevation),
                              'tolerance': tolerance, 'variants': variants})
    return {
        'generator': GENERATOR,
        'reference': reference_metadata(),
        'edgePolicy': 'edgeMode "nodata" = gdaldem default; "extrapolate" = -compute_edges (<-> ours edgeMode).',
        'nodataPolicy': 'null = nodata. Windows with a nodata neighbour are nodata by default; with '
                        '-compute_edges the neighbour is replaced by the center.',
        'conventionMap': {
            'tpi': 'center - mean(8 neighbours) <-> topographicPositionIndex',
            'tri-riley': 'sqrt(sum (n - c)^2), gdaldem default since 3.3 <-> terrainRuggednessAlgorithm "riley"',
            'tri-wilson': 'mean |n - c| <-> "wilson"',
            'roughness': 'max - min of the 3x3 window <-> roughness',
            'units': 'elevation units; independent of cell size (non-square case omitted)',
        },
        'cases': fixture_cases,
    }


def reference_metadata():
    return {
        'kind': 'gdaldem (real GDAL library code, not a reimplementation)',
        'entryPoint': 'GDALDEMProcessing from the libgdal bundled in the rasterio wheel, called through ctypes',
        'gdal': rasterio.__gdal_version__,
        'packages': {'rasterio': version('rasterio'), 'numpy': version('numpy'), 'python': sys.version.split()[0]},
        'unavailable': {'gdal python bindings': 'no gdal-config/libgdal headers', 'richdem': 'no wheel for Python 3.14'},
    }


def write(name, payload):
    path = os.path.join(FIXTURE_DIR, name)
    with open(path, 'w') as handle:
        json.dump(payload, handle, separators=(',', ':'))
        handle.write('\n')
    print(f'wrote {path} ({os.path.getsize(path)} bytes)')


if __name__ == '__main__':
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    all_cases = make_cases()
    with tempfile.TemporaryDirectory() as directory:
        write('terrain-derivatives.json', build_derivatives(directory, all_cases))
        write('terrain-ruggedness.json', build_ruggedness(directory, all_cases))
