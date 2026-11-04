#!/usr/bin/env python3
"""ElastoGrade MIS: label-free RF motion learning and probabilistic risk analysis.

One runnable file; no imports from the pre-existing elastograde package. The
three syllabus units are implemented below, alongside the imaging experiment.
Risk means failure of a declared mechanics-quality rule. Pathology labels are
absent from the supplied data, so no output is a histological tumor grade.

Quick start: python elastography_mis.py demo --device cpu
Full data:   python elastography_mis.py run --data-root data --device cuda
See README.md for definitions, assumptions, GPU setup and every output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import sys
import time
import warnings
import zlib

try:
    import numpy as np
    import scipy
    from scipy.ndimage import gaussian_filter, map_coordinates
    from scipy.signal import hilbert, savgol_filter
    from scipy.special import logsumexp
    import matplotlib
    matplotlib.use('Agg')  # Works from PowerShell, without a notebook/display.
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Circle
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 confusion_matrix, precision_recall_curve,
                                 roc_auc_score, roc_curve)
    from sklearn.preprocessing import StandardScaler
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit('Missing scientific dependency: ' + str(exc) + '\n'
                     'Install numpy scipy matplotlib scikit-learn and a suitable '
                     'PyTorch wheel. See README.md.') from exc

VERSION = '1.1.1'
ROLES = ('train', 'validation', 'calibration', 'test')
FEATURE_NAMES = ['reconstruction loss', 'cycle error', 'strain dispersion',
                 'fold fraction', 'out-of-frame fraction', 'flow magnitude']
SYLLABUS = {
    'Unit 1': ['Probabilistic graphical models', 'Probabilistic AI',
               'Bayesian networks', 'Representation learning in Bayesian networks',
               'Inference in Bayesian networks'],
    'Unit 2': ['Markov networks', 'Independencies in Markov networks',
               'Hidden Markov models'],
    'Unit 3': ['Markov random fields', 'Decision networks',
               'From Bayesian networks to Markov networks'],
}


class RFDataQualityError(ValueError):
    """A source cannot be safely represented as RF; keep diagnostic evidence."""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = {} if details is None else details


def validate_rf_storage_values(array):
    """Validate ORIGINAL precision before any float32 conversion.

    Finite float64 is not sufficient: corrupt MAT samples can be around 1e307
    among ordinary RF amplitudes of a few hundred. Global scaling or clipping
    would conceal corruption, so unsafe sources are explicitly quarantined.
    """
    array = np.asarray(array)
    if (not np.isrealobj(array) or not np.issubdtype(array.dtype, np.number)
            or not array.size):
        raise RFDataQualityError('RF values must be a nonempty real numeric array.', {'reason': 'invalid_numeric_type'})
    finite = np.isfinite(array)
    if not finite.all():
        raise RFDataQualityError('Original RF source contains NaN or infinity.',
                                 {'reason': 'nonfinite_source', 'nonfinite_samples': int(array.size-finite.sum()),
                                  'original_dtype': str(array.dtype), 'original_shape': list(array.shape)})
    lower, upper = float(array.min()), float(array.max())
    limit = float(np.finfo(np.float32).max)
    if lower < -limit or upper > limit:
        unsafe = (array < -limit) | (array > limit)
        columns = np.flatnonzero(unsafe.any(axis=tuple(range(array.ndim-1)))) if array.ndim > 1 else np.flatnonzero(unsafe)
        absolute = np.abs(array.astype(np.float64, copy=False))
        quantiles = np.quantile(absolute, [.5, .999])
        details = {'reason': 'float32_out_of_range', 'original_dtype': str(array.dtype),
                   'original_shape': list(array.shape), 'minimum': lower, 'maximum': upper,
                   'unsafe_samples': int(unsafe.sum()), 'unsafe_fraction': float(unsafe.mean()),
                   'affected_lateral_columns': columns.tolist(),
                   'absolute_median': float(quantiles[0]), 'absolute_q999': float(quantiles[1])}
        raise RFDataQualityError('Original RF values exceed float32 range; source may contain corrupt scanlines.', details)
    return array


def json_safe(value):
    """Produce standards-compliant JSON, representing unavailable metrics by null."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if not isinstance(name, str) or re.fullmatch(r'auto|cpu|cuda(?::\d+)?', name) is None:
        raise ValueError('Device must be auto, cpu, cuda, or cuda:N.')
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable in PyTorch. Run doctor and install '
                         'the CUDA wheel from the official PyTorch selector, or use --device cpu.')
    device = torch.device(name)
    if device.type == 'cuda' and device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f'CUDA device {device.index} does not exist; detected {torch.cuda.device_count()} device(s).')
    return device


def normalize_rf_pair(pair):
    """Shared, stable RF normalization used identically in training and tiling.

    Float64 statistics avoid overflow on finite high-amplitude source arrays.
    RMS scaling retains weak RF signals instead of clipping their scale to 1e-6.
    Constant individual frames cannot supply registration evidence and fail with
    a clear data error. The returned float32 array never aliases its input.
    """
    array = np.asarray(pair)
    if array.ndim != 3 or array.shape[0] != 2 or min(array.shape[1:]) < 2:
        raise ValueError('RF pair must have shape [2,z,x].')
    if not np.isrealobj(array) or not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError('RF pair must contain finite real numeric values.')
    values = np.array(array, dtype=np.float64, copy=True)
    peak = float(np.max(np.abs(values)))
    if not peak:
        raise ValueError('Constant RF frame cannot support registration; inspect the source or choose a larger crop.')
    values /= peak  # Bound intermediate arithmetic even for large finite values.
    if np.any(np.std(values, axis=(1, 2)) <= np.finfo(np.float64).eps):
        raise ValueError('Constant RF frame cannot support registration; inspect the source or choose a larger crop.')
    values -= values.mean()
    scale = float(np.sqrt(np.mean(values*values)))
    return np.clip(values/scale, -10, 10).astype(np.float32)


def environment_info():
    return {'python': sys.version, 'platform': platform.platform(),
            'numpy': np.__version__, 'scipy': scipy.__version__,
            'torch': torch.__version__, 'cuda_build': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


# ---------------------------------------------------------------------------
# RF mechanics: coordinates are [z,x]; flow channels are [ux,uz] in pixels.
# Warping convention: warped_post(x,z) = post(x+ux,z+uz).
# ---------------------------------------------------------------------------
def warp_tensor(image, flow):
    batch, _, height, width = image.shape
    z, x = torch.meshgrid(torch.arange(height, device=image.device),
                          torch.arange(width, device=image.device), indexing='ij')
    xx = x[None].float() + flow[:, 0].float()
    zz = z[None].float() + flow[:, 1].float()
    grid = torch.stack((2 * xx / max(width - 1, 1) - 1,
                        2 * zz / max(height - 1, 1) - 1), dim=-1)
    warped = F.grid_sample(image.float(), grid, mode='bilinear',
                           padding_mode='border', align_corners=True)
    inside = ((xx >= 0) & (xx <= width - 1) & (zz >= 0) & (zz <= height - 1))[:, None]
    return warped, inside


def resize_flow(flow, shape):
    height, width = flow.shape[-2:]
    result = F.interpolate(flow, size=shape, mode='bilinear', align_corners=False)
    scale = result.new_tensor([shape[1] / width, shape[0] / height])[None, :, None, None]
    return result * scale


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, (5, 3), padding=(2, 1)),
                         nn.GroupNorm(1, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.SiLU())


class RFRegistrationNet(nn.Module):
    """Three-scale shared encoder and local correlation, with bounded residuals.

    No downloaded weights. A coarse-to-fine field is measured in pixels at each
    scale; resize_flow scales both coordinates correctly. The pooled encoder is
    also a learned, pathology-free representation for exploratory phenotypes.
    """
    def __init__(self, width=12, radius=2, dropout=0.1):
        super().__init__()
        self.width, self.radius, self.dropout = width, radius, dropout
        self.encoders = nn.ModuleList([conv_block(1, width), conv_block(width, width),
                                       conv_block(width, width)])
        channels = width + (2 * radius + 1) ** 2 + 2
        self.refiners = nn.ModuleList([
            nn.Sequential(conv_block(channels, width), nn.Dropout2d(dropout),
                          nn.Conv2d(width, 2, 3, padding=1)) for _ in range(3)])
        for layer in self.refiners:
            nn.init.zeros_(layer[-1].weight)
            nn.init.zeros_(layer[-1].bias)

    def encode(self, image):
        features = []
        for index, encoder in enumerate(self.encoders):
            if index:
                image = F.avg_pool2d(image, 2)
            image = encoder(image)
            features.append(image)
        return features

    def correlation(self, a, b):
        a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
        r, height, width = self.radius, a.shape[-2], a.shape[-1]
        padded = F.pad(b, (r, r, r, r))
        return torch.cat([(a * padded[:, :, dz:dz+height, dx:dx+width]).sum(1, keepdim=True)
                          for dz in range(2*r+1) for dx in range(2*r+1)], dim=1)

    def forward(self, pre, post):
        a, b = self.encode(pre), self.encode(post)
        flow = pre.new_zeros((len(pre), 2, *a[-1].shape[-2:]))
        for level in (2, 1, 0):
            flow = resize_flow(flow, a[level].shape[-2:])
            moved, _ = warp_tensor(b[level], flow)
            corr = self.correlation(a[level], moved.to(a[level].dtype))
            residual = self.refiners[level](torch.cat((a[level], corr, flow.to(a[level].dtype)), 1))
            flow = flow.float() + 2.0 * torch.tanh(residual.float())
        return flow, a[-1].mean((2, 3)).float()


def local_correlation(a, b, window=9):
    """Signed local RF correlation; low-variance windows carry weak evidence."""
    pool = lambda t: F.avg_pool2d(t, window, stride=1, padding=window//2)
    ma, mb = pool(a), pool(b)
    va = (pool(a*a) - ma*ma).clamp_min(0)
    vb = (pool(b*b) - mb*mb).clamp_min(0)
    covariance = pool(a*b) - ma*mb
    return (covariance / torch.sqrt(va*vb + 1e-5)).clamp(-1, 1)


def masked_average(value, mask):
    mask = mask.to(value.dtype).expand_as(value)
    # Equal sample weighting makes accumulated minibatches equivalent even
    # when individual images have different valid-support sizes.
    numerator = (value*mask).flatten(1).sum(1)
    denominator = mask.flatten(1).sum(1).clamp_min(1)
    return (numerator/denominator).mean()


def window_support(mask, window=9):
    """Keep centers whose complete NCC window is inside the valid image."""
    return F.avg_pool2d(mask.float(), window, 1, window//2) >= 1-1e-6


def coordinate_overflow(flow):
    """Differentiable out-of-image displacement cost, in pixels, at each point."""
    height, width = flow.shape[-2:]
    z, x = torch.meshgrid(torch.arange(height, device=flow.device),
                          torch.arange(width, device=flow.device), indexing='ij')
    xx, zz = x[None]+flow[:, 0].float(), z[None]+flow[:, 1].float()
    return (F.relu(-xx)+F.relu(xx-(width-1))+F.relu(-zz)+F.relu(zz-(height-1))).mean()


def flow_jacobian(flow):
    uxz, uxx = torch.gradient(flow[:, 0].float(), dim=(1, 2))
    uzz, uzx = torch.gradient(flow[:, 1].float(), dim=(1, 2))
    return (1 + uxx) * (1 + uzz) - uxz * uzx


def registration_loss(pre, post, forward, backward, smooth_weight=0.02, cycle_weight=0.1):
    """Symmetric self-supervision; no initial-flow arrays or simulation labels.

    Out-of-view samples are masked, with a separate support penalty to prevent
    learning to move difficult pixels out of view. All loss arithmetic is FP32.
    """
    pre, post, forward, backward = [v.float() for v in (pre, post, forward, backward)]
    warped, inside = warp_tensor(post, forward)
    reverse_warped, reverse_inside = warp_tensor(pre, backward)
    valid_forward, valid_reverse = window_support(inside), window_support(reverse_inside)
    # Empty support must not look like a perfect zero reconstruction loss.
    photo = 0.5 * (masked_average(1-local_correlation(pre, warped), valid_forward)
                   + 2*(~valid_forward.flatten(1).any(1)).float().mean()
                   + masked_average(1-local_correlation(post, reverse_warped), valid_reverse)
                   + 2*(~valid_reverse.flatten(1).any(1)).float().mean())
    moved_backward, cycle_inside = warp_tensor(backward, forward)
    cycle = masked_average((forward + moved_backward).abs(), inside & cycle_inside)
    smooth = sum((field[:, :, 1:] - field[:, :, :-1]).abs().mean()
                 + (field[:, :, :, 1:] - field[:, :, :, :-1]).abs().mean()
                 for field in (forward, backward)) * 0.5
    fold = 0.5 * (F.relu(0.1-flow_jacobian(forward)).mean()
                  + F.relu(0.1-flow_jacobian(backward)).mean())
    # A differentiable displacement overflow penalty complements the discrete mask.
    support = 0.5*(coordinate_overflow(forward)+coordinate_overflow(backward))
    total = photo + cycle_weight*cycle + smooth_weight*smooth + 0.01*fold + 0.01*support
    if not torch.isfinite(total):
        raise FloatingPointError('Non-finite loss. Check input values and learning rate.')
    return total, {'photo': float(photo.detach()), 'cycle': float(cycle.detach()),
                   'smooth': float(smooth.detach()), 'fold': float(fold.detach()),
                   'support': float(support.detach())}


def axial_strain(flow):
    """Local least-squares derivative, dimensionless duz/dz in pixel coordinates."""
    uz = np.asarray(flow[1], dtype=np.float64)
    window = min(15, uz.shape[0] if uz.shape[0] % 2 else uz.shape[0]-1)
    if window < 3:
        return np.gradient(uz, axis=0)
    return savgol_filter(uz, window, polyorder=1, deriv=1, axis=0).astype(np.float32)


def bmode(rf):
    env = np.abs(hilbert(np.asarray(rf), axis=0))
    return np.clip(20*np.log10(np.maximum(env, 1e-8) / max(float(env.max()), 1e-8)), -60, 0)


def phase_translation(pre, post):
    """Whole-patch phase correlation baseline; returns a constant [ux,uz] field."""
    window = np.outer(np.hanning(pre.shape[0]), np.hanning(pre.shape[1]))
    a = np.fft.rfft2((pre-pre.mean())*window)
    b = np.fft.rfft2((post-post.mean())*window)
    cross = b*np.conj(a)
    cross /= np.maximum(np.abs(cross), 1e-8)
    correlation = np.fft.irfft2(cross, s=pre.shape)
    peak = np.array(np.unravel_index(np.argmax(correlation), pre.shape), dtype=float)
    # Parabolic subpixel refinement on a periodic peak neighborhood.
    for axis, size in enumerate(pre.shape):
        center = list(np.unravel_index(np.argmax(correlation), pre.shape))
        left, right = center.copy(), center.copy()
        left[axis], right[axis] = (center[axis]-1) % size, (center[axis]+1) % size
        aa, bb, cc = correlation[tuple(left)], correlation[tuple(center)], correlation[tuple(right)]
        denominator = aa - 2*bb + cc
        if abs(denominator) > 1e-12:
            peak[axis] += float(np.clip(0.5*(aa-cc)/denominator, -0.5, 0.5))
        if peak[axis] > size/2:
            peak[axis] -= size
    flow = np.zeros((2, *pre.shape), np.float32)
    flow[0], flow[1] = peak[1], peak[0]
    return flow


def quality_maps(pre, post, flow, backward, device):
    """Measured evidence for the feature-derived proxy, with explicit support."""
    pre, post, flow, backward = [np.asarray(v) for v in (pre, post, flow, backward)]
    if (pre.ndim != 2 or post.shape != pre.shape or flow.shape != (2, *pre.shape)
            or backward.shape != flow.shape or min(pre.shape) < 9
            or not all(np.isfinite(v).all() for v in (pre, post, flow, backward))):
        raise ValueError('Quality analysis requires matching finite 2D RF frames and [2,z,x] flows.')
    a = torch.as_tensor(pre, device=device)[None, None].float()
    b = torch.as_tensor(post, device=device)[None, None].float()
    u = torch.as_tensor(flow, device=device)[None].float()
    v = torch.as_tensor(backward, device=device)[None].float()
    with torch.no_grad():
        moved, inside = warp_tensor(b, u)
        reverse, valid_cycle = warp_tensor(v, u)
        residual = (1-local_correlation(a, moved))[0, 0].cpu().numpy()
        cycle = (u+reverse).square().sum(1).sqrt()[0].cpu().numpy()
        jacobian = flow_jacobian(u)[0].cpu().numpy()
    inside_array = inside[0, 0].cpu().numpy()
    out_of_frame_fraction = float(1-inside_array.mean())
    valid = window_support(inside)[0, 0].cpu().numpy()
    # Keep the strain derivative's edge estimates out of quantitative summaries.
    valid[:5] = valid[-5:] = False
    valid[:, :5] = valid[:, -5:] = False
    if not valid.any():
        raise ValueError('No valid interior pixels; increase patch size or reduce motion.')
    strain = axial_strain(flow)
    features = [float(residual[valid].mean()), float(np.median(cycle[valid])),
                float(np.median(np.abs(strain[valid]-np.median(strain[valid])))),
                float((jacobian[valid] <= 0).mean()), out_of_frame_fraction,
                float(np.median(np.sqrt((flow*flow).sum(0))[valid]))]
    return np.array(features), {'residual': residual, 'cycle': cycle, 'strain': strain,
                                'jacobian': jacobian, 'valid': valid,
                                'warped': moved[0, 0].cpu().numpy()}


def proxy_failure(features, residual_threshold=0.45, cycle_threshold=1.0):
    """Operational rule, NOT observed motion error. Thresholds are assumptions.

    The BN predicts this feature-derived event, so its calibration/ROC quantify
    rule imitation, not independent registration accuracy or clinical validity.
    Keep this caveat alongside every report and metric.
    """
    x = np.asarray(features)
    return ((x[..., 0] > residual_threshold) | (x[..., 1] > cycle_threshold)
            | (x[..., 3] > 0.01) | (x[..., 4] > 0.25)).astype(int)



# ---------------------------------------------------------------------------
# Real data: schema audit, bounded-memory preparation and grouped partitions.
# ---------------------------------------------------------------------------
from collections import Counter, defaultdict
from scipy.io import loadmat, whosmat
from scipy.io.matlab import MatReadError


def audit_data(data_root, output_dir, strict=True):
    """Inspect every MAT header without loading the complete dataset into RAM.

    The header describes array layout, not acquisition identity or physical
    units. In particular, existing flow initializers are never reference truth.
    A malformed file fails the audit; it is never silently skipped.
    """
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError('Data directory does not exist: ' + str(root))
    entries, domains = [], {}
    paths = sorted(root.rglob('*.mat'))
    if not paths:
        raise ValueError('No MATLAB files found below ' + str(root))
    for index, path in enumerate(paths):
        relative = path.relative_to(root).as_posix()
        read_error = None
        try:
            variables = [{'name': name, 'shape': list(shape), 'dtype': dtype}
                         for name, shape, dtype in whosmat(path)]
        except Exception as exc:
            if strict:
                raise ValueError(f'Cannot inspect MAT file {relative}: {exc}') from exc
            variables = []
            read_error = f'{type(exc).__name__}: {exc}'
        stat = path.stat()
        entries.append({'path': relative, 'domain': relative.split('/')[0],
                        'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                        'variables': variables, 'read_error': read_error})
        if (index + 1) % 250 == 0:
            print(f'Audit: {index + 1}/{len(paths)} MAT headers', flush=True)
    for domain in sorted({entry['domain'] for entry in entries}):
        selected = [entry for entry in entries if entry['domain'] == domain]
        schemas = Counter(json.dumps(entry['variables'], sort_keys=True) for entry in selected)
        domains[domain] = {'files': len(selected), 'bytes': sum(entry['bytes'] for entry in selected),
                           'schemas': [{'count': count, 'variables': json.loads(key)}
                                       for key, count in schemas.items()]}
    audit = {'data_root': str(root), 'mat_files': len(entries),
             'total_bytes': sum(entry['bytes'] for entry in entries),
             'domains': domains, 'files': entries,
             'notes': ['Headers do not establish histopathology grades, spacing, or patient identity.',
                       'phantom: Data[2,z,x] is used; flow_f_init/flow_b_init are excluded.',
                       'simulation: RF leading axes need interpretation; axial/lateral companions are excluded.',
                       'test: rf_pre[96,96,10] has unknown provenance and is excluded from this experiment.']}
    write_json(Path(output_dir) / 'data_audit.json', audit)
    return audit


def _source_signature(path):
    stat = Path(path).stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _resumable_rows(state, extraction):
    """Fast restart check; changed cache files are rebuilt from the MAT source.

    Content hashes are established at creation. Size/mtime checks avoid reading
    every cached pixel on every restart; they are not an adversarial integrity
    guarantee. Newly encountered existing cache content is checked below.
    """
    if not state or state.get('extraction') != extraction:
        return None
    rows = state.get('records', [])
    if not rows:
        return None
    for item in rows:
        path = Path(item['path'])
        if not path.is_file() or item.get('cache_signature') != _source_signature(path):
            return None
        try:
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            valid = list(array.shape) == item['shape'] and array.dtype == np.dtype('float32')
            del array
        except (OSError, ValueError):
            return None
        if not valid:
            return None
    return rows


def _read_group_map(path, root):
    """CSV accepts source,group or path,group. Paths are relative to data root.

    Group IDs come from the user's acquisition records. Merely supplying a CSV
    cannot independently verify them, which is stated in output provenance.
    """
    if path is None:
        return {}
    result = {}
    with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        key = 'source' if 'source' in columns else 'path'
        if key not in columns or 'group' not in columns:
            raise ValueError('Group CSV requires columns source,group (or path,group).')
        for row in reader:
            source, group = row[key].strip(), row['group'].strip()
            if not source or not group:
                raise ValueError('Empty source or group in group-map CSV.')
            candidate = Path(source)
            if candidate.is_absolute():
                try:
                    source = candidate.resolve().relative_to(root).as_posix()
                except ValueError as exc:
                    raise ValueError('Group-map source lies outside data root: ' + source) from exc
            else:
                source = source.replace('\\', '/').removeprefix('./')
            if source in result and result[source] != group:
                raise ValueError('Conflicting acquisition groups for ' + source)
            result[source] = group
    return result


def _geometry_name(path):
    match = re.match(r'^(model\d+)(?:_|\.)', Path(path).name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError('Cannot identify simulation geometry from filename: ' + str(path))
    return match.group(1).lower()


def _simulation_sources(paths):
    """First visit one frequency file per geometry, then additional frequencies.

    A small --limit therefore covers independent geometries instead of selecting
    two frequencies of the same geometry before any new geometry is visited.
    """
    groups = defaultdict(list)
    for path in paths:
        groups[_geometry_name(path)].append(path)
    names = sorted(groups, key=lambda name: int(name[5:]))
    for name in names:
        groups[name].sort(key=lambda path: path.name)
    return [groups[name][index] for index in range(max(map(len, groups.values()), default=0))
            for name in names if index < len(groups[name])]


def _cache_pair(pair, cache):
    """Hash canonical float32 data and save it once as a memory-mappable NPY.

    This content hash catches exact duplicates across different filenames and
    frequencies after conversion. Similar frames still need acquisition groups.
    Temporary files are atomically replaced only after a complete write.
    """
    pair = np.asarray(pair)
    if pair.ndim != 3 or pair.shape[0] != 2 or min(pair.shape[1:]) < 2:
        raise ValueError('RF pair must have shape [2,z,x] with nontrivial spatial dimensions.')
    validate_rf_storage_values(pair)
    pair = np.ascontiguousarray(pair, dtype=np.float32)
    if any(float(frame.min()) == float(frame.max()) for frame in pair):
        raise RFDataQualityError('A complete RF frame is constant after storage conversion.',
                                 {'reason': 'constant_frame_after_storage_conversion'})
    mean = float(pair.mean(dtype=np.float64))
    scale = float(np.sqrt(np.mean((pair.astype(np.float64) - mean) ** 2)))
    if not np.isfinite(scale) or scale <= 0:
        raise RFDataQualityError('Constant/empty RF pair cannot support registration.',
                                 {'reason': 'constant_after_storage_conversion'})
    digest = hashlib.sha256()
    digest.update(str(tuple(pair.shape)).encode('ascii'))
    digest.update(memoryview(pair).cast('B'))
    content_hash = digest.hexdigest()
    target = cache / 'pairs' / (content_hash + '.npy')
    valid_existing = False
    if target.exists():
        # Never accept an existing file solely because its filename is a hash.
        try:
            existing = np.load(target, mmap_mode='r', allow_pickle=False)
            if existing.shape == pair.shape and existing.dtype == np.dtype('float32'):
                check = hashlib.sha256()
                check.update(str(tuple(existing.shape)).encode('ascii'))
                check.update(memoryview(existing).cast('B'))
                valid_existing = check.hexdigest() == content_hash
            del existing
        except (OSError, ValueError, EOFError):
            valid_existing = False
    if not valid_existing:
        temporary = target.with_suffix('.npy.tmp')
        mapped = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=pair.shape)
        mapped[:] = pair
        mapped.flush()
        del mapped
        temporary.replace(target)
    return {'path': str(target.resolve()), 'sha256': content_hash,
            'shape': list(pair.shape), 'pair_mean': mean, 'pair_rms_centered': scale,
            'cache_signature': _source_signature(target)}


def _split_prepared_records(records, seed):
    """Merge duplicate/acquisition lineages before any split is assigned.

    Union-find preserves shared simulation geometry across frequencies and
    merges acquisition groups if they contain an identical cached pair. Pairs
    with identical content become one record carrying all source aliases.
    """
    parent = {}

    def find(item):
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(left, right):
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    hashes = {}
    for record in records:
        groups = record['lineage_groups']
        for group in groups:
            union(groups[0], group)
        previous = hashes.setdefault(record['sha256'], groups[0])
        union(previous, groups[0])
    unique = {}
    for record in records:
        group = find(record['lineage_groups'][0])
        alias = {'source': record['source'], 'source_id': record['id'],
                 'domain': record['domain'], 'lineage_groups': record['lineage_groups'],
                 'pair_selection': record.get('pair_selection')}
        if record['sha256'] in unique:
            unique[record['sha256']]['aliases'].append(alias)
        else:
            item = dict(record)
            item['group'], item['aliases'] = group, [alias]
            unique[record['sha256']] = item
    output = list(unique.values())
    by_domain = defaultdict(set)
    component_domains = defaultdict(set)
    for item in output:
        for alias in item['aliases']:
            component_domains[item['group']].add(alias['domain'])
    # Assign a mixed-domain component once, to its deterministic owner domain.
    # It remains indivisible; a final domain/role audit refuses missing coverage.
    for group, names in component_domains.items():
        by_domain[sorted(names)[0]].add(group)
    assignments = {}
    rng = np.random.default_rng(seed)
    for domain, groups in sorted(by_domain.items()):
        groups = sorted(groups)
        if len(groups) < 4:
            raise ValueError(f'{domain}: only {len(groups)} independent groups after duplicate consolidation. '
                             'At least four are required; increase --limit or review group metadata.')
        groups = [groups[index] for index in rng.permutation(len(groups))]
        # Reserve one group for each role before approximating 60/15/10/15.
        count = len(groups)
        counts = [max(1, int(count * fraction)) for fraction in (0.60, 0.15, 0.10)]
        while sum(counts) >= count:
            largest = int(np.argmax(counts))
            counts[largest] -= 1
        counts.append(count - sum(counts))
        start = 0
        for role, amount in zip(('train', 'validation', 'calibration', 'test'), counts):
            for group in groups[start:start + amount]:
                assignments[group] = role
            start += amount
    for item in output:
        item['split'] = assignments[item['group']]
        item['id'] = item['domain'] + '_' + item['sha256'][:20]
    for domain in {alias['domain'] for item in output for alias in item['aliases']}:
        roles = {item['split'] for item in output
                 if any(alias['domain'] == domain for alias in item['aliases'])}
        if roles != {'train', 'validation', 'calibration', 'test'}:
            raise ValueError(f'Duplicate consolidation leaves {domain} without all four partitions.')
    return sorted(output, key=lambda item: (item['domain'], item['source'], item['id']))


def write_data_rejections(cache, rejections, source_count, policy):
    """Persist every exclusion immediately, even if preparation later stops."""
    report = {'policy': policy, 'selected_source_files': source_count,
              'quarantined_source_count': len(rejections), 'sources': list(rejections.values()),
              'raw_file_action': 'preserved',
              'interpretation': 'Excluded corrupt/unrepresentable source files; no RF values were clipped, interpolated or replaced.'}
    write_json(Path(cache)/'data_rejections.json', report)
    return report


def prepare_data(data_root, cache_dir, seed=42, group_map=None, limit=0,
                 simulation_frame_axis=1, simulation_pairs_per_file=20,
                 domains=('phantom', 'simulation'), invalid_sources='quarantine'):
    """Prepare both RF domains, using no displacement or pathology labels.

    Only one MAT source is loaded at a time (the largest RF variable is roughly
    200 MiB). Full-resolution pairs are cached on disk, not retained in RAM.
    Resume metadata uses source size/mtime and extraction settings; it is not a
    claim that a source checksum has been independently verified.
    """
    if simulation_frame_axis not in (0, 1) or simulation_pairs_per_file < 0 or limit < 0:
        raise ValueError('Invalid frame axis, pair cap, or source limit.')
    if invalid_sources not in ('quarantine', 'error'):
        raise ValueError('Invalid-source policy must be quarantine or error.')
    domains = tuple(dict.fromkeys(domains))
    if not domains or any(domain not in ('phantom', 'simulation') for domain in domains):
        raise ValueError('Select phantom and/or simulation domains.')
    root, cache = Path(data_root).resolve(), Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    (cache / 'pairs').mkdir(exist_ok=True)
    (cache / 'preparation').mkdir(exist_ok=True)
    audit = audit_data(root, cache, strict=invalid_sources == 'error')
    groups = _read_group_map(group_map, root)
    sources = []
    for domain in domains:
        paths = sorted((root / domain).glob('*.mat'))
        if domain == 'simulation':
            paths = _simulation_sources([path for path in paths if path.stem.endswith('_all')])
        if limit:
            paths = paths[:limit]
        if not paths:
            raise ValueError('No supported RF source files in ' + str(root / domain))
        sources.extend((domain, path) for path in paths)
    notes = ['RF motion learning is self-supervised; neither pathology grades nor true displacement targets are supplied.',
             'Initial flows and simulation axial/lateral companion files are excluded from training and accuracy claims.',
             'data/test is inventoried but excluded because its sequence/provenance contract is unknown.',
             'Exact float32 pair duplicates are consolidated before group splitting. Near-duplicates require acquisition IDs.',
             '60/15/10/15 are target group fractions; rounding reserves at least one group per role in each domain.']
    if 'phantom' in domains:
        if groups:
            notes.append('Phantom acquisition IDs are user supplied through CSV and have not been independently verified.')
            missing = [path.relative_to(root).as_posix() for domain, path in sources
                       if domain == 'phantom' and path.relative_to(root).as_posix() not in groups]
            if missing:
                raise ValueError('Group CSV must cover every selected phantom source; first missing: ' + missing[0])
        else:
            notes.append('PROVISIONAL PHANTOM SPLIT: file identities are used because acquisition IDs are absent. '
                         'Correlated frames may remain across partitions; results do not establish acquisition-level generalization.')
    if 'simulation' in domains:
        notes.append(f'UNVERIFIED SIMULATION AXIS ASSUMPTION: RF leading axis {simulation_frame_axis} is ordered deformation; '
                     f'axis {1-simulation_frame_axis} is realization. Adjacent selected frames are paired. '
                     'All realizations/frequencies of modelN share one geometry group.')
    print('\n'.join(notes), flush=True)
    # Conservative disk preflight uses header shapes and completed-source state.
    # Missing pairs may deduplicate later, so this is an upper bound, not a claim
    # that every selected sample will consume a distinct file on disk.
    schema_by_source = {item['path']: item['variables'] for item in audit['files']}
    header_errors = {item['path']: item['read_error'] for item in audit['files'] if item.get('read_error')}
    required_bytes = 0
    rejections = {}
    source_states = {}
    for domain, path in sources:
        relative = path.relative_to(root).as_posix()
        extraction = {'version': 2, 'data_root': str(root), 'source': relative,
                      'signature': _source_signature(path), 'simulation_frame_axis': simulation_frame_axis,
                      'simulation_pairs_per_file': simulation_pairs_per_file}
        state_key = hashlib.sha256(relative.encode()).hexdigest()[:24]
        state_path = cache / 'preparation' / (state_key + '.json')
        state = read_json(state_path) if state_path.exists() else None
        if relative in header_errors:
            rejection = {'source': relative, 'domain': domain, 'source_signature': extraction['signature'],
                         'message': 'Cannot read MAT header: '+header_errors[relative],
                         'reason': 'unreadable_mat_source', 'action': 'source_quarantined'}
            state = {'extraction': extraction, 'status': 'quarantined', 'records': [],
                     'quality_validation_version': 1, 'rejection': rejection}
            write_json(state_path, state)
        if (state and state.get('extraction') == extraction and state.get('status') == 'quarantined'
                and state.get('quality_validation_version') == 1):
            rejection = state['rejection']
            rejections[relative] = rejection
            write_data_rejections(cache, rejections, len(sources), invalid_sources)
            if invalid_sources == 'error':
                raise RFDataQualityError(relative+': '+rejection['message'], rejection)
        rows = _resumable_rows(state, extraction)
        source_states[relative] = (extraction, state_path, rows)
        if rows is None and relative not in rejections:
            key = 'Data' if domain == 'phantom' else 'RF'
            shape = next((item['shape'] for item in schema_by_source[relative] if item['name'] == key), None)
            if shape is None or len(shape) != (3 if domain == 'phantom' else 4):
                raise ValueError(f'{relative}: missing or unsupported {key} array shape.')
            pairs = 1 if domain == 'phantom' else shape[1-simulation_frame_axis] * (shape[simulation_frame_axis]-1)
            if domain == 'simulation' and simulation_pairs_per_file:
                pairs = min(pairs, simulation_pairs_per_file)
            required_bytes += pairs * (2 * shape[-2] * shape[-1] * 4 + 256)
    free_bytes = shutil.disk_usage(cache).free
    if required_bytes + 1024**2 > free_bytes:
        raise ValueError(f'Insufficient free cache space: conservative additional requirement '
                         f'{required_bytes/1024**3:.2f} GiB; available {free_bytes/1024**3:.2f} GiB. '
                         'Use --limit or reduce --simulation-pairs-per-file, or choose another --cache drive.')
    print(f'Cache preflight: at most {required_bytes/1024**3:.2f} additional GiB; '
          f'{free_bytes/1024**3:.2f} GiB free.', flush=True)
    prepared = []
    for source_index, (domain, path) in enumerate(sources):
        relative = path.relative_to(root).as_posix()
        extraction, state_path, rows = source_states[relative]
        if relative in rejections:
            print(f'QUARANTINED {relative}: {rejections[relative]["message"]} (recorded source unchanged)', flush=True)
            continue
        if rows is None:
            array = contents = None
            try:
                key = 'Data' if domain == 'phantom' else 'RF'
                try:
                    contents = loadmat(path, variable_names=[key])
                except (OSError, ValueError, EOFError, zlib.error, MatReadError) as exc:
                    raise RFDataQualityError('Cannot read MAT source: '+str(exc),
                                             {'reason': 'unreadable_mat_source',
                                              'reader_error': type(exc).__module__+'.'+type(exc).__name__}) from exc
                if key not in contents:
                    raise ValueError(f'{relative} lacks required array {key}.')
                array = contents[key]
                if not np.issubdtype(array.dtype, np.floating):
                    raise ValueError(f'{relative}: expected real floating-point RF data.')
                validate_rf_storage_values(array)
                rows = []
                if domain == 'phantom':
                    if array.ndim != 3 or array.shape[0] != 2:
                        raise ValueError(f'{relative}: expected Data[2,z,x], found {array.shape}.')
                    selections = [(None, None)]
                else:
                    if array.ndim != 4 or min(array.shape[:2]) < 2:
                        raise ValueError(f'{relative}: expected RF[a,b,z,x], found {array.shape}.')
                    realizations, frames = array.shape[1-simulation_frame_axis], array.shape[simulation_frame_axis]
                    selections = [(realization, frame) for realization in range(realizations) for frame in range(frames-1)]
                    if simulation_pairs_per_file and len(selections) > simulation_pairs_per_file:
                        chosen = np.linspace(0, len(selections)-1, simulation_pairs_per_file, dtype=int)
                        selections = [selections[index] for index in chosen]
                for realization, frame in selections:
                    if domain == 'phantom':
                        pair, selection = array, None
                    elif simulation_frame_axis == 1:
                        pair = array[realization, frame:frame+2]
                        selection = {'realization': realization, 'frames': [frame, frame+1], 'frame_axis': 1}
                    else:
                        pair = array[frame:frame+2, realization]
                        selection = {'realization': realization, 'frames': [frame, frame+1], 'frame_axis': 0}
                    item = _cache_pair(pair, cache)
                    item.update(id=f'{relative}:{realization}:{frame}', domain=domain, source=relative,
                                source_signature=extraction['signature'], pair_selection=selection)
                    rows.append(item)
                write_json(state_path, {'extraction': extraction, 'records': rows})
            except RFDataQualityError as exc:
                rejection = {'source': relative, 'domain': domain, 'source_signature': extraction['signature'],
                             'message': str(exc), 'action': 'source_quarantined', **exc.details}
                rejections[relative] = rejection
                write_json(state_path, {'extraction': extraction, 'status': 'quarantined', 'records': [],
                                        'quality_validation_version': 1, 'rejection': rejection})
                write_data_rejections(cache, rejections, len(sources), invalid_sources)
                if invalid_sources == 'error':
                    raise RFDataQualityError(relative+': '+str(exc), rejection) from exc
                print(f'QUARANTINED {relative}: {exc}; see {cache / "data_rejections.json"}', flush=True)
                continue
            finally:
                pair = None  # Release a slice that can otherwise retain an entire RF sequence.
                del array, contents
        # Group maps and split seeds can change without recopying RF arrays.
        for item in rows:
            item = dict(item)
            native = ('simulation:' + _geometry_name(path) if domain == 'simulation' else 'phantom:file:' + relative)
            item['lineage_groups'] = [native]
            if relative in groups:
                item['lineage_groups'].append('acquisition:' + groups[relative])
            prepared.append(item)
        print(f'Prepare {source_index+1}/{len(sources)}: {relative}, {len(rows)} pairs', flush=True)
    quality = write_data_rejections(cache, rejections, len(sources), invalid_sources)
    if rejections:
        notes.append(f'DATA QUALITY: {len(rejections)} corrupt/unrepresentable sources quarantined; exclusions recorded in data_rejections.json.')
    records = _split_prepared_records(prepared, seed)
    manifest = {'version': VERSION, 'data_root': str(root), 'cache_root': str(cache),
                'seed': seed, 'domains': list(domains), 'records': records, 'notes': notes,
                'axes_assumption': {'spatial': '[z,x]', 'simulation_frame_axis': simulation_frame_axis,
                                    'simulation_verified': False},
                'group_map': str(Path(group_map).resolve()) if group_map else None,
                'source_files': len(sources), 'pairs_before_deduplication': len(prepared),
                'data_quality': quality,
                'unique_pairs': len(records), 'duplicates_consolidated': len(prepared)-len(records),
                'float32_payload_bytes': sum(math.prod(item['shape'])*4 for item in records),
                'audit_mat_files': audit['mat_files'], 'pathology_grade_available': False,
                'split_counts': {domain: dict(Counter(item['split'] for item in records if item['domain'] == domain))
                                 for domain in domains}}
    write_json(cache / 'manifest.json', manifest)
    return manifest


# ---------------------------------------------------------------------------
# MIS Units 1--3: probabilistic representations, inference and research risk.
# This section has no dependency on the older multi-file project. NumPy,
# scipy.special.logsumexp and itertools are imported at the top of this file.
# ---------------------------------------------------------------------------

def _pgm_matrix(X, columns=None, allow_empty=False):
    """Check observations before fitting or evaluating probabilistic models."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] == 0 or (len(X) == 0 and not allow_empty):
        raise ValueError('PGM observations must be a nonempty N x D matrix.')
    if columns is not None and X.shape[1] != columns:
        raise ValueError('Feature count differs from the fitted risk model.')
    if not np.isfinite(X).all():
        raise ValueError('PGM observations must be finite; missing evidence uses bayesian_query().')
    return X


def _pgm_mixture_log_joint(binary, weights, emission):
    return (binary @ np.log(emission).T
            + (1-binary) @ np.log1p(-emission).T + np.log(weights)[None, :])


def fit_risk_model(X, y, seed=42):
    """Fit ONLY training observations; callers must split by group beforehand.

    Unit 1 representation learning: a two-component Bernoulli mixture learns
    latent evidence patterns without using y. EM optimizes its likelihood.
    Component IDs have no intrinsic meaning or quality/grade ordering.

    Unit 1 supervised BN: F=operational proxy failure, Z=latent pattern,
    Xi=binarized feature i. The factorization is
        P(F,Z,X) = P(F) P(Z|F) product_i P(Xi|F,Z).
    Smoothed soft-count CPDs use the mixture responsibilities. This is a
    transparent two-stage estimator, not joint maximum-likelihood BN EM.
    The caller defines y from a declared operational rule, not pathology.
    This function cannot infer provenance from arrays: only TRAIN data belong
    here. Validation/calibration/test observations must not affect fitting.
    """
    X = _pgm_matrix(X)
    y = np.asarray(y)
    if y.shape != (len(X),) or not np.isin(y, [0, 1]).all():
        raise ValueError('One binary operational-failure label is required per training row.')
    y = y.astype(int)
    threshold = np.median(X, axis=0)  # Frozen training-only discretizer.
    binary = (X > threshold).astype(float)
    rng = np.random.default_rng(seed)
    states = 2
    weights = np.full(states, 1/states)
    emission = rng.uniform(0.2, 0.8, size=(states, X.shape[1]))
    history = []
    # The small numerical bounds prevent log(0); they are not clinical priors.
    for iteration in range(100):
        log_joint = _pgm_mixture_log_joint(binary, weights, emission)
        normalization = logsumexp(log_joint, axis=1)
        history.append(float(normalization.mean()))
        responsibility = np.exp(log_joint-normalization[:, None])
        count = responsibility.sum(axis=0)
        next_weights = np.maximum(count/len(X), 1e-8)
        next_weights /= next_weights.sum()
        next_emission = np.clip((responsibility.T @ binary)/np.maximum(count[:, None], 1e-12), 1e-6, 1-1e-6)
        change = max(float(np.max(abs(next_weights-weights))),
                     float(np.max(abs(next_emission-emission))))
        weights, emission = next_weights, next_emission
        if change < 1e-7:
            break
    log_joint = _pgm_mixture_log_joint(binary, weights, emission)
    normalization = logsumexp(log_joint, axis=1)
    history.append(float(normalization.mean()))
    responsibility = np.exp(log_joint-normalization[:, None])

    # Dirichlet/Beta(1,1) pseudocounts keep every assignment possible, including
    # small training sets or a training split containing only one failure class.
    alpha = 1.0
    prior = (np.bincount(y, minlength=2)+alpha)/(len(y)+2*alpha)
    cpd_z = np.zeros((2, states))
    cpd_x = np.zeros((2, states, X.shape[1]))
    for failure in range(2):
        membership = responsibility*(y == failure)[:, None]
        count = membership.sum(axis=0)
        cpd_z[failure] = (count+alpha)/(count.sum()+states*alpha)
        cpd_x[failure] = (membership.T @ binary+alpha)/(count[:, None]+2*alpha)
    return {
        'schema_version': 1, 'fit_role': 'train', 'seed': int(seed),
        'endpoint': 'declared_operational_proxy_failure_not_tumor_grade',
        'factorization': 'P(F) P(Z|F) product_i P(Xi|F,Z)',
        'estimator': 'unsupervised_EM_representation_then_supervised_soft_count_CPDS',
        'feature_count': int(X.shape[1]), 'training_rows': int(len(X)),
        'training_class_counts': np.bincount(y, minlength=2).tolist(),
        'thresholds': threshold.tolist(), 'discretization': 'Xi=1 iff feature_i > training_median_i',
        'prior': prior.tolist(), 'cpd_z': cpd_z.tolist(), 'cpd_x': cpd_x.tolist(),
        'pseudocount': alpha,
        'representation': {'weights': weights.tolist(), 'emission': emission.tolist(),
                           'log_likelihood_history': history,
                           'interpretation': 'Unordered latent evidence patterns, not tumor grades.'},
        'probability_status': 'Uncalibrated model probability until independent calibration.',
        'warning': ('Training data contain one outcome class; discrimination is not established.'
                    if len(np.unique(y)) < 2 else None),
    }


def _pgm_parameters(model):
    """Validate a JSON-loaded model instead of assuming checkpoint integrity."""
    d = int(model['feature_count'])
    prior = np.asarray(model['prior'], float)
    cpd_z = np.asarray(model['cpd_z'], float)
    cpd_x = np.asarray(model['cpd_x'], float)
    threshold = np.asarray(model['thresholds'], float)
    if (d < 1 or prior.shape != (2,) or cpd_z.ndim != 2 or cpd_z.shape[0] != 2
            or cpd_z.shape[1] < 1 or cpd_x.shape != (2, cpd_z.shape[1], d)
            or threshold.shape != (d,)):
        raise ValueError('Incompatible Bayesian-risk CPD dimensions.')
    if not all(np.isfinite(a).all() for a in (prior, cpd_z, cpd_x, threshold)):
        raise ValueError('Bayesian-risk parameters must be finite.')
    if (np.any(prior <= 0) or np.any(cpd_z <= 0) or np.any(cpd_x <= 0)
            or np.any(cpd_x >= 1) or not np.isclose(prior.sum(), 1)
            or not np.allclose(cpd_z.sum(axis=1), 1)):
        raise ValueError('Invalid Bayesian-risk CPD probabilities.')
    return threshold, prior, cpd_z, cpd_x


def predict_risk(model, X):
    """Exact P(F=1|X), summing hidden Z in log space; never refits thresholds."""
    threshold, prior, cpd_z, cpd_x = _pgm_parameters(model)
    X = _pgm_matrix(X, len(threshold), allow_empty=True)
    binary = (X > threshold).astype(float)
    # N x F x Z; independent feature factors are accumulated in log space.
    scores = (np.einsum('nd,fkd->nfk', binary, np.log(cpd_x))
              + np.einsum('nd,fkd->nfk', 1-binary, np.log1p(-cpd_x))
              + np.log(prior)[None, :, None] + np.log(cpd_z)[None, :, :])
    marginal = logsumexp(scores, axis=2)
    return np.exp(marginal[:, 1]-logsumexp(marginal, axis=1))


def bayesian_query(model, variable='F', evidence=None):
    """Exact marginal for F, Z, or Xi; omit missing binary evidence entirely.

    Evidence is already discretized, e.g. {'X0': 1, 'X2': 0}. Unobserved
    feature factors sum to one. Thus even a many-feature query only sums the
    tiny F x Z state space. Unknown keys/categories raise an error.
    """
    threshold, prior, cpd_z, cpd_x = _pgm_parameters(model)
    evidence = {} if evidence is None else dict(evidence)
    domains = {'F': 2, 'Z': cpd_z.shape[1]}
    domains.update({'X'+str(i): 2 for i in range(len(threshold))})
    if variable not in domains:
        raise ValueError('Unknown query variable: '+str(variable))
    for key, value in evidence.items():
        if key not in domains or not isinstance(value, (int, np.integer, bool)) or not 0 <= int(value) < domains[key]:
            raise ValueError('Unknown evidence key or invalid category: '+str(key))

    def log_mass(assignment):
        terms = []
        for failure in range(2):
            for latent in range(cpd_z.shape[1]):
                if ('F' in assignment and assignment['F'] != failure) or ('Z' in assignment and assignment['Z'] != latent):
                    continue
                score = np.log(prior[failure])+np.log(cpd_z[failure, latent])
                for key, value in assignment.items():
                    if key.startswith('X'):
                        probability = cpd_x[failure, latent, int(key[1:])]
                        score += np.log(probability if value else 1-probability)
                terms.append(score)
        return float(logsumexp(terms)) if terms else -np.inf

    denominator = log_mass(evidence)
    if not np.isfinite(denominator):
        raise ValueError('Impossible evidence.')
    result = np.zeros(domains[variable])
    for value in range(domains[variable]):
        if variable in evidence and evidence[variable] != value:
            continue
        result[value] = np.exp(log_mass(dict(evidence, **{variable: value}))-denominator)
    return result/result.sum()


def markov_separated(edges, first, second, given=()):
    """Undirected graph separation: remove given nodes, then search paths."""
    first, second, given = set(first), set(second), set(given)
    if first & second or first & given or second & given:
        raise ValueError('Conditional-independence node sets must be disjoint.')
    adjacency = {}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    seen = set(given)
    stack = list(first)
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        if node in second:
            return False
        seen.add(node)
        stack.extend(adjacency.get(node, set())-seen)
    return True


def moralize_parents(parents):
    """Marry co-parents and drop directions; CPDs remain family potentials."""
    edges = set()
    seen = set()
    for child, family in parents.items():
        if not set(family) <= seen or len(set(family)) != len(family):
            raise ValueError('Parents must form a topologically ordered DAG.')
        edges.update(tuple(sorted((parent, child))) for parent in family)
        edges.update(tuple(sorted(pair)) for pair in itertools.combinations(family, 2))
        seen.add(child)
    return [list(edge) for edge in sorted(edges)]


def categorical_hmm(initial, transition, emission, observations, ordered=False):
    """Unit 2: log-forward/backward and Viterbi on an explicitly ordered sequence."""
    if not ordered:
        raise ValueError('HMM requires verified temporal order or an explicit synthetic sequence.')
    initial, transition, emission = (np.asarray(a, float) for a in (initial, transition, emission))
    if (initial.ndim != 1 or len(initial) < 1 or transition.shape != (len(initial), len(initial))
            or emission.ndim != 2 or emission.shape[0] != len(initial) or emission.shape[1] < 1):
        raise ValueError('Incompatible HMM dimensions.')
    for a in (initial, transition, emission):
        if not np.isfinite(a).all() or np.any(a <= 0) or not np.allclose(a.sum(axis=-1), 1):
            raise ValueError('HMM requires finite, positive, normalized probabilities.')
    raw_observations = np.asarray(observations)
    if (raw_observations.ndim != 1 or len(raw_observations) < 1
            or not np.isfinite(raw_observations).all()
            or not np.isin(raw_observations, np.arange(emission.shape[1])).all()):
        raise ValueError('HMM observations must be nonempty legal emission categories.')
    observations = raw_observations.astype(int)
    n, k = len(observations), len(initial)
    log_transition = np.log(transition)
    log_emission = np.log(emission[:, observations].T)
    forward = np.empty((n, k))
    forward[0] = np.log(initial)+log_emission[0]
    for t in range(1, n):
        forward[t] = log_emission[t]+logsumexp(forward[t-1, :, None]+log_transition, axis=0)
    backward = np.zeros((n, k))
    for t in range(n-2, -1, -1):
        backward[t] = logsumexp(log_transition+log_emission[t+1, None, :]+backward[t+1, None, :], axis=1)
    filtered = np.exp(forward-logsumexp(forward, axis=1, keepdims=True))
    smoothed = np.exp(forward+backward-logsumexp(forward+backward, axis=1, keepdims=True))
    score = np.empty((n, k))
    score[0] = forward[0]
    pointer = np.zeros((n, k), int)
    for t in range(1, n):
        candidate = score[t-1, :, None]+log_transition
        pointer[t] = candidate.argmax(axis=0)
        score[t] = candidate.max(axis=0)+log_emission[t]
    path = np.zeros(n, int)
    path[-1] = score[-1].argmax()
    for t in range(n-2, -1, -1):
        path[t] = pointer[t+1, path[t+1]]
    return {'filtered': filtered.tolist(), 'smoothed': smoothed.tolist(),
            'viterbi': path.tolist(), 'log_likelihood': float(logsumexp(forward[-1]))}


def expected_risk_decision(probability, loss=None):
    """Unit 3: minimize expected operational cost; columns = nonfailure,failure."""
    probability = np.asarray(probability, float)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Failure probabilities must be finite and between zero and one.')
    loss = np.array([[0., 8.], [1., 1.5], [2.5, .5]]) if loss is None else np.asarray(loss, float)
    if loss.shape != (3, 2) or not np.isfinite(loss).all():
        raise ValueError('Expected a finite 3-action x 2-state loss table.')
    probabilities = np.stack((1-probability, probability), axis=-1)
    expected = probabilities @ loss.T
    index = expected.argmin(axis=-1)
    actions = np.asarray(['accept_mechanics', 'reacquire', 'expert_review'])
    return {'action': actions[index].tolist(), 'action_index': index.tolist(),
            'expected_cost': expected.tolist(), 'loss': loss.tolist(),
            'cost_provenance': 'Illustrative research handling costs, not a clinical policy.'}


def mrf_refine(probability, image, beta=0.6, iterations=8):
    """Contrast-sensitive four-neighbor Potts ICM; labels denote risk patterns.

    E(y) = sum_i -log P_i(y_i) + beta sum_(i,j) w_ij [y_i != y_j].
    The image determines fixed positive edge weights. Red/black checkerboard
    updates are independent within each color on a four-neighbor grid, so
    each half-sweep cannot increase energy. This is local MAP optimization,
    not exact marginal inference or a guarantee of global optimality.
    """
    probability, image = np.asarray(probability, float), np.asarray(image, float)
    if (probability.ndim != 2 or 0 in probability.shape or image.shape != probability.shape
            or not np.isfinite(probability).all() or not np.isfinite(image).all()
            or np.any((probability < 0) | (probability > 1))):
        raise ValueError('MRF needs matching finite 2D image and [0,1] probabilities.')
    if not np.isfinite(beta) or beta < 0 or not isinstance(iterations, (int, np.integer)) or iterations < 0:
        raise ValueError('MRF beta must be nonnegative and iterations a nonnegative integer.')
    p = np.clip(probability, 1e-8, 1-1e-8)
    unary = np.stack((-np.log1p(-p), -np.log(p)), axis=-1)
    vertical_difference, horizontal_difference = np.diff(image, axis=0), np.diff(image, axis=1)
    differences = np.concatenate((vertical_difference.ravel(), horizontal_difference.ravel()))
    scale = max(float(np.sqrt(np.mean(differences**2))) if differences.size else 0.0, 1e-6)
    vertical = beta*np.exp(-0.5*(vertical_difference/scale)**2)
    horizontal = beta*np.exp(-0.5*(horizontal_difference/scale)**2)
    labels = (probability > 0.5).astype(np.int64)
    row, column = np.indices(labels.shape)
    parity = (row+column) % 2

    def energy():
        return float(unary[row, column, labels].sum()
                     + (vertical*(labels[1:] != labels[:-1])).sum()
                     + (horizontal*(labels[:, 1:] != labels[:, :-1])).sum())

    energies = [energy()]
    for _ in range(iterations):
        changes = 0
        for color in (0, 1):
            costs = unary.copy()
            for state in (0, 1):
                costs[1:, :, state] += vertical*(labels[:-1] != state)
                costs[:-1, :, state] += vertical*(labels[1:] != state)
                costs[:, 1:, state] += horizontal*(labels[:, :-1] != state)
                costs[:, :-1, state] += horizontal*(labels[:, 1:] != state)
            proposed = costs.argmin(axis=-1)
            # Keep old labels on ties to prevent harmless oscillation.
            improving = costs[row, column, proposed] < costs[row, column, labels]-1e-12
            update = (parity == color) & improving & (proposed != labels)
            changes += int(update.sum())
            labels[update] = proposed[update]
        current = energy()
        if current > energies[-1]+1e-7*max(1, abs(energies[-1])):
            raise RuntimeError('MRF coordinate updates increased the fixed energy.')
        energies.append(current)
        if changes == 0:
            break
    return labels, energies


def pgm_demonstrations(model, X, seed=42):
    """JSON-ready syllabus evidence plus independent small-model verification.

    The HMM is intentionally a separately generated educational sequence.
    The current data's filenames are not treated as measured temporal order.
    """
    threshold, prior, cpd_z, cpd_x = _pgm_parameters(model)
    X = _pgm_matrix(X, len(threshold), allow_empty=True)
    binary = (X > threshold).astype(float)
    representation = model['representation']
    weights = np.asarray(representation['weights'])
    emissions = np.asarray(representation['emission'])
    latent_log = _pgm_mixture_log_joint(binary, weights, emissions)
    responsibility = np.exp(latent_log-logsumexp(latent_log, axis=1, keepdims=True))
    d = min(len(threshold), 5)  # At most 128 assignments; never exponential in all features.
    parents = {'F': [], 'Z': ['F']}
    parents.update({'X'+str(i): ['F', 'Z'] for i in range(d)})
    edges = moralize_parents(parents)
    joint, factor_product, assignments = [], [], []
    for failure, latent in itertools.product(range(2), range(cpd_z.shape[1])):
        for values in itertools.product(range(2), repeat=d):
            # Independent scalar and vectorized evaluations of the family factors.
            value = float(prior[failure]*cpd_z[failure, latent])
            for i, observation in enumerate(values):
                value *= float(cpd_x[failure, latent, i] if observation else 1-cpd_x[failure, latent, i])
            factors = [prior[failure], cpd_z[failure, latent]]
            factors.extend(np.where(np.asarray(values), cpd_x[failure, latent, :d], 1-cpd_x[failure, latent, :d]).tolist())
            joint.append(value)
            factor_product.append(float(np.prod(factors)))
            assignments.append((failure, latent, *values))
    partition = float(np.sum(factor_product))
    normalized = np.asarray(factor_product)/partition
    equality_error = float(np.max(abs(np.asarray(joint)-normalized)))
    evidence = {'X0': 1}
    enumerated = np.zeros(2)
    for assignment, probability in zip(assignments, joint):
        if assignment[2] == 1:
            enumerated[assignment[0]] += probability
    enumerated /= enumerated.sum()
    queried = bayesian_query(model, 'F', evidence)
    query_error = float(np.max(abs(enumerated-queried)))
    if equality_error > 1e-10 or abs(partition-1) > 1e-10 or query_error > 1e-10:
        raise AssertionError('BN exact inference or moralization verification failed.')

    # Synthetic ordered quality sequence: state 0=lower-risk assumed condition,
    # state 1=degraded assumed condition. Its true states exist only in this demo.
    rng = np.random.default_rng(seed)
    initial = np.array([0.8, 0.2])
    transition = np.array([[0.93, 0.07], [0.16, 0.84]])
    emission = np.array([[0.80, 0.15, 0.05], [0.10, 0.25, 0.65]])
    states, observations = [], []
    for t in range(48):
        state = int(rng.choice(2, p=initial if t == 0 else transition[states[-1]]))
        states.append(state)
        observations.append(int(rng.choice(3, p=emission[state])))
    hmm = categorical_hmm(initial, transition, emission, observations, ordered=True)
    # Exhaustive short-path posterior is independent evidence for both forward-
    # backward and Viterbi. It is feasible only for this tiny teaching case.
    short = observations[:5]
    paths = list(itertools.product(range(2), repeat=len(short)))
    masses = []
    for path in paths:
        mass = initial[path[0]]*emission[path[0], short[0]]
        for t in range(1, len(short)):
            mass *= transition[path[t-1], path[t]]*emission[path[t], short[t]]
        masses.append(mass)
    masses = np.asarray(masses)
    exact_posterior = np.zeros((len(short), 2))
    for path, mass in zip(paths, masses/masses.sum()):
        exact_posterior[np.arange(len(short)), path] += mass
    short_result = categorical_hmm(initial, transition, emission, short, ordered=True)
    hmm_error = float(np.max(abs(exact_posterior-np.asarray(short_result['smoothed']))))
    viterbi_matches = list(paths[int(masses.argmax())]) == short_result['viterbi']
    if hmm_error > 1e-10 or not viterbi_matches:
        raise AssertionError('HMM exhaustive-path verification failed.')
    hmm.update({'origin': 'educational_synthetic_ordered_sequence', 'states': states,
                'observations': observations, 'initial': initial.tolist(),
                'transition': transition.tolist(), 'emission': emission.tolist(),
                'exhaustive_posterior_max_error': hmm_error,
                'exhaustive_viterbi_matches': bool(viterbi_matches)})
    grid = np.linspace(0, 1, 101)
    decision = expected_risk_decision(grid)
    decision['failure_probability'] = grid.tolist()
    decision['actions'] = ['accept_mechanics', 'reacquire', 'expert_review']
    sensitivity = []
    for false_accept in (2., 5., 10., 20.):
        loss = np.array([[0., false_accept], [1., 1.5], [2.5, .5]])
        sensitivity.append({'false_accept_cost': false_accept,
                            'action_index': expected_risk_decision(grid, loss)['action_index']})
    decision['sensitivity'] = sensitivity
    return {
        'representation': {**representation, 'posterior': responsibility.tolist(),
                           'observation_count': int(len(X))},
        'bn': {'factorization': model['factorization'], 'parents': parents,
               'displayed_feature_count': d, 'full_feature_count': len(threshold),
               'failure_probability': predict_risk(model, X).tolist(),
               'missing_evidence_query': evidence, 'query_posterior': queried.tolist(),
               'enumeration_posterior': enumerated.tolist(), 'query_max_error': query_error,
               'prior_failure': float(prior[1])},
        'markov': {'moral_edges': edges, 'partition_function': partition,
                   'joint_max_error': equality_error,
                   'enumerated_feature_count': d,
                   'independence_X0_X1_given_F_Z': (markov_separated(edges, ['X0'], ['X1'], ['F', 'Z']) if d >= 2 else None),
                   'independence_X0_X1_without_conditioning': (markov_separated(edges, ['X0'], ['X1']) if d >= 2 else None),
                   'collider_DAG': [['A', 'C'], ['B', 'C']],
                   'collider_moral_edges': moralize_parents({'A': [], 'B': [], 'C': ['A', 'B']}),
                   'new_coparent_edge': ['A', 'B'],
                   'note': 'Original CPDs remain clique potentials. Moralization is not triangulation.'},
        'hmm': hmm, 'decision': decision,
    }



# ---------------------------------------------------------------------------
# Training: mixed precision, accumulation, deterministic splits and resume.
# ---------------------------------------------------------------------------
def manifest_records(manifest):
    return manifest['records'] if isinstance(manifest, dict) else manifest


def manifest_fingerprint(records):
    fields = [{k: r.get(k) for k in ('id', 'path', 'group', 'split', 'domain', 'sha256')}
              for r in records]
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


class TrainingPairs(Dataset):
    """Open one cached pair on demand; each worker holds only a single crop.

    Evaluation uses fixed center crops for reproducible, bounded-memory results.
    Full-field tiling is a separate optional inference command.
    """
    def __init__(self, records, patch, training=False, patches_per_pair=1, seed=42):
        self.records = records
        self.patch = tuple(patch)
        self.training = training
        self.patches_per_pair = patches_per_pair if training else 1
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return len(self.records) * self.patches_per_pair

    def __getitem__(self, index):
        record_index = index // self.patches_per_pair
        record = self.records[record_index]
        source = np.load(record['path'], mmap_mode='r', allow_pickle=False)
        height, width = source.shape[-2:]
        ph, pw = self.patch
        if height < ph or width < pw:
            raise ValueError(f'Patch {self.patch} exceeds cached shape {source.shape}: {record["path"]}')
        if self.training:
            rng = np.random.default_rng(self.seed + self.epoch*1000003 + index)
            z, x = int(rng.integers(height-ph+1)), int(rng.integers(width-pw+1))
        else:
            z, x = (height-ph)//2, (width-pw)//2
        pair = normalize_rf_pair(source[:, z:z+ph, x:x+pw])
        return {'pre': torch.from_numpy(pair[0:1].copy()),
                'post': torch.from_numpy(pair[1:2].copy()), 'index': record_index}


def load_manifest(cache_dir):
    manifest = read_json(Path(cache_dir)/'manifest.json')
    if not isinstance(manifest, dict) or not isinstance(manifest.get('records'), list) or not manifest['records']:
        raise ValueError('Empty cache manifest. Run prepare first.')
    records = manifest['records']
    seen, ids, content_roles, path_roles = {}, set(), {}, {}
    for record in records:
        required = {'id', 'path', 'group', 'split', 'domain', 'shape', 'sha256'}
        if not isinstance(record, dict) or not required <= set(record):
            raise ValueError('Manifest record lacks required identity, grouping or shape fields.')
        if not all(isinstance(record[key], str) and record[key].strip()
                   for key in ('id', 'path', 'group', 'split', 'domain', 'sha256')):
            raise ValueError('Manifest identity, path, group and domain must be nonempty strings.')
        if record['id'] in ids:
            raise ValueError('Duplicate record ID in manifest: '+record['id'])
        ids.add(record['id'])
        if record['split'] not in ROLES:
            raise ValueError('Invalid split in manifest')
        if record['group'] in seen and seen[record['group']] != record['split']:
            raise ValueError('Group leakage in manifest: ' + record['group'])
        seen[record['group']] = record['split']
        path = Path(record['path'])
        if not path.is_file():
            raise ValueError('Missing cached pair: ' + record['path'])
        for identity, mapping, label in ((record['sha256'], content_roles, 'content hash'),
                                         (str(path.resolve()).casefold(), path_roles, 'cache path')):
            if identity in mapping and mapping[identity] != record['split']:
                raise ValueError('Cross-partition duplicate '+label+' in manifest.')
            mapping[identity] = record['split']
        if record.get('cache_signature') is not None and record['cache_signature'] != _source_signature(path):
            raise ValueError('Cached pair changed; run prepare to rebuild it: '+str(path))
        try:
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            shape = tuple(record['shape'])
            good = (array.ndim == 3 and array.shape[0] == 2 and min(array.shape[1:]) >= 2
                    and array.dtype == np.dtype('float32') and array.shape == shape)
            del array
        except (OSError, ValueError, TypeError, EOFError) as exc:
            raise ValueError('Unreadable cached pair; run prepare to rebuild it: '+str(path)) from exc
        if not good:
            raise ValueError('Cached pair shape/dtype differs from manifest: '+str(path))
    for role in ROLES:
        if not any(r['split'] == role for r in records):
            raise ValueError('Missing '+role+' partition; prepare at least four independent groups.')
    return manifest


def make_loader(records, args, training=False):
    dataset = TrainingPairs(records, args.patch, training=training,
                            patches_per_pair=args.patches_per_pair, seed=args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=training,
                        num_workers=args.workers, pin_memory=args.device_resolved.type == 'cuda',
                        persistent_workers=False, generator=generator,
                        drop_last=False)
    return dataset, loader


def checkpoint_save(path, value):
    temporary = Path(path).with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def checkpoint_load(path, device='cpu'):
    # Checkpoints created here contain tensors and primitive containers only.
    # weights_only avoids executing pickled model classes from an untrusted file.
    return torch.load(path, map_location=device, weights_only=True)


def accumulation_sample_weight(step, max_batches, total_samples, batch_size, accumulation):
    """Sample-correct weighting when the last accumulated batch is smaller."""
    start = (step//accumulation)*accumulation
    end = min(start+accumulation, max_batches)
    samples = min(total_samples, end*batch_size)-start*batch_size
    this_batch = min(batch_size, total_samples-step*batch_size)
    if samples <= 0 or this_batch <= 0:
        raise ValueError('Invalid gradient accumulation batch indices.')
    return this_batch/samples


def train_pipeline(args, manifest=None):
    device = args.device_resolved
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = manifest if manifest is not None else load_manifest(args.cache)
    records = manifest_records(manifest)
    fingerprint = manifest_fingerprint(records)
    training = [r for r in records if r['split'] == 'train']
    validation = [r for r in records if r['split'] == 'validation']
    if not training or not validation:
        raise ValueError('Training and validation partitions must both be non-empty.')
    seed_everything(args.seed)
    model = RFRegistrationNet(args.width, args.radius, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    amp_enabled = device.type == 'cuda' and not args.no_amp
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)
    dataset, loader = make_loader(training, args, training=True)
    _, validation_loader = make_loader(validation, args)
    architecture = {'width': args.width, 'radius': args.radius, 'dropout': args.dropout}
    config = json_safe({k: v for k, v in vars(args).items() if k != 'device_resolved'})
    history, start, best, stale = [], 0, float('inf'), 0
    if args.resume:
        if Path(args.resume).resolve().parent != out.resolve():
            raise ValueError('Resume into the original --out directory so best.pt and history stay consistent.')
        checkpoint = checkpoint_load(args.resume, device)
        if checkpoint['architecture'] != architecture or checkpoint['manifest_fingerprint'] != fingerprint:
            raise ValueError('Resume requires the same model architecture and data/split manifest.')
        if checkpoint.get('training_contract') != training_contract(args):
            raise ValueError('Resume requires the same objective/preprocessing version, patch, seed, '
                             'batch/accumulation and pilot-step settings. Start a new --out for a version 1.0 checkpoint.')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scaler']:
            scaler.load_state_dict(checkpoint['scaler'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start, best = checkpoint['epoch']+1, checkpoint['best']
        history, stale = checkpoint['history'], checkpoint['stale']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if device.type == 'cuda' and checkpoint.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cuda_rng']])
        if start >= args.epochs:
            print(f'No training needed: checkpoint already contains {start} completed epochs.', flush=True)
            if not (out/'best.pt').exists():
                raise ValueError('Original best.pt is missing from the resume directory.')
            return out/'best.pt'
    elif (out/'last.pt').exists():
        raise ValueError('This output folder already has training checkpoints. Use --resume or a new --out.')
    write_json(out/'config.json', config)
    write_json(out/'environment.json', environment_info())
    write_json(out/'run_manifest.json', manifest)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    previous_seconds = float(history[-1].get('seconds', 0)) if history else 0.0
    write_json(out/'status.json', {'status': 'running', 'completed_epochs': start,
                                  'resume': str(out/'last.pt') if start else None})
    try:
        for epoch in range(start, args.epochs):
            epoch_started = time.perf_counter()
            dataset.epoch = epoch
            # Epoch-derived sampler makes resume order independent of prior processes.
            loader.generator.manual_seed(args.seed+epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            totals, batch_counts, component_sums = [], [], {}
            max_batches = min(len(loader), args.steps_per_epoch or len(loader))
            for step, batch in enumerate(loader):
                if step >= max_batches:
                    break
                pre, post = batch['pre'].to(device, non_blocking=True), batch['post'].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    forward, _ = model(pre, post)
                    backward, _ = model(post, pre)
                loss, terms = registration_loss(pre, post, forward, backward,
                                                args.smooth_weight, args.cycle_weight)
                # The final partial accumulation group gets its actual divisor.
                weight = accumulation_sample_weight(step, max_batches, len(dataset), args.batch_size, args.accumulation)
                scaler.scale(loss*weight).backward()
                if (step+1) % args.accumulation == 0 or step+1 == max_batches:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=not amp_enabled)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                totals.append(float(loss.detach()))
                batch_counts.append(len(pre))
                for key, value in terms.items():
                    component_sums[key] = component_sums.get(key, 0.0)+value*len(pre)
                if (step+1) % args.log_every == 0:
                    print(f'Epoch {epoch+1}/{args.epochs}, batch {step+1}/{max_batches}, loss {np.mean(totals):.4f}', flush=True)
            model.eval()
            val, validation_counts = [], []
            with torch.inference_mode():
                for step, batch in enumerate(validation_loader):
                    if args.validation_batches and step >= args.validation_batches:
                        break
                    pre, post = batch['pre'].to(device), batch['post'].to(device)
                    forward, _ = model(pre, post)
                    backward, _ = model(post, pre)
                    loss, _ = registration_loss(pre, post, forward, backward,
                                                 args.smooth_weight, args.cycle_weight)
                    val.append(float(loss))
                    validation_counts.append(len(pre))
            score = float(np.average(val, weights=validation_counts))
            scheduler.step(score)
            improved = score < best-1e-7
            if improved:
                best, stale = score, 0
            else:
                stale += 1
            epoch_seconds = time.perf_counter()-epoch_started
            row = {'epoch': epoch+1, 'train_loss': float(np.average(totals, weights=batch_counts)), 'validation_loss': score,
                   'lr': optimizer.param_groups[0]['lr'], 'seconds': previous_seconds+time.perf_counter()-started,
                   'epoch_seconds': epoch_seconds, 'training_patches': sum(batch_counts),
                   'patches_per_second_including_validation': sum(batch_counts)/max(epoch_seconds, 1e-9),
                   **{'train_'+key: value/sum(batch_counts) for key, value in component_sums.items()}}
            history.append(row)
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scaler': scaler.state_dict(), 'scheduler': scheduler.state_dict(),
                     'architecture': architecture, 'epoch': epoch, 'best': best, 'stale': stale,
                     'history': history, 'manifest_fingerprint': fingerprint,
                     'training_contract': training_contract(args), 'torch_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                     'config': config, 'version': VERSION}
            checkpoint_save(out/'last.pt', state)
            if improved:
                checkpoint_save(out/'best.pt', state)
            write_json(out/'history.json', history)
            print(f'Epoch {epoch+1}: train={row["train_loss"]:.5f}, validation={score:.5f}', flush=True)
            if args.patience and stale >= args.patience:
                print('Early stopping: validation objective has stopped improving.', flush=True)
                break
    except KeyboardInterrupt:
        write_json(out/'status.json', {'status': 'interrupted', 'completed_epochs': len(history),
                                      'resume': str(out/'last.pt') if (out/'last.pt').exists() else None})
        raise
    except Exception as exc:
        write_json(out/'status.json', {'status': 'failed', 'completed_epochs': len(history),
                                      'error_type': type(exc).__name__, 'error': str(exc),
                                      'resume': str(out/'last.pt') if (out/'last.pt').exists() else None})
        raise
    peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    write_json(out/'status.json', {'status': 'completed', 'epochs': len(history), 'best_validation_loss': best,
                                  'elapsed_seconds': time.perf_counter()-started, 'peak_cuda_bytes': peak,
                                  'parameters': sum(p.numel() for p in model.parameters()),
                                  'pathology_validation': False})
    return out/'best.pt'


def training_contract(args):
    return {'loss_version': 2, 'normalization_version': 2,
            **{k: getattr(args, k) for k in ('patch', 'seed', 'batch_size', 'accumulation',
                                            'patches_per_pair', 'smooth_weight', 'cycle_weight',
                                            'steps_per_epoch', 'validation_batches')}}


def load_trained_model(path, device):
    checkpoint = checkpoint_load(path, device)
    model = RFRegistrationNet(**checkpoint['architecture']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint


def sample_record_indices(records, maximum, seed):
    """Cap expensive report inference while retaining each domain/split group.

    `maximum=0` evaluates every record. Subsets are saved verbatim in CSV; the
    report always records that each record contributes one fixed center crop.
    """
    if not maximum or len(records) <= maximum:
        return list(range(len(records)))
    rng = np.random.default_rng(seed)
    buckets = {}
    for index, r in enumerate(records):
        buckets.setdefault((r['split'], r['domain']), []).append(index)
    if maximum < len(buckets):
        raise ValueError('--eval-limit must cover every split/domain bucket or be zero.')
    for values in buckets.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < maximum and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(selected) < maximum:
                selected.append(buckets[key].pop())
    return sorted(selected)


def representative_example_indices(records, budget):
    """Round-robin domains and acquisition groups for held-out diagnostics."""
    buckets = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        if record['split'] == 'test':
            buckets[record['domain']][record['group']].append(index)
    ordered = {}
    for domain, groups in sorted(buckets.items()):
        queues = [indices.copy() for _, indices in sorted(groups.items())]
        ordered[domain] = []
        while any(queues):
            for queue in queues:
                if queue: ordered[domain].append(queue.pop(0))
    selected = []
    while len(selected) < budget and any(ordered.values()):
        for domain in sorted(ordered):
            if ordered[domain] and len(selected) < budget:
                selected.append(ordered[domain].pop(0))
    return selected


def fit_calibrator(probabilities, y):
    """Platt calibration uses only calibration groups; one-class sets skip it."""
    y, probabilities = validate_binary_predictions(y, probabilities)
    p = np.clip(probabilities, 1e-5, 1-1e-5)
    if len(y) < 8 or len(np.unique(y)) < 2:
        return {'status': 'skipped: fewer than 8 rows or one outcome class', 'coef': None}
    estimator = LogisticRegression(C=1.0, solver='lbfgs')
    estimator.fit(np.log(p/(1-p))[:, None], y)
    return {'status': 'fitted on calibration partition', 'coef': float(estimator.coef_[0, 0]),
            'intercept': float(estimator.intercept_[0])}


def apply_calibrator(calibrator, probability):
    probability = np.asarray(probability, float)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Calibration inputs must be finite probabilities in [0,1].')
    p = np.clip(probability, 1e-5, 1-1e-5)
    if calibrator['coef'] is None:
        return p
    logit = calibrator['coef']*np.log(p/(1-p)) + calibrator['intercept']
    return 1/(1+np.exp(-np.clip(logit, -40, 40)))


def validate_binary_predictions(y, probabilities):
    target, probability = np.asarray(y), np.asarray(probabilities, float)
    if (target.ndim != 1 or probability.shape != target.shape or not np.isin(target, [0, 1]).all()
            or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1))):
        raise ValueError('Require matching 1D binary outcomes and finite [0,1] probabilities.')
    return target.astype(int), probability


def classification_metrics(y, p):
    y, p = validate_binary_predictions(y, p)
    if not len(y):
        return {'n': 0, 'reason': 'empty partition'}
    output = {'n': len(y), 'prevalence': float(y.mean()), 'brier': float(brier_score_loss(y, p)),
              'accuracy_at_0.5': float(((p >= 0.5) == y).mean()),
              'confusion_matrix': confusion_matrix(y, p >= 0.5, labels=[0, 1]).tolist(),
              'roc_auc': None, 'average_precision': None}
    if len(np.unique(y)) == 2:
        output['roc_auc'] = float(roc_auc_score(y, p))
        output['average_precision'] = float(average_precision_score(y, p))
    else:
        output['ranking_metric_status'] = 'unavailable: only one outcome class'
    bins = np.minimum((p*10).astype(int), 9)
    output['ece_10_bins'] = float(sum(np.mean(bins == k)*abs(p[bins == k].mean()-y[bins == k].mean())
                                     for k in range(10) if np.any(bins == k)))
    return output


def grouped_bootstrap(values, groups, seed=42, repetitions=500):
    values, groups = np.asarray(values, float), np.asarray(groups)
    if (values.ndim != 1 or groups.shape != values.shape or not np.isfinite(values).all()
            or not isinstance(repetitions, (int, np.integer)) or repetitions <= 0):
        raise ValueError('Bootstrap needs matching finite observations/groups and a positive repetition count.')
    if not len(values):
        return {'mean': None, 'ci95': None, 'groups': 0, 'status': 'empty observations'}
    unique = np.unique(groups)
    if len(unique) < 2:
        return {'mean': float(values.mean()), 'ci95': None, 'groups': len(unique)}
    # Equal-group weighting: long sequences do not count as more independent evidence.
    group_means = np.array([values[groups == g].mean() for g in unique])
    rng = np.random.default_rng(seed)
    means = group_means[rng.integers(len(unique), size=(repetitions, len(unique)))].mean(1)
    return {'mean': float(group_means.mean()), 'ci95': np.quantile(means, [0.025, 0.975]).tolist(),
            'groups': len(unique), 'weighting': 'equal group'}


def clinical_ordinal_extension(label_path, rows, embeddings, out):
    """Optional pathology-linked ordinal research head; encoder stays label-free.

    CSV contract: id,patient_id,grade, where grade is a zero-based ordered integer.
    Patients must already respect the imaging manifest split. The program never
    infers grades from filenames, strain thresholds, or unsupervised clusters.
    This extension requires external labels, absent from the supplied dataset.
    """
    from sklearn.metrics import cohen_kappa_score
    embeddings = np.asarray(embeddings, float)
    if (not rows or embeddings.ndim != 2 or len(embeddings) != len(rows)
            or not np.isfinite(embeddings).all() or len({row['id'] for row in rows}) != len(rows)):
        raise ValueError('Clinical modeling requires unique imaging IDs and matching finite embeddings.')
    with Path(label_path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'id', 'patient_id', 'grade'} <= set(reader.fieldnames or []):
            raise ValueError('Clinical CSV requires id,patient_id,grade columns.')
        labels = list(reader)
    by_id, patient_roles = {}, {}
    row_by_id = {row['id']: index for index, row in enumerate(rows)}
    unused = []
    for label in labels:
        if any(label.get(key) is None or not label[key].strip() for key in ('id', 'patient_id', 'grade')):
            raise ValueError('Every clinical CSV row needs id,patient_id,grade values.')
        key = label['id'].strip()
        label['patient_id'] = label['patient_id'].strip()
        if key in by_id: raise ValueError('Duplicate clinical label id: '+key)
        if not label['patient_id'].strip(): raise ValueError('Empty patient_id in clinical CSV.')
        try: grade = int(label['grade'])
        except ValueError as exc: raise ValueError('Grade must be an integer for '+key) from exc
        if grade < 0: raise ValueError('Grades must be zero-based nonnegative integers.')
        by_id[key] = (label['patient_id'], grade)
        if key not in row_by_id:
            unused.append(key); continue
        role = rows[row_by_id[key]]['split']
        patient = label['patient_id']
        if patient in patient_roles and patient_roles[patient] != role:
            raise ValueError('Patient crosses dataset partitions: '+patient+'. Rebuild --group-map before clinical modeling.')
        patient_roles[patient] = role
    selected = [index for index, row in enumerate(rows) if row['id'] in by_id]
    if not selected: raise ValueError('No clinical CSV IDs match evaluated RF pairs. Use --eval-limit 0 if needed.')
    target = np.array([by_id[rows[i]['id']][1] for i in selected])
    role = np.array([rows[i]['split'] for i in selected])
    train = role == 'train'; test = role == 'test'
    grades = np.unique(target)
    if len(grades) < 2 or not np.array_equal(grades, np.arange(len(grades))):
        raise ValueError('Clinical grades must cover a contiguous ordered scale 0..K-1 with at least two grades.')
    if not train.any() or not test.any() or not np.array_equal(np.unique(target[train]), grades):
        raise ValueError('Clinical head needs training/test patients and every grade represented in training.')
    standardizer = StandardScaler().fit(embeddings[selected][train])
    features = standardizer.transform(embeddings[selected])
    survival, heads = [], []
    for threshold in range(len(grades)-1):
        head = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        # Weight patients equally, rather than allowing repeated frames to dominate.
        training_patients = [by_id[rows[i]['id']][0] for i, keep in zip(selected, train) if keep]
        counts = {patient: training_patients.count(patient) for patient in set(training_patients)}
        weights = np.array([1/counts[patient] for patient in training_patients])
        weights *= len(weights)/weights.sum()
        head.fit(features[train], target[train] > threshold, sample_weight=weights)
        survival.append(head.predict_proba(features)[:, 1])
        heads.append({'threshold': threshold, 'coef': head.coef_.tolist(), 'intercept': head.intercept_.tolist()})
    # Separate cumulative logistic heads can cross. This deterministic monotonic
    # projection enforces P(G>k+1)<=P(G>k); this is not probability calibration.
    cumulative = np.minimum.accumulate(np.stack(survival, axis=1), axis=1)
    probabilities = np.column_stack((1-cumulative[:, 0], -np.diff(cumulative, axis=1), cumulative[:, -1]))
    prediction = probabilities.argmax(1)
    metrics = {'labeled_pairs': len(selected), 'patients': len(patient_roles), 'unused_label_ids': unused,
               'warning': 'External pathology-linked research head; requires independent clinical validation.',
               'per_partition': {}}
    for partition in ROLES:
        mask = role == partition
        if not mask.any(): continue
        metrics['per_partition'][partition] = {
            'pairs': int(mask.sum()), 'accuracy': float((prediction[mask] == target[mask]).mean()),
            'ordinal_mae': float(np.abs(prediction[mask]-target[mask]).mean()),
            'quadratic_kappa': float(cohen_kappa_score(target[mask], prediction[mask], labels=grades, weights='quadratic'))
                               if len(np.unique(target[mask])) > 1 else None,
            'confusion_matrix': confusion_matrix(target[mask], prediction[mask], labels=grades).tolist()}
    write_json(Path(out)/'clinical_ordinal_metrics.json', metrics)
    write_json(Path(out)/'clinical_ordinal_model.json', {'heads': heads, 'grades': grades,
                                                      'scaler_mean': standardizer.mean_, 'scaler_scale': standardizer.scale_,
                                                      'probability_calibration': 'not established'})
    with (Path(out)/'clinical_ordinal_predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['id', 'patient_id', 'split', 'grade', 'prediction']+['p_grade_'+str(g) for g in grades])
        for j, i in enumerate(selected):
            writer.writerow([rows[i]['id'], by_id[rows[i]['id']][0], role[j], target[j], prediction[j]]+probabilities[j].tolist())
    return metrics


def evaluate_pipeline(args, checkpoint_path=None):
    out, device = Path(args.out), args.device_resolved
    if args.clinical_labels and args.eval_limit:
        raise ValueError('Clinical patient-split audit requires --eval-limit 0 so every manifest record is checked.')
    manifest = load_manifest(args.cache)
    records = manifest_records(manifest)
    checkpoint_path = checkpoint_path or args.checkpoint or out/'best.pt'
    model, checkpoint = load_trained_model(checkpoint_path, device)
    if not getattr(args, 'patch_explicit', False):
        args.patch = list(checkpoint.get('training_contract', {}).get('patch', args.patch))
    if checkpoint['manifest_fingerprint'] != manifest_fingerprint(records):
        raise ValueError('Evaluation manifest differs from the training split manifest.')
    indices = sample_record_indices(records, args.eval_limit, args.seed)
    sampled = [records[i] for i in indices]
    dataset = TrainingPairs(sampled, args.patch, seed=args.seed)
    rows, embeddings, examples = [], [], []
    diagnostic_budget = max(args.examples, 0 if args.skip_robustness else args.robustness_samples)
    example_order = representative_example_indices(sampled, diagnostic_budget)
    selected_examples = set(example_order)
    collected_examples = {}
    started = time.perf_counter()
    for i, record in enumerate(sampled):
        batch = dataset[i]
        a, b = batch['pre'][None].to(device), batch['post'][None].to(device)
        with torch.inference_mode():
            u, embedding = model(a, b)
            v, _ = model(b, a)
        pre, post = a[0, 0].cpu().numpy(), b[0, 0].cpu().numpy()
        flow, backward = u[0].cpu().numpy(), v[0].cpu().numpy()
        features, maps = quality_maps(pre, post, flow, backward, device)
        zero_features, _ = quality_maps(pre, post, np.zeros_like(flow), np.zeros_like(flow), device)
        phase = phase_translation(pre, post)
        phase_features, _ = quality_maps(pre, post, phase, -phase, device)
        # Stochastic spread is a sensitivity signal, never a calibrated error bar.
        spread = 0.0
        if args.mc_samples > 1:
            model.train()
            draws = []
            with torch.inference_mode():
                for _ in range(args.mc_samples):
                    draw, _ = model(a, b)
                    draws.append(draw[0].cpu().numpy())
            model.eval()
            spread = float(np.std(np.stack(draws), axis=0).mean())
        row = {'id': record['id'], 'group': record['group'], 'domain': record['domain'],
               'split': record['split'], 'source': record.get('source', ''),
               'zero_residual': zero_features[0], 'phase_residual': phase_features[0],
               'strain_median': float(np.median(maps['strain'][maps['valid']])),
               'mc_spread_pixels': spread,
               **{name: float(value) for name, value in zip(FEATURE_NAMES, features)}}
        if 'analytic_truth_flow_pixels' in record:
            truth = np.asarray(record['analytic_truth_flow_pixels'])[:, None, None]
            support = maps['valid']
            row['analytic_epe_pixels'] = float(np.sqrt(((flow-truth)**2).sum(0))[support].mean())
            row['analytic_zero_epe_pixels'] = float(np.sqrt((truth**2).sum()))
            row['analytic_phase_epe_pixels'] = float(np.sqrt(((phase-truth)**2).sum(0))[support].mean())
            row['analytic_strain_mae'] = float(np.abs(maps['strain'][support]).mean())
        row['proxy_failure'] = int(proxy_failure(features, args.residual_threshold, args.cycle_threshold))
        rows.append(row)
        embeddings.append(embedding[0].cpu().numpy())
        if i in selected_examples:
            collected_examples[i] = {'record': row.copy(), 'pre': pre, 'post': post, 'flow': flow,
                                      'maps': maps, 'phase': phase, 'backward': backward}
        if (i+1) % args.log_every == 0 or i+1 == len(sampled):
            print(f'Evaluation {i+1}/{len(sampled)} RF pairs', flush=True)
    study_examples = [collected_examples[index] for index in example_order]
    examples = study_examples[:args.examples]
    X = np.array([[r[name] for name in FEATURE_NAMES] for r in rows])
    y = np.array([r['proxy_failure'] for r in rows], int)
    roles = np.array([r['split'] for r in rows])
    train_mask, calibration_mask, test_mask = [roles == role for role in ('train', 'calibration', 'test')]
    risk_model = fit_risk_model(X[train_mask], y[train_mask], seed=args.seed)
    raw_risk = predict_risk(risk_model, X)
    calibrator = fit_calibrator(raw_risk[calibration_mask], y[calibration_mask])
    risk = apply_calibrator(calibrator, raw_risk)
    # Baseline: reconstruction alone mapped monotonically to [0,1].
    for row, p, raw in zip(rows, risk, raw_risk):
        row['risk_raw'], row['risk_calibrated'] = float(raw), float(p)
        costs = np.array([[0, 8], [1, 1.5], [2.5, 0.5]]) @ np.array([1-p, p])
        row['action'] = ['accept mechanics', 'reacquire', 'review'][int(costs.argmin())]
        row['expected_cost'] = float(costs.min())
    embedding_array = np.stack(embeddings)
    scaler = StandardScaler().fit(embedding_array[train_mask])
    scaled = scaler.transform(embedding_array)
    n_components = min(3, int(train_mask.sum()), len(np.unique(np.round(scaled[train_mask], 6), axis=0)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clusters = KMeans(n_clusters=max(1, n_components), random_state=args.seed, n_init=10).fit(scaled[train_mask])
        labels = clusters.predict(scaled)
        pca = PCA(n_components=min(2, scaled.shape[1], int(train_mask.sum()))).fit(scaled[train_mask])
        coordinates = pca.transform(scaled)
    if coordinates.shape[1] == 1:
        coordinates = np.pad(coordinates, ((0, 0), (0, 1)))
    for row, label in zip(rows, labels):
        row['mechanical_phenotype'] = int(label)
    report = {'version': VERSION, 'endpoint': 'feature-derived operational proxy failure',
              'endpoint_warning': 'Risk metrics measure imitation of a declared quality rule; '
                                  'they do not establish displacement accuracy or cancer grading.',
              'crop_protocol': 'one fixed center crop per selected RF pair',
              'patch': args.patch, 'selected_pairs': len(rows), 'available_pairs': len(records),
              'thresholds': {'residual': args.residual_threshold, 'cycle_pixels': args.cycle_threshold,
                             'fold_fraction': 0.01, 'invalid_fraction': 0.25},
              'calibrator': calibrator, 'environment': environment_info(),
              'elapsed_seconds': time.perf_counter()-started, 'manifest_notes': manifest.get('notes', []),
              'metrics': {role: classification_metrics(y[roles == role], risk[roles == role]) for role in ROLES},
              'domain_metrics': {domain: classification_metrics(y[test_mask & (np.array([r['domain'] for r in rows]) == domain)],
                                                               risk[test_mask & (np.array([r['domain'] for r in rows]) == domain)])
                                 for domain in sorted(set(r['domain'] for r in rows))},
              'test_grouped_statistics': {
                  field: grouped_bootstrap([r[field] for r in rows if r['split'] == 'test'],
                                           [r['group'] for r in rows if r['split'] == 'test'], args.seed)
                  for field in ('reconstruction loss', 'zero_residual', 'phase_residual', 'proxy_failure')},
              'pathology_grade_available': False, 'simulation_displacement_labels_used': False}
    report['normalization_version'] = 2
    report['checkpoint_training_version'] = checkpoint.get('version', 'unknown')
    if checkpoint.get('training_contract', {}).get('normalization_version', 1) != 2:
        report['checkpoint_compatibility_note'] = ('This checkpoint was trained with earlier normalization. '
                                                 'Current evaluation uses version 2; retrain for a matched-protocol comparison.')
    report['checkpoint_path'] = str(Path(checkpoint_path).resolve())
    report['example_protocol'] = 'Held-out domains and acquisition groups selected in round-robin order.'
    report['data_quality'] = manifest.get('data_quality', {'quarantined_source_count': 0, 'sources': []})
    write_json(out/'data_rejections.json', report['data_quality'])
    report['robustness_study'] = run_robustness_study(model, study_examples, risk_model, calibrator, args)
    report['analytic_benchmark'] = analytic_motion_benchmark(model, args)
    report['elapsed_seconds'] = time.perf_counter()-started
    write_json(out/'robustness_results.json', report['robustness_study'])
    write_json(out/'analytic_benchmark.json', report['analytic_benchmark'])
    write_json(out/'evaluation_config.json', {k: v for k, v in vars(args).items() if k != 'device_resolved'})
    if any('analytic_epe_pixels' in r for r in rows):
        report['analytic_fixture_test_metrics'] = {
            field: float(np.mean([r[field] for r in rows if r['split'] == 'test']))
            for field in ('analytic_epe_pixels', 'analytic_zero_epe_pixels', 'analytic_phase_epe_pixels', 'analytic_strain_mae')}
        report['analytic_fixture_note'] = 'Analytic translations verify the software; they are not validated acoustic simulations.'
    demonstrations = pgm_demonstrations(risk_model, X[test_mask], seed=args.seed)
    write_json(out/'risk_model.json', {'model': risk_model, 'calibrator': calibrator,
                                      'feature_names': FEATURE_NAMES, 'endpoint': report['endpoint']})
    write_json(out/'summary.json', report)
    write_json(out/'pgm_results.json', demonstrations)
    with (out/'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out/'representations.npz', embeddings=embedding_array, pca=coordinates,
                        phenotypes=labels, roles=roles)
    if args.clinical_labels:
        clinical = clinical_ordinal_extension(args.clinical_labels, rows, embedding_array, out)
        report['clinical_extension'] = clinical
        report['pathology_grade_available'] = True
        write_json(out/'summary.json', report)
    for i, example in enumerate(examples):
        np.savez_compressed(out/f'example_{i:02d}_maps.npz', pre=example['pre'], post=example['post'],
                            flow=example['flow'], phase=example['phase'], **example['maps'])
    create_report(args, rows, examples, report, demonstrations, risk_model, coordinates, labels)
    print('Report: ' + str((out/'report.html').resolve()), flush=True)
    return report



# ---------------------------------------------------------------------------
# White-background scientific figures, numeric exports and offline HTML report.
# ---------------------------------------------------------------------------
def figure_style():
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'savefig.facecolor': 'white', 'text.color': '#172438',
                         'axes.labelcolor': '#172438', 'axes.edgecolor': '#8a96a6',
                         'xtick.color': '#49566b', 'ytick.color': '#49566b',
                         'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
                         'grid.color': '#e0e5ec', 'grid.alpha': 0.7,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.constrained_layout.use': True, 'image.cmap': 'viridis'})


# ---------------------------------------------------------------------------
# Extended measured-result figures. These helpers use frozen fitted models;
# no figure selection, threshold sweep, or display transform refits a model.
# ---------------------------------------------------------------------------
def _result_ecdf(values):
    """Return exact empirical CDF coordinates, including single-valued samples."""
    values = np.asarray(values, float)
    values = np.sort(values[np.isfinite(values)])
    return values, np.arange(1, len(values)+1, dtype=float)/max(len(values), 1)


def _profile_quantiles(values, valid):
    """Depth-wise within-image spread; not an uncertainty interval for a mean."""
    values, valid = np.asarray(values, float), np.asarray(valid, bool)
    result = np.full((3, values.shape[0]), np.nan, float)
    counts = np.zeros(values.shape[0], int)
    for depth in range(values.shape[0]):
        selected = values[depth][valid[depth] & np.isfinite(values[depth])]
        counts[depth] = len(selected)
        if len(selected):
            result[:, depth] = np.quantile(selected, [0.25, 0.5, 0.75])
    return result, counts


def additional_report_figures(args, rows, examples, summary, demonstrations,
                              risk_model, coordinates, phenotypes, save):
    """Add twelve result families plus three diagnostics per saved RF example.

    `save(fig,name,caption)` is the parent report's PNG/PDF/gallery callback.
    Return numerical data for `extended_results.json`; original pixel arrays
    remain in example_XX_maps.npz, with their paths referenced below. All risk
    endpoints continue to mean the declared feature-derived quality rule.
    """
    colors = ['#1765ad', '#e68a27', '#24998c', '#b64f73', '#7252a2', '#777777']
    result = {'scope': 'measured result diagnostics; operational proxy, not clinical risk',
              'figures_added': [], 'sample_ids': [str(row['id']) for row in rows]}
    if not rows:
        return result
    roles = np.asarray([row['split'] for row in rows])
    domains = np.asarray([row['domain'] for row in rows])
    test = roles == 'test'
    domain_names = sorted(set(domains[test]))
    features = np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows], float)
    risk = np.asarray([row['risk_calibrated'] for row in rows], float)
    raw_risk = np.asarray([row['risk_raw'] for row in rows], float)
    events = np.asarray([row['proxy_failure'] for row in rows], int)
    phenotypes = np.asarray(phenotypes, int)
    threshold = np.asarray(risk_model['thresholds'], float)

    def store(fig, name, caption):
        fig.set_facecolor('white')
        save(fig, name, caption)
        result['figures_added'].append(name)

    def unavailable(ax, message):
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha='center', va='center', wrap=True)
        ax.set_axis_off()

    def safe_boxplot(ax, samples, labels, ylabel):
        """Boxplots work for constant/singleton samples where KDEs can fail."""
        good = [(np.asarray(sample, float), str(label)) for sample, label in zip(samples, labels)
                if len(sample)]
        if not good:
            unavailable(ax, 'No held-out observations available')
            return
        boxes = ax.boxplot([sample for sample, _ in good], patch_artist=True, showmeans=True,
                           medianprops={'color': '#14283e', 'linewidth': 1.5},
                           meanprops={'marker': 'o', 'markerfacecolor': 'white',
                                      'markeredgecolor': '#14283e', 'markersize': 4})
        for index, patch in enumerate(boxes['boxes']):
            patch.set_facecolor(colors[index % len(colors)]); patch.set_alpha(0.45)
        ax.set_xticks(range(1, len(good)+1), [label for _, label in good])
        ax.set_ylabel(ylabel); ax.grid(axis='y', alpha=0.4)
        ax.tick_params(axis='x', rotation=15)

    # 1. Unit 1: one observation can update the failure posterior while other
    # features remain unobserved and are correctly marginalized.
    posterior_zero, posterior_one = [], []
    prior = float(bayesian_query(risk_model, 'F')[1])
    for index in range(len(FEATURE_NAMES)):
        posterior_zero.append(float(bayesian_query(risk_model, 'F', {'X'+str(index): 0})[1]))
        posterior_one.append(float(bayesian_query(risk_model, 'F', {'X'+str(index): 1})[1]))
    fig, ax = plt.subplots(figsize=(12, 5))
    positions = np.arange(len(FEATURE_NAMES))
    ax.bar(positions-0.18, posterior_zero, 0.36, color=colors[0], label='Only Xi=0 observed')
    ax.bar(positions+0.18, posterior_one, 0.36, color=colors[1], label='Only Xi=1 observed')
    ax.axhline(prior, color=colors[2], linestyle='--', label='Prior failure probability')
    ax.set_xticks(positions, FEATURE_NAMES, rotation=25, ha='right')
    ax.set(ylim=(0, 1), ylabel='Raw BN proxy-failure probability',
           title='Bayesian inference with one observed feature')
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.01)); ax.grid(axis='y', alpha=0.3)
    store(fig, '23_bn_single_evidence', 'Xi uses its frozen training median. Every other feature is marginalized, so these are exact single-evidence BN queries, not causal effects or feature importance.')
    result['bn_single_evidence'] = {'feature_names': FEATURE_NAMES, 'training_thresholds': threshold.tolist(),
                                    'prior': prior, 'posterior_Xi_0': posterior_zero,
                                    'posterior_Xi_1': posterior_one}

    # 2. Exact missing-evidence inference, on actual minimum/maximum-score test
    # cases. The disclosure order is fixed and is not a greedy feature search.
    fig, ax = plt.subplots(figsize=(11, 4))
    progressive = []
    held_out = np.flatnonzero(test)
    selected = []
    if len(held_out):
        selected = list(dict.fromkeys([int(held_out[np.argmin(raw_risk[test])]),
                                       int(held_out[np.argmax(raw_risk[test])])]))
    for selected_index in selected:
        evidence, curve = {}, [prior]
        binary = (features[selected_index] > threshold).astype(int)
        for feature_index, value in enumerate(binary):
            evidence['X'+str(feature_index)] = int(value)
            curve.append(float(bayesian_query(risk_model, 'F', evidence)[1]))
        label = str(rows[selected_index]['id'])
        ax.plot(np.arange(len(curve)), curve, 'o-', label=label[:34])
        progressive.append({'id': label, 'binary_evidence': binary.tolist(), 'raw_posteriors': curve,
                            'all_evidence_probability': float(raw_risk[selected_index])})
    if selected:
        ax.set_xticks(range(len(FEATURE_NAMES)+1), ['Prior']+['+ X'+str(i) for i in range(len(FEATURE_NAMES))])
        ax.legend(fontsize=8)
        ax.set(ylim=(0, 1), xlabel='Cumulative evidence, in fixed feature order',
               ylabel='Raw BN proxy-failure probability', title='How a posterior changes as evidence arrives')
        ax.grid(alpha=0.3)
    else:
        unavailable(ax, 'No held-out cases for evidence disclosure')
    store(fig, '24_bn_progressive_missing_evidence', 'An omitted observation is marginalized rather than replaced by zero. Curves use actual held-out cases with lowest/highest raw BN scores; the fixed evidence order is not an importance ranking.')
    result['bn_progressive_evidence'] = progressive

    # Group aggregation is reused below. Each group has one vote regardless of
    # the number of frames sampled from it; independence remains a data premise.
    group_names = sorted(set(str(row['group']) for row in rows if row['split'] == 'test'))
    groups = []
    for name in group_names:
        indices = np.asarray([i for i, row in enumerate(rows) if row['split'] == 'test' and str(row['group']) == name], int)
        group = {'group': name, 'pairs': len(indices),
                 'domains': sorted(set(str(domains[i]) for i in indices)),
                 'risk_mean': float(risk[indices].mean()),
                 'proxy_event_fraction': float(events[indices].mean()),
                 'features': features[indices].mean(axis=0).tolist()}
        for key in ('zero_residual', 'phase_residual', 'reconstruction loss'):
            group[key] = float(np.mean([rows[i][key] for i in indices]))
        groups.append(group)
    result['test_groups'] = groups

    # 3. Paired baseline differences use identical groups and retain each group's
    # pairing in the bootstrap. No unpaired pixel-level significance is claimed.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    comparison = {}
    if groups:
        neural = np.asarray([group['reconstruction loss'] for group in groups])
        rng = np.random.default_rng(int(getattr(args, 'seed', 42)))
        for index, (key, label) in enumerate([('zero_residual', 'Zero motion'), ('phase_residual', 'Phase translation')]):
            baseline = np.asarray([group[key] for group in groups])
            difference = baseline-neural
            interval = None
            if len(groups) >= 2:
                bootstrap = difference[rng.integers(len(groups), size=(1000, len(groups)))].mean(axis=1)
                interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
            axes[0].scatter(baseline, neural, label=label, alpha=0.7, color=colors[index])
            mean = float(difference.mean())
            error = np.maximum(np.asarray([[mean-interval[0]], [interval[1]-mean]]), 0) if interval else None
            axes[1].errorbar(index, mean, yerr=error, fmt='o', capsize=6, color=colors[index])
            # A descriptive exact sign test, with ties removed; this does not
            # repair unverified acquisition independence or correct multiplicity.
            from scipy.stats import binomtest
            positive, negative = int((difference > 1e-12).sum()), int((difference < -1e-12).sum())
            pvalue = float(binomtest(positive, positive+negative, 0.5).pvalue) if positive+negative else None
            comparison[key] = {'group_differences': difference.tolist(), 'mean_difference': mean,
                               'bootstrap_ci95': interval, 'positive_groups': positive,
                               'negative_groups': negative, 'ties': int(len(groups)-positive-negative),
                               'exploratory_two_sided_sign_test_p': pvalue}
        bounds = [float(np.min([group[key] for group in groups])) for key in ('zero_residual', 'phase_residual', 'reconstruction loss')]
        upper = max(float(np.max([group[key] for group in groups])) for key in ('zero_residual', 'phase_residual', 'reconstruction loss'))
        lower = min(bounds)
        axes[0].plot([lower, upper], [lower, upper], '--', color='#888888', label='Equal loss')
        axes[0].set(xlabel='Baseline group-mean reconstruction loss', ylabel='Neural group-mean reconstruction loss',
                    title='Paired comparison by manifest group'); axes[0].legend(fontsize=9)
        axes[1].axhline(0, color='#888888', linestyle='--')
        axes[1].set_xticks([0, 1], ['Zero minus neural', 'Phase minus neural'])
        axes[1].set(ylabel='Equal-group mean loss difference', title='Paired group bootstrap, 95% intervals')
    else:
        for ax in axes: unavailable(ax, 'No held-out groups')
    store(fig, '25_paired_group_baseline_differences', 'Positive differences favor neural reconstruction. Group bootstrap preserves pairing. Supports can differ after warping; these are reconstruction proxies, not reference displacement errors. Exploratory sign tests in JSON are unadjusted and rely on group independence.')
    result['paired_baselines'] = comparison

    # 4. ECDFs reveal distribution tails that mean bars can conceal.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ecdfs = {'pairs': {}, 'groups': {}}
    for index, (key, label) in enumerate([('zero_residual', 'Zero motion'), ('phase_residual', 'Phase translation'), ('reconstruction loss', 'Neural')]):
        pair_values = [row[key] for row in rows if row['split'] == 'test']
        group_values = [group[key] for group in groups]
        for ax, values, scope in zip(axes, (pair_values, group_values), ('pairs', 'groups')):
            x, cdf = _result_ecdf(values)
            if len(x): ax.step(x, cdf, where='post', label=label, color=colors[index])
            ecdfs[scope][key] = {'loss': x.tolist(), 'fraction_at_or_below': cdf.tolist()}
    for ax, scope in zip(axes, ('Held-out RF pairs', 'Equal-weight held-out groups')):
        ax.set(xlabel='Reconstruction loss', ylabel='Empirical cumulative fraction', ylim=(0, 1.02), title=scope)
        ax.legend(); ax.grid(alpha=0.3)
    store(fig, '26_reconstruction_ecdf', 'ECDFs use every selected held-out pair, then one mean per manifest group. A curve farther left has lower reconstruction loss; accuracy cannot be inferred without reference motion.')
    result['reconstruction_ecdf'] = ecdfs

    # 5. Boxplots remain well defined for a single or constant observation.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    box_fields = [('reconstruction loss', 1, '1 - local RF correlation'),
                  ('cycle error', 1, 'Cycle error (pixels)'),
                  ('strain_median', 100, 'Median axial strain (%)'),
                  ('strain dispersion', 100, 'Strain MAD (%)')]
    box_data = {}
    for ax, (field, multiplier, ylabel) in zip(axes.flat, box_fields):
        samples = [[float(row[field])*multiplier for row in rows if row['split'] == 'test' and row['domain'] == domain] for domain in domain_names]
        safe_boxplot(ax, samples, domain_names, ylabel)
        ax.set_title(field)
        box_data[field] = {'display_multiplier': multiplier, 'domain_values': dict(zip(domain_names, samples))}
    store(fig, '27_domain_quality_boxplots', 'Held-out distribution summaries expose spread and outliers. Boxplots are descriptive; source domains can differ in noise, motion, loading, and pairing assumptions.')
    result['domain_quality_boxplots'] = box_data

    # 6. Change the DECLARED RULE in a sensitivity analysis without refitting the
    # BN or selecting thresholds using held-out labels.
    residual_grid = np.linspace(0, 2, 31)
    cycle_grid = np.linspace(0, 3, 31)
    scope_names = ['all held-out'] + domain_names
    fig, axes = plt.subplots(1, len(scope_names), figsize=(5*len(scope_names), 4), squeeze=False)
    sensitivity = {}
    for ax, scope in zip(axes.flat, scope_names):
        mask = test if scope == 'all held-out' else test & (domains == scope)
        values = features[mask]
        prevalence = np.full((len(cycle_grid), len(residual_grid)), np.nan)
        if len(values):
            other_flags = (values[:, 3] > 0.01) | (values[:, 4] > 0.25)
            for cycle_index, cycle_threshold in enumerate(cycle_grid):
                prevalence[cycle_index] = ((values[:, 0, None] > residual_grid[None, :])
                                           | (values[:, 1, None] > cycle_threshold)
                                           | other_flags[:, None]).mean(axis=0)
            image = ax.imshow(prevalence, origin='lower', aspect='auto', vmin=0, vmax=1, cmap='magma',
                              extent=[residual_grid[0], residual_grid[-1], cycle_grid[0], cycle_grid[-1]])
            current = summary.get('thresholds', {})
            ax.axvline(current.get('residual', 0.45), color='cyan', linestyle='--', linewidth=1)
            ax.axhline(current.get('cycle_pixels', 1.0), color='cyan', linestyle='--', linewidth=1)
            ax.set_xlim(residual_grid[0], residual_grid[-1]); ax.set_ylim(cycle_grid[0], cycle_grid[-1])
            fig.colorbar(image, ax=ax, label='Fraction violating modified rule')
        else:
            unavailable(ax, 'No held-out observations')
        ax.set(xlabel='Reconstruction-rule threshold', ylabel='Cycle-rule threshold (pixels)', title=scope)
        sensitivity[scope] = {'pairs': int(len(values)), 'proxy_event_fraction': prevalence.tolist()}
    store(fig, '28_quality_rule_threshold_sensitivity', 'This sensitivity plot changes the definition of the operational event; it does not measure model discrimination or tune a clinical cutoff. Cyan lines mark the current declared thresholds. Folding/support limits stay fixed.')
    result['rule_threshold_sensitivity'] = {'residual_grid': residual_grid.tolist(), 'cycle_grid_pixels': cycle_grid.tolist(), 'scopes': sensitivity}

    # 7. Failure causes overlap, so bars are not stacked or represented as a
    # partition of the population.
    rule_names = ['Reconstruction', 'Cycle consistency', 'Folding', 'Out of frame']
    rule_limits = summary.get('thresholds', {})
    flags = np.column_stack((features[:, 0] > rule_limits.get('residual', 0.45),
                             features[:, 1] > rule_limits.get('cycle_pixels', 1.0),
                             features[:, 3] > rule_limits.get('fold_fraction', 0.01),
                             features[:, 4] > rule_limits.get('invalid_fraction', 0.25)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    causes = {}
    bar_width = 0.8/max(len(domain_names), 1)
    for index, domain in enumerate(domain_names):
        mask = test & (domains == domain)
        fractions = flags[mask].mean(axis=0)
        centers = np.arange(4)+(index-(len(domain_names)-1)/2)*bar_width
        axes[0].bar(centers, fractions, width=bar_width, color=colors[index % len(colors)], label=domain)
        counts = np.bincount(flags[mask].sum(axis=1), minlength=5)
        axes[1].plot(range(5), counts/max(int(mask.sum()), 1), 'o-', label=domain, color=colors[index % len(colors)])
        causes[domain] = {'pairs': int(mask.sum()), 'component_event_fraction': fractions.tolist(),
                          'number_of_simultaneous_flags_counts': counts.tolist()}
    axes[0].set_xticks(range(4), rule_names, rotation=20, ha='right')
    axes[0].set(ylim=(0, 1.05), ylabel='Held-out pair fraction', title='Which quality-rule components fail?')
    axes[1].set(xticks=range(5), xlabel='Simultaneous failed components', ylabel='Held-out pair fraction', title='Overlap among quality indicators')
    for ax in axes: ax.legend(); ax.grid(axis='y', alpha=0.3)
    store(fig, '29_quality_rule_failure_causes', 'Flags use the same features that define the proxy target. Causes can overlap and indicate rule violations, not independently verified physical failure mechanisms.')
    result['rule_failure_components'] = {'names': rule_names, 'domains': causes}

    # 8. A fixed risk-score cutoff gives an acceptance/flagging trade-off. Empty
    # accepted sets have undefined event rates, not an invented zero-risk value.
    cutoffs = np.linspace(0, 1, 101)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    operating = {}
    for domain_index, domain in enumerate(domain_names):
        mask = test & (domains == domain)
        probabilities, outcomes = risk[mask], events[mask]
        acceptance, accepted_event, captured_event = [], [], []
        for cutoff in cutoffs:
            accepted = probabilities <= cutoff
            acceptance.append(float(accepted.mean()))
            accepted_event.append(float(outcomes[accepted].mean()) if accepted.any() else np.nan)
            captured_event.append(float(outcomes[~accepted].sum()/outcomes.sum()) if outcomes.sum() else np.nan)
        color = colors[domain_index % len(colors)]
        axes[0].plot(cutoffs, acceptance, label=domain+' accepted', color=color)
        axes[0].plot(cutoffs, captured_event, '--', label=domain+' rule events flagged', color=color)
        axes[1].plot(cutoffs, accepted_event, color=color, label=domain)
        operating[domain] = {'acceptance_fraction': acceptance, 'accepted_proxy_event_fraction': accepted_event,
                             'fraction_of_rule_events_flagged': captured_event}
    for ax in axes: ax.set(xlabel='Accept if predicted proxy risk <= cutoff', ylim=(-0.02, 1.02)); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    axes[0].set(ylabel='Fraction', title='Operating cutoff: acceptance and event capture')
    axes[1].set(ylabel='Rule-event fraction among accepted pairs', title='Accepted quality-rule events')
    store(fig, '30_risk_cutoff_operating_curves', 'These curves describe cutoff trade-offs on held-out proxy outcomes. Blank segments mean no accepted pairs or no event denominator. This is a research handling rule, not patient triage.')
    result['risk_operating_cutoffs'] = {'cutoffs': cutoffs.tolist(), 'domains': operating}

    # 9. Calibration movement is recorded for the same actual held-out scores.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    calibration_movement = {}
    for domain_index, domain in enumerate(domain_names):
        mask = test & (domains == domain)
        before, after = raw_risk[mask], risk[mask]
        color = colors[domain_index % len(colors)]
        axes[0].scatter(before, after, color=color, alpha=0.7, label=domain)
        delta = after-before
        x, cdf = _result_ecdf(delta)
        axes[1].step(x, cdf, where='post', color=color, label=domain)
        calibration_movement[domain] = {'raw_probability': before.tolist(), 'calibrated_probability': after.tolist(),
                                        'change': delta.tolist(), 'change_ecdf': cdf.tolist(), 'change_ecdf_x': x.tolist()}
    axes[0].plot([0, 1], [0, 1], '--', color='#888888')
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel='Raw BN probability', ylabel='After calibration step', title='How calibration changes held-out scores')
    axes[1].axvline(0, color='#888888', linestyle='--')
    axes[1].set(xlabel='Calibrated minus raw probability', ylabel='Empirical cumulative fraction', title='Distribution of score adjustment')
    for ax in axes: ax.legend(); ax.grid(alpha=0.3)
    store(fig, '31_calibration_score_movement', 'The adjustment was fitted only on calibration groups. Identity movement is expected when calibration was skipped; the calibrator status remains in summary.json.')
    result['calibration_score_movement'] = {'status': summary.get('calibrator', {}).get('status'), 'domains': calibration_movement}

    # 10. Entropy measures ambiguity between latent evidence patterns. It is not
    # calibrated displacement uncertainty or a Bayesian neural-network posterior.
    representation = risk_model['representation']
    weights = np.asarray(representation['weights'], float)
    emissions = np.asarray(representation['emission'], float)
    binary = (features > threshold).astype(float)
    log_membership = (binary @ np.log(emissions).T + (1-binary) @ np.log1p(-emissions).T + np.log(weights)[None, :])
    membership = np.exp(log_membership-logsumexp(log_membership, axis=1, keepdims=True))
    entropy = -np.sum(membership*np.log2(np.maximum(membership, 1e-15)), axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    safe_boxplot(axes[0], [entropy[test & (domains == domain)] for domain in domain_names], domain_names, 'Latent membership entropy (bits)')
    axes[0].set_title('Ambiguity of learned evidence patterns')
    for domain_index, domain in enumerate(domain_names):
        mask = test & (domains == domain)
        axes[1].scatter(entropy[mask], risk[mask], label=domain, color=colors[domain_index % len(colors)], alpha=0.7)
    axes[1].set(xlabel='Latent membership entropy (bits)', ylabel='Proxy-failure probability', title='Latent ambiguity versus operational score')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    store(fig, '32_latent_pattern_entropy', 'Entropy is computed from the train-fitted Bernoulli mixture. A low-entropy evidence pattern may still be confidently unreliable; entropy alone is not an error probability.')
    result['latent_membership'] = {'sample_ids': [str(row['id']) for row in rows],
                                   'posterior': membership.tolist(), 'entropy_bits': entropy.tolist()}

    # 11. A domain-composition view can expose acquisition-driven clusters.
    cluster_names = sorted(set(int(value) for value in phenotypes[test]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = np.zeros((len(cluster_names), len(domain_names)), int)
    cluster_values = []
    for cluster_index, cluster in enumerate(cluster_names):
        cluster_mask = test & (phenotypes == cluster)
        cluster_values.append(risk[cluster_mask])
        for domain_index, domain in enumerate(domain_names):
            counts[cluster_index, domain_index] = int((cluster_mask & (domains == domain)).sum())
    bottom = np.zeros(len(cluster_names), float)
    for domain_index, domain in enumerate(domain_names):
        fraction = counts[:, domain_index]/np.maximum(counts.sum(axis=1), 1)
        axes[0].bar(range(len(cluster_names)), fraction, bottom=bottom, label=domain, color=colors[domain_index % len(colors)])
        bottom += fraction
    axes[0].set(xticks=range(len(cluster_names)), xticklabels=['Cluster '+str(label) for label in cluster_names],
                ylim=(0, 1.05), ylabel='Within-cluster domain fraction', title='Do mechanical clusters track acquisition domain?')
    axes[0].legend()
    safe_boxplot(axes[1], cluster_values, ['Cluster '+str(label) for label in cluster_names], 'Proxy-failure probability')
    axes[1].set(ylim=(-0.02, 1.02), title='Operational scores across descriptive clusters')
    store(fig, '33_phenotype_domain_composition', 'Clusters were fitted only on training embeddings. A domain-dominated cluster is evidence of acquisition structure, not a tissue diagnosis. Cluster indices have no ordinal meaning.')
    result['phenotype_domain_composition'] = {'clusters': cluster_names, 'domains': domain_names,
                                             'counts': counts.tolist(), 'cluster_test_risk': [values.tolist() for values in cluster_values]}

    # 12. A group failure atlas keeps difficult groups visible. Percentile ranks
    # are display transforms within the evaluated test subset, never new labels.
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4.5, min(len(groups), 12)*0.38+1.8)),
                             gridspec_kw={'width_ratios': [4, 1.3]})
    atlas = []
    if groups:
        from scipy.stats import rankdata
        group_features = np.asarray([group['features'] for group in groups], float)
        ranks = np.column_stack([rankdata(group_features[:, feature], method='average')/len(groups)
                                 for feature in range(group_features.shape[1])])
        order = np.argsort(-np.asarray([group['risk_mean'] for group in groups]), kind='stable')
        shown = order[:12]
        image = axes[0].imshow(ranks[shown], cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        axes[0].set_xticks(range(len(FEATURE_NAMES)), FEATURE_NAMES, rotation=28, ha='right')
        axes[0].set_yticks(range(len(shown)), [groups[index]['group'][-40:] for index in shown], fontsize=8)
        axes[0].set_title('Highest-score groups: feature percentile ranks')
        fig.colorbar(image, ax=axes[0], label='Rank fraction within evaluated test groups', shrink=0.7)
        axes[1].barh(range(len(shown)), [groups[index]['risk_mean'] for index in shown], color=colors[0])
        axes[1].set(xlim=(0, 1), ylim=(len(shown)-0.5, -0.5), xlabel='Mean predicted proxy risk', yticks=[])
        axes[1].set_title('Same groups')
        for index in order:
            atlas.append({'group': groups[index]['group'], 'mean_proxy_risk': groups[index]['risk_mean'],
                          'feature_means': group_features[index].tolist(), 'feature_rank_fractions': ranks[index].tolist()})
    else:
        for ax in axes: unavailable(ax, 'No held-out groups')
    store(fig, '34_group_failure_atlas', 'Groups are ordered by predicted proxy risk, with at most twelve displayed. Heatmap colors are within-test feature ranks; native-unit means for every group are exported in JSON. Phantom grouping remains provisional without acquisition IDs.')
    result['group_failure_atlas'] = {'feature_names': FEATURE_NAMES, 'ordered_groups': atlas}

    # Use shared strain histogram bins across all saved examples. The middle
    # 99% display range prevents one outlier flattening every histogram. Tail
    # counts are explicitly retained; raw unclipped strain is saved in NPZ.
    strain_values = [np.asarray(example['maps']['strain'], float)[np.asarray(example['maps']['valid'], bool)]*100
                     for example in examples]
    valid_strain = np.concatenate([value[np.isfinite(value)] for value in strain_values]) if any(len(value) for value in strain_values) else np.array([])
    if len(valid_strain):
        low, high = np.quantile(valid_strain, [0.005, 0.995])
        if high-low < 1e-8:
            low, high = float(low)-0.05, float(high)+0.05
        strain_edges = np.linspace(low, high, 41)
    else:
        strain_edges = np.linspace(-1, 1, 41)
    result['example_strain_histogram_bin_edges_percent'] = strain_edges.tolist()
    result['example_diagnostics'] = []
    for example_index, example in enumerate(examples):
        maps = example['maps']
        valid = np.asarray(maps['valid'], bool)
        flow = np.asarray(example['flow'], float)
        height, width = valid.shape
        depth = np.arange(height)
        example_data = {'id': str(example['record']['id']), 'raw_maps': f'example_{example_index:02d}_maps.npz',
                        'profiles': {}, 'shape': [height, width]}

        # E1: depth profiles show lateral heterogeneity and support, without
        # presenting within-image spread as confidence in a physical estimate.
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        profile_fields = [('axial_flow_pixels', flow[1], 'Axial flow (pixels)'),
                          ('axial_strain_percent', np.asarray(maps['strain'])*100, 'Axial strain (%)'),
                          ('reconstruction_loss', maps['residual'], '1 - local RF correlation'),
                          ('cycle_error_pixels', maps['cycle'], 'Cycle error (pixels)')]
        for ax, (name, values, ylabel) in zip(axes.flat, profile_fields):
            quartiles, support = _profile_quantiles(values, valid)
            ax.fill_between(depth, quartiles[0], quartiles[2], color=colors[0], alpha=0.2, label='Lateral interquartile range')
            ax.plot(depth, quartiles[1], color=colors[0], label='Lateral median')
            ax.set(xlabel='Axial sample', ylabel=ylabel, title=name.replace('_', ' ')); ax.grid(alpha=0.3)
            example_data['profiles'][name] = {'axial_sample': depth.tolist(), 'q25': quartiles[0].tolist(),
                                               'median': quartiles[1].tolist(), 'q75': quartiles[2].tolist(),
                                               'valid_lateral_samples': support.tolist()}
        axes[0, 0].legend(fontsize=8)
        fig.suptitle('Depth-dependent mechanics and quality: '+str(example['record']['id'])[:48])
        store(fig, f'35_example_{example_index:02d}_depth_profiles', 'Profiles summarize valid lateral samples at each depth. Shading is spatial interquartile spread, not an uncertainty interval or pathology-linked tissue estimate. Border depths with no support are left blank.')

        # E2: visualize the actual vector field with an explicitly reported
        # display magnification; the line plot retains signed RF phase.
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].imshow(bmode(example['pre']), cmap='gray', vmin=-60, vmax=0, aspect='auto')
        stride_z, stride_x = max(1, height//16), max(1, width//12)
        zz, xx = np.mgrid[0:height:stride_z, 0:width:stride_x]
        ux, uz = flow[0, ::stride_z, ::stride_x], flow[1, ::stride_z, ::stride_x]
        magnitude = np.sqrt(ux**2+uz**2)
        q95 = float(np.quantile(magnitude, 0.95))
        magnification = float(np.clip(3/max(q95, 1e-8), 1, 20))
        arrows = axes[0].quiver(xx, zz, ux*magnification, uz*magnification, color='#ffd64d',
                                angles='xy', scale_units='xy', scale=1, width=0.003)
        vector_key = axes[0].quiverkey(arrows, 0.5, 0.91, magnification,
                                       f'1 pixel\n(arrows x{magnification:.1f})', labelpos='E', fontproperties={'size': 8})
        vector_key.text.set_bbox({'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85, 'pad': 2})
        axes[0].set(xlabel='Lateral sample', ylabel='Axial sample', title='Displacement vectors over pre-RF envelope')
        center = width//2
        for values, label, color in ((example['pre'], 'Pre RF', colors[0]),
                                     (example['post'], 'Post RF', colors[1]),
                                     (maps['warped'], 'Warped post RF', colors[2])):
            axes[1].plot(depth, np.asarray(values)[:, center], label=label, color=color, linewidth=0.9, alpha=0.9)
        axes[1].set(xlabel='Axial sample', ylabel='Normalized signed RF amplitude', title=f'Center RF line, lateral sample {center}')
        axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
        store(fig, f'36_example_{example_index:02d}_vector_field_rf_alignment', 'Arrow magnification is explicitly reported and changes only display length. RF waveforms preserve carrier phase; a central line is an inspection aid rather than a complete 2D registration metric.')
        example_data['quiver'] = {'axial_sample': zz.tolist(), 'lateral_sample': xx.tolist(),
                                   'ux_pixels': ux.tolist(), 'uz_pixels': uz.tolist(),
                                   'display_magnification': magnification}
        example_data['center_rf_line'] = {'lateral_sample': center, 'axial_sample': depth.tolist(),
                                          'pre': np.asarray(example['pre'])[:, center].tolist(),
                                          'post': np.asarray(example['post'])[:, center].tolist(),
                                          'warped_post': np.asarray(maps['warped'])[:, center].tolist()}

        # E3: Jacobian <= 0 indicates a non-orientation-preserving local warp;
        # validity support is displayed separately from mechanics values.
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        jacobian = np.asarray(maps['jacobian'], float)
        spread = max(float(np.quantile(np.abs(jacobian-1), 0.99)), 0.05)
        image = axes[0].imshow(jacobian, cmap='RdBu_r', vmin=1-spread, vmax=1+spread, aspect='auto')
        axes[0].set_title('Warp Jacobian determinant'); fig.colorbar(image, ax=axes[0], extend='both')
        axes[1].imshow(valid, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        folded = (jacobian <= 0) & valid
        if folded.any():
            overlay = np.ma.masked_where(~folded, np.ones_like(jacobian))
            axes[1].imshow(overlay, cmap=matplotlib.colors.ListedColormap(['#df3f46']), alpha=0.9, aspect='auto')
        axes[1].set_title(f'Valid support; folding in red ({int(folded.sum())})')
        sample_strain = strain_values[example_index]
        sample_strain = sample_strain[np.isfinite(sample_strain)]
        counts, _ = np.histogram(sample_strain, bins=strain_edges)
        axes[2].stairs(counts, strain_edges, color=colors[0], fill=True, alpha=0.6)
        axes[2].axvline(0, color='#888888', linestyle='--')
        underflow, overflow = int((sample_strain < strain_edges[0]).sum()), int((sample_strain > strain_edges[-1]).sum())
        axes[2].set(xlabel='Axial strain (%)', ylabel='Valid pixels', title='Strain histogram with shared bins')
        axes[2].text(0.03, 0.97, f'Outside displayed bins: {underflow+overflow}', transform=axes[2].transAxes, va='top', fontsize=9)
        for ax in axes[:2]: ax.set(xlabel='Lateral sample', ylabel='Axial sample')
        store(fig, f'37_example_{example_index:02d}_jacobian_support_strain_histogram', 'Jacobian colors show a symmetric range around one, clipped at the 99th percentile of deviation. Red pixels mark Jacobian <= 0 inside valid support. Shared histogram bins show the central 99% strain range across saved examples; tail counts and raw maps remain available.')
        example_data['jacobian_support'] = {'valid_pixels': int(valid.sum()), 'total_pixels': int(valid.size),
                                             'folded_valid_pixels': int(folded.sum()),
                                             'jacobian_quantiles': np.quantile(jacobian, [0, 0.01, 0.5, 0.99, 1]).tolist(),
                                             'display_limits': [1-spread, 1+spread]}
        example_data['strain_histogram'] = {'bin_edges_percent': strain_edges.tolist(), 'counts': counts.tolist(),
                                             'underflow': underflow, 'overflow': overflow,
                                             'valid_strain_quantiles_percent': np.quantile(sample_strain, [0, 0.01, 0.5, 0.99, 1]).tolist() if len(sample_strain) else None}
        result['example_diagnostics'].append(example_data)
    return json_safe(result)


# ---------------------------------------------------------------------------
# Measured perturbation sensitivity and independent analytic motion fixtures.
# These studies never fit, recalibrate, or choose thresholds on held-out data.
# ---------------------------------------------------------------------------
def _study_prediction(model, pre, post, device):
    """Return deterministic forward/reverse fields with synchronized latency."""
    a = torch.as_tensor(np.ascontiguousarray(pre), device=device)[None, None].float()
    b = torch.as_tensor(np.ascontiguousarray(post), device=device)[None, None].float()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        forward, _ = model(a, b)
        backward, _ = model(b, a)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    seconds = time.perf_counter()-started
    return forward[0].cpu().numpy(), backward[0].cpu().numpy(), seconds


def run_robustness_study(model, examples, risk_model, calibrator, args):
    """Frozen-model post-acquisition perturbations on held-out example crops.

    Noise strength is relative to the clean post-frame RMS. Gain and axial
    stripe corruption affect only the post frame after the normal shared pair
    normalization. They stress acquisition inconsistency, not a new normalized
    dataset. Displacement change measures sensitivity, not true motion error.
    """
    budget = int(getattr(args, 'robustness_samples', 3))
    if budget < 0:
        raise ValueError('--robustness-samples cannot be negative.')
    if getattr(args, 'skip_robustness', False) or budget == 0 or not examples:
        return {'status': 'skipped', 'reason': 'disabled or no held-out examples', 'rows': []}
    device, records = args.device_resolved, []
    original_training = model.training
    model.eval()
    try:
        for index, example in enumerate(examples[:budget]):
            pre, post = np.array(example['pre'], np.float32), np.array(example['post'], np.float32)
            if pre.shape != post.shape or not np.isfinite(pre).all() or not np.isfinite(post).all():
                raise ValueError('Robustness study requires matching finite RF crops.')
            clean, reverse, clean_seconds = _study_prediction(model, pre, post, device)
            clean_features, clean_maps = quality_maps(pre, post, clean, reverse, device)
            rng = np.random.default_rng(args.seed+104729*(index+1))
            # The same noise realization is scaled across strengths for paired
            # comparisons; it is independent between held-out RF pairs.
            noise = rng.normal(size=post.shape).astype(np.float32)
            rms = max(float(np.sqrt(np.mean(post.astype(np.float64)**2))), 1e-6)
            for condition, strengths in (('noise_rms', (0., .05, .15, .30)),
                                         ('post_gain', (.5, 1., 1.5)),
                                         ('post_stripe_fraction', (0., .10, .25))):
                for strength in strengths:
                    changed = post.copy()
                    baseline = ((condition != 'post_gain' and strength == 0)
                                or (condition == 'post_gain' and strength == 1))
                    if condition == 'noise_rms':
                        changed += noise*(rms*strength)
                    elif condition == 'post_gain':
                        changed *= strength
                    elif strength:
                        # Centered axial dropout has a known spatial extent.
                        count = max(1, int(round(post.shape[0]*strength)))
                        start = (post.shape[0]-count)//2
                        changed[start:start+count, :] = 0
                    if baseline:
                        flow, features, maps, seconds = clean, clean_features, clean_maps, clean_seconds
                    else:
                        flow, backward, seconds = _study_prediction(model, pre, changed, device)
                        features, maps = quality_maps(pre, changed, flow, backward, device)
                    common = clean_maps['valid'] & maps['valid']
                    delta = np.sqrt(np.sum((flow-clean)**2, axis=0))
                    raw = float(predict_risk(risk_model, features[None])[0])
                    probability = float(apply_calibrator(calibrator, [raw])[0])
                    record = {'id': example['record']['id'], 'domain': example['record']['domain'],
                              'group': example['record'].get('group', example['record']['id']),
                              'condition': condition, 'strength': float(strength),
                              'proxy_failure': int(proxy_failure(features, args.residual_threshold, args.cycle_threshold)),
                              'risk_raw': raw, 'risk_calibrated': probability,
                              'flow_change_pixels': float(delta[common].mean()) if common.any() else None,
                              'common_support_fraction': float(common.mean()),
                              'inference_seconds': float(seconds),
                              **{name: float(value) for name, value in zip(FEATURE_NAMES, features)}}
                    records.append(record)
    finally:
        model.train(original_training)
    return {'status': 'completed', 'rows': records, 'examples': min(len(examples), budget),
            'normalization': 'Clean shared pair normalization retained; perturb post frame afterward.',
            'interpretation': 'Held-out perturbation sensitivity; no real-data displacement truth is available.',
            'risk_model_status': 'Frozen training fit and calibration; no adaptation to perturbations.'}


def analytic_motion_benchmark(model, args):
    """Known translation accuracy on continuous texture; not an acoustic model.

    For analytic texture T, pre(z,x)=T(z,x), post(z,x)=T(z-uz,x-ux).
    Therefore post(z+uz,x+ux)=pre(z,x) exactly before added noise. The known
    reference-to-moving flow is [ux,uz], and its axial strain is zero. No RF
    archive metadata or simulated displacement convention enters this fixture.
    """
    if getattr(args, 'skip_robustness', False):
        return {'status': 'skipped', 'rows': [], 'reason': 'disabled'}
    device = args.device_resolved
    height, width = max(32, min(96, args.patch[0])), max(32, min(64, args.patch[1]))
    z, x = np.mgrid[:height, :width]
    rng = np.random.default_rng(args.seed+65537)
    amplitudes = rng.uniform(.3, 1., 20)
    fz, fx = rng.uniform(.10, .30, 20), rng.uniform(-.09, .09, 20)
    phase = rng.uniform(0, 2*np.pi, 20)

    def texture(zz, xx):
        value = np.zeros((height, width), np.float64)
        for amplitude, axial, lateral, offset in zip(amplitudes, fz, fx, phase):
            value += amplitude*np.sin(2*np.pi*(axial*zz+lateral*xx)+offset)
        return value

    pre = texture(z, x)
    noise_direction = rng.normal(size=pre.shape)
    rows = []
    original_training = model.training
    model.eval()
    try:
        for magnitude in (0., .5, 1., 2., 4.):
            uz, ux = float(magnitude), float(magnitude*.25)
            clean_post = texture(z-uz, x-ux)
            for noise_strength in (0., .10, .30):
                post = clean_post+noise_direction*float(pre.std())*noise_strength
                pair = normalize_rf_pair(np.stack((pre, post)))
                flow, backward, seconds = _study_prediction(model, pair[0], pair[1], device)
                _, maps = quality_maps(pair[0], pair[1], flow, backward, device)
                truth = np.array([ux, uz], np.float32)[:, None, None]
                margin = int(math.ceil(max(abs(ux), abs(uz))))+5
                support = np.zeros((height, width), bool)
                support[margin:height-margin, margin:width-margin] = True
                if not support.any():
                    raise ValueError('Analytic benchmark has no valid interior support.')
                phase_flow = phase_translation(pair[0], pair[1])
                zero = np.zeros_like(flow)
                values = {'neural': flow, 'phase': phase_flow, 'zero': zero}
                row = {'axial_shift_pixels': uz, 'lateral_shift_pixels': ux,
                       'true_magnitude_pixels': float(np.hypot(ux, uz)),
                       'noise_rms': noise_strength, 'shape': [height, width],
                       'evaluation_pixels': int(support.sum()), 'inference_seconds': seconds,
                       'neural_strain_mae': float(np.abs(maps['strain'][support]).mean())}
                for name, estimate in values.items():
                    error = np.sqrt(np.sum((estimate-truth)**2, axis=0))[support]
                    row[name+'_epe_pixels'] = float(error.mean())
                rows.append(row)
    finally:
        model.train(original_training)
    return {'status': 'completed', 'rows': rows, 'origin': 'analytic_continuous_texture_translation',
            'claim_boundary': 'Software/reference-fixture accuracy only; not validated acoustic or clinical performance.',
            'truth': 'pre=T(z,x); post=T(z-uz,x-ux); reference-to-moving flow=[ux,uz]; true strain=0.',
            'evaluation': 'Fixed truth-defined interior margin; identical support for all three estimators.'}


def robustness_report_figures(study, benchmark, save):
    """Five white-background result figures; numeric points remain in JSON."""
    import matplotlib.pyplot as plt
    records = study.get('rows', [])
    palette = ['#1765ad', '#e68a27', '#24998c']
    if records:
        conditions = ['noise_rms', 'post_gain', 'post_stripe_fraction']
        condition_names = ['Post-frame Gaussian noise', 'Post-frame gain mismatch', 'Post-frame axial dropout']
        for fields, name, caption in [
            (['reconstruction loss', 'cycle error'], '30_perturbation_quality', 'Measured reconstruction and cycle sensitivity. Gain mismatch is applied after shared normalization; local correlation can be gain invariant.'),
            (['risk_calibrated', 'flow_change_pixels'], '31_perturbation_risk_flow', 'Frozen BN proxy risk and mean flow change from clean input. Flow change is sensitivity, not displacement error.')]:
            fig, axes = plt.subplots(2, 3, figsize=(13, 7))
            for column, (condition, title) in enumerate(zip(conditions, condition_names)):
                subset = [r for r in records if r['condition'] == condition]
                strengths = sorted({r['strength'] for r in subset})
                for row_index, field in enumerate(fields):
                    ax = axes[row_index, column]
                    for domain_index, domain in enumerate(sorted({r['domain'] for r in subset})):
                        means, lows, highs = [], [], []
                        for strength in strengths:
                            values = [r[field] for r in subset if r['domain'] == domain and r['strength'] == strength and r[field] is not None]
                            means.append(float(np.mean(values)) if values else np.nan)
                            lows.append(float(np.min(values)) if values else np.nan)
                            highs.append(float(np.max(values)) if values else np.nan)
                        color = palette[domain_index % len(palette)]
                        ax.plot(strengths, means, 'o-', label=domain, color=color)
                        ax.fill_between(strengths, lows, highs, color=color, alpha=.12)
                    ax.set(xlabel=condition, ylabel=field, title=title if row_index == 0 else '')
                    ax.grid(alpha=.3); ax.legend(fontsize=8)
            save(fig, name, caption+' Lines are pair means; shading is the observed pair range, not a confidence interval.')
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        labels, costs = [], []
        for condition in conditions:
            values = [r['inference_seconds']*1000 for r in records if r['condition'] == condition]
            labels.append(condition.replace('post_', '').replace('_', ' ')); costs.append(values)
        axes[0].boxplot(costs, showfliers=True)
        axes[0].set_xticks(range(1, len(labels)+1), labels)
        axes[0].tick_params(axis='x', rotation=15); axes[0].set(ylabel='Forward + reverse inference (ms)', title='Measured inference latency')
        for i, condition in enumerate(conditions):
            subset = [r for r in records if r['condition'] == condition]
            levels = sorted({r['strength'] for r in subset})
            axes[1].plot(levels, [np.mean([r['proxy_failure'] for r in subset if r['strength'] == value]) for value in levels], 'o-', label=labels[i])
        axes[1].legend(fontsize=8); axes[1].set(xlabel='Perturbation parameter (condition-specific)', ylabel='Proxy-event fraction', title='Quality-rule event sensitivity')
        save(fig, '32_perturbation_runtime_events', 'CUDA timings synchronize the device. Latencies include two network directions, exclude quality-map computation, and are not a hardware benchmark; event parameters have different units.')
    rows = benchmark.get('rows', [])
    if rows:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, noise in zip(axes, sorted({r['noise_rms'] for r in rows})):
            subset = sorted([r for r in rows if r['noise_rms'] == noise], key=lambda r: r['axial_shift_pixels'])
            for name, color in zip(['zero', 'phase', 'neural'], palette):
                ax.plot([r['true_magnitude_pixels'] for r in subset], [r[name+'_epe_pixels'] for r in subset], 'o-', label=name, color=color)
            ax.set(xlabel='True translation magnitude (pixels)', ylabel='Mean EPE (pixels)', title=f'Analytic texture: noise RMS = {noise}')
            ax.legend(); ax.grid(alpha=.3)
        save(fig, '33_analytic_translation_accuracy', 'Independent known translations of a continuous synthetic texture. All estimators use the same truth-defined interior. These are fixture results, not real RF accuracy or clinical evidence.')
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        shifts = sorted({r['axial_shift_pixels'] for r in rows})
        levels = sorted({r['noise_rms'] for r in rows})
        strain = np.array([[next(r['neural_strain_mae']*100 for r in rows if r['axial_shift_pixels'] == shift and r['noise_rms'] == noise) for shift in shifts] for noise in levels])
        image = axes[0].imshow(strain, aspect='auto', cmap='magma')
        axes[0].set_xticks(range(len(shifts)), shifts); axes[0].set_yticks(range(len(levels)), levels)
        axes[0].set(xlabel='True axial translation (pixels)', ylabel='Noise RMS', title='False strain under true rigid translation')
        fig.colorbar(image, ax=axes[0], label='Mean absolute axial strain (%)')
        for name, color in zip(['zero', 'phase', 'neural'], palette):
            values = np.sort([r[name+'_epe_pixels'] for r in rows])
            axes[1].step(values, np.arange(1, len(values)+1)/len(values), where='post', label=name, color=color)
        axes[1].set(xlabel='Fixture mean EPE (pixels)', ylabel='Empirical fraction of fixture cases', title='Accuracy distribution across controlled cases')
        axes[1].legend(); axes[1].grid(alpha=.3)
        save(fig, '34_analytic_strain_and_error_ecdf', 'Rigid translations have exactly zero strain; nonzero estimated strain is an artifact in this reference fixture. ECDF weights each controlled case equally.')


def create_report(args, rows, examples, summary, demonstrations, risk_model, coordinates, phenotypes):
    figure_style()
    out = Path(args.out)
    figure_dir = out/'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    gallery = []

    def save(fig, name, caption):
        fig.savefig(figure_dir/(name+'.png'), dpi=170, facecolor='white', bbox_inches='tight')
        fig.savefig(figure_dir/(name+'.pdf'), facecolor='white', bbox_inches='tight')
        plt.close(fig)
        gallery.append({'file': 'figures/'+name+'.png', 'pdf': 'figures/'+name+'.pdf', 'caption': caption})

    def note(ax, message):
        ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes, wrap=True)
        ax.set_axis_off()

    roles = np.array([r['split'] for r in rows])
    domains = np.array([r['domain'] for r in rows])
    test = roles == 'test'
    y = np.array([r['proxy_failure'] for r in rows])
    p = np.array([r['risk_calibrated'] for r in rows])
    raw = np.array([r['risk_raw'] for r in rows])
    colors = ['#1765ad', '#e68a27', '#24998c', '#b64f73']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, domain in enumerate(sorted(set(domains))):
        counts = [int(((roles == role) & (domains == domain)).sum()) for role in ROLES]
        axes[0].bar(np.arange(4)+index*0.35, counts, width=0.35, label=domain, color=colors[index])
    axes[0].set_xticks(np.arange(4)+0.175, ROLES, rotation=15)
    axes[0].set_ylabel('Evaluated RF pairs'); axes[0].legend()
    axes[0].set_title('Report subset by domain and partition')
    group_counts = [len(set(r['group'] for r in rows if r['split'] == role)) for role in ROLES]
    axes[1].bar(ROLES, group_counts, color=colors)
    axes[1].set_ylabel('Distinct manifest groups'); axes[1].set_title('Groups represented in report')
    save(fig, '01_dataset_partitions', 'Pair counts and group counts in the evaluated subset. Phantom groups remain provisional without acquisition metadata.')

    history_path = out/'history.json'
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if history_path.exists():
        history = read_json(history_path)
        for key in ('train_loss', 'validation_loss'):
            axes[0].plot([r['epoch'] for r in history], [r[key] for r in history], 'o-', label=key)
        axes[0].legend(); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Self-supervised objective')
        axes[1].plot([r['epoch'] for r in history], [r['lr'] for r in history], 'o-', color=colors[2])
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Learning rate')
    else:
        for ax in axes: note(ax, 'No training history present in this output directory')
    axes[0].set_title('Optimization and held-out validation'); axes[1].set_title('Learning-rate schedule')
    save(fig, '02_learning_curves', 'Training crops are random; validation crops are fixed. Their objectives are not identical sampling distributions.')

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, name in zip(axes.flat, FEATURE_NAMES):
        for index, domain in enumerate(sorted(set(domains))):
            values = np.array([r[name] for r in rows])[test & (domains == domain)]
            if len(values): ax.hist(values, bins=15, alpha=0.55, label=domain, color=colors[index])
        ax.set_title(name); ax.set_ylabel('Test pairs'); ax.grid(axis='y')
    axes[0, 0].legend()
    save(fig, '03_quality_distributions', 'Held-out mechanics evidence. These indicators are proxies when reference displacement is unavailable.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, domain in enumerate(sorted(set(domains))):
        mask = test & (domains == domain)
        axes[0].scatter(np.array([r['reconstruction loss'] for r in rows])[mask],
                        np.array([r['cycle error'] for r in rows])[mask], label=domain, alpha=0.65, color=colors[i])
        axes[1].scatter(np.array([r['strain_median'] for r in rows])[mask]*100,
                        np.array([r['strain dispersion'] for r in rows])[mask]*100,
                        label=domain, alpha=0.65, color=colors[i])
    axes[0].set(xlabel='1 − local RF correlation', ylabel='Cycle error (pixels)', title='Reconstruction versus cycle consistency')
    axes[1].set(xlabel='Median axial strain (%)', ylabel='Strain MAD (%)', title='Mechanical phenotype features')
    axes[0].legend(); axes[1].legend()
    save(fig, '04_quality_relationships', 'Scatterplots describe empirical relationships; correlation alone cannot establish conditional independence.')

    fig, ax = plt.subplots(figsize=(8, 5))
    matrix = np.array([[r[name] for name in FEATURE_NAMES] for r in rows if r['split'] == 'test'])
    if len(matrix) > 1:
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = np.nan_to_num(np.corrcoef(matrix.T))
        image = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(len(FEATURE_NAMES)), FEATURE_NAMES, rotation=40, ha='right')
        ax.set_yticks(range(len(FEATURE_NAMES)), FEATURE_NAMES)
        fig.colorbar(image, ax=ax, label='Pearson correlation')
    else: note(ax, 'At least two test pairs are needed')
    ax.set_title('Observed feature dependence')
    save(fig, '05_feature_correlations', 'Feature correlations, with constant-feature entries displayed as zero. Graphical-model independence assumptions are evaluated separately.')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, domain in enumerate(sorted(set(domains))):
        mask = test & (domains == domain)
        values = np.array([[r[key] for key in ('zero_residual', 'phase_residual', 'reconstruction loss')] for r in rows])[mask]
        if len(values):
            axes[0].plot(range(3), values.mean(0), 'o-', label=domain, color=colors[i])
            improvement = values[:, 0]-values[:, 2]
            axes[1].hist(improvement, bins=15, alpha=0.55, label=domain, color=colors[i])
    axes[0].set_xticks(range(3), ['Zero motion', 'Phase translation', 'Neural registration'])
    axes[0].set_ylabel('Mean reconstruction loss'); axes[0].set_title('Measured baseline comparison'); axes[0].legend()
    axes[1].axvline(0, color='#555', linestyle='--'); axes[1].set_xlabel('Zero loss − neural loss')
    axes[1].set_title('Positive values favor neural reconstruction'); axes[1].legend()
    save(fig, '06_registration_baselines', 'Proxy comparisons on the same center crops. Lower reconstruction loss does not by itself establish more accurate displacement.')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if len(np.unique(y[test])) == 2:
        for probability, label, color in ((raw, 'Uncalibrated BN', colors[1]), (p, 'Calibrated BN', colors[0])):
            fpr, tpr, _ = roc_curve(y[test], probability[test]); axes[0].plot(fpr, tpr, label=label, color=color)
            precision, recall, _ = precision_recall_curve(y[test], probability[test]); axes[1].plot(recall, precision, label=label, color=color)
        axes[0].plot([0, 1], [0, 1], '--', color='#aaa'); axes[0].legend(); axes[1].legend()
    else:
        for ax in axes: note(ax, 'Ranking curve unavailable:\nheld-out proxy outcomes contain one class')
    axes[0].set(xlabel='False-positive rate', ylabel='True-positive rate', title='Proxy-rule ROC')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Proxy-rule precision–recall')
    save(fig, '07_proxy_roc_pr', 'Ranking measures quality-rule imitation using the same evidence that defines the rule. They are not independent accuracy metrics.')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    edges = np.linspace(0, 1, 11)
    for probabilities, label, color in ((raw, 'Raw BN', colors[1]), (p, 'Calibrated BN', colors[0])):
        means, rates = [], []
        for index in range(10):
            mask = test & (probabilities >= edges[index]) & (probabilities <= edges[index+1] if index == 9 else probabilities < edges[index+1])
            if mask.any(): means.append(probabilities[mask].mean()); rates.append(y[mask].mean())
        axes[0].plot(means, rates, 'o-', label=label, color=color)
    axes[0].plot([0, 1], [0, 1], '--', color='#aaa'); axes[0].legend()
    axes[0].set(xlabel='Mean predicted proxy probability', ylabel='Observed rule-event frequency', title='Held-out reliability diagram')
    axes[1].hist(p[test & (y == 0)], bins=edges, alpha=0.6, label='Rule passes', color=colors[2])
    axes[1].hist(p[test & (y == 1)], bins=edges, alpha=0.6, label='Rule fails', color=colors[3])
    axes[1].set(xlabel='Predicted proxy-failure probability', ylabel='Test pairs', title='Risk-score distributions'); axes[1].legend()
    save(fig, '08_proxy_calibration', 'Platt calibration is fitted on calibration groups only. One-class or tiny calibration sets retain raw BN probabilities.')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for probability, label in ((p, 'BN probability'), (np.array([r['reconstruction loss'] for r in rows]), 'Residual only')):
        order = np.argsort(probability[test], kind='stable')
        events = y[test][order]
        coverage = np.arange(1, len(events)+1)/len(events)
        axes[0].plot(coverage, np.cumsum(events)/np.arange(1, len(events)+1), label=label)
    axes[0].set(xlabel='Pair coverage (lowest scores accepted first)', ylabel='Accepted proxy-event fraction', title='Selective proxy risk'); axes[0].legend()
    # Group-level selection gives each manifest group equal weight.
    test_groups = sorted(set(r['group'] for r in rows if r['split'] == 'test'))
    group_p = np.array([np.mean([r['risk_calibrated'] for r in rows if r['group'] == g]) for g in test_groups])
    group_y = np.array([np.mean([r['proxy_failure'] for r in rows if r['group'] == g]) for g in test_groups])
    order = np.argsort(group_p)
    axes[1].plot(np.arange(1, len(order)+1)/len(order), np.cumsum(group_y[order])/np.arange(1, len(order)+1), 'o-')
    axes[1].set(xlabel='Manifest-group coverage', ylabel='Mean group proxy-event fraction', title='Equal-group selective proxy risk')
    save(fig, '09_selective_risk_coverage', 'Coverage is descriptive for the evaluated subset. Phantom file groups do not guarantee acquisition independence.')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cm = confusion_matrix(y[test], p[test] >= 0.5, labels=[0, 1])
    image = axes[0].imshow(cm, cmap='Blues'); fig.colorbar(image, ax=axes[0])
    for (z, x), value in np.ndenumerate(cm): axes[0].text(x, z, str(value), ha='center', va='center')
    axes[0].set(xticks=[0, 1], yticks=[0, 1], xlabel='Predicted rule event', ylabel='Actual rule event', title='Proxy confusion matrix at 0.5')
    actions = ['accept mechanics', 'reacquire', 'review']
    axes[1].bar(actions, [sum(r['action'] == action for r in rows if r['split'] == 'test') for action in actions], color=colors[:3])
    axes[1].set_ylabel('Test RF pairs'); axes[1].set_title('Illustrative decisions'); axes[1].tick_params(axis='x', rotation=15)
    save(fig, '10_proxy_confusion_decisions', 'Actions refer to research mechanics handling. Costs and the probability threshold are illustrative.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    prob = np.linspace(0, 1, 201)
    losses = np.array([[0, 8], [1, 1.5], [2.5, 0.5]])
    costs = np.stack((1-prob, prob), axis=1) @ losses.T
    for i, action in enumerate(actions): axes[0].plot(prob, costs[:, i], label=action, color=colors[i])
    axes[0].legend(); axes[0].set(xlabel='Proxy-failure probability', ylabel='Expected illustrative loss', title='Decision network: minimum expected loss')
    fn_costs = np.linspace(2, 16, 61)
    decisions = []
    for cost in fn_costs:
        modified = losses.copy(); modified[0, 1] = cost
        decisions.append((np.stack((1-prob, prob), axis=1) @ modified.T).argmin(1))
    image = axes[1].imshow(decisions, origin='lower', aspect='auto', extent=[0, 1, 2, 16], cmap=matplotlib.colors.ListedColormap(colors[:3]), vmin=-0.5, vmax=2.5)
    fig.colorbar(image, ax=axes[1], ticks=[0, 1, 2], label='Action index: 0 accept, 1 reacquire, 2 review')
    axes[1].set(xlabel='Proxy-failure probability', ylabel='Cost of accepting a failed result', title='Decision sensitivity to assumed costs')
    save(fig, '11_decision_network_costs', 'Chance node = operational failure; decision = action; utility = negative cost. All costs are assumptions, not clinical recommendations.')

    # Draw the actual learned BN and its moral graph using the implemented structure.
    nodes = ['F', 'Z'] + ['X'+str(i) for i in range(len(FEATURE_NAMES))]
    positions = {'F': (-0.5, 1.4), 'Z': (0.5, 1.4)}
    positions.update({node: (i-2.5, 0) for i, node in enumerate(nodes[2:])})
    directed_edges = [('F', 'Z')] + [(parent, node) for node in nodes[2:] for parent in ('F', 'Z')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, directed in zip(axes, (True, False)):
        for left, right in directed_edges:
            ax.add_patch(FancyArrowPatch(positions[left], positions[right], arrowstyle='-|>' if directed else '-',
                                         mutation_scale=12, color='#9eabbc', shrinkA=17, shrinkB=17))
        for node, (x, z) in positions.items():
            ax.add_patch(Circle((x, z), 0.22, facecolor='#e8f1fc', edgecolor=colors[0], zorder=3))
            ax.text(x, z, node, ha='center', va='center', zorder=4)
        ax.set(xlim=(-3.1, 3.1), ylim=(-0.6, 2)); ax.set_axis_off()
        ax.set_aspect('equal')
    axes[0].set_title('Learned Bayesian network'); axes[1].set_title('Moralized Markov graph; CPDs kept as factors')
    save(fig, '12_bayesian_and_moral_graphs', 'F = proxy failure, Z = latent evidence pattern, Xi = binarized quality feature. Co-parents F and Z are already joined in this DAG; a separate collider example in pgm_results.json demonstrates a newly added marriage edge.')

    fig, ax = plt.subplots(figsize=(11, 4))
    cpds = np.asarray(risk_model.get('emission', risk_model.get('cpd_x', [])))
    representation = risk_model.get('representation', {})
    # Accept the explicit EM table returned by the PGM implementation.
    if isinstance(representation, dict):
        emission = representation.get('emission', representation.get('p', representation.get('probabilities')))
    else: emission = None
    if emission is None:
        emission = risk_model.get('mixture_p', risk_model.get('latent_emissions'))
    if emission is not None and np.asarray(emission).ndim == 2:
        table = np.asarray(emission)
        image = ax.imshow(table, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(FEATURE_NAMES)), FEATURE_NAMES, rotation=25, ha='right')
        ax.set_yticks(range(table.shape[0]), ['Latent '+str(i) for i in range(table.shape[0])])
        fig.colorbar(image, ax=ax, label='P(binary evidence=1 | latent component)')
    else: note(ax, 'Latent representation emission parameters are recorded in risk_model.json')
    ax.set_title('Unit 1: learned latent evidence representation')
    save(fig, '13_latent_representation', 'Bernoulli-mixture components describe quality patterns and have no intrinsic ordering or tumor-grade interpretation.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    likelihood = representation.get('log_likelihood_history', [])
    axes[0].plot(likelihood, color=colors[0])
    axes[0].set(xlabel='EM iteration', ylabel='Mean observed log likelihood', title='Train-only latent representation learning')
    conditional = np.asarray(risk_model['cpd_x']).reshape(-1, len(FEATURE_NAMES))
    image = axes[1].imshow(conditional, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    axes[1].set_yticks(range(4), ['F=0, Z=0', 'F=0, Z=1', 'F=1, Z=0', 'F=1, Z=1'])
    axes[1].set_xticks(range(len(FEATURE_NAMES)), FEATURE_NAMES, rotation=30, ha='right')
    axes[1].set_title('Learned Bayesian evidence CPDs')
    fig.colorbar(image, ax=axes[1], label='P(Xi=1 | F,Z)')
    save(fig, '13b_em_and_bayesian_cpds', 'EM fits an unlabeled Bernoulli mixture; smoothed soft counts then fit the supervised proxy-event BN. This is a two-stage estimator.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for domain in sorted(set(domains)):
        mask = domains == domain
        axes[0].scatter(coordinates[mask, 0], coordinates[mask, 1], label=domain, alpha=0.6)
    image = axes[1].scatter(coordinates[:, 0], coordinates[:, 1], c=phenotypes, cmap='tab10', alpha=0.65)
    axes[0].legend(); fig.colorbar(image, ax=axes[1], label='Exploratory cluster index')
    for ax in axes: ax.set(xlabel='Train-fitted PC1', ylabel='Train-fitted PC2')
    axes[0].set_title('CNN representation: domain structure'); axes[1].set_title('Label-free mechanical phenotypes')
    save(fig, '14_cnn_representation_phenotypes', 'Standardization, PCA and clustering are fitted on training embeddings. Cluster IDs are descriptive and are not histological grades.')

    fig, ax = plt.subplots(figsize=(9, 4))
    for label in np.unique(phenotypes):
        values = [r['strain_median']*100 for r, c in zip(rows, phenotypes) if c == label and r['split'] == 'test']
        if values: ax.scatter(np.full(len(values), label), values, alpha=0.6, label='Cluster '+str(label))
    ax.set(xlabel='Unordered phenotype cluster', ylabel='Median strain (%)', title='Test mechanics across descriptive clusters')
    save(fig, '15_phenotype_mechanics', 'Strain is a deformation measurement under unknown loading. It does not supply absolute stiffness or tumor grade.')

    hmm = demonstrations.get('hmm', {})
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    observations = hmm.get('observations', [])
    filtered = np.asarray(hmm.get('filtered', []))
    smoothed = np.asarray(hmm.get('smoothed', []))
    states = hmm.get('states', hmm.get('hidden_states', []))
    if len(observations) and filtered.ndim == 2:
        axes[0].step(range(len(observations)), observations, where='mid', label='Observed symbol', color=colors[1])
        if len(states): axes[0].step(range(len(states)), states, where='mid', label='Synthetic hidden state', alpha=0.7)
        axes[0].legend()
        for i in range(filtered.shape[1]):
            axes[1].plot(filtered[:, i], '--', label=f'Filtered state {i}')
            axes[1].plot(smoothed[:, i], label=f'Smoothed state {i}')
        axes[1].legend(ncol=2)
        axes[2].step(range(len(observations)), hmm.get('viterbi', []), where='mid', color=colors[2])
    else:
        for ax in axes: note(ax, 'HMM arrays are available in pgm_results.json')
    axes[0].set_title('Unit 2: explicitly synthetic ordered HMM sequence')
    axes[0].set_ylabel('Symbol / state'); axes[1].set_ylabel('Posterior probability')
    axes[2].set(xlabel='Synthetic sequence step', ylabel='Viterbi state')
    save(fig, '16_hmm_filter_smooth_viterbi', 'HMM filtering uses past evidence; smoothing uses the complete sequence. This educational example does not infer time from unrelated phantom filenames.')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, values, title, xlabel in zip(axes, [hmm['transition'], hmm['emission']],
                                       ['Assumed HMM transitions', 'Assumed HMM emissions'],
                                       ['Next hidden state', 'Observed symbol']):
        values = np.asarray(values)
        image = ax.imshow(values, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        for (z, x), value in np.ndenumerate(values): ax.text(x, z, f'{value:.2f}', ha='center', va='center')
        ax.set(title=title, xlabel=xlabel, ylabel='Current hidden state')
        ax.set_xticks(range(values.shape[1])); ax.set_yticks(range(values.shape[0]))
        fig.colorbar(image, ax=ax)
    save(fig, '16b_hmm_parameters', 'HMM parameters are explicit educational assumptions, not fitted temporal measurements from the RF archive.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    markov = demonstrations.get('markov', {})
    lines = ['Factor-preservation verification',
             f'Partition function Z = {markov["partition_function"]:.12f}',
             f'Maximum joint error = {markov["joint_max_error"]:.3g}',
             '', 'Graph separation in the moral graph',
             'X0 independent of X1 given {F,Z}: '+str(markov['independence_X0_X1_given_F_Z']),
             'Separated without conditioning: '+str(markov['independence_X0_X1_without_conditioning']),
             '', 'Factor tables remain the original BN CPDs.',
             'Moralization does not imply triangulation.']
    axes[0].text(0.02, 0.95, '\n'.join(lines), va='top', fontsize=10, linespacing=1.8, transform=axes[0].transAxes)
    axes[0].set_axis_off(); axes[0].set_title('Unit 2/3: probabilities and independencies')
    collider_positions = {'A': (-1, 1), 'B': (1, 1), 'C': (0, -0.5)}
    for left, right in [('A', 'C'), ('B', 'C')]:
        axes[1].add_patch(FancyArrowPatch(collider_positions[left], collider_positions[right], arrowstyle='-',
                                         mutation_scale=15, color='#718298', shrinkA=20, shrinkB=20))
    axes[1].plot([-1, 1], [1, 1], '--', color=colors[1], linewidth=2.5, label='New co-parent edge')
    for node, position in collider_positions.items():
        axes[1].add_patch(Circle(position, 0.22, facecolor='#e8f1fc', edgecolor=colors[0], zorder=3))
        axes[1].text(*position, node, ha='center', va='center', zorder=4)
    axes[1].set(xlim=(-1.6, 1.6), ylim=(-1, 1.7)); axes[1].set_aspect('equal'); axes[1].set_axis_off()
    axes[1].legend(loc='lower center'); axes[1].set_title('Moralizing collider A → C ← B')
    save(fig, '17_markov_independence_checks', 'The moral graph encodes separation statements. Multiplying the original family CPDs preserves the BN joint; moralization alone does not define new probabilities.')

    fig, ax = plt.subplots(figsize=(10, 4))
    fields = ['reconstruction loss', 'zero_residual', 'phase_residual', 'proxy_failure']
    for i, field in enumerate(fields):
        stat = summary['test_grouped_statistics'][field]
        mean = stat['mean']; interval = stat['ci95']
        ax.errorbar(i, mean, yerr=np.array([[mean-interval[0]], [interval[1]-mean]]) if interval else None,
                    fmt='o', capsize=5, color=colors[i])
    ax.set_xticks(range(len(fields)), fields, rotation=15)
    ax.set_ylabel('Equal-group mean'); ax.set_title('Grouped bootstrap intervals (95%, descriptive)')
    save(fig, '18_grouped_bootstrap', 'Groups are resampled, not individual pixels. Intervals reflect only the evaluated subset and chosen manifest grouping; mixed endpoints retain their native units.')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter([r['mc_spread_pixels'] for r in rows if r['split'] == 'test'], p[test], alpha=0.7)
    axes[0].set(xlabel='MC dropout mean spread (pixels)', ylabel='Proxy-failure probability', title='Stochastic sensitivity versus operational risk')
    if args.mc_samples <= 1: axes[0].text(0.05, 0.95, 'MC dropout disabled; use --mc-samples 8', transform=axes[0].transAxes, va='top')
    comparison = [summary['metrics'][role].get('brier', np.nan) for role in ROLES]
    axes[1].bar(ROLES, comparison, color=colors); axes[1].set(ylabel='Brier score', title='Proxy calibration by partition')
    save(fig, '19_uncertainty_and_split_metrics', 'Dropout spread is a sensitivity measure, not a calibrated probability of displacement error.')

    for index, example in enumerate(examples):
        maps = example['maps']; pre, post = example['pre'], example['post']; flow = example['flow']
        fig, axes = plt.subplots(2, 4, figsize=(16, 9))
        panels = [(bmode(pre), 'Pre RF: B-mode display', 'gray', -60, 0),
                  (bmode(post), 'Post RF: B-mode display', 'gray', -60, 0),
                  (bmode(maps['warped']), 'Warped post: B-mode display', 'gray', -60, 0),
                  (maps['residual'], '1 − local RF correlation', 'magma', 0, 2),
                  (flow[0], 'Lateral displacement (pixels)', 'RdBu_r', -8, 8),
                  (flow[1], 'Axial displacement (pixels)', 'RdBu_r', -8, 8),
                  (maps['strain']*100, 'Axial strain (%)', 'RdBu_r', -5, 5),
                  (maps['cycle'], 'Cycle consistency (pixels)', 'magma', 0, 3)]
        for ax, (values, title, cmap, low, high) in zip(axes.flat, panels):
            image = ax.imshow(values, cmap=cmap, vmin=low, vmax=high, aspect='auto')
            ax.set_title(title); ax.set_xlabel('Lateral sample'); ax.set_ylabel('Axial sample')
            fig.colorbar(image, ax=ax, shrink=0.7)
        fig.suptitle('Held-out RF pair: '+example['record']['id'])
        save(fig, f'20_example_{index:02d}_elastography', 'Fixed display limits facilitate comparison; raw numeric maps retain unclipped values. This is mechanics analysis without pathology grades.')
        # Conditional spatial MRF on residual-derived unaries, not lesion segmentation.
        probability = np.clip(maps['residual']/2, 1e-4, 1-1e-4)
        probability[~maps['valid']] = 0.99
        unary = (probability >= 0.5).astype(int)
        refined, energies = mrf_refine(probability, bmode(pre)/60, beta=0.6, iterations=8)
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        axes[0].imshow(probability, vmin=0, vmax=1, cmap='magma', aspect='auto'); axes[0].set_title('Residual-derived risk unary')
        axes[1].imshow(unary, vmin=0, vmax=1, cmap='Blues', aspect='auto'); axes[1].set_title('Independent binary labels')
        axes[2].imshow(refined, vmin=0, vmax=1, cmap='Blues', aspect='auto'); axes[2].set_title('Four-neighbor Potts ICM')
        axes[3].plot(energies, 'o-', color=colors[0]); axes[3].set(xlabel='ICM sweep', ylabel='Energy', title='Non-increasing energy')
        save(fig, f'21_example_{index:02d}_mrf', 'Unit 3: feature-conditioned spatial MRF. ICM finds a local minimum; labels indicate spatial quality patterns, not tumors.')
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        sensitivity = {}
        for ax, beta in zip(axes, [0, 0.2, 0.6, 1.2]):
            labels, energy = mrf_refine(probability, bmode(pre)/60, beta=beta, iterations=8)
            ax.imshow(labels, vmin=0, vmax=1, cmap='Blues', aspect='auto'); ax.set_title(f'Potts beta = {beta}')
            sensitivity[str(beta)] = {'energies': energy, 'positive_fraction': float(labels.mean())}
        save(fig, f'22_example_{index:02d}_mrf_sensitivity', 'Sensitivity to the assumed neighbor penalty; stronger smoothness can erase narrow quality boundaries.')
        write_json(out/f'example_{index:02d}_mrf.json', {'energies': energies, 'sensitivity': sensitivity})

    extended = additional_report_figures(args, rows, examples, summary, demonstrations,
                                         risk_model, coordinates, phenotypes, save)
    write_json(out/'extended_results.json', extended)
    robustness_report_figures(summary.get('robustness_study', {}), summary.get('analytic_benchmark', {}), save)
    write_json(out/'figure_manifest.json', gallery)
    syllabus_html = ''.join('<h3>'+html.escape(unit)+'</h3><ul>'+''.join('<li>'+html.escape(topic)+'</li>' for topic in topics)+'</ul>'
                            for unit, topics in SYLLABUS.items())
    cards = ''.join('<figure><a href="'+g['file']+'"><img loading="lazy" src="'+g['file']+'" alt="'+html.escape(g['caption'], quote=True)+'"></a>'
                    '<figcaption>'+html.escape(g['caption'])+' <a href="'+g['pdf']+'">PDF</a></figcaption></figure>' for g in gallery)
    metrics_html = '<table><tr><th>Partition</th><th>Pairs</th><th>Prevalence</th><th>Brier</th><th>ROC AUC</th></tr>'
    for role in ROLES:
        metric = summary['metrics'][role]
        metrics_html += '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
                                      (role, metric.get('n'), metric.get('prevalence'), metric.get('brier'), metric.get('roc_auc')))+'</tr>'
    metrics_html += '</table>'
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page += '<title>ElastoGrade MIS experiment report</title><style>body{font:16px/1.6 system-ui,sans-serif;color:#172438;background:white;max-width:1280px;margin:36px auto;padding:0 24px}h1,h2,h3{line-height:1.2}a{color:#1765ad}figure{margin:36px 0;border:1px solid #dde5ee;padding:16px;border-radius:12px}img{max-width:100%;height:auto}figcaption{color:#465971}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #dde5ee}.notice{background:#fff6e8;padding:18px;border-left:4px solid #e68a27}pre{white-space:pre-wrap;background:#f4f7fa;padding:16px}</style>'
    page += '<h1>Label-Free Ultrasound Elastography and Probabilistic Risk Analysis</h1><p>Single-file MIS course experiment report.</p>'
    page += '<p class="notice">'+html.escape(summary['endpoint_warning'])+' Simulation displacement labels are disabled. '
    page += 'Simulation RF pairing follows the recorded frame-axis assumption. Phantom independence is provisional until acquisition groups are supplied.</p>'
    quarantined = summary.get('data_quality', {}).get('quarantined_source_count', 0)
    if quarantined:
        page += '<p class="notice">Data-quality exclusions: '+str(quarantined)+' source file(s) were quarantined before splitting. '
        page += 'See <a href="data_rejections.json">source names, reasons and diagnostic counts</a>. Raw MAT files were preserved.</p>'
    page += '<h2>Measured outputs</h2>'+metrics_html+'<p>Protocol: '+html.escape(summary['crop_protocol'])+'. Evaluated '+str(len(rows))+' RF pairs.</p>'
    page += '<p><a href="results.csv">All pair results (CSV)</a> · <a href="summary.json">Summary JSON</a> · <a href="pgm_results.json">PGM results</a> · <a href="risk_model.json">Learned risk model</a> · <a href="extended_results.json">Extended plot data</a> · <a href="robustness_results.json">Perturbation study</a> · <a href="analytic_benchmark.json">Known-motion fixtures</a></p>'
    if 'checkpoint_compatibility_note' in summary:
        page += '<p class="notice">'+html.escape(summary['checkpoint_compatibility_note'])+'</p>'
    page += '<h2>Course syllabus</h2>'+syllabus_html+'<h2>White-background results gallery</h2>'
    page += '<p><label for="figure-filter">Find figures by topic: </label><input id="figure-filter" type="search" placeholder="e.g. strain, Bayesian, noise" style="padding:10px;min-width:260px;border:1px solid #becbdc;border-radius:6px"> <span id="figure-count"></span></p>'+cards
    page += '<script>const search=document.getElementById("figure-filter"),figures=[...document.querySelectorAll("figure")],count=document.getElementById("figure-count");function filterFigures(){let visible=0;for(const f of figures){const match=f.textContent.toLowerCase().includes(search.value.toLowerCase());f.hidden=!match;if(match)visible++;}count.textContent=visible+" / "+figures.length+" figures";}search.addEventListener("input",filterFigures);filterFigures();</script>'
    page += '<h2>Environment and provenance</h2><pre>'+html.escape(json.dumps(json_safe(summary['environment']), indent=2))+'</pre>'
    if 'clinical_extension' in summary:
        page += '<h2>External pathology-linked ordinal research extension</h2><p>Patient partition checks passed. The encoder was trained without grade labels; '
        page += 'the ordinal head uses supplied pathology labels. Clinical probability calibration and deployment validity are not established.</p><pre>'
        page += html.escape(json.dumps(json_safe(summary['clinical_extension']), indent=2))+'</pre>'
    page += '<h2>Interpretation</h2><p>RF displacement and axial strain support mechanical characterization. Without linked pathology, '
    page += 'exploratory clusters cannot establish benign/malignant status or histological tumor grades. '
    page += 'No absolute modulus is inferred from unknown stress. The synthetic HMM illustrates temporal reasoning independently of the real-data metrics.</p></html>'
    (out/'report.html').write_text(page, encoding='utf-8')



# ---------------------------------------------------------------------------
# Full-field tiled inference: displacement is blended before strain derivation.
# ---------------------------------------------------------------------------
def infer_full_field(args):
    if not args.input or not args.checkpoint:
        raise ValueError('infer requires --input and --checkpoint.')
    device = args.device_resolved
    model, checkpoint = load_trained_model(args.checkpoint, device)
    if not getattr(args, 'patch_explicit', False):
        args.patch = list(checkpoint.get('training_contract', {}).get('patch', args.patch))
    path = Path(args.input)
    if path.suffix.lower() == '.mat':
        from scipy.io import loadmat
        contents = loadmat(path, variable_names=['Data'])
        if 'Data' not in contents:
            raise ValueError('MAT inference expects phantom Data[2,z,x]. Use a prepared pair.npy for simulation.')
        pair = np.asarray(contents['Data'])
    else:
        pair = np.asarray(np.load(path, mmap_mode='r', allow_pickle=False))
    try:
        validate_rf_storage_values(pair)
    except RFDataQualityError as exc:
        raise RFDataQualityError(str(path)+': '+str(exc), exc.details) from exc
    if pair.ndim != 3 or pair.shape[0] != 2 or not np.isfinite(pair).all():
        raise ValueError('Expected a finite RF pair of shape [2,z,x].')
    height, width = pair.shape[-2:]
    ph, pw = args.patch
    if not getattr(args, 'overlap_explicit', False):
        args.tile_overlap = min(args.tile_overlap, min(ph, pw)//2)
    if height < ph or width < pw:
        raise ValueError('Patch exceeds input image size.')
    if not 0 <= args.tile_overlap < min(ph, pw):
        raise ValueError('--tile-overlap must be nonnegative and smaller than both patch dimensions.')
    display_pair = normalize_rf_pair(pair)
    def starts(size, patch):
        values = list(range(0, size-patch+1, patch-args.tile_overlap))
        if values[-1] != size-patch: values.append(size-patch)
        return values
    weights = np.maximum(np.outer(np.hanning(ph), np.hanning(pw)), 0.05).astype(np.float32)
    accumulated = np.zeros((2, height, width), np.float64)
    denominator = np.zeros((height, width), np.float64)
    tiles = [(z, x) for z in starts(height, ph) for x in starts(width, pw)]
    with torch.inference_mode():
        for index, (z, x) in enumerate(tiles):
            crop = torch.from_numpy(normalize_rf_pair(pair[:, z:z+ph, x:x+pw])).to(device)
            flow, _ = model(crop[0:1][None], crop[1:2][None])
            accumulated[:, z:z+ph, x:x+pw] += flow[0].cpu().numpy()*weights
            denominator[z:z+ph, x:x+pw] += weights
            if (index+1) % args.log_every == 0:
                print(f'Tile {index+1}/{len(tiles)}', flush=True)
    flow = (accumulated/denominator[None]).astype(np.float32)
    if not np.isfinite(flow).all():
        raise FloatingPointError('Tiled inference produced non-finite displacement; check checkpoint weights.')
    strain = axial_strain(flow)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'full_field_maps.npz', flow_pixels=flow, axial_strain=strain)
    figure_style()
    fig, axes = plt.subplots(1, 4, figsize=(14, 7))
    for ax, values, title, cmap, low, high in zip(axes,
            [bmode(display_pair[0]), flow[0], flow[1], strain*100],
            ['Pre B-mode display', 'Lateral flow (pixels)', 'Axial flow (pixels)', 'Axial strain (%)'],
            ['gray', 'RdBu_r', 'RdBu_r', 'RdBu_r'], [-60, -8, -8, -5], [0, 8, 8, 5]):
        image = ax.imshow(values, aspect='auto', cmap=cmap, vmin=low, vmax=high)
        ax.set_title(title); fig.colorbar(image, ax=ax, shrink=0.6)
    fig.savefig(out/'full_field_panel.png', dpi=180, facecolor='white', bbox_inches='tight')
    fig.savefig(out/'full_field_panel.pdf', facecolor='white', bbox_inches='tight'); plt.close(fig)
    write_json(out/'full_field_provenance.json', {'input': str(path.resolve()), 'checkpoint': str(Path(args.checkpoint).resolve()),
                                               'shape': pair.shape, 'patch': args.patch, 'overlap': args.tile_overlap,
                                               'normalization': 'Shared RF normalization per tile, matching training',
                                               'tiles': len(tiles), 'pathology_grade_available': False,
                                               'warning': 'Tiling can introduce motion discontinuities at tile boundaries; validate on held-out data.'})
    print('Full-field maps: '+str((out/'full_field_maps.npz').resolve()))


def make_demo_cache(cache, seed=42, groups=24):
    """Analytic translation fixtures for software verification, not acoustic FEM.

    Both demo domain labels are synthetic, and the manifest records that origin.
    Independent random speckle fields and a range of corruption levels exercise
    the risk interface without pretending to be a clinical dataset.
    """
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records = []
    roles = ['train']*14+['validation']*4+['calibration']*3+['test']*3
    for i in range(groups):
        height, width = 80, 64
        pre = gaussian_filter(rng.normal(size=(height, width)), (0.7, 1.0))
        z, x = np.mgrid[:height, :width]
        uz, ux = float(rng.uniform(-2.5, 2.5)), float(rng.uniform(-1.0, 1.0))
        # post(z+uz,x+ux)=pre(z,x), apart from interpolation and corruption.
        post = map_coordinates(pre, [z-uz, x-ux], order=1, mode='reflect')
        post += rng.normal(0, [0.01, 0.08, 0.35, 0.8][i % 4], pre.shape)
        if i % 6 == 0: post[:height//3] = rng.normal(0, 1, (height//3, width))
        path = cache/f'demo_{i:03d}.npy'
        np.save(path, np.stack((pre, post)).astype(np.float32), allow_pickle=False)
        records.append({'id': f'demo_{i:03d}', 'path': str(path.resolve()), 'domain': 'synthetic_fixture',
                        'group': f'fixture_{i:03d}', 'split': roles[i % len(roles)],
                        'source': 'analytic translation fixture', 'shape': [2, height, width],
                        'analytic_truth_flow_pixels': [ux, uz],
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {'version': VERSION, 'records': records,
                'notes': ['Synthetic software demonstration only; no clinical or acoustic validation.'],
                'data_origin': 'synthetic_fixture'}
    write_json(cache/'manifest.json', manifest)
    return manifest


def run_self_tests(args):
    """Independent numerical checks for the key scientific conventions."""
    import tempfile
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    def rejects(name, function, error=ValueError):
        try:
            function()
        except error:
            check(name, True)
        else:
            raise AssertionError(name+' was accepted')
    seed_everything(123)
    # Verify grid coordinates against an analytic ramp, not the implementation.
    z, x = torch.meshgrid(torch.arange(16), torch.arange(20), indexing='ij')
    ramp = (3*z + 2*x).float()[None, None]
    displacement = torch.zeros(1, 2, 16, 20)
    displacement[:, 0] = 1.25; displacement[:, 1] = -0.5
    warped, valid = warp_tensor(ramp, displacement)
    expected = ramp + 2*1.25 - 3*0.5
    check('Warp coordinate order and subpixel sign', torch.allclose(warped[valid], expected[valid], atol=2e-5))
    flow = np.zeros((2, 31, 17), np.float32)
    flow[1] = -0.02*np.arange(31)[:, None]
    check('Known compression strain derivative', np.allclose(axial_strain(flow), -0.02, atol=1e-6))
    rng = np.random.default_rng(9)
    data = rng.normal(size=(240, 6))
    labels = (data[:, 0]+0.5*data[:, 1] > 0).astype(int)
    risk = fit_risk_model(data, labels, seed=9)
    predicted = predict_risk(risk, data[:13])
    check('Learned BN returns finite legal probabilities', np.isfinite(predicted).all() and np.all((predicted >= 0) & (predicted <= 1)))
    check('Risk-model JSON round trip', np.allclose(predicted, predict_risk(json.loads(json.dumps(risk)), data[:13])))
    probability = rng.uniform(0.05, 0.95, (7, 8))
    mrf, energy = mrf_refine(probability, rng.normal(size=(7, 8)), beta=0.6, iterations=10)
    check('MRF ICM never increases energy', np.all(np.diff(energy) <= 1e-7))
    no_prior, _ = mrf_refine(probability, np.zeros_like(probability), beta=0, iterations=3)
    check('Zero-neighbor MRF agrees with unary MAP', np.array_equal(no_prior, probability > 0.5))
    reference = rng.normal(size=(64, 48))
    shifted = np.roll(reference, (2, -1), axis=(0, 1))
    estimated = phase_translation(reference, shifted)
    check('Phase baseline returns reference-to-moving flow', np.allclose(estimated[:, 20, 20], [-1, 2], atol=0.2))
    # The PGM section adds exhaustive BN/moralization/HMM checks in its demo.
    demo = pgm_demonstrations(risk, data[:8], seed=9)
    check('PGM demonstration serializes without nonfinite JSON', bool(json.dumps(json_safe(demo), allow_nan=False)))
    check('One-class evaluation suppresses undefined ROC AUC', classification_metrics(np.zeros(6, int), np.ones(6)*0.2)['roc_auc'] is None)
    # These regressions exercise invalid data and numerical invariants, rather
    # than merely copying the implementation into an expected-value function.
    options = parse_arguments(['demo', '--out=reports/equals_option_fixture', '--cache=artifacts/equals_option_fixture',
                               '--epochs=1', '--width=4', '--batch-size=1'])
    check('Equals-form CLI options retain explicit demo settings',
          options.out == Path('reports/equals_option_fixture') and options.epochs == 1 and options.width == 4 and options.batch_size == 1)
    import contextlib
    import io
    with contextlib.redirect_stderr(io.StringIO()):
        rejects('NaN quality thresholds are rejected', lambda: parse_arguments(['doctor', '--residual-threshold=nan']), SystemExit)
        rejects('Infinite learning rates are rejected', lambda: parse_arguments(['doctor', '--lr=inf']), SystemExit)
    rejects('Unsupported device syntax is rejected', lambda: resolve_device('invalid_device'))
    amplitude_fixture = rng.normal(size=(2, 32, 24))
    reference_normalized = normalize_rf_pair(amplitude_fixture)
    check('RF normalization preserves very weak and very large signal scales',
          np.allclose(reference_normalized, normalize_rf_pair(amplitude_fixture*1e-20), atol=1e-6)
          and np.allclose(reference_normalized, normalize_rf_pair(amplitude_fixture*1e20), atol=1e-6))
    rejects('Constant individual RF frames are rejected', lambda: normalize_rf_pair(np.stack((np.zeros((32, 24)), amplitude_fixture[0]))))
    check('NumPy boolean metadata serializes to strict JSON', json.dumps(json_safe({'valid': np.bool_(True)}), allow_nan=False) == '{"valid": true}')
    unit_mask = torch.ones(1, 1, 20, 24, dtype=torch.bool)
    check('NCC windows exclude padding and retain exact interior support', int(window_support(unit_mask).sum()) == (20-8)*(24-8))
    displacement_fixture = torch.zeros(1, 2, 20, 24, requires_grad=True)
    with torch.no_grad(): displacement_fixture[:, 0] = 2; displacement_fixture[:, 1] = -1
    overflow = coordinate_overflow(displacement_fixture)
    overflow.backward()
    check('Boundary penalty equals analytic coordinate overflow and has gradients',
          abs(float(overflow.detach())-(3/24+1/20)) < 1e-6 and float(displacement_fixture.grad.abs().sum()) > 0)
    values = torch.tensor([0., 10.])[:, None, None, None].expand(2, 1, 2, 2)
    masks = torch.ones_like(values, dtype=torch.bool); masks[0] = False; masks[0, 0, 0, 0] = True
    check('Masked reductions weight images equally despite different support', abs(float(masked_average(values, masks))-5) < 1e-6)
    weights = [accumulation_sample_weight(i, 2, 3, 2, 2) for i in range(2)]
    scalar = torch.tensor(0.7, requires_grad=True)
    observations = torch.tensor([1., 3., 5.])
    (scalar*observations).square().mean().backward(); expected_gradient = scalar.grad.clone()
    scalar.grad = None
    for subset, weight in zip((observations[:2], observations[2:]), weights):
        ((scalar*subset).square().mean()*weight).backward()
    check('Partial-batch gradient accumulation matches a full three-sample batch',
          torch.allclose(scalar.grad, expected_gradient, atol=1e-6) and np.allclose(weights, [2/3, 1/3]))
    calibration = fit_calibrator(np.linspace(.05, .95, 20), np.arange(20) >= 10)
    check('Two-class calibration fits finite positive log-odds slope', np.isfinite(calibration['coef']) and calibration['coef'] > 0)
    rejects('Invalid probabilities cannot be silently clipped for metrics', lambda: classification_metrics([0, 1], [0.2, 1.1]))
    check('Empty grouped bootstrap has an explicit unavailable result', grouped_bootstrap([], [])['mean'] is None)
    # Windows scanners may briefly lock a completed fixture during cleanup.
    # Cleanup is best effort; numerical/data assertions still propagate normally.
    with tempfile.TemporaryDirectory(dir=Path(args.out).resolve(), ignore_cleanup_errors=True) as temporary:
        manifest = make_demo_cache(temporary, seed=9)
        dataset = TrainingPairs(manifest['records'], (32, 24), training=True, seed=9)
        first = dataset[0]['pre']; same = dataset[0]['pre']
        check('Deterministic patch sampling within epoch', torch.equal(first, same))
        dataset.epoch = 1
        check('New epoch changes training crop', not torch.equal(first, dataset[0]['pre']))
        # Small MAT archives exercise adapters and grouping independently of the
        # real 58 GiB collection. A duplicate pair must never cross partitions.
        from scipy.io import savemat
        fixture_root = Path(temporary)/'raw'
        for domain in ('phantom', 'simulation', 'test'):
            (fixture_root/domain).mkdir(parents=True, exist_ok=True)
        phantom_arrays = [rng.normal(size=(2, 32, 24)).astype(np.float64) for _ in range(4)]
        for index, pair in enumerate(phantom_arrays + [phantom_arrays[0]]):
            savemat(fixture_root/'phantom'/f'Data_{index+1}.mat', {'Data': pair})
        simulation_arrays = []
        for index in range(4):
            array = rng.normal(size=(2, 3, 32, 24)).astype(np.float32)
            simulation_arrays.append(array)
            savemat(fixture_root/'simulation'/f'model{index+2}_all.mat', {'RF': array})
            savemat(fixture_root/'simulation'/f'model{index+2}_Dis_2048_fix.mat', {'axial': np.zeros((3, 32, 24))})
        savemat(fixture_root/'test'/'Data_1.mat', {'rf_pre': np.zeros((12, 12, 3))})
        fixture_cache = Path(temporary)/'cache'
        prepared = prepare_data(fixture_root, fixture_cache, seed=9, simulation_pairs_per_file=2)
        check('Both real-data adapters exclude displacement/test arrays', len(prepared['records']) == 12)
        check('Exact duplicate pairs consolidate before splitting', prepared['duplicates_consolidated'] == 1)
        group_roles = {}
        for record in prepared['records']:
            group_roles.setdefault(record['group'], set()).add(record['split'])
            if record['domain'] == 'simulation':
                selection = record['pair_selection']
                model_index = int(Path(record['source']).name.split('_')[0][5:])-2
                truth = simulation_arrays[model_index][selection['realization'], selection['frames']]
                check_name = 'Simulation frame-axis extraction matches source indexing'
                if not np.array_equal(np.load(record['path']), truth): raise AssertionError(check_name)
        check('Simulation frame-axis extraction matches source indexing', True)
        check('Acquisition and geometry groups never cross splits', all(len(v) == 1 for v in group_roles.values()))
        check('Each source domain has four evaluation roles', all(set(prepared['split_counts'][domain]) == set(ROLES) for domain in ('phantom', 'simulation')))
        resumed = prepare_data(fixture_root, fixture_cache, seed=9, simulation_pairs_per_file=2)
        check('Preparation resume preserves cache and split fingerprint', manifest_fingerprint(prepared['records']) == manifest_fingerprint(resumed['records']))
        check('Validated cache manifest accepts correct headers and signatures', len(load_manifest(fixture_cache)['records']) == 12)
        valid_manifest = read_json(fixture_cache/'manifest.json')
        malformed = json.loads(json.dumps(valid_manifest))
        malformed['records'][1]['id'] = malformed['records'][0]['id']
        write_json(fixture_cache/'manifest.json', malformed)
        rejects('Duplicate imaging IDs are rejected before training or labeling', lambda: load_manifest(fixture_cache))
        malformed = json.loads(json.dumps(valid_manifest))
        first = malformed['records'][0]
        opposite = next(record for record in malformed['records'] if record['split'] != first['split'])
        opposite['sha256'] = first['sha256']
        write_json(fixture_cache/'manifest.json', malformed)
        rejects('Duplicate content cannot cross data partitions', lambda: load_manifest(fixture_cache))
        write_json(fixture_cache/'manifest.json', valid_manifest)
        corrupt = _cache_pair(phantom_arrays[0], fixture_cache)
        Path(corrupt['path']).write_bytes(b'broken NPY header')
        rejects('Modified cached files are rejected before training', lambda: load_manifest(fixture_cache))
        repaired = _cache_pair(phantom_arrays[0], fixture_cache)
        check('Corrupt cache headers rebuild from intact RF source data', np.array_equal(np.load(repaired['path']), phantom_arrays[0].astype(np.float32)))
        # Rebuild preparation signatures after repairing a file, then deliberately
        # request an oversized crop to verify training failure status reporting.
        prepare_data(fixture_root, fixture_cache, seed=9, simulation_pairs_per_file=2)
        failing_args = parse_arguments(['train', '--cache', str(fixture_cache), '--out', str(Path(temporary)/'failed_training'),
                                        '--device', 'cpu', '--patch', '64', '48', '--width', '2', '--radius', '0', '--epochs', '1'])
        failing_args.device_resolved = torch.device('cpu')
        rejects('Invalid training crops fail without producing a false completed status', lambda: train_pipeline(failing_args))
        check('Failed training writes a failed status with an error reason', read_json(failing_args.out/'status.json')['status'] == 'failed')
        caps = training_contract(failing_args); failing_args.steps_per_epoch = 1
        check('Resume contract detects changes to pilot training caps', training_contract(failing_args) != caps)
        # Preserve ordinary cache encoding, but reject unsafe ORIGINAL precision
        # before a cast can create infinity or overwrite usable RF dynamic range.
        ordinary = np.ascontiguousarray(phantom_arrays[1], dtype=np.float32)
        expected_digest = hashlib.sha256()
        expected_digest.update(str(tuple(ordinary.shape)).encode('ascii'))
        expected_digest.update(memoryview(ordinary).cast('B'))
        ordinary_record = _cache_pair(phantom_arrays[1], fixture_cache)
        check('Ordinary RF cache hashes remain compatible with completed preparation', ordinary_record['sha256'] == expected_digest.hexdigest())
        bad_pair = phantom_arrays[0].copy(); bad_pair[0, 0, 0] = 1.0e307
        with np.errstate(over='raise', invalid='raise'):
            rejects('Finite sparse overflow is rejected before any unsafe float32 cast', lambda: _cache_pair(bad_pair, fixture_cache), RFDataQualityError)
            rejects('Globally unrepresentable RF is rejected without silent rescaling', lambda: _cache_pair(phantom_arrays[0]*1e300, fixture_cache), RFDataQualityError)
        infinite_pair = phantom_arrays[0].copy(); infinite_pair[0, 0, 0] = np.inf
        rejects('Original nonfinite RF receives an explicit data-quality rejection', lambda: _cache_pair(infinite_pair, fixture_cache), RFDataQualityError)
        corrupted_source = fixture_root/'phantom'/'Data_corrupt.mat'
        nan_source = fixture_root/'phantom'/'Data_nan.mat'
        savemat(corrupted_source, {'Data': bad_pair})
        nan_pair = phantom_arrays[0].copy(); nan_pair[1, 1, 1] = np.nan
        savemat(nan_source, {'Data': nan_pair})
        raw_digest = hashlib.sha256(corrupted_source.read_bytes()).hexdigest()
        stable_files = {record['path']: _source_signature(record['path']) for record in resumed['records']}
        quarantined_manifest = prepare_data(fixture_root, fixture_cache, seed=9, simulation_pairs_per_file=2)
        check('Corrupt finite and NaN sources are explicitly quarantined before splitting',
              quarantined_manifest['data_quality']['quarantined_source_count'] == 2 and len(quarantined_manifest['records']) == 12)
        diagnostics = read_json(fixture_cache/'data_rejections.json')
        check('Quarantine audit identifies source paths and precision failure diagnostics',
              {item['reason'] for item in diagnostics['sources']} == {'float32_out_of_range', 'nonfinite_source'})
        check('Quarantine preserves raw MAT bytes and all completed good cache entries',
              hashlib.sha256(corrupted_source.read_bytes()).hexdigest() == raw_digest
              and all(_source_signature(path) == signature for path, signature in stable_files.items()))
        rejects('Strict data-quality mode stops with the original source diagnostic',
                lambda: prepare_data(fixture_root, fixture_cache, seed=9, simulation_pairs_per_file=2, invalid_sources='error'), RFDataQualityError)
        savemat(corrupted_source, {'Data': rng.normal(size=(2, 32, 24)).astype(np.float32)})
        repaired_source_manifest = prepare_data(fixture_root, fixture_cache, seed=9, simulation_pairs_per_file=2)
        check('Replacing a repaired source invalidates its quarantine and includes valid RF',
              repaired_source_manifest['data_quality']['quarantined_source_count'] == 1
              and any(record['source'] == 'phantom/Data_corrupt.mat' for record in repaired_source_manifest['records']))
        compressed_source = fixture_root/'phantom'/'Data_zlib.mat'
        savemat(compressed_source, {'Data': rng.normal(size=(2, 128, 128))}, do_compression=True)
        compressed_bytes = bytearray(compressed_source.read_bytes())
        compressed_length = int.from_bytes(compressed_bytes[132:136], 'little')
        compressed_bytes[136+compressed_length-1] ^= 1  # Alter the zlib checksum, retaining the MAT header.
        compressed_source.write_bytes(compressed_bytes)
        (fixture_root/'phantom'/'Data_header.mat').write_bytes(b'broken MATLAB header')
        unreadable_manifest = prepare_data(fixture_root, fixture_cache, seed=9, simulation_pairs_per_file=2)
        check('Corrupt compressed MAT and unreadable headers are quarantined without stopping good sources',
              unreadable_manifest['data_quality']['quarantined_source_count'] == 3
              and len(unreadable_manifest['records']) == len(repaired_source_manifest['records']))
        check('Unreadable MAT files have explicit source-level exclusion reasons',
              sum(record['reason'] == 'unreadable_mat_source' for record in unreadable_manifest['data_quality']['sources']) == 2)
        # Check the optional grading contract with artificial labels only.
        # These rows verify software behavior and never enter the project data.
        clinical_rows = [{'id': f'clinical_fixture_{i}', 'split': ROLES[i//2]} for i in range(8)]
        clinical_path = Path(temporary)/'clinical.csv'
        with clinical_path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream); writer.writerow(['id', 'patient_id', 'grade'])
            writer.writerows([[r['id'], f'patient_{i}', i % 2] for i, r in enumerate(clinical_rows)])
        clinical = clinical_ordinal_extension(clinical_path, clinical_rows, rng.normal(size=(8, 5)), temporary)
        check('Optional ordinal head accepts valid patient partitions', clinical['patients'] == 8)
        try:
            # A shared patient in train and test must be rejected before fitting.
            with clinical_path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream); writer.writerow(['id', 'patient_id', 'grade'])
                writer.writerows([[f'clinical_fixture_{i}', 'shared' if i in (0, 6) else f'patient_{i}', i % 2] for i in range(8)])
            clinical_ordinal_extension(clinical_path, [{'id': f'clinical_fixture_{i}', 'split': ROLES[i//2]} for i in range(8)], rng.normal(size=(8, 5)), temporary)
        except ValueError as exc:
            check('Ordinal head rejects patient leakage', 'Patient crosses' in str(exc))
        else:
            raise AssertionError('Ordinal head accepted patient leakage')
    model = RFRegistrationNet(width=4, radius=1, dropout=0)
    pre, post = torch.randn(2, 1, 32, 24), torch.randn(2, 1, 32, 24)
    forward, _ = model(pre, post); backward, _ = model(post, pre)
    loss, _ = registration_loss(pre, post, forward, backward)
    loss.backward()
    check('Registration network learns through differentiable warping', all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
    impossible = torch.ones_like(forward)*100
    _, no_support_terms = registration_loss(pre, post, impossible, impossible)
    check('Empty image support cannot appear as perfect reconstruction', no_support_terms['photo'] >= 1.99)
    mixed = torch.cat((torch.zeros_like(forward[:1]), impossible[:1]))
    _, mixed_terms = registration_loss(pre, post, mixed, mixed)
    _, first_terms = registration_loss(pre[:1], post[:1], mixed[:1], mixed[:1])
    check('Empty-support penalty applies per image within mixed batches', abs(mixed_terms['photo']-(first_terms['photo']+2)/2) < 1e-6)
    model = model.to(args.device_resolved)
    if args.device_resolved.type == 'cuda':
        model.zero_grad(set_to_none=True)
        gpu_pre, gpu_post = pre.to(args.device_resolved), post.to(args.device_resolved)
        gpu_forward, _ = model(gpu_pre, gpu_post); gpu_backward, _ = model(gpu_post, gpu_pre)
        gpu_loss, _ = registration_loss(gpu_pre, gpu_post, gpu_forward, gpu_backward)
        gpu_loss.backward()
        torch.cuda.synchronize(args.device_resolved)
        check('CUDA registration forward/backward has finite gradients',
              all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
    benchmark = analytic_motion_benchmark(model, args)
    if benchmark['rows']:
        check('Known-motion benchmark zero-flow errors match analytic translation magnitude', all(abs(r['zero_epe_pixels']-r['true_magnitude_pixels']) < 1e-5 for r in benchmark['rows']))
    example = {'record': {'id': 'robustness_fixture', 'domain': 'synthetic_fixture', 'group': 'fixture'},
               'pre': pre[0, 0].numpy(), 'post': post[0, 0].numpy()}
    sensitivity = run_robustness_study(model, [example], risk, {'coef': None}, args)
    if sensitivity['rows']:
        check('Perturbation study keeps clean baselines at zero displacement change',
              all(row['flow_change_pixels'] == 0 for row in sensitivity['rows']
                  if row['strength'] == (1 if row['condition'] == 'post_gain' else 0)))
        check('Perturbation study restores original model training mode', model.training)
    test_report = {'passed': len(checks), 'checks': checks, 'environment': environment_info()}
    write_json(Path(args.out)/'self_test_results.json', test_report)
    print(json.dumps({'passed': len(checks), 'checks': checks}, indent=2))
    return test_report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument('command', choices=['doctor', 'audit', 'prepare', 'train', 'evaluate', 'run', 'demo', 'self-test', 'infer'])
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--cache', type=Path, default=Path('artifacts/mis_cache'))
    parser.add_argument('--out', type=Path, default=Path('reports/mis_run'))
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--domains', choices=['both', 'phantom', 'simulation'], default='both')
    parser.add_argument('--simulation-frame-axis', type=int, choices=[0, 1], default=1,
                        help='Working assumption for RF leading time axis; no displacement labels are used.')
    parser.add_argument('--simulation-pairs-per-file', type=int, default=20, help='0 selects all adjacent pairs.')
    parser.add_argument('--limit', type=int, default=0, help='Cap source RF files per domain for a pilot; 0 means all.')
    parser.add_argument('--group-map', type=Path, help='Optional CSV assigning source paths to independent acquisition groups.')
    parser.add_argument('--invalid-sources', choices=['quarantine', 'error'], default='quarantine',
                        help='Log and exclude corrupt RF sources, or stop with their diagnostic details.')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--patch', type=int, nargs=2, default=[256, 128], metavar=('AXIAL', 'LATERAL'))
    parser.add_argument('--width', type=int, default=12)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--accumulation', type=int, default=2)
    parser.add_argument('--patches-per-pair', type=int, default=2)
    parser.add_argument('--workers', type=int, default=0, help='Windows-safe default; try 2 after a successful pilot.')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping epochs; 0 disables.')
    parser.add_argument('--steps-per-epoch', type=int, default=0, help='0 processes the entire training loader.')
    parser.add_argument('--validation-batches', type=int, default=0)
    parser.add_argument('--smooth-weight', type=float, default=0.02)
    parser.add_argument('--cycle-weight', type=float, default=0.1)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--eval-limit', type=int, default=400, help='Maximum report RF pairs across all partitions; 0 all.')
    parser.add_argument('--examples', type=int, default=6)
    parser.add_argument('--mc-samples', type=int, default=1, help='Use 8 or more to plot MC-dropout sensitivity.')
    parser.add_argument('--robustness-samples', type=int, default=3, help='Held-out pairs for bounded noise/gain/dropout studies; 0 skips perturbations.')
    parser.add_argument('--skip-robustness', action='store_true', help='Skip perturbation and analytic-reference studies for a faster report.')
    parser.add_argument('--clinical-labels', type=Path, help='Optional external pathology CSV: id,patient_id,grade. Never inferred from RF.')
    parser.add_argument('--residual-threshold', type=float, default=0.45)
    parser.add_argument('--cycle-threshold', type=float, default=1.0)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--input', type=Path, help='Inference input: cached [2,z,x] .npy or phantom .mat')
    parser.add_argument('--tile-overlap', type=int, default=32)
    return parser


def parse_arguments(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    supplied = {token.split('=', 1)[0] for token in (sys.argv[1:] if argv is None else argv)
                if token.startswith('--')}
    args.patch_explicit = '--patch' in supplied
    args.overlap_explicit = '--tile-overlap' in supplied
    if args.command == 'demo':
        # Short defaults, with explicit user options retained.
        if '--cache' not in supplied: args.cache = Path('artifacts/mis_demo_cache')
        if '--out' not in supplied: args.out = Path('reports/mis_demo')
        if '--epochs' not in supplied: args.epochs = 2
        if '--patch' not in supplied: args.patch = [64, 48]
        if '--batch-size' not in supplied: args.batch_size = 2
        if '--width' not in supplied: args.width = 8
        if '--eval-limit' not in supplied: args.eval_limit = 0
    positive = ('threads', 'epochs', 'batch_size', 'width', 'accumulation', 'patches_per_pair', 'log_every', 'mc_samples')
    for key in positive:
        if getattr(args, key) <= 0: parser.error('--'+key.replace('_', '-')+' must be positive')
    for key in ('workers', 'limit', 'simulation_pairs_per_file', 'steps_per_epoch', 'validation_batches', 'eval_limit', 'examples', 'patience', 'radius', 'robustness_samples', 'tile_overlap', 'seed'):
        if getattr(args, key) < 0: parser.error('--'+key.replace('_', '-')+' cannot be negative')
    if min(args.patch) < 32: parser.error('Both patch dimensions must be at least 32 for local-window support.')
    if args.width < 2 or args.radius > 4: parser.error('Use width >= 2 and radius <= 4.')
    for key in ('lr', 'dropout', 'smooth_weight', 'cycle_weight', 'residual_threshold', 'cycle_threshold'):
        if not math.isfinite(getattr(args, key)): parser.error('--'+key.replace('_', '-')+' must be finite')
    if not 0 <= args.dropout < 1 or args.lr <= 0: parser.error('Require 0 <= dropout < 1 and lr > 0.')
    if args.smooth_weight < 0 or args.cycle_weight < 0: parser.error('Loss weights cannot be negative.')
    if not 0 <= args.residual_threshold <= 2 or args.cycle_threshold < 0:
        parser.error('Require residual threshold in [0,2] and a nonnegative cycle threshold.')
    if args.seed > 2**32-1: parser.error('--seed must be in 0..4294967295.')
    return args


def main(argv=None):
    args = parse_arguments(argv)
    args.device_resolved = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == 'doctor':
        print(json.dumps(environment_info(), indent=2))
        return
    if args.command == 'audit':
        audit = audit_data(args.data_root, args.out)
        concise = {'data_root': audit['data_root'], 'mat_files': audit['mat_files'],
                   'total_bytes': audit['total_bytes'],
                   'domains': {name: {'files': values['files'], 'bytes': values['bytes'],
                                      'distinct_schemas': len(values['schemas'])}
                               for name, values in audit['domains'].items()},
                   'details': str((args.out/'data_audit.json').resolve())}
        print(json.dumps(concise, indent=2))
        return
    if args.command in ('prepare', 'run'):
        domains = ('phantom', 'simulation') if args.domains == 'both' else (args.domains,)
        manifest = prepare_data(args.data_root, args.cache, seed=args.seed, group_map=args.group_map,
                                limit=args.limit, simulation_frame_axis=args.simulation_frame_axis,
                                simulation_pairs_per_file=args.simulation_pairs_per_file, domains=domains,
                                invalid_sources=args.invalid_sources)
        print('Prepared '+str(len(manifest_records(manifest)))+' RF pairs.', flush=True)
        if args.command == 'prepare': return
    if args.command == 'self-test':
        run_self_tests(args)
    elif args.command == 'demo':
        manifest = make_demo_cache(args.cache, args.seed)
        checkpoint = train_pipeline(args, manifest)
        evaluate_pipeline(args, checkpoint)
    elif args.command in ('train', 'run'):
        checkpoint = train_pipeline(args)
        if args.command == 'run': evaluate_pipeline(args, checkpoint)
    elif args.command == 'evaluate':
        evaluate_pipeline(args)
    elif args.command == 'infer':
        infer_full_field(args)


if __name__ == '__main__':
    # Required by Windows DataLoader multiprocessing; importing the file is safe.
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted. Resume from the last completed epoch checkpoint.', file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, FileNotFoundError, FloatingPointError) as exc:
        print('Error: '+str(exc), file=sys.stderr)
        raise SystemExit(2)
