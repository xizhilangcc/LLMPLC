#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(
        description='Route Evo2/ESM2 features to shipped classifiers and report lifestyle confidence.'
    )
    p.add_argument('--work-dir', type=Path, default=Path('work'))
    p.add_argument('--evo-h5', type=Path)
    p.add_argument('--esm-h5', type=Path)
    p.add_argument('--contig-map', type=Path)
    p.add_argument('--model', type=Path, default=ROOT / 'models/merged_best_model.ubj')
    p.add_argument('--evo-model', type=Path, default=ROOT / 'models/evo2_best_model.ubj')
    p.add_argument('--threshold', type=float, default=0.5, help='Binary virulent decision threshold.')
    p.add_argument('--confidence-threshold', type=float, default=0.85,
                   help='Strictly greater confidence required for a retained lifestyle; otherwise uncertain.')
    p.add_argument('--out-csv', type=Path)
    return p.parse_args()


def load_h5(path, width):
    with h5py.File(path, 'r') as f:
        ids = f['ids'].asstr()[:].tolist()
        features = f['features'][:].astype(np.float32)
    if features.shape != (len(ids), width):
        raise ValueError(f'{path}: expected ({len(ids)}, {width}), got {features.shape}')
    if len(set(ids)) != len(ids) or not np.isfinite(features).all():
        raise ValueError(f'{path}: duplicate IDs or non-finite features')
    return ids, features


def predict(path, X):
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    model.set_params(device='cpu', n_jobs=4)
    if model.n_features_in_ != X.shape[1]:
        raise ValueError(f'{path}: requires {model.n_features_in_} features, got {X.shape[1]}')
    return model.predict_proba(X)[:, 1].astype(float)


def annotate(df, threshold, confidence_threshold):
    score = df['score'].astype(float)
    valid = score.notna()
    pred = pd.Series(pd.NA, index=df.index, dtype='Int64')
    pred.loc[valid] = (score.loc[valid] >= threshold).astype(int)
    df['pred'] = pred
    df['virulent_probability'] = score
    df['temperate_probability'] = 1 - score
    df['binary_lifestyle'] = np.where(valid, np.where(score >= threshold, 'virulent', 'temperate'), 'unavailable')
    df['confidence'] = np.where(valid, np.where(score >= threshold, score, 1 - score), np.nan)
    df['high_confidence'] = valid & (df['confidence'] > confidence_threshold)
    df['lifestyle'] = np.where(df['high_confidence'], df['binary_lifestyle'], 'uncertain')
    return df


def main():
    args = parse_args()
    if not 0 < args.threshold < 1 or not 0.5 <= args.confidence_threshold < 1:
        raise ValueError('Require 0 < threshold < 1 and 0.5 <= confidence-threshold < 1')
    evo_path = args.evo_h5 or args.work_dir / 'evo2_embedding.h5'
    esm_path = args.esm_h5 or args.work_dir / 'esm33_embedding.h5'
    map_path = args.contig_map or args.work_dir / 'All_contig_sentence.csv'
    output = args.out_csv or args.work_dir / 'predictions.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    ids, evo = load_h5(evo_path, 4096)
    esm_ids, esm = load_h5(esm_path, 1280)
    esm_index = {ident: i for i, ident in enumerate(esm_ids)}
    if set(esm_ids) - set(ids):
        raise ValueError('ESM2 contains IDs without matching Evo2 features; check input files')
    with map_path.open(newline='', encoding='utf-8') as f:
        mapping = {}
        for row in csv.reader(f):
            if not row:
                continue
            if row[0] in mapping:
                raise ValueError(f'Duplicate contig ID in map: {row[0]}')
            mapping[row[0]] = [x for x in row[1:] if x]
    expected = [frag for fragments in mapping.values() for frag in fragments]
    if len(expected) != len(set(expected)):
        raise ValueError('Duplicate fragment IDs in contig map')
    if set(ids) - set(expected):
        raise ValueError('Embedding IDs absent from contig map; check input files')
    scores = np.full(len(ids), np.nan)
    merged_rows = [i for i, ident in enumerate(ids) if ident in esm_index]
    evo_rows = [i for i, ident in enumerate(ids) if ident not in esm_index]
    if merged_rows:
        X = np.stack([np.concatenate([esm[esm_index[ids[i]]], evo[i]]) for i in merged_rows])
        scores[merged_rows] = predict(args.model, X)
    if evo_rows:
        scores[evo_rows] = predict(args.evo_model, evo[evo_rows])
    routes = ['merged' if ident in esm_index else 'evo2_only' for ident in ids]
    fragments = annotate(pd.DataFrame({'fragment_id': ids, 'score': scores, 'model_route': routes}),
                         args.threshold, args.confidence_threshold)
    frag_out = output.with_name(output.stem + '_fragment_predictions.csv')
    fragments.to_csv(frag_out, index=False)
    score_map, route_map = dict(zip(ids, scores)), dict(zip(ids, routes))
    rows = []
    for ident, fragment_ids in mapping.items():
        used = [frag for frag in fragment_ids if frag in score_map]
        rows.append(dict(id=ident, score=float(np.mean([score_map[f] for f in used])) if used else np.nan,
                         used_fragments=len(used), total_fragments=len(fragment_ids),
                         merged_fragments=sum(route_map[f] == 'merged' for f in used),
                         evo2_only_fragments=sum(route_map[f] == 'evo2_only' for f in used),
                         status='no_full_fragment' if not fragment_ids else
                         ('no_embedding' if not used else ('partial' if len(used) < len(fragment_ids) else 'ok'))))
    columns = ['id', 'score', 'used_fragments', 'total_fragments', 'merged_fragments', 'evo2_only_fragments', 'status']
    contigs = annotate(pd.DataFrame(rows, columns=columns), args.threshold, args.confidence_threshold)
    contigs.to_csv(output, index=False)
    high_out = output.with_name(output.stem + '_high_confidence.csv')
    uncertain_out = output.with_name(output.stem + '_uncertain.csv')
    contigs.loc[contigs['high_confidence']].to_csv(high_out, index=False)
    contigs.loc[~contigs['high_confidence']].to_csv(uncertain_out, index=False)
    meta = dict(binary_threshold=args.threshold, confidence_threshold=args.confidence_threshold,
                confidence_comparison='strictly_greater', model=str(args.model.resolve()),
                evo_model=str(args.evo_model.resolve()), contigs=len(contigs),
                fragments=len(ids), high_confidence=int(contigs['high_confidence'].sum()),
                uncertain=int((~contigs['high_confidence']).sum()),
                merged_fragments=len(merged_rows), evo2_only_fragments=len(evo_rows))
    output.with_suffix('.run.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    print(f'[OK] {output}: {meta}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
