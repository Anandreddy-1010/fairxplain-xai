from __future__ import annotations
import argparse, json, os, sys, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.fairxplain.config import MODELS_DIR, REPORTS_DIR, PROCESSED_DIR, FIGURES_DIR, RANDOM_STATE, AUDIT_COLUMNS
from src.fairxplain.data import load_dataset, get_feature_columns
from src.fairxplain.features import engineer_features, prepare_target
from src.fairxplain.metrics import evaluate, report_frame, confusion_frame
from src.fairxplain.pipeline import make_pipeline, CatBoostAdapter
from src.fairxplain.fairness import audit_groups, fairness_summary

warnings.filterwarnings('ignore', category=UserWarning)


def json_dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')


def schema_for_inputs(raw_X, raw_feature_cols, categorical_cols):
    schema = {}
    for col in raw_feature_cols:
        s = raw_X[col]
        if col in categorical_cols:
            mode = s.mode(dropna=True)
            values = sorted(s.dropna().unique().tolist(), key=lambda x: str(x))
            schema[col] = {'kind':'categorical', 'values':[int(v) if isinstance(v, (np.integer,int)) else str(v) for v in values], 'default': int(mode.iloc[0]) if len(mode) and isinstance(mode.iloc[0], (np.integer,int)) else (str(mode.iloc[0]) if len(mode) else '')}
        else:
            v = pd.to_numeric(s, errors='coerce')
            schema[col] = {'kind':'numeric', 'min':float(v.min()), 'max':float(v.max()), 'median':float(v.median()), 'mean':float(v.mean())}
    return schema


def candidates(task, n_classes):
    common = dict(n_jobs=2, random_state=RANDOM_STATE)
    is_binary = task in ('binary', 'resolved_binary')
    xgb_obj = 'binary:logistic' if is_binary else 'multi:softprob'
    lgb_obj = 'binary' if is_binary else 'multiclass'
    cat_loss = 'Logloss' if is_binary else 'MultiClass'
    xgb = XGBClassifier(n_estimators=240, max_depth=4, learning_rate=0.045,
        min_child_weight=2, subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
        objective=xgb_obj, eval_metric='logloss' if task == 'binary' else 'mlogloss',
        tree_method='hist', n_jobs=2, random_state=RANDOM_STATE)
    lgb = LGBMClassifier(n_estimators=240, num_leaves=18, max_depth=7, learning_rate=0.04,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0, objective=lgb_obj,
        verbosity=-1, n_jobs=2, random_state=RANDOM_STATE)
    if is_binary:
        rf_weight = 'balanced_subsample'
        et_weight = 'balanced'
    else:
        rf_weight = None
        et_weight = None
    rf = RandomForestClassifier(n_estimators=220, max_depth=None, min_samples_leaf=2,
        max_features='sqrt', class_weight=rf_weight, n_jobs=2, random_state=RANDOM_STATE)
    et = ExtraTreesClassifier(n_estimators=220, max_depth=None, min_samples_leaf=2,
        max_features='sqrt', class_weight=et_weight, n_jobs=2, random_state=RANDOM_STATE)
    cat = CatBoostAdapter(iterations=180, depth=6, learning_rate=0.05,
        loss_function=cat_loss, random_seed=RANDOM_STATE, thread_count=2)
    return {'XGBoost':xgb, 'LightGBM':lgb, 'CatBoost':cat, 'Random Forest':rf, 'Extra Trees':et}


def train_task(df, task):
    print(f'\n=== {task.upper()} TASK ===', flush=True)
    if task == 'resolved_binary':
        # Match the published resolved-outcome task: Enrolled is excluded, not relabelled.
        df = df[df['Target'].isin(['Dropout', 'Graduate'])].copy().reset_index(drop=True)
        print('Enrolled records excluded for this separate benchmark:', int((load_dataset()['Target'] == 'Enrolled').sum()), flush=True)
    y_text = prepare_target(df['Target'], task)
    encoder = LabelEncoder(); y = encoder.fit_transform(y_text)
    labels = list(range(len(encoder.classes_)))
    X_raw = df.drop(columns=['Target']).copy()
    X = engineer_features(X_raw)
    _, categorical_cols = get_feature_columns(pd.concat([X, y_text.rename('Target')], axis=1))
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1875, stratify=y_trainval, random_state=RANDOM_STATE)
    print(f'Rows={len(df)} | train={len(X_train)} | val={len(X_val)} | untouched test={len(X_test)} | features={X.shape[1]}', flush=True)
    rows=[]; fitted={}
    for name, est in candidates(task, len(encoder.classes_)).items():
        print('Fitting', name, '...', flush=True)
        pipe = make_pipeline(est, numeric_cols, categorical_cols, use_smote=False)
        try:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_val).astype(int)
            m = evaluate(y_val, pred, labels)
            rows.append({'model':name, **m})
            fitted[name]=pipe
            print(f"  val accuracy={m['accuracy']:.4f}; macro-F1={m['macro_f1']:.4f}", flush=True)
        except Exception as e:
            print(f'  SKIPPED {name}: {type(e).__name__}: {e}', flush=True)
    if not fitted: raise RuntimeError('No model trained successfully')
    comp = pd.DataFrame(rows).sort_values(['accuracy','macro_f1'], ascending=False).reset_index(drop=True)
    best_name = str(comp.iloc[0]['model'])
    best = fitted[best_name]
    print('Selected by validation:', best_name, flush=True)
    best.fit(X_trainval, y_trainval)
    test_pred_enc = best.predict(X_test).astype(int)
    test_pred = encoder.inverse_transform(test_pred_enc)
    y_test_text = encoder.inverse_transform(y_test)
    tm = evaluate(y_test, test_pred_enc, labels)
    tm.update({'task':task,'selected_model':best_name,'classes':list(encoder.classes_),
        'test_evaluated_once':True,'split_seed':RANDOM_STATE,'train_size':len(X_train),
        'validation_size':len(X_val),'final_fit_size':len(X_trainval),'test_size':len(X_test),
        'feature_count_original':X_raw.shape[1],'feature_count_engineered':X.shape[1],
        'prediction_scope':'End-of-course outcome classification; first- and second-semester performance features included.',
        'task_definition':('Dropout vs Graduate; Enrolled class excluded' if task == 'resolved_binary' else ('Dropout vs Enrolled+Graduate' if task == 'binary' else 'Dropout vs Enrolled vs Graduate'))})
    md=MODELS_DIR/task; rd=REPORTS_DIR/task; pdst=PROCESSED_DIR/task
    for p in (md,rd,pdst,FIGURES_DIR/task): p.mkdir(parents=True,exist_ok=True)
    comp.to_csv(rd/'validation_model_comparison.csv',index=False)
    json_dump(tm,rd/'final_test_metrics.json')
    report_frame(y_test_text,test_pred,list(encoder.classes_)).to_csv(rd/'classification_report.csv',index=False)
    confusion_frame(y_test_text,test_pred,list(encoder.classes_)).to_csv(rd/'confusion_matrix.csv')
    joblib.dump(best,md/'pipeline.joblib'); joblib.dump(encoder,md/'label_encoder.joblib')
    json_dump({'task':task,'classes':list(encoder.classes_),'selected_model':best_name,
        'raw_feature_columns':list(X_raw.columns),'engineered_feature_columns':list(X.columns),
        'categorical_columns':categorical_cols,'numeric_columns':numeric_cols,
        'input_schema':schema_for_inputs(X_raw,list(X_raw.columns),categorical_cols),
        'audit_columns':[c for c in AUDIT_COLUMNS if c in X_raw.columns],
        'prediction_timing_note':'UCI dataset includes first- and second-semester academic performance; this is end-of-course outcome classification, not admission-time early warning.'},md/'model_metadata.json')
    val_export = X_val.copy(); val_export['__target'] = encoder.inverse_transform(y_val)
    val_export.to_csv(pdst/'validation_features.csv',index=False)
    ref=X_raw.loc[X_trainval.index].copy(); ref['Target']=encoder.inverse_transform(y_trainval); ref.to_csv(pdst/'train_reference.csv',index=False)
    audit=audit_groups(X_raw.loc[X_test.index], y_test_text, test_pred, list(encoder.classes_), task)
    audit.to_csv(rd/'fairness_group_metrics.csv',index=False); json_dump(fairness_summary(audit),rd/'fairness_summary.json')
    holdout=X_test.copy(); holdout['actual_target']=y_test_text; holdout['predicted_target']=test_pred; holdout.to_csv(pdst/'test_predictions.csv',index=False)
    print('\nFINAL TEST METRICS (one holdout evaluation)')
    for k,v in tm.items(): print(f'{k}: {v}')
    return {'task':task,'selected_model':best_name,'metrics':tm,'validation_comparison':comp.to_dict('records')}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--task',choices=['multiclass','binary','resolved_binary','both'],default='both')
    args=ap.parse_args()
    df=load_dataset()
    print(f'Loaded official CSV: {len(df)} rows, {df.shape[1]-1} predictors',flush=True)
    tasks=['multiclass','binary','resolved_binary'] if args.task=='both' else [args.task]
    summaries=[train_task(df,t) for t in tasks]
    (REPORTS_DIR/'compact_training_summary.json').write_text(json.dumps(summaries,indent=2,default=str),encoding='utf-8')

if __name__=='__main__': main()
