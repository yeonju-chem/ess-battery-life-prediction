"""Independent integrity checks, including future-cycle perturbation."""
from pathlib import Path
import sys,json,copy,hashlib,re,tempfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import day1_analysis as da
OUT=ROOT/'results/day1';t=pd.read_csv(OUT/'tables/cell_features.csv');split=pd.read_csv(OUT/'tables/planned_split.csv')
checks=[]
def passed(name,condition):
    assert condition,name
    checks.append({'check':name,'passed':True})
passed('unique_cell_ids',t.cell_id.is_unique)
passed('raw_and_screened_counts',len(t)==139 and int(t.analysis_eligible.sum())==115)
passed('missing_targets_not_imputed',int(t.cycle_life.isna().sum())==10)
passed('no_missing_main_feature',np.isfinite(t.loc[t.analysis_eligible,'log_dq_var']).all())
passed('no_policy_overlap_holdout',not set(split[split.planned_split.eq('train')].policy)&set(split[split.planned_split.eq('validation')].policy))
tr=split[split.planned_split.eq('train')]
passed('cv_policy_integrity',int(tr.groupby('policy').cv_fold.nunique().max())==1)
passed('cv4_and_29train_7valid',set(tr.cv_fold)=={0,1,2,3} and len(tr)==29 and len(split)-len(tr)==7)
passed('future_fields_not_features',not set(da.FEATURES)&{'cycle_life','n_cycles','max_cycle','last_qd','knee_cycle','slope_after','cell_id','batch'})
cells,_=da.load_batches(ROOT/'data')
chosen=cells  # Test the early-feature invariant for every provided cell.
base,_=da.feature_table(chosen);changed=copy.deepcopy(chosen)
for c in changed:
    s=c['summary'];s.loc[s.cycle>100,['QDischarge','IR','Tavg','Tmax','chargetime']]=999
after,_=da.feature_table(changed)
passed('perturb_future_cycles_no_feature_change',np.allclose(base[da.FEATURES].to_numpy(float),after[da.FEATURES].to_numpy(float),equal_nan=True))
for b in da.FILES:
    curves=[c for c in cells if c['batch']==b]
    passed(b+'_voltage_grids_match',all(np.allclose(c['voltage'],curves[0]['voltage']) for c in curves))
    for c in curves:
        # source index is found from actual cycle number, verified by loader assertion.
        passed_name=None
        assert c['summary'].cycle.is_monotonic_increasing
passed('cycle100_features_have_1000_points',all(len(c['qcurves'][100])==1000 for c in cells))
for b,g in t.groupby('batch'):
    for _,row in g.iterrows():
        c=next(c for c in cells if c['cell_id']==row.cell_id)
        dq=c['qcurves'][100]-c['qcurves'][10]
        assert np.isclose(np.log10(np.var(dq,ddof=1)),row.log_dq_var)
passed('independent_delta_formula_all_cells',True)
png=list((OUT/'figures').glob('*.png'));passed('all_17_figures_present',len(png)==17 and all(p.stat().st_size>1000 for p in png))
for table in (OUT/'tables').glob('*.csv'):
    pd.read_csv(table)
passed('all_csv_readable',True)
relative=pd.read_csv(OUT/'tables/delta_relative_groups.csv')
passed('relative_groups_cover_each_batch',relative.groupby('batch').size().eq(2).all() and len(relative)==6)
passed('relative_delta_direction_verified',all(g.iloc[0].median_log_dq_var>g.iloc[1].median_log_dq_var for _,g in relative.groupby('batch')))
pairs=pd.read_csv(OUT/'tables/feature_pair_correlations.csv')
passed('collinearity_all_batches_all_feature_pairs',pairs.groupby('batch').size().eq(len(da.FEATURES)*(len(da.FEATURES)-1)//2).all())
current=pd.read_csv(OUT/'tables/current_pattern_representatives.csv')
passed('representatives_match_figures_and_tables',current.cell_id.tolist()==da.representatives(t))
passed('representatives_invariant_to_input_order',da.representatives(t.sample(frac=1,random_state=42))==da.representatives(t))
for c in cells:
    a=c['current10_trace'];assert len(a)>0 and np.isfinite(a).all() and (np.diff(a[:,0])>=0).all()
passed('current_trace_time_alignment_all_cells',True)
import day1_review as review
with tempfile.TemporaryDirectory() as d:
    (Path(d)/'tables').mkdir()
    review.slope_sensitivity(chosen,base,d)
    a=pd.read_csv(Path(d)/'tables/early_slope_sensitivity_cells.csv')
    review.slope_sensitivity(changed,after,d)
    b=pd.read_csv(Path(d)/'tables/early_slope_sensitivity_cells.csv')
    passed('smoothed_slope_has_no_future_cycle_dependency',np.allclose(a.smoothed_qd_slope,b.smoothed_qd_slope,equal_nan=True))
import nbformat
for name in ['01_data_check.ipynb','02_day1_eda.ipynb']:
    nb=nbformat.read(ROOT/'notebooks'/name,as_version=4)
    passed(name+'_all_code_executed_without_error',all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in nb.cells if c.cell_type=='code'))
    passed(name+'_declarative_narrative',not re.search(r'[가-힣]+(?:습니다|합니다|입니다|됩니다|까요|세요)', '\n'.join(c.source for c in nb.cells)))
passed('scratch_unchanged_during_final_review',hashlib.sha256((ROOT/'notebooks/30-ESSHealth-scratch.ipynb').read_bytes()).hexdigest()=='785194b8095312e87e6f2b40fd7d22e81a15edda10b736743ff8c4da3ef6db7d')
(OUT/'verification.json').write_text(json.dumps({'checks':checks,'model_fit_performed':False},ensure_ascii=False,indent=2))
print('PASS:',len(checks),'integrity checks')
