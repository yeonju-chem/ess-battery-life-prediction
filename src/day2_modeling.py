"""DAY 2 regression: frozen group splits, fold-local preprocessing, external evaluation."""
from pathlib import Path
import json, hashlib, platform, importlib.metadata as metadata
from datetime import datetime, timezone
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
import day1_analysis as da

SEED=42
TARGET_MAPE=9.1
SETS={'F1':['log_dq_var'], 'F2':['log_dq_var','qd_slope','c_equiv'],
      'F3':['log_dq_var','qd_slope','c_equiv','tavg_median','ir_median']}

def stamp():return datetime.now(timezone.utc).isoformat()
def pow10(x):return np.power(10.,x)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(df,out,name):
    df.to_csv(Path(out)/'tables'/f'{name}.csv',index=False,encoding='utf-8-sig')
    return df
def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    assert np.isfinite(p).all() and (y>0).all()
    return {'MAPE_pct':100*mean_absolute_percentage_error(y,p),'MAE_cycles':mean_absolute_error(y,p),
            'RMSE_cycles':np.sqrt(mean_squared_error(y,p)),'R2':r2_score(y,p) if len(y)>1 else np.nan}

def candidates():
    cs=[{'id':'C000','family':'Dummy','set':'F1','target':'raw','params':{}}]
    for fs in SETS:
        for target in ['raw','log10']:
            entries=[('Ridge',{'alpha':a}) for a in [.1,1.,10.,100.]]
            entries += [('ElasticNet',{'alpha':a,'l1_ratio':r}) for a in [.001,.01,.1] for r in [.2,.8]]
            entries += [('RandomForest',{'max_depth':d,'min_samples_leaf':l}) for d in [2,3] for l in [3,5]]
            for family,params in entries:cs.append({'id':f'C{len(cs):03d}','family':family,'set':fs,'target':target,'params':params})
    return cs

def estimator(c):
    if c['family']=='Dummy':model=DummyRegressor(strategy='median')
    elif c['family']=='Ridge':model=Ridge(**c['params'])
    elif c['family']=='ElasticNet':model=ElasticNet(**c['params'],max_iter=20000,tol=1e-6,random_state=SEED)
    else:model=RandomForestRegressor(**c['params'],n_estimators=300,random_state=SEED,n_jobs=1)
    steps=[('impute',SimpleImputer(strategy='median',keep_empty_features=True))]
    if c['family'] in ['Ridge','ElasticNet']:steps.append(('scale',StandardScaler()))
    steps.append(('model',model));pipe=Pipeline(steps)
    return TransformedTargetRegressor(regressor=pipe,func=np.log10,inverse_func=pow10) if c['target']=='log10' else pipe

def prepare(root,out):
    root=Path(root);out=Path(out)
    for sub in ['tables','figures','models']: (out/sub).mkdir(parents=True,exist_ok=True)
    protected=[*sorted((root/'notebooks').glob('0[12]*.ipynb')),root/'notebooks/30-ESSHealth-scratch.ipynb',
               root/'src/day1_analysis.py',root/'src/day1_review.py',root/'results/DS-MINI-Design-SKALA_3반-U095_이연주.pdf']
    write_json(out/'day1_preservation.json',{str(p.relative_to(root)):digest(p) for p in protected})
    cells,manifest=da.load_batches(root/'data');t,_=da.feature_table(cells)
    old=pd.read_csv(root/'results/day1/tables/cell_features.csv').set_index('cell_id').loc[t.cell_id]
    assert np.allclose(t[da.FEATURES].to_numpy(float),old[da.FEATURES].to_numpy(float),equal_nan=True)
    assert t.analysis_eligible.tolist()==old.analysis_eligible.tolist()
    split=pd.read_csv(root/'results/day1/tables/planned_split.csv')
    b1=t[t.analysis_eligible&t.batch.eq('Batch 1')].merge(split[['cell_id','planned_split','cv_fold']],on='cell_id',validate='one_to_one')
    train=b1[b1.planned_split.eq('train')].reset_index(drop=True)
    valid=b1[b1.planned_split.eq('validation')].reset_index(drop=True)
    assert (len(train),len(valid))==(29,7) and not set(train.policy)&set(valid.policy)
    assert set(train.cv_fold)==set(range(4))
    for fold in range(4):assert not set(train[train.cv_fold.eq(fold)].policy)&set(train[train.cv_fold.ne(fold)].policy)
    csv(t,out,'cell_features');csv(split,out,'split_membership');csv(manifest,out,'source_manifest')
    counts=t.groupby('batch').agg(raw=('cell_id','size'),labels=('label_valid','sum'),screened=('analysis_eligible','sum')).reset_index()
    csv(counts,out,'cohort_counts')
    alignment=[]
    for c in cells:
        v=c['voltage'];dq=c['qcurves'][100]-c['qcurves'][10]
        assert np.allclose(v,cells[0]['voltage'])
        assert np.isclose(np.var(dq,ddof=1),np.var(dq+.1,ddof=1))
        alignment.append({'cell_id':c['cell_id'],'batch':c['batch'],'voltage_points':len(v),'voltage_min':v.min(),'voltage_max':v.max(),
                          'q10_mean':np.mean(c['qcurves'][10]),'q100_mean':np.mean(c['qcurves'][100]),
                          'dq_mean':dq.mean(),'offset_invariant_log_var':np.log10(np.var(dq+.1,ddof=1))})
    csv(pd.DataFrame(alignment),out,'voltage_alignment')
    write_json(out/'frozen_plan.json',{'created_utc':stamp(),'random_state':SEED,'feature_sets':SETS,'candidates':candidates(),
      'selection':'minimum mean 4-fold group CV MAPE on Batch 1 train 29; ties by fewer features then ID',
      'holdout':'7 Batch 1 cells; policies disjoint; evaluate once after selection',
      'final_fit':'refit selected specification on Batch 1 screened 36 for Batch 2 and Batch 3',
      'target_mape_pct':TARGET_MAPE,'split_sha256':digest(root/'results/day1/tables/planned_split.csv'),
      'external_use':'selected model and predeclared median baseline only; no retuning after external results',
      'gap_formulas':{'Train-Valid':'Valid - CV','Valid-Test':'Batch 2 - Valid','Target-Test':'Test - 9.1','Batch2-Batch3':'Batch 3 - Batch 2'},
      'known_limitation':'All batches were inspected during DAY 1 EDA; neither holdout nor external evaluation is fully blind.'})
    return cells,t,train,valid

def select_model(train,out):
    """Only the 29-cell development frame is accepted; no external/holdout arguments."""
    assert train.batch.eq('Batch 1').all() and train.planned_split.eq('train').all()
    scores=[];predictions=[];audits=[]
    for c in candidates():
        feats=SETS[c['set']]
        for fold in range(4):
            a=train[train.cv_fold.ne(fold)];b=train[train.cv_fold.eq(fold)]
            est=estimator(c)
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                est.fit(a[feats],a.cycle_life)
            pred=est.predict(b[feats]);m=metrics(b.cycle_life,pred)
            scores.append({'candidate_id':c['id'],'family':c['family'],'feature_set':c['set'],'target':c['target'],
                           'params':json.dumps(c['params'],sort_keys=True),'fold':fold,'n':len(b),**m})
            for cell,y,p in zip(b.cell_id,b.cycle_life,pred):predictions.append({'candidate_id':c['id'],'fold':fold,'cell_id':cell,'actual':y,'predicted':p})
            pipe=est.regressor_ if c['target']=='log10' else est
            assert np.allclose(pipe.named_steps['impute'].statistics_,np.nanmedian(a[feats],axis=0),equal_nan=True)
            if 'scale' in pipe.named_steps:
                assert int(pipe.named_steps['scale'].n_samples_seen_)==len(a)
                assert np.allclose(pipe.named_steps['scale'].mean_,pipe.named_steps['impute'].transform(a[feats]).mean(axis=0))
            audits.append({'candidate_id':c['id'],'fold':fold,'fit_ids':';'.join(a.cell_id),'score_ids':';'.join(b.cell_id),
                           'policy_overlap':len(set(a.policy)&set(b.policy)),'imputer_fit_rows':len(a)})
    folds=csv(pd.DataFrame(scores),out,'cv_fold_scores');csv(pd.DataFrame(predictions),out,'cv_oof_predictions')
    csv(pd.DataFrame(audits),out,'fold_preprocessing_audit')
    summary=folds.groupby(['candidate_id','family','feature_set','target','params'],as_index=False).agg(
       cv_mape=('MAPE_pct','mean'),cv_std=('MAPE_pct','std'),cv_mae=('MAE_cycles','mean'),cv_rmse=('RMSE_cycles','mean'))
    summary['n_features']=summary.feature_set.map(lambda k:len(SETS[k]))
    summary=summary.sort_values(['cv_mape','n_features','candidate_id']).reset_index(drop=True)
    csv(summary,out,'candidate_comparison')
    best=next(c for c in candidates() if c['id']==summary.iloc[0].candidate_id)
    write_json(Path(out)/'selected_model.json',{'selected_utc':stamp(),'specification':best,'features':SETS[best['set']],
                'cv_mape':float(summary.iloc[0].cv_mape),'cv_std':float(summary.iloc[0].cv_std),
                'frozen_plan_sha256':digest(Path(out)/'frozen_plan.json'),'external_metrics_seen':False})
    return best,summary

def evaluate(best,t,train,valid,out):
    feats=SETS[best['set']];internal=estimator(best).fit(train[feats],train.cycle_life)
    selected_path=Path(out)/'selected_model.json';selection_hash=digest(selected_path)
    b1=t[t.analysis_eligible&t.batch.eq('Batch 1')].copy()
    final=estimator(best).fit(b1[feats],b1.cycle_life)
    joblib.dump(internal,Path(out)/'models/holdout_model.joblib');joblib.dump(final,Path(out)/'models/final_model.joblib')
    baseline_internal=DummyRegressor(strategy='median').fit(train[feats],train.cycle_life)
    baseline_final=DummyRegressor(strategy='median').fit(b1[feats],b1.cycle_life)
    rows=[];scores=[]
    for label,g,model,baseline,nfit in [('Valid (Batch 1 Hold-out)',valid,internal,baseline_internal,len(train)),
       ('Test (Batch 2)',t[t.analysis_eligible&t.batch.eq('Batch 2')],final,baseline_final,len(b1)),
       ('Test (Batch 3)',t[t.analysis_eligible&t.batch.eq('Batch 3')],final,baseline_final,len(b1))]:
        pred=model.predict(g[feats]);bp=baseline.predict(g[feats]);assert (pred>0).all()
        scores.append({'dataset':label,'n':len(g),'fit_cells':nfit,**metrics(g.cycle_life,pred),
                       'baseline_MAPE_pct':metrics(g.cycle_life,bp)['MAPE_pct']})
        for (_,r),p,b in zip(g.iterrows(),pred,bp):
            rows.append({'dataset':label,'cell_id':r.cell_id,'batch':r.batch,'policy':r.policy,'actual':r.cycle_life,
                         'predicted':p,'residual':p-r.cycle_life,'APE_pct':100*abs(p-r.cycle_life)/r.cycle_life,
                         'baseline_predicted':b,'life_outside_train_range':bool(r.cycle_life<b1.cycle_life.min() or r.cycle_life>b1.cycle_life.max())})
    scores=csv(pd.DataFrame(scores),out,'evaluation_metrics');preds=csv(pd.DataFrame(rows),out,'predictions')
    cv=pd.read_csv(Path(out)/'tables/cv_fold_scores.csv');s=cv[cv.candidate_id.eq(best['id'])]
    cm=float(s.MAPE_pct.mean());vm=float(scores.iloc[0].MAPE_pct);b2=float(scores.iloc[1].MAPE_pct);b3=float(scores.iloc[2].MAPE_pct)
    core=[['Train (Batch 1 CV)',cm,'%',f'29셀, 정책 그룹 4-fold 평균; SD {s.MAPE_pct.std():.2f}%p'],
          ['Valid (Batch 1 Hold-out)',vm,'%', '29셀 학습 모델; 미사용 정책 7셀 평가'],
          ['Test (Batch 2)',b2,'%', 'Batch 1 전체 36셀 재학습; Batch 2 39셀'],
          ['Gap (Train-Valid)',vm-cm,'%p','Valid - CV; (+)이면 과적합 또는 분할 난도 차이 의심'],
          ['Gap (Valid-Test)',b2-vm,'%p','Batch 2 - Valid; (+)이면 일반화 저하 의심'],
          ['Gap (Target-Test)',b2-TARGET_MAPE,'%p','Batch 2 - 9.1; (+)이면 논문 기준보다 높은 오차']]
    mandatory=csv(pd.DataFrame(core,columns=['구분','MAPE (%)','단위','비고']),out,'model_performance')
    extended=core+[['Test (Batch 3)',b3,'%', '동일 최종 모델; Batch 3 40셀'],
                   ['Gap (Batch2-Batch3)',b3-b2,'%p','Batch 3 - Batch 2; (+)이면 Batch 3 오차 증가'],
                   ['Gap (Target-Test) [Batch 3]',b3-TARGET_MAPE,'%p','Batch 3 - 9.1; 과제 공통 기준과 비교']]
    csv(pd.DataFrame(extended,columns=mandatory.columns),out,'model_performance_extended')
    csv(s,out,'selected_cv_scores')
    rng=np.random.default_rng(SEED);uncertainty=[]
    for label,g in preds.groupby('dataset',sort=False):
        groups=[z.APE_pct.to_numpy() for _,z in g.groupby('policy')];vals=[]
        for _ in range(1000):vals.append(np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]).mean())
        uncertainty.append({'dataset':label,'n_policies':len(groups),'MAPE_pct':g.APE_pct.mean(),
                            'policy_bootstrap_lo':np.quantile(vals,.025),'policy_bootstrap_hi':np.quantile(vals,.975)})
    csv(pd.DataFrame(uncertainty),out,'uncertainty')
    diagnostics(t,b1,preds,feats,out)
    pipe=final.regressor_ if best['target']=='log10' else final
    model=pipe.named_steps['model']
    if hasattr(model,'coef_'):
        csv(pd.DataFrame({'feature':feats,'coefficient_standardized':model.coef_}),out,'model_explanation')
        explain={'type':'linear','intercept':float(model.intercept_),'target_scale':best['target'],
                 'scaler_mean':pipe.named_steps['scale'].mean_.tolist(),'scaler_scale':pipe.named_steps['scale'].scale_.tolist()}
    elif hasattr(model,'feature_importances_'):
        csv(pd.DataFrame({'feature':feats,'impurity_importance':model.feature_importances_}),out,'model_explanation')
        explain={'type':'tree','target_scale':best['target']}
    else:explain={'type':'constant','target_scale':best['target']}
    write_json(Path(out)/'model_explanation.json',explain)
    assert selection_hash==digest(selected_path)
    write_json(Path(out)/'evaluation_record.json',{'evaluated_utc':stamp(),'selection_sha256':selection_hash,
        'internal_fit_ids':train.cell_id.tolist(),'final_fit_ids':b1.cell_id.tolist(),
        'test_model_sha256':digest(Path(out)/'models/final_model.joblib'),'no_external_refit':True})
    return scores,preds,mandatory

def diagnostics(t,b1,preds,feats,out):
    drift=[]
    for b,g in t[t.analysis_eligible].groupby('batch'):
        for f in feats:
            ref=b1[f].dropna();v=g[f].dropna();lo,hi=ref.min(),ref.max()
            drift.append({'batch':b,'feature':f,'n':len(v),'train_min':lo,'train_max':hi,'median':v.median(),
              'mean_shift_train_sd':(v.mean()-ref.mean())/ref.std(ddof=1),'outside_train_pct':100*((v<lo)|(v>hi)).mean()})
    csv(pd.DataFrame(drift),out,'feature_drift')
    group=preds.copy();group['life_band']=pd.cut(group.actual,[-np.inf,500,1000,np.inf],right=False,labels=['<500','500-999','>=1000'])
    csv(group.groupby(['dataset','life_band'],observed=True,as_index=False).agg(n=('cell_id','size'),MAPE_pct=('APE_pct','mean'),bias_cycles=('residual','mean')),out,'error_by_life_band')
    csv(group.groupby(['dataset','policy'],as_index=False).agg(n=('cell_id','size'),MAPE_pct=('APE_pct','mean'),bias_cycles=('residual','mean')),out,'error_by_policy')
    csv(preds.sort_values('APE_pct',ascending=False).groupby('dataset',sort=False).head(5),out,'largest_errors')

def plots(best,comparison,preds,out):
    da.configure();out=Path(out)
    fig,ax=plt.subplots(figsize=(10,4.3));top=comparison.head(10).iloc[::-1]
    ax.barh(range(len(top)),top.cv_mape,xerr=top.cv_std,color='#258578',alpha=.85,capsize=2)
    ax.set_yticks(range(len(top)),[f'{r.candidate_id} {r.family} {r.feature_set} {r.target}' for r in top.itertuples()],fontsize=9)
    ax.set(xlabel='Group CV mean MAPE (%) +/- fold SD',title='Batch 1 development only | top 10 candidates')
    fig.tight_layout();da.savefig(fig,out,'01_cv_selection');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,(label,g),color in zip(axs,preds.groupby('dataset',sort=False),['#356B9E','#DF7647','#1B8979']):
        lo=min(g.actual.min(),g.predicted.min())*.9;hi=max(g.actual.max(),g.predicted.max())*1.05
        ax.scatter(g.actual,g.predicted,c=color,s=35,alpha=.8);ax.plot([lo,hi],[lo,hi],'k--',lw=1)
        ax.set(title=label,xlabel='Actual life (cycles)',ylabel='Predicted life (cycles)',xlim=(lo,hi),ylim=(lo,hi))
        ax.text(.05,.94,f'n={len(g)} | MAPE={g.APE_pct.mean():.2f}%',transform=ax.transAxes,va='top')
    fig.tight_layout();da.savefig(fig,out,'02_actual_predicted');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for label,g in preds.groupby('dataset',sort=False):
        axs[0].scatter(g.actual,g.residual,label=label,alpha=.8,s=25)
        axs[1].scatter(g.actual,g.APE_pct,label=label,alpha=.8,s=25)
    axs[0].axhline(0,c='gray',ls='--');axs[0].set(xlabel='Actual life (cycles)',ylabel='Prediction - actual (cycles)',title='Signed residual')
    axs[1].set(xlabel='Actual life (cycles)',ylabel='Absolute percentage error (%)',title='Error vs lifetime')
    for ax in axs:ax.legend(fontsize=8)
    fig.tight_layout();da.savefig(fig,out,'03_residuals');plt.close(fig)
    met=pd.read_csv(out/'tables/evaluation_metrics.csv');fig,ax=plt.subplots(figsize=(9,4));x=np.arange(3)
    ax.bar(x-.18,met.MAPE_pct,width=.36,label='Selected model',color='#168573');ax.bar(x+.18,met.baseline_MAPE_pct,width=.36,label='Median baseline',color='#BCC7CA')
    ax.axhline(TARGET_MAPE,c='#C96440',ls='--',label='Paper target 9.1%');ax.set_xticks(x,['B1 hold-out','Batch 2','Batch 3'])
    ax.set(ylabel='MAPE (%)',title='Frozen model vs predeclared baseline');ax.legend(fontsize=9)
    fig.tight_layout();da.savefig(fig,out,'04_performance');plt.close(fig)
    d=pd.read_csv(out/'tables/feature_drift.csv');mat=d.pivot(index='feature',columns='batch',values='outside_train_pct')
    fig,ax=plt.subplots(figsize=(8,max(2.8,len(mat)*.6)));im=ax.imshow(mat,aspect='auto',vmin=0,vmax=100,cmap='YlOrRd')
    ax.set_xticks(range(3),mat.columns);ax.set_yticks(range(len(mat)),mat.index);ax.set_title('Features outside Batch 1 observed range (%)')
    for i in range(len(mat)):
        for j in range(3):ax.text(j,i,f'{mat.iloc[i,j]:.1f}',ha='center',va='center')
    fig.colorbar(im,ax=ax);fig.tight_layout();da.savefig(fig,out,'05_feature_drift');plt.close(fig)

def run(root):
    root=Path(root);out=root/'results/day2'
    cells,t,train,valid=prepare(root,out)
    best,comparison=select_model(train,out)
    scores,preds,report=evaluate(best,t,train,valid,out)
    plots(best,comparison,preds,out)
    write_json(out/'environment.json',{'python':platform.python_version(),'random_state':SEED,
       'libraries':{p:metadata.version(p) for p in ['numpy','pandas','scipy','h5py','scikit-learn','matplotlib','joblib','nbformat','nbclient']}})
    print('Selected:',json.dumps(best,ensure_ascii=False));print(report.to_string(index=False));print(scores.to_string(index=False))
    return best,comparison,scores,preds

if __name__=='__main__':run(da.root_path())
