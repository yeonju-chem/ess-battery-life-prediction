"""DAY 1: provided-batch EDA, without model fitting. SKALA 3반 U095 이연주.

HDF5 references are read selectively: summary, cycles 2/10/100 Qdlin and
cycle-10 current. No multi-gigabyte full-object load is needed.
"""
from pathlib import Path
import os, re, json, hashlib, platform, importlib.metadata
import numpy as np
import pandas as pd
import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import skew, kurtosis, spearmanr

RANDOM_STATE = 42
FILES = {'Batch 1':'2017-05-12_batchdata_updated_struct_errorcorrect.mat',
         'Batch 2':'2018-02-20_batchdata_updated_struct_errorcorrect.mat',
         'Batch 3':'2018-04-12_batchdata_updated_struct_errorcorrect.mat'}
COLORS = {'Batch 1':'#2367A1','Batch 2':'#DF7647','Batch 3':'#1B8979'}
FIELDS = ['QDischarge','QCharge','IR','Tavg','Tmax','Tmin','chargetime','cycle']
FEATURES = ['log_dq_var','dq_min','dq_mean','dq_std','dq_skew','dq_kurtosis',
            'dq_area','qd_2','qd_slope','qd_change','ir_median','ir_change',
            'tavg_median','tmax_median','chargetime_median','c1','soc_switch','c2','c_equiv']
B1_UNRESOLVED = {0,1,2,3,4,8,10,12,13,22}
B3_NOISY = {2,23,32,37,42,43}

def root_path():
    if os.environ.get('ESS_PROJECT_ROOT'):
        return Path(os.environ['ESS_PROJECT_ROOT']).resolve()
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p/'data'/FILES['Batch 1']).exists(): return p
    raise FileNotFoundError('프로젝트 루트 또는 notebooks 폴더에서 실행한다.')

def configure():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold',
        'figure.dpi':110,'savefig.dpi':180,'axes.grid':True,'grid.alpha':.15})

def read_array(node):
    if node.attrs.get('MATLAB_empty',0): return np.array([], dtype=float)
    return np.asarray(node[()],dtype=float).ravel()

def at_cycle(cell, number, field='QDischarge'):
    s=cell['summary']; vals=s.loc[s.cycle.eq(number),field]
    return float(vals.iloc[0]) if len(vals) else np.nan

def plausible_q(q):
    # Broad inspection rule, not a learned bound and not an EOL cutoff.
    return np.isfinite(q) & (q>0) & (q<=1.5)

def parse_policy(policy):
    # suffixes such as -newstructure are retained in the original policy.
    m=re.match(r'^(\d+(?:\.\d+)?)C\((\d+(?:\.\d+)?)%\)-(\d+(?:\.\d+)?)C',policy)
    if not m: return (np.nan,)*4
    c1,s,c2=map(float,m.groups()); p=s/100
    # Two-stage equivalent rate over 0-80% SOC, excluding CV and rest.
    ceq=.8/(p/c1+(.8-p)/c2) if 0<=p<=.8 else np.nan
    return c1,s,c2,ceq

def load_batches(data_dir):
    cells=[]; manifest=[]
    for b,fn in FILES.items():
        path=Path(data_dir)/fn
        with h5py.File(path,'r') as f:
            g=f['batch']; n=g['summary'].shape[0]
            manifest.append({'batch':b,'file':fn,'bytes':path.stat().st_size,
                'cell_count':n,'fields':','.join(g.keys())})
            for i in range(n):
                sg=f[g['summary'][i,0]]; cg=f[g['cycles'][i,0]]
                s=pd.DataFrame({k:read_array(sg[k]) for k in FIELDS})
                assert s.cycle.is_unique and s.cycle.is_monotonic_increasing
                assert len(s)==cg['Qdlin'].shape[0], (b,i,'cycle alignment')
                policy=''.join(chr(int(x)) for x in f[g['policy_readable'][i,0]][()].ravel())
                vd=read_array(f[g['Vdlin'][i,0]])
                qcurves={}; current10=np.nan; current_trace=np.empty((0,2))
                for cyc in [2,10,100]:
                    ix=np.flatnonzero(s.cycle.to_numpy()==cyc)
                    q=read_array(f[cg['Qdlin'][int(ix[0]),0]]) if len(ix) else np.array([])
                    qcurves[cyc]=q
                    if cyc==10 and len(ix):
                        cur=read_array(f[cg['I'][int(ix[0]),0]])
                        time=read_array(f[cg['t'][int(ix[0]),0]])
                        assert len(time)==len(cur)
                        current_trace=np.column_stack([time,cur])
                        pos=cur[np.isfinite(cur)&(cur>0)]
                        if len(pos): current10=float(np.quantile(pos,.95))
                life=float(read_array(f[g['cycle_life'][i,0]])[0])
                cells.append({'cell_id':f'b{b[-1]}c{i}','batch':b,'index':i,
                    'cycle_life':life,'policy':policy,'summary':s,'voltage':vd,
                    'qcurves':qcurves,'current10_p95_A':current10,'current10_trace':current_trace})
    return cells,pd.DataFrame(manifest)

def knee_candidate(s):
    """Continuous two-line least squares on rolling median (EDA only).

    The breakpoint is an exploratory candidate, never an early-life feature.
    Require >=25% SSE reduction and faster negative slope after the knot.
    """
    z=s.loc[plausible_q(s.QDischarge),['cycle','QDischarge']].copy()
    if len(z)<150: return (np.nan,)*5
    z['smooth']=z.QDischarge.rolling(11,center=True,min_periods=5).median()
    z=z.dropna(); x=z.cycle.to_numpy()/1000; y=z.smooth.to_numpy()
    base=np.column_stack([np.ones(len(x)),x]); coef=np.linalg.lstsq(base,y,rcond=None)[0]
    sse0=np.square(y-base@coef).sum()
    best=None
    for k in np.linspace(max(.1,float(np.quantile(x,.15))),float(np.quantile(x,.85)),45):
        X=np.column_stack([np.ones(len(x)),x,np.maximum(x-k,0)])
        c=np.linalg.lstsq(X,y,rcond=None)[0]; sse=np.square(y-X@c).sum()
        if best is None or sse<best[0]:best=(sse,k,c)
    sse,k,c=best; gain=1-sse/sse0 if sse0>0 else 0
    s1=c[1]/1000; s2=(c[1]+c[2])/1000
    accepted=bool(gain>=.25 and s2<s1 and s2<-.0001)
    return k*1000,s1,s2,gain,accepted

def feature_table(cells):
    records=[]; quality=[]
    for c in cells:
        s=c['summary']; early=s.loc[s.cycle.between(2,100)].copy()
        good=early.loc[plausible_q(early.QDischarge)]
        q10=c['qcurves'][10];q100=c['qcurves'][100];v=c['voltage']
        valid=(len(q10)==len(q100)==len(v)==1000 and
               np.isfinite(q10).all() and np.isfinite(q100).all() and np.isfinite(v).all())
        row={k:c[k] for k in ['cell_id','batch','index','cycle_life','policy','current10_p95_A']}
        row.update({'n_cycles':len(s),'max_cycle':s.cycle.max(),'last_qd':s.QDischarge.iloc[-1],
            'has_cycle5':bool(s.cycle.eq(5).any()),'has_cycle100':bool(s.cycle.eq(100).any()),
            'dq_valid':valid,'n_valid_qd_2_100':len(good),'label_valid':bool(np.isfinite(c['cycle_life']) and c['cycle_life']>0)})
        row['qd_2']=at_cycle(c,2); row['qd_change']=at_cycle(c,100)-at_cycle(c,10)
        row['qd_slope']=np.polyfit(good.cycle,good.QDischarge,1)[0] if len(good)>=80 else np.nan
        for name,field in [('ir','IR'),('tavg','Tavg'),('tmax','Tmax'),('chargetime','chargetime')]:
            vals=early[field].where(early[field]>0)
            if name=='chargetime': vals=vals.where(vals<=60)
            row[name+'_median']=vals.median()
        row['ir_change']=at_cycle(c,100,'IR')-at_cycle(c,10,'IR')
        if at_cycle(c,100,'IR')<=0 or at_cycle(c,10,'IR')<=0:row['ir_change']=np.nan
        row.update(dict(zip(['c1','soc_switch','c2','c_equiv'],parse_policy(c['policy']))))
        row['policy_special']=any(x in c['policy'] for x in ['VarCharge','SLOWCYCLE'])
        for k in ['log_dq_var','dq_min','dq_max','dq_mean','dq_std','dq_skew','dq_kurtosis','dq_area']:row[k]=np.nan
        if valid:
            dq=q100-q10; variance=float(np.var(dq,ddof=1)); order=np.argsort(v)
            row.update({'log_dq_var':np.log10(variance) if variance>0 else np.nan,
                'dq_min':dq.min(),'dq_max':dq.max(),'dq_mean':dq.mean(),'dq_std':dq.std(ddof=1),
                'dq_skew':skew(dq,bias=False),'dq_kurtosis':kurtosis(dq,bias=False),
                'dq_area':np.trapezoid(dq[order],v[order])})
        row.update(dict(zip(['knee_cycle','slope_before','slope_after','knee_gain','knee_accepted'],knee_candidate(s))))
        reasons=[]
        if not row['label_valid']:reasons.append('missing_target')
        if c['batch']=='Batch 1' and c['index'] in B1_UNRESOLVED:reasons.append('unresolved_EOl_or_continuation')
        if c['batch']=='Batch 3' and c['index'] in B3_NOISY:reasons.append('author_flagged_channel')
        if row['policy_special']:reasons.append('special_protocol')
        if not valid:reasons.append('invalid_dQ')
        if len(good)<80:reasons.append('insufficient_early_QD')
        row['review_reason']=';'.join(reasons)
        row['analysis_eligible']=not bool(reasons)
        records.append(row)
        for field in FIELDS[:-1]:
            a=s[field].to_numpy()
            quality.append({'cell_id':c['cell_id'],'batch':c['batch'],'field':field,
                'n':len(a),'missing':int((~np.isfinite(a)).sum()),'zero':int((a==0).sum()),
                'qd_outside_0_1_5':int((~plausible_q(a)).sum()) if field=='QDischarge' else 0,
                'charge_over60':int((a>60).sum()) if field=='chargetime' else 0})
    return pd.DataFrame(records),pd.DataFrame(quality)

def batch_summary(t):
    rows=[]
    for b,g in t.groupby('batch'):
        y=g.loc[g.label_valid,'cycle_life'];n=len(y)
        rows.append({'batch':b,'cells':len(g),'labels':n,'missing_labels':len(g)-n,
            'cycles':int(g.n_cycles.sum()),'policies':g.policy.nunique(),'min':y.min(),
            'q25':y.quantile(.25),'median':y.median(),'mean':y.mean(),'q75':y.quantile(.75),
            'max':y.max(),'std':y.std(),'short_n':int((y<500).sum()),'long_n':int((y>1000).sum()),
            'short_pct':100*(y<500).mean(),'long_pct':100*(y>1000).mean(),
            'below550':int((y<550).sum()),'above_eq550':int((y>=550).sum()),
            'valid_dq':int(g.dq_valid.sum()),'eligible':int(g.analysis_eligible.sum())})
    return pd.DataFrame(rows)

def correlations(t):
    rows=[]
    for scope,sub in [('all_labeled',t[t.label_valid]),('quality_screened',t[t.analysis_eligible])]:
        for b,g in [('Pooled',sub),*list(sub.groupby('batch'))]:
            for feat in FEATURES:
                z=g[[feat,'cycle_life']].replace([np.inf,-np.inf],np.nan).dropna()
                if len(z)>=4 and z[feat].nunique()>1:
                    rows.append({'scope':scope,'batch':b,'feature':feat,'n':len(z),
                        'pearson':z[feat].corr(z.cycle_life),'spearman':z[feat].corr(z.cycle_life,method='spearman')})
    return pd.DataFrame(rows)

def savefig(fig,out,name):
    fig.savefig(Path(out)/'figures'/f'{name}.png',bbox_inches='tight',facecolor='white')
    return fig

def fig_distribution(t,out):
    fig,axs=plt.subplots(1,3,figsize=(13,3.5),sharex=True,sharey=True)
    for ax,(b,g) in zip(axs,t.groupby('batch')):
        y=g.loc[g.label_valid,'cycle_life']; ax.hist(y,bins=np.arange(150,2351,100),color=COLORS[b],alpha=.85,edgecolor='white')
        ax.axvline(500,c='gray',ls='--');ax.axvline(1000,c='gray',ls=':')
        ax.set(title=f'{b} | labeled {len(y)}/{len(g)}',xlabel='Recorded cycle life',xlim=(150,2300))
        ax.text(.97,.95,f'median {y.median():.1f}\n<500: {(y<500).sum()} | >1000: {(y>1000).sum()}',ha='right',va='top',transform=ax.transAxes,fontsize=9)
    axs[0].set_ylabel('Cells');fig.tight_layout();return savefig(fig,out,'01_life_distribution')

def fig_degradation(cells,t,out):
    fig,axs=plt.subplots(1,3,figsize=(13,3.7),sharex=True,sharey=True); norm=Normalize(150,2300);cmap=plt.get_cmap('viridis')
    for ax,b in zip(axs,FILES):
        for c in cells:
            if c['batch']!=b:continue
            s=c['summary'];y=s.QDischarge.where(plausible_q(s.QDischarge));life=c['cycle_life']
            ax.plot(s.cycle,y,color=cmap(norm(life)) if np.isfinite(life) else '#9CA3AF',lw=.75,alpha=.7)
        ax.axhline(.88,c='#C3444A',ls='--',lw=1);ax.set(title=b,xlabel='Cycle',ylim=(.65,1.18),xlim=(0,2350))
    axs[0].set_ylabel('Discharge capacity (Ah)');fig.tight_layout(rect=(0,0,.93,1))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.94,.2,.012,.6]),label='Recorded cycle life')
    return savefig(fig,out,'02_degradation')

def representatives(t):
    ids=[]
    for b,g in t[t.analysis_eligible].groupby('batch'):
        # Tie-break on ID so figures, tables and re-runs select the same cell.
        g=g.sort_values(['cycle_life','cell_id'],kind='stable');ids.extend([g.iloc[0].cell_id,g.iloc[len(g)//2].cell_id,g.iloc[-1].cell_id])
    return ids

def fig_representatives(cells,t,out):
    ids=representatives(t); cmap={c['cell_id']:c for c in cells};fig,axs=plt.subplots(1,3,figsize=(13,3.8),sharey=True)
    for ax,b in zip(axs,FILES):
        rows=t[t.cell_id.isin(ids)&t.batch.eq(b)].sort_values('cycle_life')
        for color,(_,r) in zip(['#D55E00','#0072B2','#009E73'],rows.iterrows()):
            s=cmap[r.cell_id]['summary'];good=plausible_q(s.QDischarge)
            ax.plot(s.cycle,s.QDischarge.where(good),c=color,label=f'{r.cell_id}: {r.cycle_life:.0f}')
            if r.knee_accepted:ax.axvline(r.knee_cycle,color=color,ls=':',alpha=.65)
        ax.axhline(.88,c='gray',ls='--');ax.set(title=b,xlabel='Cycle',ylim=(.7,1.15));ax.legend(fontsize=8)
    axs[0].set_ylabel('Discharge capacity (Ah)');fig.tight_layout();return savefig(fig,out,'03_representatives_knee')

def fig_delta(cells,t,out):
    fig,axs=plt.subplots(1,3,figsize=(13,3.8));norm=Normalize(150,2300);cmap=plt.get_cmap('viridis')
    for ax,b in zip(axs,FILES):
        subset=t[t.batch.eq(b)&t.analysis_eligible];ids=set(subset.cell_id)
        for c in cells:
            if c['cell_id'] in ids:
                ax.plot(c['voltage'],c['qcurves'][100]-c['qcurves'][10],color=cmap(norm(c['cycle_life'])),alpha=.6,lw=.8)
        ax.set(title=f'{b} | screened n={len(subset)}',xlabel='Voltage (V)');ax.axhline(0,c='gray',ls='--')
    axs[0].set_ylabel('Q100(V) - Q10(V) (Ah)');fig.tight_layout(rect=(0,0,.93,1))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.94,.2,.012,.6]),label='Recorded cycle life')
    return savefig(fig,out,'04_delta_curves')

def fig_delta_groups(cells,t,out):
    fig,axs=plt.subplots(1,3,figsize=(13,3.7));cm={c['cell_id']:c for c in cells}
    for ax,b in zip(axs,FILES):
        for label,mask,color in [('Short <500',t.cycle_life.lt(500),'#D55E00'),('Middle 500-1000',t.cycle_life.between(500,1000),'#0072B2'),('Long >1000',t.cycle_life.gt(1000),'#009E73')]:
            g=t[t.batch.eq(b)&t.analysis_eligible&mask]
            if g.empty:continue
            arr=np.stack([cm[c]['qcurves'][100]-cm[c]['qcurves'][10] for c in g.cell_id]);v=cm[g.cell_id.iloc[0]]['voltage']
            ax.plot(v,np.median(arr,axis=0),c=color,label=f'{label} (n={len(g)})')
            ax.fill_between(v,np.quantile(arr,.25,axis=0),np.quantile(arr,.75,axis=0),color=color,alpha=.15)
        ax.set(title=b,xlabel='Voltage (V)');ax.legend(fontsize=7)
    axs[0].set_ylabel('Median delta Q (Ah), band = IQR');fig.tight_layout();return savefig(fig,out,'05_delta_groups')

def fig_feature_scatter(t,out):
    fig,axs=plt.subplots(1,3,figsize=(13,3.7));sub=t[t.analysis_eligible]
    for ax,feat,label in zip(axs,['log_dq_var','qd_slope','chargetime_median'],['log10 Var(delta Q)','Early QD slope (Ah/cycle)','Median charge time (min)']):
        for b,g in sub.groupby('batch'):ax.scatter(g[feat],g.cycle_life,c=COLORS[b],s=24,alpha=.8,label=b)
        ax.set(xlabel=label,ylabel='Recorded cycle life');ax.legend(fontsize=8)
    fig.tight_layout();return savefig(fig,out,'06_feature_scatter')

def policy_table(t):
    return t[t.label_valid].groupby(['batch','policy'],as_index=False).agg(n=('cell_id','size'),mean=('cycle_life','mean'),median=('cycle_life','median'),std=('cycle_life','std'))

def fig_policy(t,out):
    fig,axs=plt.subplots(1,3,figsize=(13,5.7),sharex=True)
    p=policy_table(t)
    for ax,b in zip(axs,FILES):
        g=p[p.batch.eq(b)].sort_values('mean');y=np.arange(len(g))
        ax.barh(y,g['mean'],xerr=g['std'].fillna(0),color=COLORS[b],alpha=.8,error_kw={'elinewidth':.7,'capsize':2})
        ax.set_yticks(y,[f'{s.replace("-newstructure", " (new)")} [n={n}]' for s,n in zip(g.policy,g.n)],fontsize=6.5)
        ax.set(title=b,xlabel='Mean recorded life +/- 1 SD')
    fig.tight_layout();return savefig(fig,out,'07_policy_means')

def fig_rate(t,out):
    fig,axs=plt.subplots(1,3,figsize=(13,3.6));s=t[t.analysis_eligible]
    for ax,feat,y,label in zip(axs,['c_equiv','c_equiv','current10_p95_A'],['cycle_life','qd_slope','qd_slope'],['Two-stage equivalent C-rate','Two-stage equivalent C-rate','Cycle 10 charge-current P95 (A)']):
        for b,g in s.groupby('batch'):ax.scatter(g[feat],g[y],s=25,c=COLORS[b],alpha=.8,label=b)
        ax.set(xlabel=label,ylabel='Recorded cycle life' if y=='cycle_life' else 'Early QD slope (Ah/cycle)');ax.legend(fontsize=8)
    fig.tight_layout();return savefig(fig,out,'08_rate_and_fade')

def fig_early(cells,t,out):
    fig,axs=plt.subplots(1,3,figsize=(13,3.6),sharey=True)
    ids=set(t.loc[t.analysis_eligible,'cell_id']);norm=Normalize(150,2300);cmap=plt.get_cmap('viridis')
    for ax,b in zip(axs,FILES):
        for c in cells:
            if c['batch']!=b or c['cell_id'] not in ids:continue
            s=c['summary'];g=s[s.cycle.between(2,100)&plausible_q(s.QDischarge)]
            ax.plot(g.cycle,g.QDischarge,lw=.8,alpha=.6,color=cmap(norm(c['cycle_life'])))
        ax.set(title=b,xlabel='Cycle (2-100)')
    axs[0].set_ylabel('Discharge capacity (Ah)');fig.tight_layout();return savefig(fig,out,'11_early_cycles')

def supplementary_tables(t,corr,out):
    from sklearn.model_selection import GroupShuffleSplit,GroupKFold
    rows=[]
    for b,g in t[t.analysis_eligible].groupby('batch'):
        for x,y in [('c_equiv','qd_slope'),('current10_p95_A','qd_slope')]:
            z=g[[x,y]].dropna();rows.append({'batch':b,'x':x,'y':y,'n':len(z),'pearson':z[x].corr(z[y]),'spearman':z[x].corr(z[y],method='spearman')})
    pd.DataFrame(rows).to_csv(Path(out)/'tables/rate_fade_correlations.csv',index=False,encoding='utf-8-sig')
    g=t[t.batch.eq('Batch 1')&t.analysis_eligible].copy().reset_index(drop=True)
    tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(g,groups=g.policy))
    assert not set(g.iloc[tr].policy)&set(g.iloc[va].policy)
    g['planned_split']='validation';g.loc[tr,'planned_split']='train';g['cv_fold']=-1
    for fold,(_,test) in enumerate(GroupKFold(n_splits=4).split(g.iloc[tr],groups=g.iloc[tr].policy)):
        g.loc[tr[test],'cv_fold']=fold
    g[['cell_id','policy','planned_split','cv_fold']].to_csv(Path(out)/'tables/planned_split.csv',index=False,encoding='utf-8-sig')
    rng=np.random.default_rng(42);ci=[]
    for b,g in t[t.analysis_eligible].groupby('batch'):
        groups=[a for _,a in g.groupby('policy')];vals=[]
        for _ in range(1000):
            z=pd.concat([groups[i] for i in rng.integers(0,len(groups),len(groups))])
            vals.append(z.log_dq_var.corr(z.cycle_life))
        ci.append({'batch':b,'n_cells':len(g),'n_policy_groups':len(groups),'pearson':g.log_dq_var.corr(g.cycle_life),
                   'bootstrap_lo':np.nanquantile(vals,.025),'bootstrap_hi':np.nanquantile(vals,.975)})
    pd.DataFrame(ci).to_csv(Path(out)/'tables/delta_correlation_uncertainty.csv',index=False,encoding='utf-8-sig')
    # Readable single-batch figures for the PDF.
    p=policy_table(t)
    for b in FILES:
        g=p[p.batch.eq(b)].sort_values('mean');fig,ax=plt.subplots(figsize=(9,6))
        y=np.arange(len(g));ax.barh(y,g['mean'],xerr=g['std'].fillna(0),color=COLORS[b],alpha=.85,error_kw={'elinewidth':1,'capsize':2})
        ax.set_yticks(y,[f'{s.replace("-newstructure"," (new)")} [n={n}]' for s,n in zip(g.policy,g.n)],fontsize=9)
        ax.set(title=b+' | charging protocol',xlabel='Mean recorded cycle life +/- 1 SD')
        fig.tight_layout();savefig(fig,out,'07_policy_batch'+b[-1]);plt.close(fig)
    return g

def fig_correlations(corr,out):
    sel=corr[(corr.scope=='quality_screened')]
    fig,axs=plt.subplots(1,2,figsize=(13.5,6.2),layout='constrained')
    for ax,method in zip(axs,['pearson','spearman']):
        pivot=sel.pivot(index='feature',columns='batch',values=method).reindex(FEATURES)[['Batch 1','Batch 2','Batch 3','Pooled']]
        im=ax.imshow(pivot,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
        ax.set_xticks(range(4),pivot.columns);ax.set_yticks(range(len(pivot)),pivot.index,fontsize=8);ax.set_title(method.title()+' vs recorded life')
        for i in range(len(pivot)):
            for j in range(4):
                z=pivot.iloc[i,j]
                ax.text(j,i,f'{z:.2f}' if np.isfinite(z) else 'NA',ha='center',va='center',fontsize=7,color='white' if abs(z)>.55 else 'black')
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.6);return savefig(fig,out,'09_correlations')

def fig_collinearity(t,out):
    from day1_review import collinearity
    return collinearity(t,out)[0]

def make_tables(cells,t,q,manifest,out):
    td=Path(out)/'tables';td.mkdir(parents=True,exist_ok=True);(Path(out)/'figures').mkdir(parents=True,exist_ok=True)
    summary=batch_summary(t);corr=correlations(t)
    audits=t[['cell_id','batch','cycle_life','last_qd','n_cycles','dq_valid','analysis_eligible','review_reason']]
    for name,frame in [('cell_features',t),('batch_summary',summary),('quality_by_cell_field',q),('source_manifest',manifest),('correlations',corr),('policy_summary',policy_table(t)),('cell_audit',audits)]:
        frame.to_csv(td/f'{name}.csv',index=False,encoding='utf-8-sig')
    flags=[]
    for b,g in t[t.label_valid].groupby('batch'):
        q1,q3=g.cycle_life.quantile([.25,.75]);lo=q1-1.5*(q3-q1);hi=q3+1.5*(q3-q1)
        for _,row in g.iterrows():
            flags.append({'cell_id':row.cell_id,'batch':b,'cycle_life':row.cycle_life,
                'iqr_lower':lo,'iqr_upper':hi,'iqr_flag':'low' if row.cycle_life<lo else 'high' if row.cycle_life>hi else 'within',
                'policy':row.policy})
    pd.DataFrame(flags).to_csv(td/'lifetime_outliers.csv',index=False,encoding='utf-8-sig')
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','h5py','scipy','matplotlib','scikit-learn','nbformat','nbclient']}
    (Path(out)/'environment.json').write_text(json.dumps({'python':platform.python_version(),'platform':platform.platform(),'random_state':RANDOM_STATE,'libraries':versions},ensure_ascii=False,indent=2))
    return summary,corr

def run_all(data_dir,out):
    configure();cells,manifest=load_batches(data_dir);t,q=feature_table(cells)
    summary,corr=make_tables(cells,t,q,manifest,out)
    for fun,args in [(fig_distribution,(t,)),(fig_degradation,(cells,t)),(fig_representatives,(cells,t)),(fig_delta,(cells,t)),(fig_delta_groups,(cells,t)),(fig_feature_scatter,(t,)),(fig_policy,(t,)),(fig_rate,(t,)),(fig_correlations,(corr,)),(fig_collinearity,(t,))]:
        f=fun(*args,out);plt.close(f)
    f=fig_early(cells,t,out);plt.close(f);supplementary_tables(t,corr,out)
    from day1_review import run
    run(cells,t,out)
    print(summary.to_string(index=False));print('\nBatch 1 screened correlation:')
    print(corr[(corr.batch=='Batch 1')&(corr.scope=='quality_screened')].sort_values('pearson',key=abs,ascending=False).to_string(index=False))
    return cells,t,q,summary,corr

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    run_all(a.data,a.out)
