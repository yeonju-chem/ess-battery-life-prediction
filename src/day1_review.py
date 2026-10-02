"""Additional per-batch evidence for final DAY 1 review; no model fitting."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import day1_analysis as da

def save_table(frame,out,name):
    frame.to_csv(Path(out)/'tables'/f'{name}.csv',index=False,encoding='utf-8-sig')
    return frame

def feature_distributions(t,out):
    sub=t[t.analysis_eligible]
    cols=['log_dq_var','qd_slope','ir_median','tavg_median','chargetime_median','c_equiv']
    labels=['log10 Var(delta Q)','Early QD slope (Ah/cycle)','IR median (ohm)',
            'Tavg median (deg C)','Charge time median (min)','Equivalent C-rate']
    fig,axs=plt.subplots(2,3,figsize=(13,6.2));rows=[]
    for ax,feat,label in zip(axs.ravel(),cols,labels):
        for i,(b,g) in enumerate(sub.groupby('batch')):
            a=g[feat].dropna();bp=ax.boxplot(a,positions=[i],widths=.45,patch_artist=True,showfliers=False)
            bp['boxes'][0].set_facecolor(da.COLORS[b]);bp['boxes'][0].set_alpha(.35)
            # Deterministic horizontal spread; every cell remains visible.
            ax.scatter(i+np.linspace(-.17,.17,len(a)),a,s=9,c=da.COLORS[b],alpha=.65)
            rows.append({'batch':b,'feature':feat,'n':len(a),'missing':len(g)-len(a),
                         'min':a.min(),'q25':a.quantile(.25),'median':a.median(),
                         'q75':a.quantile(.75),'max':a.max()})
        ax.set_xticks(range(3),['B1','B2','B3']);ax.set_title(label,fontsize=11)
    fig.tight_layout();da.savefig(fig,out,'12_feature_distributions')
    return fig,save_table(pd.DataFrame(rows),out,'feature_distributions')

def relative_delta(cells,t,out):
    cm={c['cell_id']:c for c in cells};fig,axs=plt.subplots(1,3,figsize=(13,3.7));rows=[]
    for ax,(b,g) in zip(axs,t[t.analysis_eligible].groupby('batch')):
        lo,hi=g.cycle_life.quantile([.25,.75])
        for label,grp,color in [('Bottom life quartile',g[g.cycle_life<=lo],'#CE623F'),
                                ('Top life quartile',g[g.cycle_life>=hi],'#168A7B')]:
            arr=np.stack([cm[c]['qcurves'][100]-cm[c]['qcurves'][10] for c in grp.cell_id])
            v=cm[grp.cell_id.iloc[0]]['voltage']
            ax.plot(v,np.median(arr,axis=0),color=color,label=f'{label} n={len(grp)}')
            ax.fill_between(v,np.quantile(arr,.25,axis=0),np.quantile(arr,.75,axis=0),color=color,alpha=.18)
            rows.append({'batch':b,'group':label,'n':len(grp),'life_q25_cut':lo,'life_q75_cut':hi,
                         'life_min':grp.cycle_life.min(),'life_max':grp.cycle_life.max(),
                         'median_log_dq_var':grp.log_dq_var.median()})
        ax.set(title=b,xlabel='Voltage (V)');ax.legend(fontsize=8)
    axs[0].set_ylabel('Median delta Q (Ah); band = IQR');fig.tight_layout()
    da.savefig(fig,out,'13_delta_relative_groups')
    return fig,save_table(pd.DataFrame(rows),out,'delta_relative_groups')

def collinearity(t,out):
    cols=['log_dq_var','dq_std','dq_mean','dq_area','qd_slope','tavg_median','tmax_median','c_equiv']
    fig,axs=plt.subplots(1,3,figsize=(14,5.2));rows=[]
    for ax,(b,g) in zip(axs,t[t.analysis_eligible].groupby('batch')):
        z=g[cols].corr()
        im=ax.imshow(z,cmap='RdBu_r',vmin=-1,vmax=1)
        ax.set_xticks(range(len(cols)),cols,rotation=65,ha='right',fontsize=7)
        ax.set_yticks(range(len(cols)),cols,fontsize=7);ax.set_title(f'{b} | n={len(g)}')
        for i in range(len(cols)):
            for j in range(len(cols)):
                val=z.iloc[i,j]
                ax.text(j,i,f'{val:.2f}',ha='center',va='center',fontsize=6,
                        color='white' if abs(val)>.55 else 'black')
        # Full feature-pair audit, not only the selected plotting columns.
        full=g[da.FEATURES].corr()
        for i,x in enumerate(full.columns):
            for y in full.columns[i+1:]:
                val=full.loc[x,y]
                rows.append({'batch':b,'feature_x':x,'feature_y':y,'pearson':val,
                             'n':len(g[[x,y]].dropna()),'abs_r_ge_0_9':bool(abs(val)>=.9)})
    fig.tight_layout(rect=(0,0,.94,1));fig.colorbar(im,cax=fig.add_axes([.955,.32,.01,.5]))
    da.savefig(fig,out,'10_collinearity')
    return fig,save_table(pd.DataFrame(rows),out,'feature_pair_correlations')

def current_patterns(cells,t,out):
    ids=da.representatives(t);cm={c['cell_id']:c for c in cells}
    fig,axs=plt.subplots(1,3,figsize=(13,3.8));rows=[]
    for ax,b in zip(axs,da.FILES):
        for color,cell_id in zip(['#D55E00','#0072B2','#009E73'],[i for i in ids if cm[i]['batch']==b]):
            c=cm[cell_id];a=c['current10_trace'];a=a[np.isfinite(a).all(axis=1)]
            # Positive current is charging. Plot before the first negative-current discharge.
            positive=np.flatnonzero(a[:,1]>.05)
            if not len(positive):continue
            start=positive[0];ends=np.flatnonzero((np.arange(len(a))>start)&(a[:,1]<-.05))
            end=int(ends[0]) if len(ends) else len(a)
            a=a[start:end];elapsed=a[:,0]-a[0,0]
            ax.plot(elapsed,a[:,1],color=color,lw=1.3,label=f'{cell_id}: life {c["cycle_life"]:.0f}')
            rows.append({'cell_id':cell_id,'batch':b,'cycle':10,'samples':len(a),
                         'charge_window_minutes':elapsed[-1],'positive_current_p95_A':c['current10_p95_A'],
                         'policy':c['policy']})
        ax.set(title=b,xlabel='Cycle 10: elapsed charging time (min)');ax.legend(fontsize=8)
    axs[0].set_ylabel('Measured current (A)');fig.tight_layout();da.savefig(fig,out,'14_current_patterns')
    return fig,save_table(pd.DataFrame(rows),out,'current_pattern_representatives')

def evidence_tables(t,out):
    rows=[]
    for b,g in t[t.analysis_eligible].groupby('batch'):
        s=g.loc[g.cycle_life.idxmin()];peers=g[(g.policy==s.policy)&(g.cell_id!=s.cell_id)]
        for feat in ['log_dq_var','qd_slope','tavg_median','ir_median']:
            rows.append({'batch':b,'cell_id':s.cell_id,'cycle_life':s.cycle_life,'feature':feat,
                         'value':s[feat],'batch_median':g[feat].median(),
                         'percentile_le':100*g[feat].le(s[feat]).mean(),
                         'other_same_policy_n':len(peers),
                         'other_same_policy_life_median':peers.cycle_life.median()})
    save_table(pd.DataFrame(rows),out,'shortest_cell_comparison')
    cols=['batch','cell_id','cycle_life','knee_cycle','slope_before','slope_after','knee_gain','knee_accepted']
    save_table(t.loc[t.analysis_eligible,cols],out,'knee_diagnostics')
    sensitivity=da.correlations(t)
    sensitivity=sensitivity[sensitivity.feature.isin(['log_dq_var','qd_slope','chargetime_median'])]
    save_table(sensitivity,out,'screening_sensitivity')

def slope_sensitivity(cells,t,out):
    """Compare raw OLS and median-smoothed OLS using cycles 2-100 only."""
    rows=[]
    for c in cells:
        s=c['summary'];z=s.loc[s.cycle.between(2,100),['cycle','QDischarge']].copy()
        z['q']=z.QDischarge.where(da.plausible_q(z.QDischarge))
        z['smooth']=z.q.rolling(11,center=True,min_periods=5).median()
        z=z.dropna(subset=['smooth'])
        slope=np.polyfit(z.cycle,z.smooth,1)[0] if len(z)>=80 else np.nan
        rows.append({'cell_id':c['cell_id'],'smoothed_qd_slope':slope,
                     'local_deviation_gt_0_02Ah':int((z.q-z.smooth).abs().gt(.02).sum())})
    df=t[['cell_id','batch','analysis_eligible','cycle_life','qd_slope']].merge(pd.DataFrame(rows),on='cell_id',validate='one_to_one')
    save_table(df,out,'early_slope_sensitivity_cells')
    summary=[]
    for b,g in df[df.analysis_eligible].groupby('batch'):
        summary.append({'batch':b,'n':len(g),'raw_slope_life_r':g.qd_slope.corr(g.cycle_life) if len(g)>1 else np.nan,
                        'smoothed_slope_life_r':g.smoothed_qd_slope.corr(g.cycle_life) if len(g)>1 else np.nan,
                        'cells_with_local_deviation':int(g.local_deviation_gt_0_02Ah.gt(0).sum())})
    return save_table(pd.DataFrame(summary),out,'early_slope_sensitivity')

def run(cells,t,out):
    evidence_tables(t,out)
    slope_sensitivity(cells,t,out)
    for func,args in [(feature_distributions,(t,)),(relative_delta,(cells,t)),
                      (collinearity,(t,)),(current_patterns,(cells,t))]:
        fig,_=func(*args,out);plt.close(fig)
