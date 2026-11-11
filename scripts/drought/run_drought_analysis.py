"""Climate-first event analysis with explicit denominator and censoring audits."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from prepare_inputs import prepare, ROOT, RUN, OUT, YEARS

VARIABLES = ['AGB','FVC','LAI']
N_BOOT = 2000
SCENARIOS = {
    'fixed_window': {},
    'same_sensor': {'same_sensor':True},
    'same_platform': {'same_sensor':True,'same_platform':True},
    'pre2': {'pre_length':2},
    'temporal_mean': {'reducer':'mean'},
    'strict_coverage': {'coverage':.8,'lai_coverage':.2},
    'include_dry_baseline': {'exclude_dry':False},
    'event_minimum': {'event_minimum':True},
}

def weighted_median(values, weights):
    v,w=np.asarray(values,float),np.asarray(weights,float)
    ok=np.isfinite(v)&np.isfinite(w)&(w>0)
    if not ok.any(): return np.nan
    v,w=v[ok],w[ok]; order=np.argsort(v,kind='stable'); v,w=v[order],w[order]
    return float(v[np.searchsorted(np.cumsum(w),w.sum()/2)])

def block_ci(frame,col,seed=187):
    d=frame[np.isfinite(frame[col])].copy()
    if d.empty: return np.nan,np.nan,np.nan,0
    d=d.sort_values(col,kind='stable')
    v,w=d[col].to_numpy(float),d.area_km2.to_numpy(float)
    codes,blocks=pd.factorize(d.block_id,sort=True)
    n=len(blocks); med=weighted_median(v,w)
    if n<5: return med,np.nan,np.nan,n
    rng=np.random.default_rng(seed)
    draws=rng.multinomial(n,np.full(n,1/n),size=N_BOOT)
    bw=draws[:,codes]*w
    cumulative=np.cumsum(bw,axis=1)
    idx=np.argmax(cumulative>=bw.sum(axis=1)[:,None]/2,axis=1)
    lo,hi=np.quantile(v[idx],[.025,.975])
    return med,float(lo),float(hi),n

def sensor(year,var,platforms,exact=False):
    if var=='LAI':
        if exact: return str(platforms.get(year,'unknown'))
        return 'AVHRR' if year<=2013 else ('VIIRS_SNPP' if year<=2022 else 'VIIRS_NOAA20')
    if year<=1998: return 'L5_early'
    if year<=2002: return 'L5_L7'
    if year<=2011: return 'L5_late'
    if year==2012: return 'L7_SLCoff'
    if year<=2020: return 'L8'
    return 'L8_L9'

def select_events():
    c=pd.read_csv(OUT/'annual_climate_fixed_support.csv')
    p20=float(c.pre_jas_mm.quantile(.2)); amid=float(c.ai.median())
    c['core_drought']=(c.pre_jas_z<=-1)&(c.ai>amid)
    c['expanded_dry']=(c.pre_jas_mm<=p20)&(c.ai>amid)
    # Preserve all strict events even for pathological empirical distributions.
    c['analysis_dry']=c.expanded_dry|c.core_drought
    c['precip_p20_mm']=p20; c['ai_median']=amid
    c['source_2011']='original_NetCDF'
    c.to_csv(OUT/'climate_event_screen.csv',index=False)
    groups=[]
    for yr in c.loc[c.analysis_dry,'year'].astype(int):
        if groups and yr==groups[-1][-1]+1: groups[-1].append(yr)
        else: groups.append([yr])
    events=[]
    for ys in groups:
        sub=c[c.year.isin(ys)]
        events.append({'id':str(ys[0]) if len(ys)==1 else f'{ys[0]}-{ys[-1]}',
            'start':ys[0],'end':ys[-1],'years':ys,
            'tier':'core' if sub.core_drought.any() else 'expanded_only',
            'minimum_precip_z':sub.pre_jas_z.min(),'maximum_ai':sub.ai.max()})
    pd.DataFrame(events).to_csv(OUT/'event_registry.csv',index=False)
    old=pd.read_csv(ROOT/'agb_lai_resilience_breakpoint_main95_1988_2024/outputs/drought_event_selection.csv')
    comp=c.merge(old[['year','pre_jas_mm','ai','pre_jas_z']],on='year',suffixes=('_new','_old'))
    comp.to_csv(OUT/'climate_old_new_audit.csv',index=False)
    print('Events:', [(e['id'],e['tier']) for e in events],flush=True)
    return c,events

def windows(event,events,dry,var,platforms,options):
    length=options.get('pre_length',3)
    pre=list(range(max(1988,event['start']-length),event['start']))
    if options.get('exclude_dry',True): pre=[y for y in pre if y not in dry]
    nxt=min([e['start'] for e in events if e['start']>event['end']]+[2025])
    end=min(event['end']+3,nxt-1,2024)
    post=list(range(event['end']+1,end+1))
    reason='complete_3yr' if len(post)==3 else ('new_dry_episode' if nxt<=event['end']+3 else 'series_end')
    phases=pre+event['years']+post
    full_cross=len({sensor(y,var,platforms) for y in phases})>1
    if options.get('same_sensor'):
        exact=options.get('same_platform',False)
        stages={sensor(y,var,platforms,exact) for y in event['years']}
        if len(stages)>1: return [],[],True,'event_spans_sensors'
        stage=next(iter(stages))
        pre=[y for y in pre if sensor(y,var,platforms,exact)==stage]
        valid_post=[]
        for y in post:
            if sensor(y,var,platforms,exact)!=stage:
                reason='sensor_transition'; break
            valid_post.append(y)
        post=valid_post
    return pre,post,full_cross,reason

def reduce_period(pivot, years, required, reducer='median'):
    if not years: return pd.Series(np.nan,index=pivot.index)
    a=pivot.reindex(columns=years)
    value=a.mean(axis=1) if reducer=='mean' else a.median(axis=1)
    return value.where(a.notna().sum(axis=1)>=required)

def first_attainment(values, baseline, target, post_years, event_end, sustained=False):
    """First observed attainment, not an exact continuous-time recovery estimate."""
    out=np.full(len(baseline),np.nan)
    for j,year in enumerate(post_years):
        valid=np.isfinite(values[:,j])&(values[:,j]>=target*baseline)
        if sustained:
            if j+1>=len(post_years) or post_years[j+1]!=year+1: continue
            valid &= np.isfinite(values[:,j+1])&(values[:,j+1]>=target*baseline)
        found=np.isnan(out)&valid
        # For the sustained rule this is the start, confirmed in the next year.
        out[found]=year-event_end
    return out

def analyze(panel,climate,events):
    dry=set(climate.loc[climate.analysis_dry,'year'])
    support=panel.drop_duplicates('cell_id').set_index('cell_id')[['row','col','area_km2','block_id']]
    platforms=panel.drop_duplicates('year').set_index('year').platform.to_dict()
    totals=support.area_km2.sum()
    summaries=[]; ratios_all=[]; annual=[]; recoveries=[]; rec_curves=[]; winrows=[]
    for scenario,options in SCENARIOS.items():
        for var in VARIABLES:
            covcol={'AGB':'agb_valid_fraction','FVC':'fvc_valid_fraction','LAI':'lai_valid_fraction'}[var]
            threshold=options.get('lai_coverage',.1) if var=='LAI' else options.get('coverage',.5)
            pp=panel.copy(); pp.loc[~(pp[covcol]>=threshold),var]=np.nan
            pp.loc[(pp[var]<0)|~np.isfinite(pp[var]),var]=np.nan
            pivot=pp.pivot(index='cell_id',columns='year',values=var).reindex(support.index).astype('float64')
            minimum=20 if var=='AGB' else .05
            for event in events:
                pre,post,cross,reason=windows(event,events,dry,var,platforms,options)
                base={'scenario':scenario,'variable':var,'event_id':event['id'],'tier':event['tier'],
                    'cross_sensor_fixed_window':cross,'post_window_years':len(post),'censor_reason':reason}
                winrows.append({**base,'pre_years':';'.join(map(str,pre)),'event_years':';'.join(map(str,event['years'])),
                    'post_years':';'.join(map(str,post)),'pre_window_years':len(pre)})
                reducer=options.get('reducer','median')
                b=reduce_period(pivot,pre,2,reducer)
                d=reduce_period(pivot,event['years'],len(event['years']),reducer)
                if options.get('event_minimum'):
                    d=pivot[event['years']].min(axis=1).where(pivot[event['years']].notna().all(axis=1))
                a=reduce_period(pivot,post,max(1,int(np.ceil(len(post)/2))),reducer)
                b=b.where(b>minimum); d=d.where(d>minimum)
                rr=support.copy(); rr['pre']=b; rr['during']=d; rr['post']=a
                rr['resistance']=d/b; rr['recovery']=a/d; rr['resilience']=a/b
                rr.loc[b.isna(),['resistance','recovery','resilience']]=np.nan
                rr.loc[d.isna(),['recovery','resilience']]=np.nan
                for k,v in base.items(): rr[k]=v
                rr['no_observed_decline']=rr.resistance>=1
                rr['small_decline']=(rr.resistance>=.9)&(rr.resistance<1)
                rr['decline_gt10pct']=rr.resistance<.9
                ratios_all.append(rr.reset_index())
                for metric in ['resistance','recovery','resilience']:
                    val=rr[np.isfinite(rr[metric])]
                    med,lo,hi,nb=block_ci(val,metric)
                    area=val.area_km2.sum()
                    row={**base,'metric':metric,'median':med,'ci_low':lo,'ci_high':hi,
                        'n_cells':len(val),'n_blocks':nb,'valid_area_km2':area,'valid_area_fraction':area/totals}
                    if metric=='resistance' and area:
                        for key in ['no_observed_decline','small_decline','decline_gt10pct']:
                            row[key+'_area_fraction']=val.loc[val[key],'area_km2'].sum()/area
                    summaries.append(row)
                if scenario not in ['fixed_window','same_sensor']: continue
                valid=rr.resistance.notna()
                for year in list(range(max(1988,event['start']-3),event['end']+1))+post:
                    if options.get('same_sensor') and sensor(year,var,platforms)!=sensor(event['start'],var,platforms): continue
                    ar=rr.loc[valid].copy(); ar['ratio']=pivot.loc[valid,year]/b[valid]
                    good=ar[np.isfinite(ar.ratio)]
                    med,lo,hi,nb=block_ci(good,'ratio')
                    annual.append({**base,'year':year,'relative_year':year-event['start'],
                        'ratio':med,'ci_low':lo,'ci_high':hi,'n_cells':len(good),
                        'cohort_area_km2':rr.loc[valid,'area_km2'].sum(),'observed_area_km2':good.area_km2.sum(),
                        'is_event':year in event['years'],'is_baseline':year in pre})
                cohort=rr[valid & rr.decline_gt10pct].copy()
                for target,sustain in [(0.90,False),(.95,False),(1.,False),(1.,True)]:
                    arr=pivot.loc[cohort.index].reindex(columns=post).to_numpy(float)
                    times=first_attainment(arr,cohort.pre.to_numpy(),target,post,event['end'],sustain)
                    last=pivot.loc[cohort.index,event['end']].to_numpy(float)
                    at_end=last>=target*cohort.pre.to_numpy()
                    if not sustain: times[at_end]=0
                    for i,(cell,row) in enumerate(cohort.iterrows()):
                        obs=np.isfinite(arr[i]); k=int(obs.sum())
                        if not sustain and at_end[i]: status='at_target_at_event_end'
                        elif np.isfinite(times[i]): status='attainment_observed'
                        elif not len(post): status='no_followup'
                        elif not obs.all(): status='incomplete_followup_no_observed_attainment'
                        else: status='not_attained_within_observed_window'
                        recoveries.append({**base,'cell_id':cell,'target_fraction':target,'sustained_two_years':sustain,
                            'first_observed_years':times[i],'status':status,'observed_post_years':k,
                            'has_missing_post_year':not obs.all(),'area_km2':row.area_km2,
                            'event_end_already_at_target':bool(at_end[i])})
                    area=cohort.area_km2.sum()
                    if not area: continue
                    for j,year in enumerate(post):
                        # Confirmation year counts for the two-year rule.
                        confirmed=(np.isfinite(times)&(times+(1 if sustain else 0)<=year-event['end']))
                        fully_observed=np.isfinite(arr[:,:j+1]).all(axis=1)
                        missing=~confirmed & ~fully_observed
                        notyet=~confirmed & fully_observed
                        w=cohort.area_km2.to_numpy()
                        rec_curves.append({**base,'target_fraction':target,'sustained_two_years':sustain,
                            'post_year':year-event['end'],'year':year,'declined_cohort_cells':len(cohort),
                            'declined_cohort_area_km2':area,'attained_fraction':w[confirmed].sum()/area,
                            'no_observed_attainment_complete_fraction':w[notyet].sum()/area,
                            'unknown_due_missing_fraction':w[missing].sum()/area})
        print(f'Analysis complete: {scenario}',flush=True)
    outputs={'event_metric_summary':summaries,'annual_response_trajectories':annual,
             'recovery_observations':recoveries,'conditional_recovery_curves':rec_curves,'event_windows':winrows}
    for name,records in outputs.items(): pd.DataFrame(records).to_csv(OUT/f'{name}.csv',index=False)
    cells=pd.concat(ratios_all,ignore_index=True)
    cells.to_pickle(OUT/'cell_event_metrics.pkl.gz',compression='gzip')
    cells.to_csv(OUT/'cell_event_metrics.csv',index=False)
    validate(cells,pd.DataFrame(rec_curves),pd.DataFrame(recoveries))

def validate(cells,curves,recovery):
    checks={}
    d=cells.dropna(subset=['resistance','recovery','resilience'])
    error=np.max(np.abs(d.resistance*d.recovery-d.resilience)) if len(d) else np.nan
    checks['ratio_identity_max_absolute_error']=float(error)
    assert error<1e-10
    sums=curves[['attained_fraction','no_observed_attainment_complete_fraction','unknown_due_missing_fraction']].sum(axis=1)
    assert np.allclose(sums,1)
    checks['recovery_partition_sums_to_one']=True
    for _,g in curves.groupby(['scenario','variable','event_id','target_fraction','sustained_two_years']):
        assert (np.diff(g.sort_values('post_year').attained_fraction)>=-1e-12).all()
    checks['cumulative_attainment_monotonic']=True
    no_obs=recovery.status.eq('no_followup')
    assert recovery.loc[no_obs,'first_observed_years'].isna().all()
    checks['no_followup_is_not_recovered']=True
    checks['bootstrap_replicates']=N_BOOT
    (OUT/'validation_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')

def main():
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    panel=prepare(); climate,events=select_events()
    analyze(panel,climate,events)

if __name__=='__main__': main()
