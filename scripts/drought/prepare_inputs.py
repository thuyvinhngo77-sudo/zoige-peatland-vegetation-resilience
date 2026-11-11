"""Prepare traceable, physical-unit inputs; never modify the previous analysis."""
from pathlib import Path
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import xarray as xr
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'drought_reanalysis_20260916'
OUT = RUN / 'outputs'
CACHE = RUN / 'cache'
LEGACY = ROOT / 'agb_lai_resilience_breakpoint_main95_1988_2024'
PEAK = ROOT / 'agb_driver_xgboost_shap_reduced_2001_2024/outputs/peak_only_trajectory_panel_1988_2024.pkl.gz'
YEARS = np.arange(1988, 2025)

def grid_spec():
    g = json.loads((ROOT / 'agb_driver_xgboost_shap_v2/outputs/analysis_grid.json').read_text())
    g['transform'] = from_origin(g['bounds'][0], g['bounds'][3], 5000, 5000)
    return g

def warp(a, transform, crs, g):
    dst = np.full((g['height'], g['width']), np.nan, 'float32')
    reproject(a.astype('float32'), dst, src_transform=transform, src_crs=crs,
              src_nodata=np.nan, dst_transform=g['transform'], dst_crs=g['crs'],
              dst_nodata=np.nan, resampling=Resampling.average)
    return dst

def raw_climate_2011(g, variable):
    base = ROOT / 'climate_data/input'
    if variable == 'pre':
        p = base / 'China_Pre_1901_2024_from_CRUv4.09/China_Pre_1901_2024_from_CRUv4/pre_2011.nc'
    else:
        p = base / 'China_AI_1901_2024_from_CRUv4.09/China_AI_1901_2024_from_CRUv4/AI_annual_2011.nc'
    tr = Transformer.from_crs(g['crs'], 'EPSG:4326', always_xy=True)
    lon, lat = tr.transform([g['bounds'][0],g['bounds'][2]], [g['bounds'][1],g['bounds'][3]])
    with xr.open_dataset(p, decode_times=False) as ds:
        d = ds[variable].sel(lon=slice(min(lon)-.5,max(lon)+.5), lat=slice(max(lat)+.5,min(lat)-.5))
        attrs = dict(ds[variable].attrs)
        if variable == 'pre':
            # Matches the documented 0.1-mm source scaling in the existing workflow.
            d = d.isel(time=[6,7,8]).sum('time', skipna=False) * .1
        d = d.squeeze()
        a = d.values.astype('float32')
        xx, yy = d.lon.values, d.lat.values
    assert a.ndim == 2 and len(xx)>1 and len(yy)>1
    transform = from_origin(xx[0]-np.diff(xx).mean()/2, yy[0]+abs(np.diff(yy).mean())/2,
                            np.diff(xx).mean(), abs(np.diff(yy).mean()))
    return warp(a, transform, 'EPSG:4326', g), {'path':str(p),'variable':variable,'attrs':attrs}

def prepare():
    for p in (OUT,CACHE): p.mkdir(parents=True,exist_ok=True)
    dest = CACHE / 'physical_panel_1988_2024.pkl.gz'
    if dest.exists():
        print('Reusing prepared inputs',flush=True)
        return pd.read_pickle(dest)
    g = grid_spec()
    old = pd.read_pickle(LEGACY / 'cache/annual_state_panel_5km.pkl.gz')
    assert not old.duplicated(['cell_id','year']).any()
    support = old.groupby('cell_id',sort=False).agg(row=('row','first'),col=('col','first'),
                  x=('x','first'),y=('y','first'),peat_fraction=('peat_area_fraction','median'))
    support = support[support.peat_fraction>=.1].copy()
    support['area_km2'] = support.peat_fraction * 25
    support['block_id'] = (support.x//25000).astype(int).astype(str)+'_'+(support.y//25000).astype(int).astype(str)
    support.to_csv(OUT / 'fixed_spatial_support.csv')
    p = old[old.cell_id.isin(support.index)][['cell_id','year','lai','lai_valid_fraction','pre_jas_mm','ai']].copy()
    p = p.merge(support.reset_index(),on='cell_id',validate='many_to_one')
    meta=[]
    for var,col in [('pre','pre_jas_mm'),('ai','ai')]:
        arr, item = raw_climate_2011(g,var)
        idx=p.year.eq(2011)
        p.loc[idx,col]=arr[p.loc[idx,'row'],p.loc[idx,'col']]
        meta.append(item)
    peak = pd.read_pickle(PEAK).rename(columns={'peak_agb':'AGB','strict_valid_fraction':'agb_valid_fraction'})
    p = p.merge(peak[['cell_id','year','AGB','agb_valid_fraction']],on=['cell_id','year'],how='left',validate='one_to_one')
    comp = pd.read_csv(LEGACY / 'outputs/annual_input_completeness.csv')
    p=p.merge(comp[['year','platform','sensor_family','file_count','readable_completeness']],on='year',how='left',validate='many_to_one')
    p['LAI'] = p['lai']
    p['FVC']=np.nan
    p['fvc_valid_fraction']=np.nan
    for year in YEARS:
        path = ROOT / f'agb_resilience_section_3_3_1988_2025/rasters/fvc_90m/Zoige_FVC_PeakNDVI_3x3Median_90m_{year}.tif'
        with rasterio.open(path) as src:
            a=src.read(1,masked=True).filled(np.nan)
            a[(a<0)|(a>1)]=np.nan
            avg=warp(a,src.transform,src.crs,g)
            frac=warp(np.isfinite(a).astype('float32'),src.transform,src.crs,g)
        idx=p.year.eq(year)
        rows,cols=p.loc[idx,'row'].to_numpy(),p.loc[idx,'col'].to_numpy()
        p.loc[idx,'FVC']=avg[rows,cols]
        p.loc[idx,'fvc_valid_fraction']=np.minimum(1,frac[rows,cols]/p.loc[idx,'peat_fraction'].to_numpy())
        print(f'FVC strict-mask aggregation {year}',flush=True)
    p.sort_values(['cell_id','year']).to_pickle(dest,compression='gzip')
    rows=[]
    for year,d in p.groupby('year'):
        r={'year':year,'fixed_cells':len(d),'peat_area_km2':d.area_km2.sum()}
        for col in ['pre_jas_mm','ai']:
            valid=np.isfinite(d[col]); r[col]=np.average(d.loc[valid,col],weights=d.loc[valid,'area_km2'])
        rows.append(r)
    climate=pd.DataFrame(rows)
    climate['pre_jas_z']=(climate.pre_jas_mm-climate.pre_jas_mm.mean())/climate.pre_jas_mm.std(ddof=1)
    climate.to_csv(OUT / 'annual_climate_fixed_support.csv',index=False)
    quality=[]
    for year,d in p.groupby('year'):
        for var,cov,thr in [('AGB','agb_valid_fraction',.5),('FVC','fvc_valid_fraction',.5),('LAI','lai_valid_fraction',.1)]:
            ok=np.isfinite(d[var]) & (d[cov]>=thr)
            quality.append({'year':year,'variable':var,'valid_cells':int(ok.sum()),
                'valid_peat_area_km2':d.loc[ok,'area_km2'].sum(),'valid_area_fraction':d.loc[ok,'area_km2'].sum()/d.area_km2.sum(),
                'median_valid_fraction':d[cov].median(),'platform':d.platform.iloc[0] if var=='LAI' else 'Landsat'})
    pd.DataFrame(quality).to_csv(OUT / 'annual_data_coverage.csv',index=False)
    metadata={'created_by':'prepare_inputs.py','climate_2011_replaced':meta,'response_years':[1988,2024],
        'AGB_source':str(PEAK),'AGB_band_in_original_rasters':3,'LAI_support':'native NOAA approximately 5km, not peat-only subpixels',
        'FVC_support':'mean of existing strict-mask 90m FVC values per 5km grid',
        'climate_support':'fixed peat fractions, independent of annual vegetation coverage',
        'protocol_sha256':hashlib.sha256((RUN/'analysis_protocol.md').read_bytes()).hexdigest()}
    (OUT/'input_manifest.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    return p

if __name__=='__main__':
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    prepare()
