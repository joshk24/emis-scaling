import os
import numpy as np
import xarray as xr
import pandas as pd
import glob
import pyproj

lstfilter = True
proj4 = '+proj=lcc +lat_0=40.0 +lon_0=-97.0 +lat_1=33.0 +lat_2=45.0 +x_0=2952000.0 +y_0=2772000.0 +R=6370000.0 +to_meter=36000.0 +no_defs'

xr.set_options(keep_attrs=True)
MWAIR = 28.9628e-3  # kg/mol
lslice = slice(None, 14)
d1h = pd.to_timedelta('3600s')
droot = '/work/MOD3DEV/jkumm/EMBER/CMAQ/36US3'

# import environmental variables
start_date = os.environ.get("start_date")
end_date = os.environ.get("end_date")
current_iteration = os.environ.get("iteration")

print(f"Running script with:")
print(f" Start Date: {start_date}")
print(f" End Date: {end_date}")
print(f" Iteration no. : {current_iteration}")

current = int(current_iteration)
prev = current - 1

def iopen(path, varks, nvar=None):
    if nvar is None:
        nvar = len(varks)
    vslice = slice(None, nvar)
    rawf = xr.open_dataset(path, decode_cf=False, mode='rs')
    if 'TFLAG' in rawf:
        varks = ['TFLAG'] + varks
        f = rawf[varks].isel(LAY=lslice, VAR=vslice)
    else:
        f = rawf[varks].isel(lev=lslice).rename(**{
            'lev': 'LAY', 'time': 'TSTEP',
            'south-north': 'ROW', 'west-east': 'COL',
        })
    ref = pd.to_datetime(f'{f.SDATE:07d}T{f.STIME:06d}', format='%Y%jT%H%M%S')
    f.coords['TSTEP'] = [ref + i * d1h for i in range(f.sizes['TSTEP'])]
    f.coords['ROW'] = np.arange(f.sizes['ROW']) + 0.5
    f.coords['COL'] = np.arange(f.sizes['COL']) + 0.5
    f.attrs['crs_proj4'] = proj4
    return f

def popen(incpath):
    print(f"Processing {incpath}", flush=True)
    date = pd.to_datetime(incpath.split('.')[-1], format='%Y%m%d%H')
    mpath = f'{droot}/input/mcip/metcro3d_{date:%Y%m%d}.ncf'
    c100path = f'{droot}/output/2023fires.v2/NOASSIM/CMAQv54_cb6r5_ae7_aq.36US3.35.NOASSIM.CONC.{date:%Y%j}'
    c085path = f'{droot}/output/2023fires.v2/15noxcut/CMAQv54_cb6r5_ae7_aq.36US3.35.15noxcut.CONC.{date:%Y%j}'
    bkgpath = f'{droot}/output/2023fires.v2/ASSIM{prev}_FINAL/CMAQv54_cb6r5_ae7_aq.36US3.35.ASSIM{prev}_FINAL.CONC.{date:%Y%j}'
    # ocnpath = f'/work/ROMO/lrt/cmaq/36US3/land/ssmask.36US3.ncf'
    lwpath = f'/work/MOD3DEV/jkumm/EMBER/CMAQ/36US3/input/mcip/aqm.t12z.grdcro2d_20230501.ncf'
    tomipath = f'{droot}/scripts/cmaqsatproc/tropomino2/TropOMINO2_{date:%Y-%m-%d}_36US3.nc'
    cmaqpath = f'{droot}/scripts/cmaqsatproc/files/TropOMINO2_{date:%Y-%m-%d}_36US3_CMAQ.nc'

    # Define path to previous iteration scale factors
    oldpath = f'{droot}/run/final_scale_factors/SCALE{prev}_FINAL/DAILY_SCALE_{date:%Y%m%d}.nc'
    
    # Define file paths dynamically based on the date
    emispath = f'/work/MOD3DEV/jkumm/EMBER/CMAQ/36US3/output/2023fires.v2/BASE0/{date:%Y%j}'
    anthro_files = [
        f'{emispath}/CCTM_DESID1_ALL_CMAQv54_cb6r5_ae7_aq.36US3.35.BASE0.nc',
        f'{emispath}/CCTM_DESID1_CMV12_CMAQv54_cb6r5_ae7_aq.36US3.35.BASE0.nc',
        f'{emispath}/CCTM_DESID1_CMV3_CMAQv54_cb6r5_ae7_aq.36US3.35.BASE0.nc',
        f'{emispath}/CCTM_DESID1_EGU_CMAQv54_cb6r5_ae7_aq.36US3.35.BASE0.nc',
        f'{emispath}/CCTM_DESID1_NONEGU_CMAQv54_cb6r5_ae7_aq.36US3.35.BASE0.nc',
        f'{emispath}/CCTM_DESID1_PT_OILGAS_CMAQv54_cb6r5_ae7_aq.36US3.35.BASE0.nc',
        f'{emispath}/CCTM_DESID1_PT_OTH_CMAQv54_cb6r5_ae7_aq.36US3.35.BASE0.nc',
        f'{emispath}/CCTM_DESID1_RWC_CMAQv54_cb6r5_ae7_aq.36US3.35.BASE0.nc'
    ]
    fire_files = [
        f'{emispath}/CCTM_DESID1_PT_RXFIRES_CMAQv54_cb6r5_ae7_aq.36US3.35.BASE0.nc',
        f'{emispath}/CCTM_DESID1_PT_WILDFIRES_CMAQv54_cb6r5_ae7_aq.36US3.35.BASE0.nc',
        f'{emispath}/CCTM_DESID1_PT_FIRES_MXCA_CMAQv54_cb6r5_ae7_aq.36US3.35.BASE0.nc'
    ]
    natural_files = [
        f'{emispath}/CCTM_DESID1_BIOG_CMAQv54_cb6r5_ae7_aq.36US3.35.BASE0.nc',
        f'{emispath}/CCTM_DESID1_LTNG_CMAQv54_cb6r5_ae7_aq.36US3.35.BASE0.nc'
    ]
    if not os.path.exists(incpath):
        return None
  
    if not all(os.path.exists(p) for p in [mpath, c100path, c085path, bkgpath, tomipath, cmaqpath, lwpath, oldpath]):
        print(f"Missing input files for {incpath}", flush=True)
        return None
    if not all(os.path.exists(p) for p in anthro_files + fire_files + natural_files):
        print(f"Missing emissions files for {incpath}", flush=True)
        return None
    

    incf = iopen(incpath, ['INCNO2', 'NO2'])

    zf = iopen(mpath, ['DENS', 'ZF'], nvar=7).sel(TSTEP=incf.TSTEP)
    zf['DZ'] = zf['ZF'].copy()
    zf['DZ'][:, 1:] -= zf['DZ'][:, :-1].data

    c100f = iopen(c100path, ['NO2']).sel(TSTEP=incf.TSTEP)
    c085f = iopen(c085path, ['NO2']).sel(TSTEP=incf.TSTEP)
    bkgf = iopen(bkgpath, ['NO2']).sel(TSTEP=incf.TSTEP)
    l3f = xr.open_dataset(tomipath)['nitrogendioxide_tropospheric_column']
    qf = xr.open_dataset(cmaqpath)['VCDNO2_CMAQ_TOMI']
    lwf = xr.open_dataset(lwpath)['LWMASK']

    oldf = xr.open_dataset(oldpath)['SCALE']

    # calculating land mask (no adjustments over open ocean)
    land_mask = np.round(lwf.isel(LAY=0, TSTEP=0))

    l3vcd = l3f * 6.02214e+19
    qvcd = qf * 6.02214e+19
    #anthro_ds = xr.open_dataset(anthro_file)[['NO', 'NO2', 'HONO']]
    anthro_ds = [xr.open_dataset(file)[['NO', 'NO2', 'HONO']] for file in anthro_files]
    fire_ds = [xr.open_dataset(file)[['NO', 'NO2', 'HONO']] for file in fire_files]
    natural_ds = [xr.open_dataset(file)['NO'] for file in natural_files]
    
    NOX_CONVERSION_FACTOR = 3600 * 24 * 0.014
    def calculate_nox(dataset, variables, conversion_factor):
      return sum(dataset[var].mean(dim=('LAY', 'TSTEP')) for var in variables) * conversion_factor

    nox_anthro = sum(calculate_nox(anthro, ['NO', 'NO2', 'HONO'], NOX_CONVERSION_FACTOR) for anthro in anthro_ds)
    nox_fires = sum(calculate_nox(fire, ['NO', 'NO2', 'HONO'], NOX_CONVERSION_FACTOR) for fire in fire_ds)
    nox_natural = sum(natural.mean(dim=('LAY', 'TSTEP')) * NOX_CONVERSION_FACTOR for natural in natural_ds)

    def pvcd(f, zf, key, inkey='NO2', **attrs):
        zf[key] = (
            f[inkey] * 1e-6
            * zf['DENS'] / MWAIR
            * zf['DZ']
        )
        zf[key].attrs.update(
            long_name=key.ljust(16),
            units='mol/m2'.ljust(16),
            **attrs
        )

    pvcd(c100f, zf, 'PVCDNO2_100', var_desc='Partial VCD NO2 F(E*1.0, M)'.ljust(80))
    pvcd(c085f, zf, 'PVCDNO2_085', var_desc='Partial VCD NO2 F(E*.85, M)'.ljust(80))
    pvcd(bkgf, zf, 'PVCDNO2_BKG', var_desc='Partial VCD NO2 AF(E, M)'.ljust(80))
    pvcd(incf, zf, 'PVCDNO2_UPD', var_desc='Partial VCD NO2 Updated'.ljust(80))
    pvcd(incf, zf, 'PVCDNO2_INC', inkey='INCNO2', var_desc='Partial VCD NO2 Increment'.ljust(80))

    outf = zf[['TFLAG', 'PVCDNO2_100', 'PVCDNO2_085', 'PVCDNO2_BKG', 'PVCDNO2_UPD', 'PVCDNO2_INC']].sum('LAY', keepdims=True)
    dpvcd = (outf['PVCDNO2_085'] - outf['PVCDNO2_100'])
    beta = (-0.15 * outf['PVCDNO2_100'] / dpvcd).where(dpvcd != 0).fillna(0)
    q1, q2, q3 = np.asarray(beta.quantile([.25, .5, .75]), dtype='f')
    iqr = q3 - q1
    ub = q2 + iqr
    lb = max(q2 - iqr, 0)
    outf['BETA'] = beta.where(lambda x: x > lb).fillna(0).where(lambda x: x < ub).fillna(0)

    # Calculate SCALE values before applying any mask
    dpvcd = (outf['PVCDNO2_UPD'] - outf['PVCDNO2_BKG'])
    outf['SCALE'] = np.maximum(1 + (outf['BETA'] * dpvcd / outf['PVCDNO2_BKG']), 1e-6)

    # Calculate total NOx and fractions
    nox_total = nox_anthro + nox_fires + nox_natural
    anthro_fraction = xr.where(nox_total > 0, nox_anthro / nox_total, 0)
    fire_fraction = xr.where(nox_total > 0, nox_fires / nox_total, 0)

 # Combine all masks into a single mask
    combined_mask = ((anthro_fraction >= 0.5) | (fire_fraction >= 0.5)) & (land_mask > 0) & (np.maximum(qvcd, l3vcd) >= 1.2e15)  # Retained the second VCD mask

    # Apply the combined mask to the SCALE variable
    outf['SCALE'] = xr.where(combined_mask, outf['SCALE'], 1.0)

    print(f"SCALE {prev}: {oldf.isel(LAY=0, TSTEP=0).mean() * 1000}")
    print(f"SCALE {current}: {outf['SCALE'].mean()}")

    # Multiply by previous SCALE
    prev_scale = oldf.isel(LAY=0, TSTEP=0) * 1000  # Extract previous scale factor
    outf['SCALE'] = prev_scale * outf['SCALE']
    
    print(f"NEW SCALE: {outf['SCALE'].mean()}")
    print(f"Change: {outf['SCALE'].mean() - prev_scale.mean()}")

    # Smooth SCALE using a rolling mean
    scale_raw = outf['SCALE']
    scale_smoothed = outf['SCALE'].rolling(ROW=3, COL=3, center=True, min_periods=4).mean()

    # Combine raw and smoothed SCALE values
    outf['SCALE'] = 0.9 * scale_raw + 0.1 * scale_smoothed

    # Apply clip and fillna after smoothing
    outf['SCALE'] = outf['SCALE'].clip(min=0.5, max=2.0).fillna(1.0)

    # Finalize SCALE updates
    outf['SCALE'] = outf['SCALE'] / 1000  # Divide by 1000 after clip and fillna
    outf['SCALE'] = outf['SCALE'].transpose('TSTEP', 'LAY', 'ROW', 'COL')  # Ensure correct dimension order

    # Add the required attributes to SCALE
    outf['SCALE'].attrs['long_name'] = 'Scaling Factor'.ljust(16)
    outf['SCALE'].attrs['units'] = 'unitless'.ljust(16)  # Add units attribute
    outf['SCALE'].attrs['var_desc'] = 'Emission Scaling Factor'.ljust(80)  # Optional, for clarity

    # Ensure all other variables have required attributes
    for k in outf.data_vars:
        v = outf[k]
        v.attrs.setdefault('long_name', k.ljust(16))
        v.attrs.setdefault('units', 'unknown'.ljust(16))  # Default units if not already set

    varkeys = [k.ljust(16) for k in outf.data_vars if k not in 'TFLAG']
    outf.attrs['NVARS'] = np.int32(len(varkeys))
    outf.attrs['VAR-LIST'] = ''.join(varkeys)
    outf.attrs['NLAYS'] = np.int32(1)
    outf.attrs['VGLVLS'] = zf.VGLVLS[[0, -1]]
    return outf

dates = pd.date_range(start_date, end_date)

for date in dates:
    inctmpl = f'{droot}/gsi_scripts/ASSIM{prev}_FINAL/*/incno2.{date:%Y%m%d}??'
    incpaths = sorted(glob.glob(inctmpl))
    if len(incpaths) == 0:
        continue
    pf = None
    mypaths = []
    for incpath in incpaths:
        print(incpath, flush=True)
        f = popen(incpath)
        if f is None:
            continue
        mypaths.append(incpath.replace(droot, ''))
        if pf is None:
            pf = f
        else:
            pf = xr.concat([pf, f], dim='TSTEP')
    if pf is None:
        continue

    output_dir = f'./SCALE{current}_FINALv2'
    os.makedirs(output_dir, exist_ok=True)

    pf.attrs['FILEDESC'] = (
        f'{droot}:'.ljust(79) + '\n'
        + ''.join([
            (' - ' + p).ljust(79) + '\n' for p in mypaths
        ])
    )
    t25 = pd.date_range(date, date + pd.to_timedelta('86400s'), freq='1h')
    hf = pf.interp(TSTEP=t25, method="nearest", kwargs={"fill_value": "extrapolate"})
    hf['TFLAG'] = ('TSTEP', 'VAR', 'DATE-TIME'), pf['TFLAG'].isel(TSTEP=[0] * 25).data, pf['TFLAG'].attrs
    hf['TFLAG'][:, :, 0] = hf.TSTEP.dt.strftime('%Y%j').astype('i')
    hf['TFLAG'][:, :, 1] = hf.TSTEP.dt.strftime('%H%M%S').astype('i')
    hf.drop_vars(['ROW', 'COL', 'TSTEP']).to_netcdf(f'./{output_dir}/HOURLY_SCALE_{date:%Y%m%d}.nc')
    if lstfilter:
        Y, X = xr.broadcast(pf.ROW, pf.COL)
        proj = pyproj.Proj(pf.crs_proj4)
        LON, LAT = proj(X, Y, inverse=True)
        LON = X * 0 + LON
        DT = (LON / 15).astype('timedelta64[h]')
        LST_TSTEP = (pf.TSTEP + DT).transpose('TSTEP', 'ROW', 'COL')
        LST_H = LST_TSTEP.dt.strftime('%H').astype('f')
        LST_H += LST_TSTEP.dt.strftime('%M').astype('f') / 60.
        pf = pf.where((LST_H > 11.5) & (LST_H < 13.5))
        print(pf.mean())
    df = pf.mean('TSTEP', keepdims=True).drop_vars(['ROW', 'COL'])
    df['SCALE'] = df['SCALE'].fillna(0.001)
    df.attrs['TSTEP'] = np.int32(0)
    df['TFLAG'] = pf['TFLAG'].isel(TSTEP=slice(None, 1)).mean(dim=['ROW', 'COL'], keepdims=False)
    df['TFLAG'][:, :, :] = 0
    df['TFLAG'] = df['TFLAG'].astype('i')
    df.to_netcdf(f'./{output_dir}/DAILY_SCALE_{date:%Y%m%d}.nc')
    print(f'./{output_dir}/DAILY_SCALE_{date:%Y%m%d}.nc')

