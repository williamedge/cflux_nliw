import os
import numpy as np
import xarray as xr
# import pandas as pd
import arviz as az
import cmocean.cm as cm
from fielddata.seawater_absorbtion import seawater_absorbtion
from fielddata.sw_svel import sw_svel
import wootils.filters as fl
from wootils.inpex23_filtfunc import nans_back
from wootils.plotnice import horz_stack
# from datetime import datetime
from afloat.pca import rotate_2D, PCA_2D



def psi(R, R_crit):
    psi_val = (1 + 1.35*(R/R_crit) + (2.5*(R/R_crit))**3.2) / (1.35*(R/R_crit) + (2.5*(R/R_crit))**3.2)
    return psi_val


def load_traces_new(workDir, fieldtrip):
    trace_1 = az.from_netcdf(os.path.join(workDir, 'analysis_files', 'insitu_cal',\
                            (fieldtrip + '_SIG_NTU0359_trace_arviz.nc')))
    trace_2 = az.from_netcdf(os.path.join(workDir, 'analysis_files', 'lab_cal',\
                            (fieldtrip + '_NTU0359_Lab_trace_arviz.nc')))
    return trace_1, trace_2


def load_ntu(workDir, ntu_sn, mooring, fieldtrip='RS2019', year=2019):

    # Set the NTU data dir
    ntu_dir = os.path.join(workDir, 'data', 'field', fieldtrip, 'processed_data')

    ntu_file = os.path.join(ntu_dir, mooring, 'NTU', (ntu_sn + '.nc'))
    ds_ntu = xr.open_dataset(ntu_file, decode_times=True)
    ds_ntu.close()
    
    # Calculate the days between 1970-01-01 and 2019-01-01
    dsince = np.datetime64(str(year) + '-01-01') - np.datetime64('1970-01-01')
    dsince = dsince / np.timedelta64(1, 'D')
    ds_ntu['Datenum'] = ds_ntu['Datenum'].values - dsince
    ds_ntu['Datenum'] = np.datetime64('2019-01-01') +\
                            np.array(ds_ntu['Datenum'].values*24*60*60, dtype='timedelta64[s]')
    ntu_flag = ds_ntu['MasterBadFlag']==0
    return ds_ntu['NTU'][ntu_flag]


def rotate_event(u, v):
    if len(np.shape(u)) != len(np.shape(v)):
        raise ValueError('u and v must have the same dimensions')
    # Convert to numpy from xarray
    if isinstance(u, xr.DataArray):
        u_np = u.values
        v_np = v.values
    else:
        u_np = u
        v_np = v
    if len(np.shape(u_np)) > 1:
        u_np = u_np.flatten()
        v_np = v_np.flatten()
    theta_w, _, _, _ = PCA_2D(u_np, v_np)
    v_r, u_r = rotate_2D(v_np, u_np, -theta_w)
    # Convert back to xarray
    if isinstance(u, xr.DataArray):
        u_r = xr.DataArray(u_r.reshape(u.shape), dims=u.dims, coords=u.coords)
        v_r = xr.DataArray(v_r.reshape(u.shape), dims=v.dims, coords=v.coords)
    return u_r, v_r


def rotate_data(ds):
    pca_BT = PCA_2D(ds['enu'].isel(cartesian_axes=0).mean(dim='height'), ds['enu'].isel(cartesian_axes=1).mean(dim='height'))
    return rotate_2D(ds['enu'].isel(cartesian_axes=0), ds['enu'].isel(cartesian_axes=1), -pca_BT[0]), -pca_BT[0]


def confirm_front(ssc_obj, idxs, nbeams=4, thin=[24,4], vmax=30):
    fig, ax = horz_stack(nbeams)
    for ix, x in enumerate(ax):
        ssc_obj.isel(beam=ix, time=np.arange(idxs[0], idxs[1]))[::thin[0],::thin[1]].T.plot(ax=x, cmap=cm.turbid,\
                                                                       add_colorbar=False,\
                                                                       vmax=vmax)
        x.set_xlabel('')
        x.set_ylabel('')
        x.set_title('b' + str(ix+1))
        x.set_xticks([])
        if ix > 0:
            x.set_yticks([])
    return fig, ax



def load_c(echo, ds_temp, workDir, fieldtrip, bs_cutoff=None, bs_corrected=False):

    trace_2, trace_3 = load_traces_new(workDir, fieldtrip)

    P_atm = 15.6821  # RS2019 pressure at 148 m BSL [ATM], including 1 ATM from ambient

    S = 35.3 # [psu]

    # beam frequency (Hz)
    f = 1000*1000

    # transducer radius
    at = 1.75

    R = np.copy(echo['height'].values) # - 0.33

    temp_ix = (ds_temp['time'].values >= echo.time.values[0]) &\
                (ds_temp['time'].values <= echo.time.values[-1])

    # Check if xarray dataset
    if isinstance(ds_temp, xr.Dataset):
        ds_temp = ds_temp['Temperature']
    ind_nan = np.any(np.isnan(ds_temp.values[1:,temp_ix]), axis=0)  

    a_zi = ds_temp[1:,temp_ix][:,~ind_nan].interp(time=echo['time'].values, \
                                                            depth=echo['height'].values,\
                                                            method='nearest',\
                                                                kwargs={"fill_value": 'extrapolate'})
    
    # Estimate the variable absorbtion
    alpha_var = seawater_absorbtion(f, a_zi, P_atm)

    # Calculate the 2-way dB change
    absorb_var = 2 * alpha_var.T * R

    if echo.shape[1] == 1:
        sig_atten = echo + absorb_var.values
    else:
        # try to broadcast the absorb_var to the echo shape
        try:
            sig_atten = echo + absorb_var.values.T
        except:
            sig_atten = echo + np.repeat(np.squeeze(absorb_var.values)[:,:,np.newaxis], echo.shape[-1], axis=-1)
        # sig_atten = echo + np.repeat(np.squeeze(absorb_var.values)[:,:,np.newaxis], echo.shape[-1], axis=-1)

    ## Calculate beam spreading with nearfield correction
    # Calculate the speed of sound
    T_Av = np.nanmean(a_zi, axis=0)                                 # average T over height
    S_Av = S                                                        # Contant salinity
    int_depth = 149.7                                               # depth of instrument
    cdash = sw_svel(S_Av, T_Av, int_depth)                          # m/s

    # Calculate acoustic wavelength
    gamma = cdash / f

    # Calculate R critical
    R_crit = (np.pi * (at/100)**2) / np.nanmean(gamma)
    print('Nearfield correction applied to cells within ' + str(np.round(R_crit,2)) + ' m')

    # Calculate near-field correction
    psi_val = psi(R, R_crit)

    beam_psi = 20 * np.log10(R*psi_val)
    # beam_psi = 20 * np.log10(R)

    beam_adj = beam_psi
    beam_adj[beam_psi<0] = 0

    if len(sig_atten.shape) > 2:
        beam_near = sig_atten + np.tile(beam_adj, (echo.shape[-1],1)).T 
    else:
        beam_near = sig_atten.T + beam_adj   
    beam_near = beam_near.rename('turb_BS')

    if bs_cutoff is not None:
        cut_ind = echo.values.T < bs_cutoff
        cut_hgt = 1.3

        beam_n = beam_near.copy()

        if len(sig_atten.shape) < 3:
            for ix, sdd in enumerate(beam_n.height):
                if sdd < cut_hgt :
                    t_nan = beam_n[:,ix]
                    t_nan[~cut_ind[:,ix]] = np.nan
                    beam_n[:,ix] = t_nan 
                # beam_n[:,beam_n.height < cut_hgt] = np.nan

    else:
        beam_n = beam_near.copy()

    ntu_b = np.mean(trace_3.posterior['a_intercept'])
    ntu_m = np.mean(trace_3.posterior['b_slope'])

    sig_b = np.mean(trace_2.posterior['a_intercept'])
    sig_m = np.mean(trace_2.posterior['b_slope'])

    sig_log = beam_n * sig_m + sig_b
    sig_ntu = 10**sig_log
    sig_ssc = sig_ntu * (ntu_m/2.33) + ntu_b

    sig_val = sig_ssc.values
    sig_val[sig_val<0] = 0.0

    sig_ssc = sig_ssc.rename('turb_SSC')
    if len(sig_atten.shape) < 3:
        try:
            sig_ssc['turb_SSC'] = (('time', 'height'), sig_val)
        except:
            sig_ssc['turb_SSC'] = (('time', 'height'), sig_val.T)
    else:
        sig_ssc['turb_SSC'] = (('time', 'height', 'beam'), sig_val)

    if bs_corrected:
        return sig_ssc, beam_n.values
    else:
        return sig_ssc



def get_sigfiles(wav_tx, wav_bef, wav_aft, nc_files):
    sig_fil = []
    for wt in wav_tx:
        for nx, nc in enumerate(nc_files):
            ds = xr.open_dataset(nc)
            ds.close() 

            if ((wt - wav_bef >= ds.time.values[0]) & (wt - wav_bef <= ds.time.values[-1]))\
                & (wt + wav_aft <= ds.time.values[-1]):
                sig_fil.append(os.path.split(nc)[1])
                break
            elif ((wt - wav_bef >= ds.time.values[0]) & (wt - wav_bef <= ds.time.values[-1]))\
                & (wt + wav_aft > ds.time.values[-1]):
                sig_fil.append([os.path.split(nc)[1], os.path.split(nc_files[nx+1])[1]])
                break
        else:
            continue
    return sig_fil


def decompose_xarray(x_arr, da_lim, filt_low=34, filt_high=6, max_tgap=10, max_zgap=5, t_dim='time', z_dim='z'):

    # Get timestep in minutes
    ds_tstep = np.median(np.diff(x_arr[t_dim].values)) / np.timedelta64(1, 'm')

    # Interpolate small gaps and zero-pad the rest
    data_na = x_arr.interpolate_na(dim=t_dim, max_gap=np.timedelta64(max_tgap,'m'))
    data_na = x_arr.interpolate_na(dim=z_dim, max_gap=max_zgap)

    # requires conversion back to numpy for 2D indexing
    data_np = data_na.values
    data_np[np.isnan(data_np)] = 0.0

    # back to xarray
    data_zp = xr.DataArray(data=data_np, coords={z_dim:x_arr[z_dim].values, t_dim:x_arr[t_dim].values})

    u_drift = xr.apply_ufunc(fl.filter1d, data_zp, filt_low, ds_tstep, 'lowpass',\
                input_core_dims=[[t_dim],[],[],[]], output_core_dims=[[t_dim]],\
                vectorize=True)
    
    u_bchf = xr.apply_ufunc(fl.filter1d, data_zp, filt_high, ds_tstep, 'highpass',\
                input_core_dims=[[t_dim],[],[],[]], output_core_dims=[[t_dim]],\
                vectorize=True)
    
    u_tide_comb = data_zp - u_drift - u_bchf

    # Put nans back in before depth-averaging
    u_tide_comb = nans_back(u_tide_comb, x_arr)

    u_bchf = nans_back(u_bchf, x_arr)

    u_tide = u_tide_comb[u_tide_comb[z_dim]>da_lim, :].mean(dim=z_dim)
    u_itide = u_tide_comb - u_tide

    return u_tide, u_itide, u_drift, u_bchf





# def load_traces(workDir, fieldtrip):

#     # Load data for model SIG_OBS
#     ntu_sn = '0359'

#     ds_fname = os.path.join(workDir, 'analysis_files', 'insitu_cal',\
#                             (fieldtrip + '_insitu_cal_SIG_NTU' + ntu_sn + '.nc'))
#     ds_t = xr.open_dataset(ds_fname)

#     m_one_x = ds_t['sig_echo'].values
#     m_one_y = ds_t['lis_conc'].values

#     inds = m_one_x.argsort()
#     df = pd.DataFrame({'sig_amp': m_one_x[inds], 'ntu_log':m_one_y[inds]})

#     with pm.Model() as model:

#         # Convert the import data to theano.tensor types
#         xin = pm.Data('x_data', m_one_x[inds])
#         yin = pm.Data('y_data', m_one_y[inds])

#         a = pm.Normal('a_intercept', mu=0, sigma=10)
#         b = pm.Normal('b_slope', mu=0, sigma=10)

#         # Define the uncertainty on the Y data
#         sigma_y = pm.HalfNormal('sigma_y', 10)

#         # Define the mean function (linear regression)
#         y_est = a + b * xin

#         # Define the model for the likelihood
#         likelihood = pm.Normal('likelihood', mu=y_est, sigma=sigma_y, observed=yin)

#     trace_fname = 'RS2019_SIG_NTU0359_trace'
#     trace_full = os.path.join(workDir, 'analysis_files', 'insitu_cal', trace_fname)
#     with model:
#         # trace_2 = pm.backends.ndarray.load_trace(trace_full)
#         trace_2 = pm.load_trace(trace_full)
#     print('Trace 2 loaded') 

#     runno = 'Run_1'

#     # Load data for model
#     df = pd.read_pickle(os.path.join(workDir, 'analysis_files', 'lab_cal_files',\
#                         (fieldtrip + '_Cal_NTU' + ntu_sn + '_' + runno + '_alldata.pkl')))

#     with pm.Model() as model_3:

#         xin_3 = pm.Data('x_data', df['NTU'].values)
#         yin_3 = pm.Data('y_data', df['mass_conc'].values)

#         a = pm.Normal('a_intercept', mu=0, sd=10)
#         b = pm.Normal('b_slope', mu=0, sd=10)
#         sigma_y = pm.HalfNormal('sigma_y', 1)
#         y_est = a + b * xin_3
#         likelihood = pm.Normal('likelihood', mu=y_est, sd=sigma_y, observed=yin_3)

#     trace_fname = (fieldtrip + '_NTU' + ntu_sn + '_Lab_trace')
#     trace_full = os.path.join(workDir, 'analysis_files', 'lab_cal', trace_fname)

#     with model_3:
#         trace_3 = pm.load_trace(trace_full)
#     print('Trace 3 loaded')

#     return trace_2, trace_3