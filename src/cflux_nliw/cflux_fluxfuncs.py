import os
import numpy as np
import xarray as xr
from afloat.pca import PCA_2D, rotate_2D
from d2spike.utils import nan_gauss_xr, nan_gauss
from cflux_basefuncs import load_traces_new
from cflux_basefuncs import rotate_event



def reynolds_decomp(vel2D, sigma):
    if isinstance(vel2D, xr.DataArray):
        v_mean = nan_gauss_xr(vel2D, sigma)
    elif isinstance(vel2D, np.ndarray):
        v_mean = nan_gauss(vel2D, sigma)
    else:
        raise ValueError('vel2D must be either xarray or numpy array')
    v_turb = vel2D - v_mean
    return v_mean, v_turb


def adcp_theta(u_bchf, v_bchf, ds_sig):
    txx = (u_bchf['time'] > ds_sig['time'].values[0]) & (u_bchf['time'] <= ds_sig['time'].values[-1])
    rowx = np.any(np.isnan(u_bchf.values[:,txx]), axis=1)
    colx = np.any(np.isnan(v_bchf.values[:,txx][~rowx,:]), axis=0)
    pca_temp = PCA_2D(u_bchf[:,txx][~rowx,~colx].values.flatten(),\
                      v_bchf[:,txx][~rowx,~colx].values.flatten(),\
                      ellipse_stdevs=1)
    return pca_temp


# def sig_theta(ds, skp=4*60):
#     ds_u = ds['enu'].sel(cartesian_axes=1).isel(height=ds.height>2.7).mean(dim='height')[::skp]
#     ds_v = ds['enu'].sel(cartesian_axes=2).isel(height=ds.height>2.7).mean(dim='height')[::skp]
#     u_mean = ds_u.mean(dim='time').values
#     v_mean = ds_v.mean(dim='time').values
#     pca_temp = PCA_2D(ds_u - u_mean, ds_v - v_mean, ellipse_stdevs=1)
#     return pca_temp 


def get_sums(x, ds):
    t_l = calc_flux(ds, x)
    bx_h = np.diff(t_l[0].height.values)[0]
    nanx = np.any(np.any(np.isnan(np.stack(t_l)), axis=0), axis=-1)
    dc_dt_sum,_,_ = height_integrals(t_l[0].values[...,np.newaxis][~nanx,:], bx_h)
    ba_hg_med, _, _ = height_integrals((t_l[1].values +
                                            t_l[2].values +
                                            t_l[3].values)[...,np.newaxis][~nanx,:], 
                                            bx_h)
    return dc_dt_sum - ba_hg_med


def calc_flux(ds, c_wave):

    u_r_mean, v_r_mean = rotate_event(ds['u_mean'], ds['v_mean'])

    t_step = np.diff(ds['time'].values)[0].astype('timedelta64[s]').astype('float')

    # First the horizontal velocity
    uc_z_mean = t_step * u_r_mean * ds['c_mean'].differentiate('time', datetime_unit='s') / c_wave

    # Now the mean vertical vel
    wc_z_mean = t_step * ds['w_mean'] * -1*ds['c_mean'].differentiate('height')

    # And the Reynolds flux
    wc_z_turb = t_step * -1*ds['wc_turb'].differentiate('height')

    # Also the time derivative of C
    dc_dt = t_step * ds['c_mean'].differentiate('time', datetime_unit='s')

    return [dc_dt, wc_z_mean, wc_z_turb, uc_z_mean]




def calc_flux_UQ(ds, w_turb, c_samples, c_samples_thin, c_wave, filt_z=4, filt_x=300, thin_v=2, thin_h=10):

    t_step = np.diff(ds['time'].values)[0].astype('timedelta64[s]').astype('float')

    # First the horizontal velocity
    uc_z_mean = t_step * -1*ds['u_mean'] * c_samples_thin.differentiate('time', datetime_unit='s') / c_wave

    # Now the mean vertical vel
    wc_z_mean = t_step * ds['w_mean'] * -1*c_samples_thin.differentiate('height')
    print('wc_z_mean calculated')

    # And the Reynolds flux
    print(w_turb.shape)
    print(c_samples.shape)
    # wc_turb_raw = w_turb.values * c_samples.values
    # print(wc_turb_raw.shape)
    wc_turb = np.full_like(c_samples.values[::thin_v, ::thin_h, :][:-1,:,:], np.nan)
    print(wc_turb.shape)

    for ii in range(c_samples.shape[-1]):
        wc_turb[:,:,ii] = nan_gauss(w_turb.values * c_samples.values[:,:,ii],\
                                    [filt_z,filt_x], axis=[0,1])[::thin_v, ::thin_h][:-1,:]
    print(wc_turb.shape)

    # wc_turb = wc_turb[::thin_v, ::thin_h][:-1,:,:]
    wc_turb_xr = xr.DataArray(data=wc_turb, dims=['height', 'time', 'sample'],\
                              coords={'height': w_turb['height'].values[::thin_v][:-1],\
                                      'time': w_turb['time'].values[::thin_h],\
                                      'sample': np.arange(wc_turb.shape[-1])})
    wc_z_turb = t_step * -1*wc_turb_xr.differentiate('height')
    print('wc_z_turb calculated')

    # Also the time derivative of C
    dc_dt = t_step * c_samples_thin.differentiate('time', datetime_unit='s')

    return [dc_dt, wc_z_mean, wc_z_turb, uc_z_mean]  


def sample_fluxterms(ds_flux, beam5_turb, ds_echo, backscatter, n_samples=10,\
                     work_dir=None, field_trip='RS2019', thin_v=2, thin_h=10,\
                     gf_h=4, gf_t=300):
    
    if work_dir is None:
        work_dir = r'/mnt/c/Users/00099894/OneDrive - The University of Western Australia/UWA/PhD'

    _, w_turb = reynolds_decomp(beam5_turb, [gf_h,gf_t])

    trace_2, trace_3 = load_traces_new(os.path.join(work_dir, 'pl'), field_trip)

    trace_2.posterior['sigma_y'] = trace_2.posterior['sigma_y']/5
    sig_ntu = sample_ppc_full_new(trace_2.posterior, backscatter.T, n_samples)
    trace_3.posterior['b_slope'] = trace_3.posterior['b_slope']/2.33
    trace_3.posterior['sigma_y'] = trace_3.posterior['sigma_y']
    ntu_ssc = sample_ppc_full_new(trace_3.posterior, 10**sig_ntu, 1)

    ssc_samples_xr = xr.DataArray(data=ntu_ssc, dims=['height', 'time', 'sample'],\
                                coords={'height': ds_echo['height'],\
                                        'time': ds_echo['time'],\
                                        'sample': np.arange(n_samples)})
    ssc_samples_xr = ssc_samples_xr.where(ssc_samples_xr > 0, 0)
    print('SSC sampled')

    ssc_prime = np.full_like(ssc_samples_xr.values, np.nan)
    for ii in range(ssc_samples_xr.shape[-1]):
        ssc_prime[:,:,ii] = ssc_samples_xr.values[:,:,ii] - nan_gauss(ssc_samples_xr.values[:,:,ii], [gf_h,gf_t], axis=[0,1])
    ssc_prime_xr = xr.DataArray(data=ssc_prime, dims=['height', 'time', 'sample'],\
                            coords={'height': ssc_samples_xr['height'].values,\
                                    'time': ssc_samples_xr['time'].values,\
                                    'sample': ssc_samples_xr['sample'].values})
    print('SSC\' calculated')

    # Reshape dimensions
    ssc_samples_xr_thin = ssc_samples_xr[::thin_v, ::thin_h, :][:-1,:,:]

    s_est_samples = np.random.normal(ds_flux.attrs['est_celerity'], ds_flux.attrs['obs_celerity'][1], size=n_samples)

    terms_list = calc_flux_UQ(ds_flux, w_turb, ssc_prime_xr, ssc_samples_xr_thin, s_est_samples)   

    return  terms_list, s_est_samples



def calc_settling(ds, ws=0.0006):
    t_step = np.diff(ds['time'].values)[0].astype('timedelta64[s]').astype('float')
    wc_ws = t_step * (ds['w_mean'] + ws) * -1*ds['c_mean'].differentiate('height')
    return wc_ws


def range_bounds(flux_term):

    dcdt_var = np.median(flux_term, axis=-1)

#     dcdt_var = np.nanvar(np.nansum(flux_term, axis=0), axis=0)
#     print(dcdt_var.shape)
    
#     # Find median variance
#     dcdt_ix = list(dcdt_var).index(np.percentile(dcdt_var, 50, interpolation='nearest'))
#     print(dcdt_ix)
    dcdt_rg_med = np.percentile(dcdt_var, 97.5, axis=0) - \
                    np.percentile(dcdt_var, 2.5, axis=0)
    
#     dcdt_ix = list(dcdt_var).index(np.percentile(dcdt_var, 2.5, interpolation='nearest'))
#     dcdt_rg_low = np.percentile(flux_term[:,:,dcdt_ix], 97.5, axis=0) - \
#                     np.percentile(flux_term[:,:,dcdt_ix], 2.5, axis=0)
    
#     dcdt_ix = list(dcdt_var).index(np.percentile(dcdt_var, 97.5, interpolation='nearest'))
#     dcdt_rg_upp = np.percentile(flux_term[:,:,dcdt_ix], 97.5, axis=0) - \
#                     np.percentile(flux_term[:,:,dcdt_ix], 2.5, axis=0)
    
    return dcdt_rg_med#, dcdt_rg_low, dcdt_rg_upp

def time_integrals(flux_term, t_step):

    dcdt_tint = np.sum(flux_term, axis=0) * t_step

    dcdt_tint_med = np.median(dcdt_tint, axis=-1)
    dcdt_tint_low = np.percentile(dcdt_tint, 2.5, axis=-1)
    dcdt_tint_upp = np.percentile(dcdt_tint, 97.5, axis=-1)
    
    return dcdt_tint_med, dcdt_tint_low, dcdt_tint_upp

def height_integrals(flux_term, hg_step):

    dcdt_tint = np.sum(flux_term, axis=0) * hg_step

    dcdt_tint_med = np.median(dcdt_tint, axis=-1)
    dcdt_tint_low = np.percentile(dcdt_tint, 2.5, axis=-1)
    dcdt_tint_upp = np.percentile(dcdt_tint, 97.5, axis=-1)
    
    return dcdt_tint_med, dcdt_tint_low, dcdt_tint_upp


def sample_ppc_full_new(trace_var, eval_data, sample_len):
                
    # Loop through and sample the data
    # lm = lambda x, samp: samp['a_intercept'].values + samp['b_slope'].values * x +\
    #       np.random.normal(loc=0, scale=samp['sigma_y'].values)
    lm = lambda x, samp, sig: samp['a_intercept'].values + samp['b_slope'].values * x # + sig
                      
    # Initialize the sampling variable with nans
    if hasattr(eval_data, '__len__'):

        if eval_data.ndim == 1:
            ppc_full = np.full((len(eval_data), sample_len), np.nan)
            rand_chain = np.random.randint(0, high=len(trace_var.chain.values), size=sample_len)
            rand_draw = np.random.randint(0, high=len(trace_var.draw.values), size=sample_len)

            for iy, (rc, ri) in enumerate(zip(rand_chain, rand_draw)): 
                rand_sample = trace_var.isel(chain=rc, draw=ri)
                rand_sig = np.random.normal(loc=0, scale=rand_sample['sigma_y'].values, size=1)
                ppc_full[:, iy] = lm(eval_data, rand_sample, rand_sig) 
            
            
        elif (eval_data.ndim == 2) & (sample_len == 1):
            ppc_full = np.full_like(eval_data, np.nan)
            rand_chain = np.random.randint(0, high=len(trace_var.chain.values), size=sample_len)
            rand_draw = np.random.randint(0, high=len(trace_var.draw.values), size=sample_len)

            # for iy, (rc, ri) in enumerate(zip(rand_chain, rand_draw)): 
            rand_sample = trace_var.isel(chain=rand_chain, draw=rand_draw)
            rand_sig = np.random.normal(loc=0, scale=rand_sample['sigma_y'].values, size=1)
            ppc_full = lm(eval_data, rand_sample, rand_sig)            
            
            
        elif (eval_data.ndim == 2) & (sample_len!=1):
            ppc_full = np.full((np.shape(eval_data)[0], np.shape(eval_data)[1], sample_len), np.nan)
            rand_chain = np.random.randint(0, high=len(trace_var.chain.values), size=sample_len)
            rand_draw = np.random.randint(0, high=len(trace_var.draw.values), size=sample_len)
            
            for iz, (rc, ri) in enumerate(zip(rand_chain, rand_draw)): 
                rand_sample = trace_var.isel(chain=rc, draw=ri)                
                rand_sig = np.random.normal(loc=0, scale=rand_sample['sigma_y'].values, size=1)
                ppc_full[:,:, iz] = lm(eval_data, rand_sample, rand_sig)
                
            
        elif (eval_data.ndim == 3) & (sample_len==1):
            ppc_full = np.full_like(eval_data, np.nan)
            rand_chain = np.random.randint(0, high=len(trace_var.chain.values), size=np.shape(eval_data)[-1])
            rand_draw = np.random.randint(0, high=len(trace_var.draw.values), size=np.shape(eval_data)[-1])
            
            for iz, (rc, ri) in enumerate(zip(rand_chain, rand_draw)): 
                rand_sample = trace_var.isel(chain=rc, draw=ri)                
                rand_sig = np.random.normal(loc=0, scale=rand_sample['sigma_y'].values, size=1)
                ppc_full[:,:,iz] = lm(eval_data[:,:,iz], rand_sample, rand_sig)

    return ppc_full


def get_solibore_times():
    # I did this by hand and later regretted it
    wav_tx = np.array([np.datetime64('2019-03-07T13:50:00'),\
                    #    np.datetime64('2019-03-07T21:10:00'),\
                    np.datetime64('2019-03-09T12:20:00'),\
                    np.datetime64('2019-03-09T16:10:00'),\
                    np.datetime64('2019-03-12T14:36:00'),\
                    np.datetime64('2019-03-16T05:15:00'),\
                    np.datetime64('2019-03-18T11:37:00'),\
                    np.datetime64('2019-03-19T13:14:00'),\
                    np.datetime64('2019-03-20T01:45:00'),\
                    np.datetime64('2019-03-20T12:18:00'),\
                    np.datetime64('2019-03-20T23:56:00'),\
                    np.datetime64('2019-03-22T11:42:00'),\
                    np.datetime64('2019-03-23T12:45:00'),\
                    np.datetime64('2019-03-24T16:26:00'),\
                    np.datetime64('2019-03-25T01:50:00'),\
                    np.datetime64('2019-03-25T05:40:00'),\
                    np.datetime64('2019-03-25T17:35:00'),\
                    np.datetime64('2019-03-26T05:50:00'),\
                    np.datetime64('2019-03-26T15:45:00'),\
                    np.datetime64('2019-03-26T17:05:00'),\
                    np.datetime64('2019-03-27T05:55:00'),\
                    np.datetime64('2019-03-27T18:15:00'),\
                    np.datetime64('2019-03-28T07:00:00'),\
                    np.datetime64('2019-03-28T17:20:00'),\
                    np.datetime64('2019-03-28T19:00:00'),\
                    np.datetime64('2019-03-29T09:10:00'),\
                    np.datetime64('2019-03-29T17:00:00'),\
                    np.datetime64('2019-03-29T19:00:00'),\
                    np.datetime64('2019-03-30T09:20:00'),\
                    np.datetime64('2019-03-30T21:00:00'),\
                    np.datetime64('2019-03-31T09:15:00'),\
                    np.datetime64('2019-04-01T00:10:00'),\
                    np.datetime64('2019-04-01T11:10:00'),\
                    np.datetime64('2019-04-02T00:25:00'),\
                    np.datetime64('2019-04-02T11:50:00'),\
                    np.datetime64('2019-04-03T01:10:00'),\
                    np.datetime64('2019-04-03T12:20:00'),\
                    np.datetime64('2019-04-04T00:25:00'),\
                    np.datetime64('2019-04-04T13:10:00'),\
                    np.datetime64('2019-04-04T22:30:00'),\
                    np.datetime64('2019-04-05T01:15:00'),\
                    np.datetime64('2019-04-05T13:10:00'),\
                    np.datetime64('2019-04-05T23:15:00'),\
                    np.datetime64('2019-04-06T01:10:00'),\
                    np.datetime64('2019-04-06T13:25:00'),\
                    np.datetime64('2019-04-07T01:45:00'),\
                    np.datetime64('2019-04-07T13:05:00'),\
                    np.datetime64('2019-04-08T02:20:00'),\
                    np.datetime64('2019-04-08T14:25:00'),\
                    ])
    return wav_tx


# def get_nliw_times():
#     wav_tx = np.array([np.datetime64('2019-03-11T10:30:00'),\
#                         np.datetime64('2019-03-12T23:20:00'),\
#                         np.datetime64('2019-03-13T12:20:00'),\
#                         np.datetime64('2019-03-13T23:50:00'),\
#                         np.datetime64('2019-03-15T00:10:00'),\
#                         np.datetime64('2019-03-15T12:20:00'),\
#                         np.datetime64('2019-03-15T23:00:00'),\
#                         np.datetime64('2019-03-17T15:20:00')])
#     return wav_tx

def get_nliw_times():
    wav_tx = np.array([np.datetime64('2019-03-12T23:20:00'),\
                        np.datetime64('2019-03-13T12:20:00'),\
                        np.datetime64('2019-03-13T23:50:00'),\
                        np.datetime64('2019-03-14T11:00:00'),\
                        np.datetime64('2019-03-15T00:10:00'),\
                        np.datetime64('2019-03-15T12:20:00'),\
                        np.datetime64('2019-03-15T23:00:00'),\
                        np.datetime64('2019-03-16T12:10:00'),\
                        np.datetime64('2019-03-16T19:45:00'),\
                        np.datetime64('2019-03-17T06:00:00'),\
                        np.datetime64('2019-03-17T15:20:00')])
    return wav_tx


def get_nliw_fronts():
    return np.array([[np.datetime64('2019-03-13T00:00'), np.datetime64('2019-03-13T00:21'), np.datetime64('2019-03-13T00:50')],
                    [np.datetime64('2019-03-13T12:15'), np.datetime64('2019-03-13T12:32'), np.datetime64('2019-03-13T13:45')],
                    [np.datetime64('2019-03-13T23:45'), np.datetime64('2019-03-14T00:11'), np.datetime64('2019-03-14T01:09')],
                    [np.datetime64('2019-03-14T11:00'), np.datetime64('2019-03-14T11:26'), np.datetime64('2019-03-14T12:34')],
                    [np.datetime64('2019-03-15T00:00'), np.datetime64('2019-03-15T00:31'), np.datetime64('2019-03-15T01:45')],
                    [np.datetime64('2019-03-15T12:05'), np.datetime64('2019-03-15T12:36'), np.datetime64('2019-03-15T13:45')],
                    [np.datetime64('2019-03-15T23:00'), np.datetime64('2019-03-15T23:36'), np.datetime64('2019-03-16T00:30')],
                    [np.datetime64('2019-03-16T12:30'), np.datetime64('2019-03-16T12:45'), np.datetime64('2019-03-16T13:45')],
                    [np.datetime64('2019-03-16T19:52'), np.datetime64('2019-03-16T20:05'), np.datetime64('2019-03-16T21:20')],
                    [np.datetime64('2019-03-17T06:00'), np.datetime64('2019-03-17T06:11'), np.datetime64('2019-03-17T07:00')],
                    [np.datetime64('2019-03-17T15:45'), np.datetime64('2019-03-17T16:01'), np.datetime64('2019-03-17T17:30')]])


# def get_nliw_fronts():
#     return np.array([[np.datetime64('2019-03-11T10:48'), np.datetime64('2019-03-11T11:08'), np.datetime64('2019-03-11T12:00')],\
#                     [np.datetime64('2019-03-13T00:00'), np.datetime64('2019-03-13T00:21'), np.datetime64('2019-03-13T00:50')],\
#                     [np.datetime64('2019-03-13T12:15'), np.datetime64('2019-03-13T12:31'), np.datetime64('2019-03-13T13:45')],
#                     [np.datetime64('2019-03-13T23:45'), np.datetime64('2019-03-14T00:11'), np.datetime64('2019-03-14T01:09')],
#                     # [np.datetime64('2019-03-14T11:00'), np.datetime64('2019-03-14T11:26'), np.datetime64('2019-03-14T12:34')],
#                     [np.datetime64('2019-03-15T00:00'), np.datetime64('2019-03-15T00:31'), np.datetime64('2019-03-15T01:45')],
#                     [np.datetime64('2019-03-15T12:05'), np.datetime64('2019-03-15T12:36'), np.datetime64('2019-03-15T13:45')],
#                     [np.datetime64('2019-03-15T23:00'), np.datetime64('2019-03-15T23:36'), np.datetime64('2019-03-16T00:30')],\
#                     [np.datetime64('2019-03-17T15:45'), np.datetime64('2019-03-17T16:19'), np.datetime64('2019-03-17T17:30')]])


# wav_tx = np.array([np.datetime64('2019-03-11T10:30:00'),\
#                    np.datetime64('2019-03-12T23:30:00'),\
#                    np.datetime64('2019-03-13T12:20:00'),\
#                    np.datetime64('2019-03-13T23:50:00'),\
#                    np.datetime64('2019-03-14T11:00:00'),\
#                    np.datetime64('2019-03-14T23:50:00'),\
#                    np.datetime64('2019-03-15T00:30:00'),\
#                    np.datetime64('2019-03-15T12:30:00'),\
#                    np.datetime64('2019-03-15T23:00:00'),\
#                    np.datetime64('2019-03-17T16:00:00')])


def get_solibore_fronts():
    return 100*np.array([[34, 39],
                     [34, 38],
                     [35, 40],
                     [37, 41],
                     [30, 35],
                     [31, 34],
                     [57, 62],
                     [40, 45],
                     [26, 30],
                     [30, 34],
                     [40, 46],
                     [39, 43],
                     [35, 39],
                     [28, 34],
                     [26, 31],
                     [40, 44],
                     [40, 44],
                     [38, 42],
                     [38, 42],
                     [38, 42],
                     [31, 36],
                     [40, 44],
                     [35, 39],
                     [-1, -1],
                     [52, 55],
                     [43, 48],
                     [38, 45],
                     [48, 55],
                     [45, 52],
                     [53, 60],
                     [44, 54],
                     [40, 44],
                     [-1, -1],
                     [43, 52],
                     [40, 46],
                     [-1, -1],
                     [40, 57],
                     [42, 47],
                     [39, 43],
                     [48, 52],
                     [53, 62],
                     [60, 74],
                     [34, 40],
                     [34, 39],
                     [33, 39],
                     [33, 38],
                     [39, 42],
                     [35, 40],
                     ])


