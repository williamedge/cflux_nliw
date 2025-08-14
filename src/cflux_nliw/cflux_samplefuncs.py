import os
import numpy as np
import xarray as xr
import h5py
from pyddcurves.models import double_tanh
from iwaves import IWaveModes
from iwaves.utils.density import InterpDensity

from cflux_fluxfuncs import reynolds_decomp
from cflux_basefuncs import load_traces_new
from d2spike.utils import nan_gauss_xr


def sample_density(h5file, gridpoints=100, samples=100, zmin=-150, noise=False):
    """
    Sample fitted density from trace file
    """
    
    with h5py.File(h5file,'r') as f:
        
        data = f['/data']
        time = data['time'][:]
        z_std = data['z_std']
        rho_std = data['rho_std']
        rho_mu = data['rho_mu']
        
        if zmin is None:
            zmin = data['z'][:].min()*z_std
        
        if gridpoints is None:
            zout = data['z'][:][:26] * z_std
        else:
            zout = np.linspace(zmin,0,gridpoints)
        
        rhoall = np.zeros((len(time), len(zout), samples)).astype(float)
        tstep = range(len(time))
        nparams, nt, nsamples = f['beta_samples'].shape
        
        if samples is None:
            samples = nsamples

        beta = f['beta_samples'][:]
        
        for ix, rand_loc in enumerate(np.random.randint(0, nsamples, samples)):
            for ts in tstep:
                rhotmp = double_tanh([beta[ii,ts,rand_loc] for ii in range(6)], zout/z_std)
                if noise:
                    # rhotmp += np.random.normal(0, data['sigma_curve'], len(zout))
                    rhotmp += np.random.normal(0, 0.08, len(zout))
                rhoall[ts,:,ix] = rhotmp*rho_std+rho_mu
    
    time_np = np.datetime64('1970-01-01') + time.astype('timedelta64[ns]')
    return rhoall, zout, time_np


def sample_celerity(time, zout, rhoall, nsamp, mode=0):
    dz = np.median(np.diff(zout))
    cel = np.full((len(time), nsamp), np.nan)

    # Loop through samples (axis=-1)
    for iss in range(nsamp):

        # Loop through time (axis=0)
        for itt in range(len(time)):
            iw = IWaveModes(rhoall[itt,:,iss], zout[:26], density_class=InterpDensity, density_func='double_tanh')
            _, c1, _, _ = iw(zout.min(), dz, mode)
            cel[itt,iss] = c1
    return cel


from chp3_fluxfuncs import reynolds_decomp, sample_ppc_full_new
from chp3_basefuncs import load_traces_new
from d2spike.utils import nan_gauss, nan_gauss_xr


def sample_ssc(backscatter, trace_2, trace_3):
    sig_ntu = run_sample(trace_2.posterior, backscatter, 1)
    ntu_ssc = run_sample(trace_3.posterior, 10**sig_ntu, 1)
    return ntu_ssc


def run_sample(trace_var, eval_data, sample_len):
    lm = lambda x, samp, sig: samp['a_intercept'].values + samp['b_slope'].values * x + sig

    rand_chain = np.random.randint(0, high=len(trace_var.chain.values), size=sample_len)
    rand_draw = np.random.randint(0, high=len(trace_var.draw.values), size=sample_len)

    rand_sample = trace_var.isel(chain=rand_chain, draw=rand_draw)
    rand_sig = np.random.normal(loc=0, scale=rand_sample['sigma_y'].values[0], size=1)
    ppc_full = lm(eval_data, rand_sample, rand_sig)
    return ppc_full


def make_xr(xarr, n_samples):
    return xr.DataArray(data=np.full((xarr.shape[0], xarr.shape[1], n_samples), np.nan),\
                        dims=['height', 'time', 'sample'],\
                        coords={'height': xarr.height, 'time': xarr.time, 'sample': np.arange(n_samples)})

def calc_fluxes(ssc_mean, ssc_turb, w_turb, ds_crop, c_est, thin_v, thin_t, gf_h, gf_t):
    # Calc Reynolds flux and thin
    wc_raw = w_turb.values * ssc_turb
    wc_raw = xr.DataArray(data=wc_raw, dims=['height', 'time'],\
                            coords={'height': w_turb['height'].values, 'time': w_turb['time'].values})
    wc_z_turb = nan_gauss_xr(-1*wc_raw.differentiate('height'),\
                                [gf_h, gf_t], axis=[0,1])[::thin_v, ::thin_t][:-1,:]

    # Calc the horz flux
    uc_z_mean = ds_crop['u_mean'] * ssc_mean.differentiate('time', datetime_unit='s') / c_est

    # Calc the vert flux
    wc_z_mean = -1 * ds_crop['w_mean'] * ssc_mean.differentiate('height')

    # Calculate the time rate of change
    dc_dt = ssc_mean.differentiate('time', datetime_unit='s')
    
    t_list_sample = [dc_dt, wc_z_mean, wc_z_turb, uc_z_mean]
    return t_list_sample


def sample_loop_ssc(ds_crop, w_turb, beam_cor, n_samp=10, work_dir=None, field_trip=None,\
                    thin_v=2, thin_t=10, gf_h=4, gf_t=300, sp='est_celerity'):

    if work_dir is None:
        work_dir = r'/mnt/c/Users/00099894/OneDrive - The University of Western Australia/UWA/PhD'
    if field_trip is None:
        field_trip = 'RS2019'    
    
    trace_2, trace_3 = load_traces_new(os.path.join(work_dir, 'pl'), field_trip)
    trace_2.posterior['sigma_y'] = trace_2.posterior['sigma_y']/3
    trace_3.posterior['b_slope'] = trace_3.posterior['b_slope']/2.33
    trace_3.posterior['sigma_y'] = trace_3.posterior['sigma_y']/2

    # t_step = np.diff(ds_crop['time'].values)[0].astype('timedelta64[ns]')/np.timedelta64(1,'s')
    # t_step_turb = np.diff(w_turb['time'].values)[0].astype('timedelta64[ns]')/np.timedelta64(1,'s')

    # ds_crop = ds.isel(time=tx_sig)
    uc_z_mean = make_xr(ds_crop['u_mean'], n_samp)
    wc_z_mean = make_xr(ds_crop['u_mean'], n_samp)
    wc_z_turb = make_xr(ds_crop['u_mean'], n_samp)
    dc_dt = make_xr(ds_crop['u_mean'], n_samp)

    ### Now the loop
    for ii in range(n_samp):

        # Sample the wave speed once
        if sp == 'est_celerity':
            c_est = np.random.normal(ds_crop.attrs[sp],\
                                    ds_crop.attrs['obs_celerity'][1],\
                                    size=1)
        else:
            c_est = np.random.normal(ds_crop.attrs[sp][0],\
                                    ds_crop.attrs['obs_celerity'][1],\
                                    size=1)

        # Sample SSC once
        ssc_samp = sample_ssc(beam_cor.T, trace_2, trace_3)
        ssc_samp[ssc_samp < 0.0] = 0.0

        # Reynolds decomp
        ssc_mean, ssc_turb = reynolds_decomp(ssc_samp, [gf_h,gf_t])
        print('Reynolds decomp number ' + str(ii+1) + ' sampled')

        # Thin the mean ssc
        ssc_mean = xr.DataArray(data=ssc_mean[::thin_v, ::thin_t][:-1,:], dims=['height', 'time'],\
                                coords={'height': ds_crop['height'].values, 'time': ds_crop['time'].values})

        # Calc Reynolds flux and thin
        t_samp = calc_fluxes(ssc_mean, ssc_turb, w_turb, ds_crop, c_est, thin_v, thin_t, gf_h, gf_t)
        dc_dt[:,:,ii] = t_samp[0]
        wc_z_mean[:,:,ii] = t_samp[1]
        wc_z_turb[:,:,ii] = t_samp[2]
        uc_z_mean[:,:,ii] = t_samp[3]

        # wc_raw = w_turb.values * ssc_turb
        # wc_raw = xr.DataArray(data=wc_raw, dims=['height', 'time'],\
        #                         coords={'height': w_turb['height'].values, 'time': w_turb['time'].values})
        # wc_z_turb[:,:,ii] = nan_gauss_xr(-1*wc_raw.differentiate('height'),\
        #                             [gf_h, gf_t], axis=[0,1])[::thin_v, ::thin_t][:-1,:]

        # # Calc the horz flux
        # uc_z_mean[:,:,ii] = ds_crop['u_mean'] * ssc_mean.differentiate('time', datetime_unit='s') / c_est

        # # Calc the vert flux
        # wc_z_mean[:,:,ii] = -1*ds_crop['w_mean'] * ssc_mean.differentiate('height')

        # # Calculate the time rate of change
        # dc_dt[:,:,ii] = ssc_mean.differentiate('time', datetime_unit='s')

    t_list = [dc_dt, wc_z_mean, wc_z_turb, uc_z_mean] 
    return t_list



# def samp_ssc(beam_near, n_samp, workDir, fieldtrip):

#     trace_2, trace_3 = load_traces_new(workDir, fieldtrip)

#     # Loop through and sample the data
#     lm = lambda x, samp: samp['a_intercept'] + samp['b_slope'] * x
#     lm2 = lambda x, samp: samp['a_intercept'] + samp['b_slope']/2.5 * x

#     rand_init = np.random.randint(0, high=len(trace_2), size=n_samp)
#     rand_twot = np.random.randint(0, high=len(trace_3), size=n_samp)

#     ssc_shape = beam_near.shape
#     sig_n_C = np.full((ssc_shape[0], ssc_shape[1], n_samp), np.nan)

#     for iz, (ri, rt) in enumerate(zip(rand_init, rand_twot)):

#         rs_trb_oo = np.random.normal(loc=0, scale=trace_2[ri]['sigma_y']/5, size=ssc_shape)
#         c_turb_first = lm(beam_near, trace_2[ri]) #+ rs_trb_oo

#         rs_trb_tt = np.random.normal(loc=0, scale=trace_3[rt]['sigma_y'], size=ssc_shape)
#         c_turb = lm2(10**c_turb_first, trace_3[rt]) #+ rs_trb_tt

#         c_turb_val = c_turb.values
#         c_turb_val[np.isnan(c_turb_val)] = np.nanmax(c_turb_val)
#         sig_n_C[:,:,iz] = c_turb_val
        
#     return sig_n_C






# def sample_ppc_full(trace_var, eval_data, sample_len):
                
#     # Loop through and sample the data
#     lm = lambda x, samp: samp['a_intercept'] + samp['b_slope'] * x + np.random.normal(loc=0, scale=samp['sigma_y'])
                  
#     # Initialize the sampling variable with nans
#     if hasattr(eval_data, '__len__'):

#         if eval_data.ndim == 1:
#             ppc_full = np.full((len(eval_data), sample_len), np.nan)
#             rand_init = np.random.randint(0, high=len(trace_var), size=sample_len)

#             for iy, ri in enumerate(rand_init): 
#                 rand_sample = trace_var[ri]
#                 ppc_full[:, iy] = lm(eval_data, rand_sample)
            
            
#         elif (eval_data.ndim == 2) & (sample_len == 1):
#             ppc_full = np.full_like(eval_data, np.nan)
#             rand_init = np.random.randint(0, high=len(trace_var), size=np.shape(eval_data)[1]) 
            
#             for iy, ri in enumerate(rand_init): 

#                 rand_sample = trace_var[ri]
#                 ppc_full[:, iy] = lm(eval_data[:, iy], rand_sample)            
            
            
#         elif (eval_data.ndim == 2) & (sample_len!=1):
#             ppc_full = np.full((np.shape(eval_data)[0], np.shape(eval_data)[1], sample_len), np.nan)
#             rand_init = np.random.randint(0, high=len(trace_var), size=sample_len)
            
#             for iz, ri in enumerate(rand_init): 
#                 rand_sample = trace_var[ri]
                
#                 ppc_full[:,:, iz] = lm(eval_data, rand_sample)
                
            
#         elif (eval_data.ndim == 3) & (sample_len==1):
#             ppc_full = np.full_like(eval_data, np.nan)
#             rand_init = np.random.randint(0, high=len(trace_var), size=np.shape(eval_data)[-1]) 
            
#             for iz, ri in enumerate(rand_init): 
#                 rand_sample = trace_var[ri]
                
#                 ppc_full[:,:,iz] = lm(eval_data[:,:,iz], rand_sample)

#     return ppc_full