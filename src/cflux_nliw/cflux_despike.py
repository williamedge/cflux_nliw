import numpy as np
from d2spike.despike import flag_corr

def round_one(w_c, beam_data, gf_sig, re_val, sw_vals, skip_pair, verbose, full_output, max_z, max_t):

    # Use a 2D Gaussian filter to find the mean values (works much better than 1D lowpass with heavy spiking)
    # w_gf = xr.DataArray(data=nan_gauss(w_c.values, gf_sig), coords=w_c.coords)
    w_gf = w_c.floatda.gaussian_filter(gf_sig)

    # Subtract the background values and despike
    w_gn = (w_c - w_gf).copy()
    for ii, wd in enumerate(w_gn.T):
        if ii==0:
            print(wd.shape)
        w_gn[:,ii], _ = wd.floatda.despike_gn23(full_output=full_output,\
            sw_thresh=sw_vals, skip_pair=skip_pair, verbose=verbose)

    # Call 2D indexing reinstatement
    re_ix = np.abs(beam_data - w_gf) < re_val
    w_gn = w_gn.floatda.reinstate_threshold((beam_data - w_gf), re_ix)

    # Interpolate gaps of 1 (timestep and spatial bin)
    w_int = w_gn + w_gf.T
    try:
        w_int = w_int.interpolate_na(dim='height', method='cubic', max_gap=max_z)
    except:
        print('Failed to interpolate along height axis')
    try:
        w_int = w_int.interpolate_na(dim='time', method='cubic', max_gap=np.timedelta64(max_t,'ms'))
    except:
        print('Failed to interpolate along time axis')
    return w_int


def full_pipe(beam_data, corr_data, corrflag, qc0_val, gf_sig, re1=0.05, re2=0.01,\
                sw_vals=0.5, skip_pair=[-1], verbose=False, full_output=False, max_z=0.07, max_t=600):

    '''
    Call the full de-spiking pipeline from raw data up to small gap interpolation.
    Large gap interpolation is handled later. 
    '''

    # Flag data below a correlation threshold (and initiate the despike class)
    # w_c = flag_corr(beam_data, corr_data, corrflag).T
    w_c = beam_data.floatda.qc0_lowcorr(corr_data, corrflag)

    # Also flag values that are physically unreasonable
    w_c = w_c.floatda.qc0_flags(val=qc0_val)

    # Dont send any rows that are >50% nans (just nan all)
    w_c_nanrows = (np.sum(np.isnan(w_c), axis=0) / w_c.values.shape[0]) > 0.5

    # Flag round 1
    w_int1 = w_c.copy()
    w_int1[:,~w_c_nanrows] = round_one(w_c[:,~w_c_nanrows], beam_data, gf_sig, re_val=re1,\
                                    sw_vals=sw_vals, skip_pair=skip_pair,\
                                    verbose=verbose, full_output=full_output, max_z=max_z, max_t=max_t)
    w_int1[:,w_c_nanrows] = np.nan

    # Flag round 2
    w_int1_nanrows = (np.sum(np.isnan(w_int1), axis=0) / w_int1.values.shape[0]) > 0.5
    w_int2 = w_int1.copy()
    w_int2[:,~w_int1_nanrows] = round_one(w_int1[:,~w_int1_nanrows], beam_data, gf_sig,\
                                    re_val=re2, sw_vals=sw_vals, skip_pair=skip_pair,\
                                    verbose=verbose, full_output=full_output, max_z=max_z, max_t=max_t)
    w_int2[:,w_int1_nanrows] = np.nan

    return w_int2