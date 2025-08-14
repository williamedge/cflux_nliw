import numpy as np
import xarray as xr
from wootils.sigint import crosscorr

from cflux_basefuncs import load_c
from cflux_plotfuncs import plot_corrlags
from cflux_basefuncs import confirm_front


def beam_crosscorr(ssc_obj, hx, lag_window, nbeams=4):
    rs_all = np.full((len(hx), nbeams, nbeams), np.nan)
    corr_all = np.full((len(hx), nbeams, nbeams), np.nan)

    for izz, iz in enumerate(hx):
        for ix in np.arange(nbeams):
            for iy in np.arange(nbeams):
                rs = crosscorr(ssc_obj.isel(beam=ix, height=iz).values,\
                               ssc_obj.isel(beam=iy, height=iz).values,\
                               lag_window)
                corr_all[izz,ix,iy] = np.max(rs)
                rs_all[izz,ix,iy] = lag_window - 1 - np.argmax(rs)
    return rs_all, corr_all


def get_beam_distances(adcp_distance, theta=25, beampos=[0,90,180,270]):
    phi = -1*np.array(beampos)
    r = adcp_distance

    xb = np.full((len(r), phi.shape[0]), np.nan)
    yb = np.full((len(r), phi.shape[0]), np.nan)
    zb = np.full((len(r), phi.shape[0]), np.nan)

    for ix, b in enumerate(phi):
        xb[:,ix] = r*np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(b))
        yb[:,ix] = r*np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(b))
        zb[:,ix] = -r*np.cos(np.deg2rad(theta))
    return xb, yb, zb, phi



@xr.register_dataarray_accessor("sscda")
class SSCArray():
    def __init__(self, da):
        if 'units' in da.attrs:
            self.units = da.attrs['units']
        else:
            self.units = '?'
        dims = da.dims
        if not dims[0].lower() == 'time':
            raise(Exception("First dimension must be time"))
        self._obj = da

    @property
    def _da(self):
        return self._obj

    @property
    def dims(self):
        return self._obj.dims

    @property
    def other_dims(self):
        return [dim for dim in self._obj.dims if (not dim.lower()=='time')]

    @property
    def coords(self):
        return self._obj.coords

    def __repr__(self):
        return self._da.__repr__()

    ### Class methods ###
    def set_workdir(self, workdir):
        self.workdir = workdir

    def set_fieldtrip(self, fieldtrip):
        self.fieldtrip = fieldtrip

    def calibrate_echo(self, ds_temp, bc_cut=None, bs_adj=False):
        return load_c(self._da, ds_temp, self.workdir, self.fieldtrip, bs_cutoff=bc_cut, bs_corrected=bs_adj)

    def mask_bsdata(self, cutoff=100, replace=0.0):
        self = self._da.where(self._da < cutoff)
        if replace is not None:
            self = self.fillna(replace)
        return self

    def beam_xcorr(self, hx, lag_window, nbeams=4):
        rs_all, corr_all = beam_crosscorr(self._da, hx, lag_window, nbeams=nbeams)
        return rs_all, corr_all
    
    def plot_front(self, idxs, nbeams=4, thin=[24,4], vmax=30):
        fig, ax = confirm_front(self._da, idxs, nbeams=nbeams, thin=thin, vmax=vmax)
        return fig, ax
    
    def plot_xcorr(self, zx, lag_window, zx_all=None):
        # Check zx is int or length one
        if isinstance(zx, list) | isinstance(zx, np.ndarray):
            if len(zx) > 1:
                raise(Exception("zx must be int or length one list/array"))
            else:
                zx = zx[0]
        fig, ax = plot_corrlags(self._da, zx, lag_window, zx_all=zx_all)
        return fig, ax




# def crosscorr(datax, datay, lag):
#     return datax.corr(datay.shift(lag))

# def get_beam_distances(adcp_distance):
#     phi = -1*np.array([0,90,180,270])
#     r = adcp_distance
#     theta = 25 # degrees
#     wave_angle = np.arange(-180,179,1)
#     c_guess = np.arange(0.15,1.0,0.01)
#     dtime = 0.25

#     xb = np.full(phi.shape, np.nan)
#     yb = np.full(phi.shape, np.nan)
#     zb = np.full(phi.shape, np.nan)

#     for ix, b in enumerate(phi):
#         xb[ix] = r*np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(b))
#         yb[ix] = r*np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(b))
#         zb[ix] = -r*np.cos(np.deg2rad(theta))
#     return xb, yb, zb, phi


# def beam_crosscorr(df, lag_window, phi):
#     rs_all = np.full((len(phi), len(phi)), np.nan)
#     corr_all = np.full((len(phi), len(phi)), np.nan)

#     for ix in np.arange(len(phi)):
#         for iy in np.arange(len(phi)):
#             rs = [crosscorr(df['b' + str(int(ix+1))],\
#                                 df['b' + str(int(iy+1))], lag_window)\
#                                 for lag_window in range(-lag_window, lag_window)]
#             corr_all[ix,iy] = np.max(rs)
#             rs_all[ix,iy] = np.ceil(len(rs)/2) - np.argmax(rs)
#     return rs_all, corr_all


# def theta_2_dist(theta, xb, yb):
#     theta_deg = np.rad2deg(theta)
#     Phi2 = 90 - theta_deg

#     # Distance from the beams to the origin in the wave direction
#     d = xb*np.cos(np.deg2rad(Phi2)) + yb*np.sin(np.deg2rad(Phi2))  

#     # Stack the distances
#     distance = np.array([d[1] - d[0],\
#                         d[2] - d[0],\
#                         d[3] - d[0],\
#                         d[2] - d[1],\
#                         d[3] - d[1],\
#                         d[3] - d[2]])
#     return distance


# def extract_front(ssc_arr, var_lvl=10, rolltime=100, t_bef=30, t_aft=120):
#     ssc_var = ssc_arr.rolling(time=100, center=True).var()
#     ssc_time = ssc_arr.time[ssc_var > var_lvl].values[0] - np.timedelta64(t_bef,'s')
#     ssc_tx = np.argmin(np.abs(ssc_arr.time.values - ssc_time))
#     ssc_ix = (ssc_arr.time.values >= ssc_time - np.timedelta64(t_bef,'s')) &\
#                 (ssc_arr.time.values <= ssc_time + np.timedelta64(t_aft,'s'))
#     return ssc_ix


# def plot_rs_all(df, lag_window):
#     fig, ax = plt.subplots(3, 3, figsize=(6,6))

#     for ix, x_row in enumerate(ax.T):
#         for iy, x_col in enumerate(x_row):
#             if ix + iy + 2 < 5:
#                 # Loop through all pairs
#                 rs = [crosscorr(df['b' + str(int(ix+1))],\
#                                 df['b' + str(int(ix+iy+2))], lag_window)\
#                                 for lag_window in range(-lag_window, lag_window)]
            
#                 offset = np.ceil(len(rs)/2) - np.argmax(rs)

#                 x_row[ix+iy].plot(rs)
#                 x_row[ix+iy].axvline(np.ceil(len(rs)/2), color='k', linestyle='--', label='Centre')
#                 x_row[ix+iy].axvline(np.argmax(rs), color='r', linestyle='--', label='Peak synchrony')

#                 x_row[ix+iy].spines['right'].set_visible(False)
#                 x_row[ix+iy].spines['top'].set_visible(False)
#                 x_row[ix+iy].set_xticklabels('')
#                 x_row[ix+iy].set_yticklabels('')          
#                 x_row[ix+iy].annotate(offset, xy=(0.05, 0.85), xycoords=x_row[ix+iy].transAxes)
                
#                 if x_row[ix+iy] in ax[-1]:
#                     x_row[ix+iy].set_xlabel('b' + str(int(ix+1)))             
#             if ix==0:
#                 x_row[ix+iy].set_ylabel('b' + str(int(ix+iy+2)))            
#             if ix > iy:
#                 x_col.remove()
#     return fig, ax
