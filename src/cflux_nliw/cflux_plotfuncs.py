import numpy as np
from wootils.plotnice import vert_stack, plot_align, basic_ts
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmocean.cm as cm
from matplotlib import gridspec
from matplotlib import dates as mdates
from matplotlib.patches import Ellipse
from wootils.sigint import crosscorr
from wootils.linalg import polar_mean, polar_stddev
from afloat.pca import rotate_2D

import sys
sys.path.insert(0, '')
from cflux_fluxfuncs import height_integrals



def plot_minimal_setup(ds_t, u_mean, w_mean, ssc_total, ssc_vmax=20, ulim=[-0.19,0.59], wlim=[-0.039,0.039]):
    
    temp_ix = (ds_t['time'].values >= u_mean.time.values[0]) &\
            (ds_t['time'].values <= u_mean.time.values[-1])
    ds_t = ds_t.isel(time=temp_ix)

    fig, ax = vert_stack(3, hsize=10, vsize=1.6)

    # Plot temperature
    xr_t = ax[0].contourf(ds_t['time'].values, ds_t['depth'].values[1:],\
                        ds_t['Temperature'].values[1:,:],\
                        levels=np.arange(18,32,1), cmap=plt.cm.viridis)
    xr_tc = ax[0].contour(xr_t, colors='k', linewidths=0.1)
    cb = fig.colorbar(xr_t, ax=ax[0], pad=0.01, ticks=np.arange(20,29.1,3))
    mpl.colorbar.ColorbarBase.set_label(cb, 'Temperature\n[$^\circ$C]')

    # Plot U
    u_mean.sel(height=4.5, method='nearest').plot(ax=ax[1], c='k')
    ax[1].set_ylim(ulim)
    ax[1].grid()

    # Plot W
    ax2 = ax[1].twinx()
    w_mean.sel(height=4.5, method='nearest').plot(ax=ax2, c='r')
    ax2.set_ylim(wlim)
    ax2.set_title('')

    # Plot E
    ssc_total.plot(cmap=cm.turbid,\
                        vmin=0,\
                        vmax=ssc_vmax,\
                        ax=ax[2], center=0,\
                        cbar_kwargs={'pad':0.01, 'label':'C'})
    for x in ax:
        x.set_ylabel('m ASB')
        if x == ax[-1]:
            x.set_ylim(2.2, 6.8)
    ax[1].set_ylabel('U')
    ax2.set_ylabel('W')
    basic_ts(ssc_total.time.values, ax)
    plot_align(ax)
    return fig, ax


def plot_corrlags(ssc_obj, zx, lag_window, zx_all=None):

    # print(zx)
    fig, ax = plt.subplots(3, 3, figsize=(6,6))

    for ix, x_row in enumerate(ax.T):
        for iy, x_col in enumerate(x_row):
            if ix + iy + 2 < 5:
                rs = crosscorr(ssc_obj.isel(beam=ix, height=zx).values,\
                                ssc_obj.isel(beam=ix+iy+1, height=zx).values,\
                                lag_window)
                offset = lag_window - 1 - np.argmax(rs)

                x_row[ix+iy].plot(rs)
                x_row[ix+iy].axvline(lag_window - 1, color='k', linestyle='--', label='Centre')
                x_row[ix+iy].axvline(np.argmax(rs), color='r', linestyle='--', label='Peak synchrony')

                if zx_all is not None:
                    for zz in zx_all:
                        rs = crosscorr(ssc_obj.isel(beam=ix, height=zz).values,\
                                        ssc_obj.isel(beam=ix+iy+1, height=zz).values,\
                                        lag_window)
                        x_row[ix+iy].plot(rs, alpha=0.5, lw=0.5)

                x_row[ix+iy].spines['right'].set_visible(False)
                x_row[ix+iy].spines['top'].set_visible(False)
                x_row[ix+iy].set_xticks([])
                x_row[ix+iy].set_yticks([])
                x_row[ix+iy].set_xticklabels('')
                x_row[ix+iy].set_yticklabels('')          
                x_row[ix+iy].annotate(offset, xy=(0.05, 0.85), xycoords=x_row[ix+iy].transAxes)
                
                if x_row[ix+iy] in ax[-1]:
                    x_row[ix+iy].set_xlabel('b' + str(int(ix+1)))             
            if ix==0:
                x_row[ix+iy].set_ylabel('b' + str(int(ix+iy+2)))            
            if ix > iy:
                x_col.remove()
    return fig, ax


def plot_beam_distance(xb, yb):
    fig, ax = plt.subplots(1,1)
    plt.scatter(xb, yb)
    for xx in np.arange(4):
        plt.text(xb[xx], yb[xx]+0.1, str(xx+1))

    # Add arrows with the distance between points
    plt.arrow(xb[0], yb[0], xb[2]*2, yb[2]*2,\
                color='k', length_includes_head=True, head_width=0.1,\
                head_length=0.1)
    plt.text(0.3, 0.1, f'{2*xb[0]:.2f}')
    plt.arrow(xb[1], yb[1], xb[3]*2, yb[3]*2,\
                color='k', length_includes_head=True, head_width=0.1,\
                head_length=0.1)
    plt.text(0.1, 0.4, f'{2*yb[3]:.2f}')
    plt.arrow(xb[2], yb[2], xb[3] - xb[2], yb[3],\
                color='k', length_includes_head=True, head_width=0.1,\
                head_length=0.1)
    plt.text(xb[2]/2, yb[3]/2, f'{np.sqrt(xb[0]**2 + yb[3]**2):.2f}')

    return fig, ax




def plot_polar_dirs():
    # Plot these as an ellipse on a polar plot
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111, projection='polar')

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 0.8)
    # ax.set_yticklabels('')
    # ax.set_xticklabels('')
    ax.set_rlabel_position(0)
    return fig, ax


def add_ellipse(ax, trace, colix, int_heading):
    # Get the standard deviation of the posterior
    c_std = np.std(trace['c'])
    c_mean = np.mean(trace['c'])

    # Compute the mean + stddev angle
    theta_mean = polar_mean(trace['theta']) + int_heading
    theta_std = polar_stddev(trace['theta'])

    # Create the ellipse patch
    ellipse = Ellipse((theta_mean, c_mean), 2*theta_std, 2*c_std, color=colix, fill=True, alpha=0.5, lw=0)
    ax.add_patch(ellipse)



def plot_setup(ds_temp, u_mean, w_mean, ssc_total, ssc_vmax=20, ulim=0.59, wlim=0.039):
    fig, ax = vert_stack(4, hsize=14, vsize=2)

    # Plot temperature
    xr_t = ax[0].contourf(ds_temp['time'].values, ds_temp['depth'].values[1:],\
                        ds_temp['Temperature'].values[1:,:],\
                        levels=np.arange(18,32,1), cmap=plt.cm.viridis)
    xr_tc = ax[0].contour(xr_t, colors='k', linewidths=0.1)
    cb = fig.colorbar(xr_t, ax=ax[0], pad=0.01, ticks=np.arange(20,29.1,3))
    mpl.colorbar.ColorbarBase.set_label(cb, 'Temperature\n[$^\circ$C]')

    # Plot U
    u_mean.plot(cmap='PuOr',\
                            vmin=-ulim,\
                            vmax=ulim,\
                            ax=ax[1], center=0,\
                            cbar_kwargs={'pad':0.01, 'label':'$\overline{U}$'})

    # Plot W
    w_mean.plot(cmap='PuOr',\
                            vmin=-wlim,\
                            vmax=wlim,\
                            ax=ax[2], center=0,\
                            cbar_kwargs={'pad':0.01, 'label':'$\overline{W}$'})

    # Plot E
    ssc_total.plot(cmap=cm.turbid,\
                        vmin=0,\
                        vmax=ssc_vmax,\
                        ax=ax[3], center=0,\
                        cbar_kwargs={'pad':0.01, 'label':'C'})
    for x in ax:
        x.set_ylabel('m ASB')
        if x != ax[0]:
            x.set_ylim(2.0, ssc_total.height[-3])
    basic_ts(ssc_total.time.values, ax)
    plot_align(ax)
    return fig, ax


def plot_decomp(mean, turb, mean_lim=None, mean_cm='PuOr'):
    if mean_lim is None:
         mean_lim=[-2*np.max(np.abs(mean)),2*np.max(np.abs(mean))]
    fig, ax = vert_stack(2, hsize=14, vsize=2)
    mean[::4,::32].plot(cmap=mean_cm, vmin=mean_lim[0], vmax=mean_lim[1],\
                        ax=ax[0], center=0,\
                        cbar_kwargs={'pad':0.01})
    turb[::4,::32].plot(cmap='PuOr', vmin=-mean_lim[1], vmax=mean_lim[1],\
                    ax=ax[1], center=0, cbar_kwargs={'pad':0.01})
    for x in ax:
        x.set_ylabel('m ASB')
        x.set_ylim(2.0, turb.height[-3])
    basic_ts(mean.time.values, ax)
    plot_align(ax)
    return fig, ax


def plot_fluxterms(ucz, wcz, wcz_turb, lim=0.06):
    fig, ax = vert_stack(3, hsize=14, vsize=2)
    ucz[::4,::32].plot(cmap='PuOr', vmin=-lim, vmax=lim,\
                    ax=ax[0], center=0, cbar_kwargs={'pad':0.01})
    wcz[::4,::32].plot(cmap='PuOr', vmin=-lim, vmax=lim,\
                    ax=ax[1], center=0, cbar_kwargs={'pad':0.01})
    wcz_turb[::4,::32].plot(cmap='PuOr', vmin=-lim, vmax=lim,\
                    ax=ax[2], center=0, cbar_kwargs={'pad':0.01})
    for x in ax:
        x.set_ylabel('m ASB')
        x.set_ylim(2.0, ucz.height[-1])
    plot_align(ax)
    basic_ts(ucz.time.values, ax)
    return fig, ax


def plot_profiles(ds, terms_list, ax, cb_labels, ii):

    lss = ['-', '--', '-.', ':','-', '--', '-.', ':','-']
    t_step = np.diff(ds.time.values)[0].astype('timedelta64[s]').astype(int)
    s_dist = ds.height.values
    sn_ix = (ds.time.values >= ds['s'].values[0]) & (ds.time.values <= ds['s'].values[1])
    ed_ix = (ds.time.values >= ds['s'].values[1]) & (ds.time.values <= ds['s'].values[2])

    for x, f_term, cbl in zip(ax, terms_list, cb_labels):

        ti_t = np.sum(f_term[:,sn_ix], axis=1) * t_step
        x.plot(ti_t, s_dist, c='royalblue', lw=1, ls=lss[ii])

        ed_t = np.sum(f_term[:,ed_ix], axis=1) * t_step    
        x.plot(ed_t, s_dist, c='darkorange', lw=1, ls=lss[ii])
        
        x.set_xlim(-49, 49)
        x.set_ylim(2.4, s_dist[-3])
        x.grid()
        x.set_title(cbl)
        
        if ii==6:
            if x != ax[0]:
                x.set_yticklabels('')
            else:
                x.set_ylabel('m ASB')
    return ax


def plot_shields(ds_temp, ds, ds_turb):

    fig, ax = vert_stack(3)

    # Top subplot
    temp_ix = (ds_temp['time'].values >= ds.time.values[0]) &\
               (ds_temp['time'].values <= ds.time.values[-1])
    plot_tempcontours(ds_temp['Temperature'].isel(time=temp_ix), fig, ax[0])
    xx = ax[0].twinx()
    xx.plot(ds['time'].values, ds['mode_1'].values, 'w')
    xx.set_ylim([-130/2, 130/2])
    xx.set_yticks([])

    # Plot C and Shields
    ax[1].plot(ds['time'].values, ds['c_mean'].sel(height=3.0, method='nearest').values, 'r')

    w_turb = ds_turb['beam5'].sel(height=3.0, method='nearest') -\
             ds_turb['beam5'].sel(height=3.0, method='nearest').rolling(time=300*4, center=True).mean()
    t_isw = 1025 * (w_turb**2).rolling(time=4*300, center=True).construct('tmp').quantile(.95, dim='tmp')
    xx1 = ax[1].twinx()
    xx1.plot(ds_turb['time'].values, t_isw.values, 'k')


    shields_bot = (1025*1.22 - 1025) * 9.81 * 0.0001
    ax[2].plot(ds_turb['time'].values, (t_isw/shields_bot).values, 'k')
    ax[2].axhline(0.7, color='grey', linestyle='--')

    basic_ts(ds.time.values, ax)
    plot_align(ax)

    return fig, ax


def plot_lines(ds_temp, ds, ds_turb):
    fig, ax = vert_stack(4)

    # Top subplot
    temp_ix = (ds_temp['time'].values >= ds.time.values[0]) &\
               (ds_temp['time'].values <= ds.time.values[-1])
    plot_tempcontours(ds_temp['Temperature'].isel(time=temp_ix), fig, ax[0])
    xx = ax[0].twinx()
    xx.plot(ds['time'].values, ds['mode_1'].values, 'w')
    xx.set_ylim([-130/2, 130/2])
    xx.set_yticks([])

    # Next subplot (U)
    u_bc, v_bc = rotate_2D(ds['u_bchf'], ds['v_bchf'], ds.attrs['wave_direction'])
    u_tt, v_tt = rotate_2D(ds['u_adcp'], ds['v_adcp'], ds.attrs['wave_direction'])
    # u_bt, v_bt = rotate_2D(ds['u_tide'], ds['v_tide'], ds.attrs['wave_direction'])
    # u_it, v_it = rotate_2D(ds['u_itide'], ds['v_itide'], ds.attrs['wave_direction'])

    # ax[1].plot(ds['time'].values, ds['u_tide'].values, 'r')
    ax[1].plot(ds['time'].values, (v_tt - v_bc).values, 'k', linewidth=2.5)
    # ax[1].plot(ds['time'].values, (ds['u_itide'] + u_bc).values, 'grey', linewidth=1)
    # ax[1].plot(ds['time'].values, (u_bc).values, 'grey', linewidth=1)
    ax[1].plot(ds['time'].values, (v_tt).values, 'red', linewidth=1)
    ax[1].set_ylim([-0.75, 0.75])

    u_turb, v_turb = rotate_2D(ds_turb['enu'].isel(cartesian_axes=0),\
                               ds_turb['enu'].isel(cartesian_axes=1),\
                               ds.attrs['sig_direction'])
    ax[2].plot(ds_turb['time'].values, v_turb.sel(height=3.0, method='nearest'), 'r', linewidth=1)
    ax[2].plot(ds['time'].values, ds['u_mean'].sel(height=3.0, method='nearest'), 'k', linewidth=2.5)
    ax[2].set_ylim([-0.75, 0.75])
    

    # Next subplot (C)
    ax[3].plot(ds['time'].values, ds['c_mean'].sel(height=1.0, method='nearest').values, 'k')
    ax[3].plot(ds['time'].values, ds['c_mean'].sel(height=3.0, method='nearest').values, 'r')
    ax[3].plot(ds['time'].values, ds['c_mean'].sel(height=6.0, method='nearest').values, 'orange')
    
    basic_ts(ds.time.values, ax)
    plot_align(ax)

    return fig, ax


def plot_tempcontours(d_arr, fig, ax, levels=np.arange(18,32,1), ticks=np.arange(20,29.1,3)):
    xr_t = ax.contourf(d_arr['time'].values, d_arr['depth'].values[1:],\
                        d_arr.values[1:,:],\
                        levels=levels, cmap=plt.cm.viridis)
    xr_tc = ax.contour(xr_t, colors='k', linewidths=0.1)
    cb = fig.colorbar(xr_t, ax=ax, pad=0.01, ticks=ticks)
    mpl.colorbar.ColorbarBase.set_label(cb, 'Temperature\n[$^\circ$C]')
    return xr_t, xr_tc



def plot_diffprofiles(diff, y_data, x_indexes, y_indexes, dmin):

    x1, x2 = x_indexes
    x1  = np.argmin(np.abs(diff.time.values - x1))
    x2  = np.argmin(np.abs(diff.time.values - x2))
    
    y1, y2 = y_indexes
    y1  = np.argmin(np.abs(diff.height.values - y1))
    y2  = np.argmin(np.abs(diff.height.values - y2))

    y_m = np.median(diff[y1:y2,x1:x2].T, axis=0)
    y_s = np.array([mad(diff[xcx,x1:x2].T) for xcx in np.arange(y1,y2)])/2


    fig, ax = plt.subplots(2,1, figsize=(4.5,6.5), gridspec_kw={'height_ratios':[1,6], 'hspace':0.03})

    diff.plot(cmap='PuOr', vmin=-dmin, vmax=dmin,\
                            ax=ax[0], center=0, cbar_kwargs={'pad':0.01})
    ax[0].set_xticklabels('')

    vline = ax[0].axvline(diff.time[x1].values, c='w', lw=2.0, ls='--')
    vline = ax[0].axvline(diff.time[x2].values, c='w', lw=2.0, ls='--')

    scatter = ax[1].scatter(diff[y1:y2,x1:x2].T,\
                            np.tile(y_data[y1:y2], (diff[y1:y2,x1:x2].shape[1],1)),\
                            s=2, c='grey', alpha=0.02)
    ax[1].errorbar(y_m, y_data[y1:y2], xerr=y_s,\
                 markersize=5, c='red', alpha=1, ls=' ')
    ax[1].scatter(y_m, y_data[y1:y2],\
                 s=30, c='w', edgecolor='red', zorder=10)

    ax[1].set_xscale('log')
    ax[1].set_xlim(10e-5, 10e-2)
    for x in ax:
        x.set_ylim(y_data[y1], y_data[y2])
    plot_align(ax)
    return fig, ax


def plot_all_terms_paper_V2(terms_list, cb_labels, cbar_min=0.06):

    bx_h = np.diff(terms_list[0].height.values)[0]
    nanx = np.any(np.any(np.isnan(np.stack(terms_list)), axis=0), axis=-1)

    fig = plt.figure(figsize=(7,6.5), constrained_layout=False)
    gs = gridspec.GridSpec(5, 1, height_ratios=[1,1,1,1,0.6], hspace=0.1)
    ax = np.empty((5,), dtype='object')
    ax[0] = plt.subplot(gs[0])
    ax[1] = plt.subplot(gs[1])
    ax[2] = plt.subplot(gs[2])
    ax[3] = plt.subplot(gs[3])
    ax[4] = plt.subplot(gs[4])

    for x, f_term, cbl in zip(ax[0:4], terms_list, cb_labels):
        f_term.plot(cmap='PuOr', vmin=-cbar_min, vmax=cbar_min,\
                        ax=x, center=0,\
                        cbar_kwargs={'pad':0.01 , 'label':cbl})

        x.set_ylabel('m ASB')
        x.set_ylim(2.4, f_term.height[-1])

    ## Plot height integral - dC/dt
    C_hg_med, _, _ = height_integrals(terms_list[0].values[...,np.newaxis][~nanx,:], bx_h)
    ax[4].plot(terms_list[0].time.values, C_hg_med, c='orange')

    # Plot height integral - sum of other terms
    ba_hg_med, _, _ = height_integrals((terms_list[1].values +
                                        terms_list[2].values +
                                        terms_list[3].values)[...,np.newaxis][~nanx,:], 
                                        bx_h)
    ax[4].plot(terms_list[0].time.values, ba_hg_med, c='purple')
    ax[4].grid()
    lim = 1.15 * np.max(np.abs(ba_hg_med))
    ax[4].set_ylim(-lim, lim)

    plot_align(ax)
    basic_ts(terms_list[0].time.values, ax)

    myFmt = mdates.DateFormatter('%H:%M')
    ax[4].xaxis.set_major_formatter(myFmt)

    ax[0].text(0.05, 0.85, '(a)', horizontalalignment='center',\
        verticalalignment='center', transform=ax[0].transAxes, size=14)
    ax[1].text(0.05, 0.85, '(b)', horizontalalignment='center',\
        verticalalignment='center', transform=ax[1].transAxes, size=14)
    ax[2].text(0.05, 0.85, '(c)', horizontalalignment='center',\
        verticalalignment='center', transform=ax[2].transAxes, size=14)
    ax[3].text(0.05, 0.85, '(d)', horizontalalignment='center',\
        verticalalignment='center', transform=ax[3].transAxes, size=14)
    ax[4].text(0.05, 0.75, '(e)', horizontalalignment='center',\
        verticalalignment='center', transform=ax[4].transAxes, size=14)
    return fig, ax



def plot_all_terms_paper(terms_list, cb_labels, sig_ntime, xnan, bx_h):
    fig = plt.figure(figsize=(7,6.5), constrained_layout=False)
    gs = gridspec.GridSpec(5, 1, height_ratios=[1,1,1,1,0.6], hspace=0.1)
    ax = np.empty((5,), dtype='object')
    ax[0] = plt.subplot(gs[0])
    ax[1] = plt.subplot(gs[1])
    ax[2] = plt.subplot(gs[2])
    ax[3] = plt.subplot(gs[3])

    ax[4] = plt.subplot(gs[4])

    vmm = -0.06
    vxx = 0.06
    #############################################################################

    for x, f_term, cbl in zip(ax[0:4], terms_list, cb_labels):
        
        ### Subplot pcolor
        t_med = np.median(f_term, axis=-1)
        xr_dtmean = x.pcolor(sig_ntime, bx_h, t_med[xnan,:], cmap='PuOr',\
                                    vmin=vmm, vmax=vxx)
        cb = fig.colorbar(xr_dtmean, ax=x, pad=0.01)
        mpl.colorbar.ColorbarBase.set_label(cb, cbl,\
                                            rotation=0, ha='left', va='center')

        # Pcolor config
        x.set_ylim(2.4, bx_h[-1])
        x.set_xlim(st_min, ed_min)
    #     x.set_yticklabels('')
        x.set_xticklabels('')
        x.set_ylabel('m ASB')
        x.set_xticks(np.arange(np.datetime64('2019-03-13T12:00'),\
                        np.datetime64('2019-03-13T14:01'),\
                        np.timedelta64(30, 'm')))
        
    ###############################################

    #### Plot other stuff

    ## Plot height integral - dC/dt
    C_hg_med, C_hg_low, C_hg_upp = height_integrals(dc_dt_uq.values[xnan,:,:], np.diff(bx_h)[0])
    ax[4].plot(sig_ntime, C_hg_med, c='orange')
    ax[4].fill_between(sig_ntime, C_hg_low, y2=C_hg_upp, color='orange', alpha=0.3)

    # Plot height integral - sum of other terms
    ba_hg_med, b_hg_low, ba_hg_upp = height_integrals(wc_z_turb_uq.values[xnan,:,:] +\
                                                    wc_z_uq.values[xnan,:,:] +\
                                                    uc_z_uq.values[xnan,:,:], np.diff(bx_h)[0])
    ax[4].plot(sig_ntime, ba_hg_med, c='purple')
    ax[4].fill_between(sig_ntime, b_hg_low, y2=ba_hg_upp, color='purple', alpha=0.3)

    #############################################################################

    ## Plot config
    ax_zero = ax[0].get_position().bounds
    ax_one = ax[4].get_position().bounds
    ax[4].set_position([ax_zero[0], ax_one[1], ax_zero[2], ax_one[3]])
    ax[4].set_xlim(st_min, ed_min)
    ax[4].set_ylim(-0.25, 0.25)
    ax[4].grid()
    ax[4].text(1.15, 0.5, '$\\int_{0}^{H}$', horizontalalignment='center',\
        verticalalignment='center', transform=ax[4].transAxes, size=14)
    ax[4].yaxis.set_label_position("right")
    ax[4].yaxis.tick_right()
    ax[4].set_xticks(np.arange(np.datetime64('2019-03-13T12:00'),\
                        np.datetime64('2019-03-13T14:01'),\
                        np.timedelta64(30, 'm')))
    myFmt = mdates.DateFormatter('%H:%M')
    ax[4].xaxis.set_major_formatter(myFmt)

    ax[0].text(0.05, 0.85, '(a)', horizontalalignment='center',\
        verticalalignment='center', transform=ax[0].transAxes, size=14)
    ax[1].text(0.05, 0.85, '(b)', horizontalalignment='center',\
        verticalalignment='center', transform=ax[1].transAxes, size=14)
    ax[2].text(0.05, 0.85, '(c)', horizontalalignment='center',\
        verticalalignment='center', transform=ax[2].transAxes, size=14)
    ax[3].text(0.05, 0.85, '(d)', horizontalalignment='center',\
        verticalalignment='center', transform=ax[3].transAxes, size=14)
    ax[4].text(0.05, 0.75, '(e)', horizontalalignment='center',\
        verticalalignment='center', transform=ax[4].transAxes, size=14)
    return fig, ax


def plot_profiles_paper(terms_list, cb_labels, s_dist, xnan, bx_h, sn_ix, ed_ix):
    fig, ax = plt.subplots(1, 4, figsize=(14,5), gridspec_kw={'wspace':0.07})

    for x, f_term, cbl in zip(ax, terms_list, cb_labels):
        time = f_term['time'].values
        f_term = f_term.values
        
    #     if x == ax[2]:
    #         f_term = f_term + ws_div_all

        t_step = np.diff(time)[0].astype('timedelta64[s]').astype('float')
        t_med = np.nanmedian(f_term, axis=-1)
        ti_t = np.sum(t_med[xnan,:][:,sn_ix], axis=1) * t_step
    #     ti_ma = np.convolve(ti_t, np.ones(5), 'valid') / 5
        
        ti_all_05 = np.percentile(np.sum(f_term[xnan,:,:][:,sn_ix], axis=1)*t_step, 2.5, axis=-1)
    #     co_all_05 = np.convolve(ti_all_05, np.ones(5), 'valid') / 5
        ti_all_95 = np.percentile(np.sum(f_term[xnan,:,:][:,sn_ix], axis=1)*t_step, 97.5, axis=-1)
    #     co_all_95 = np.convolve(ti_all_95, np.ones(5), 'valid') / 5
        
        # x.plot(ti_t, s_dist[xnan], c='royalblue')
        x.fill_betweenx(s_dist[xnan], ti_all_05, ti_all_95, color='royalblue', alpha=0.3)
        x.text(0.75, 0.9, str(int(np.sum(ti_t*bx_h))), transform=x.transAxes,\
            c='royalblue')

        ed_t = np.sum(t_med[xnan,:][:,ed_ix], axis=1) * t_step
        # ed_ma = np.convolve(ed_t, np.ones(5), 'valid') / 5
        
        ti_all_05 = np.percentile(np.nansum(f_term[xnan,:,:][:,ed_ix], axis=1)*t_step, 2.5, axis=-1)
        # co_all_05 = np.convolve(ti_all_05, np.ones(5), 'valid') / 5
        ti_all_95 = np.percentile(np.nansum(f_term[xnan,:,:][:,ed_ix], axis=1)*t_step, 97.5, axis=-1)
        # co_all_95 = np.convolve(ti_all_95, np.ones(5), 'valid') / 5    
        
        # x.plot(ed_t, s_dist[xnan], c='darkorange')
        x.fill_betweenx(s_dist[xnan], ti_all_05, ti_all_95, color='darkorange', alpha=0.3)
        x.text(0.75, 0.8, str(int(np.sum(ed_t*bx_h))), transform=x.transAxes,\
            c='darkorange')
        
        x.set_xlim(-80, 80)
        x.set_ylim(s_dist[xnan][0], s_dist[xnan][-1])
        x.grid()
        x.set_title(cbl)
        
        if x != ax[0]:
            x.set_yticklabels('')
        else:
            x.set_ylabel('m ASB')
            
    ax[0].text(0.1, 0.93, '(a)', horizontalalignment='center',\
        verticalalignment='center', transform=ax[0].transAxes, size=14)
    ax[1].text(0.1, 0.93, '(b)', horizontalalignment='center',\
        verticalalignment='center', transform=ax[1].transAxes, size=14)
    ax[2].text(0.1, 0.93, '(c)', horizontalalignment='center',\
        verticalalignment='center', transform=ax[2].transAxes, size=14)
    ax[3].text(0.1, 0.93, '(d)', horizontalalignment='center',\
        verticalalignment='center', transform=ax[3].transAxes, size=14)
    return fig, ax
