import numpy as np
import cmocean.cm as cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from wootils.plotnice import plot_axislabels, plot_align, basic_ts, vert_stack
from d2spike.utils import nan_gauss_xr
import matplotlib.patheffects as path_effects


# def draw_symbols(ax, xd, ypos, c='k', fs=12, symbol=r'$\blacktriangledown$', ec=None):
#     # Check if xd is length 1
#     if np.size(xd) == 1:
#         ax.text(xd, ypos, symbol, fontsize=fs, ha='center', c=c, ec=ec)
#     else:
#         for xt in xd:
#             ax.text(xt, ypos, symbol, fontsize=fs, ha='center', c=c, ec=ec)  
            
def draw_symbols(ax, xd, ypos, c='k', fs=12, symbol=r'$\blacktriangledown$', ec=None):
    # Check if xd is length 1
    if np.size(xd) == 1:
        txt = ax.text(xd, ypos, symbol, fontsize=fs, ha='center', c=c)
        if ec is not None:
            txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground=ec)])
    else:
        for xt in xd:
            txt = ax.text(xt, ypos, symbol, fontsize=fs, ha='center', c=c)
            if ec is not None:
                txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground=ec)])  


def add_symbols(xd, ax, spine='top', c='k', fs=12, symbol=r'$\blacktriangledown$', ec=None):
    # Check if ax is length 1
    if np.size(ax) == 1:
        top_pos = ax.get_ylim()[1]
        draw_symbols(ax, xd, top_pos, c=c, fs=fs, symbol=symbol, ec=ec)
    else:
        for x in ax:
            top_pos = x.get_ylim()[1]
            draw_symbols(x, xd, top_pos, c=c, fs=fs, symbol=symbol, ec=ec)


def plot_one_endform(ax, ed, sol_tx, wav_tx, ev_sol, ev_wave, pos='topleft', h_ratios=None):

    # Add symbols to axes
    symax = [ax[0], ax[-1]]
    add_symbols(sol_tx[sol_tx < ed], ax=symax, c='k', fs=12)
    add_symbols(wav_tx[wav_tx < ed], ax=symax, c='g', fs=12)
    add_symbols(sol_tx[ev_sol], ax=symax, c='k', fs=20)
    add_symbols(wav_tx[ev_wave], ax=symax, c='g', fs=20)

    plot_axislabels(ax, pos=pos, h_ratios=h_ratios)
    return None


def plot_papersetup(u_mean, w_mean, ds_temp, u_full, w_full, c_full, land_temp, ssc_good, sig_mean_ssc,\
                    ulinemin=0.89, wtick=0.1, umin=0.79, wmin=0.149, cmin=49):
    h_rat = [1,3,2,2,2,1]
    fig, ax = vert_stack(6, hsize=8, vsize=7/6, hspace=0.1, h_ratio=h_rat)
    par = ax[0].twinx()
    par2 = ax[5].twinx()
    axf = np.append(ax, [par, par2]).flatten()

    # Plot mean U and W at single height
    u_mean.plot(ax=ax[0], c='k', linewidth=1.5)
    w_mean.plot(ax=par, c='r', linewidth=1.5)
    ax[0].set_ylabel('$\overline{U}$\n[m s$^{-1}$]')
    ax[0].set_ylim(-ulinemin, ulinemin)
    par.set_ylabel('$\overline{W}$\n[m s$^{-1}$]')
    par.yaxis.label.set_color('red')
    par.tick_params(axis='y', colors='red')
    par.set_ylim(-wtick*(ulinemin/0.5), wtick*(ulinemin/0.5))

    # Plot temperature
    ds_temp.plot.contourf(ax=ax[1], cmap=plt.cm.viridis, levels=np.arange(18,31,1),\
                          cbar_kwargs={'pad':0.01, 'label':'Temperature\n[$^\circ$C]'})
    ds_temp.plot.contour(ax=ax[1], colors='k', linewidths=0.1,\
                         levels=np.arange(18,31,1), add_colorbar=False)

    # Plot U
    u_full.plot(ax=ax[2], cmap='PuOr', vmin=-umin, vmax=umin,\
                cbar_kwargs={'pad': 0.01, 'label': '$U$ [m s$^{-1}$]'})

    # Plot W
    w_full.plot(ax=ax[3], cmap=cm.balance, vmin=-wmin, vmax=wmin,\
                cbar_kwargs={'pad': 0.01, 'label': '$W$ [m s$^{-1}$]'})

    # Plot C
    ax[4].fill_between(c_full.time, 0.5, 1.5, color='grey', alpha=1)
    c_full.plot(ax=ax[4], cmap=cm.turbid, vmin=0, vmax=cmin,\
                cbar_kwargs={'pad': 0.01, 'label': 'C\n[g m$^{-3}$]'})
    ax[4].fill_between(c_full.time, 0.5, 0.9, color='grey', alpha=1)

    # Line plots
    land_temp.plot(ax=ax[5], c='blue', linewidth=1.5)
    ssc_good.plot(ax=par2, c='k', linewidth=1.5)
    sig_mean_ssc.plot(ax=par2, c='grey', linewidth=1.5)
    par2.set_ylim(0, 120)
    par2.set_xlabel('')
    ax[5].yaxis.label.set_color('blue')
    ax[5].tick_params(axis='y', colors='blue')
    ax[5].set_ylabel('T [$^\circ$C]\n0.3 mab')
    par2.set_ylabel('C\n[g m$^{-3}$]')

    #####################
    plot_align(axf)
    basic_ts(u_mean.time.values, axf)

    for xiax, x in enumerate(ax):
        if (xiax > 0) & (xiax < 5):
            x.set_ylabel('m ASB')

    myFmt = mdates.DateFormatter('%H:%M')
    ax[5].xaxis.set_major_formatter(myFmt)
    plot_axislabels(ax, pos='topleft', h_ratios=h_rat)
    ax[0].grid()
    return fig, ax



def plot_paperflux(terms_list, cb_labels, ylim=(2.4, 7.42), cbar_min=0.149, pltgf_t=30, pltgf_z=4, sumlim=10):

    h_rat = [1,1,1,1,0.6]
    qtile = 0.341
    fig, ax = vert_stack(5, hsize=8, vsize=7/5, hspace=0.1, h_ratio=h_rat)

    for x, f_term, cbl in zip(ax[:-1], terms_list, cb_labels):
        # f_term_plt = nan_gauss_xr(f_term.mean(dim='sample'), [pltgf_z, pltgf_t])
        f_term.median(dim='sample').plot(cmap='PuOr', vmin=-cbar_min, vmax=cbar_min,\
                        ax=x, center=0,\
                        cbar_kwargs={'pad':0.01 , 'label':cbl})
        x.set_ylabel('m ASB')
        x.set_ylim(ylim[0], ylim[1])

    ## Plot height integral - dC/dt
    # dc_dt_int = nan_gauss_xr(terms_list[0], [pltgf_z, pltgf_t], axis=[0,1]).sel(height=slice(ylim, None)).integrate('height')
    dc_dt_int = terms_list[0].sel(height=slice(ylim[0], ylim[1])).integrate('height')
 
    ax[4].plot(terms_list[0].time.values, dc_dt_int.median(dim='sample'), c='orange', zorder=10)
    ax[4].fill_between(terms_list[0].time.values, dc_dt_int.quantile(0.5-qtile, dim='sample'),\
                        dc_dt_int.quantile(0.5+qtile, dim='sample'), color='orange', alpha=0.5, zorder=9)

    # Plot height integral - sum of other terms
    # dc_oth_int = nan_gauss_xr((terms_list[1] + terms_list[2] + terms_list[3]), [pltgf_z, pltgf_t], axis=[0,1])\
    #                         .sel(height=slice(ylim, None)).integrate('height')
    dc_oth_int = (terms_list[1] + terms_list[2] + terms_list[3])\
                            .sel(height=slice(ylim[0], ylim[1])).integrate('height')
        
    ax[4].plot(terms_list[0].time.values, dc_oth_int.median(dim='sample'), c='purple', zorder=8)
    ax[4].fill_between(terms_list[0].time.values, dc_oth_int.quantile(0.5-qtile, dim='sample', skipna=False),\
                        dc_oth_int.quantile(0.5+qtile, dim='sample', skipna=False), color='purple', alpha=0.5, zorder=7)

    ax[4].grid()
    # lim = 1.5 * np.nanmax(np.abs([dc_oth_int.quantile(0.5-qtile, dim='sample', skipna=False),\
    #                               dc_dt_int.quantile(0.5-qtile, dim='sample')]))
    ax[4].set_ylim(-sumlim, sumlim)
    ax[4].set_ylabel('[g m$^{-2}$ s$^{-1}$]')

    plot_align(ax)
    basic_ts(terms_list[0].time.values, ax)

    myFmt = mdates.DateFormatter('%H:%M')
    ax[4].xaxis.set_major_formatter(myFmt)
    plot_axislabels(ax, pos='topleft', h_ratios=h_rat)

    return fig, ax



# def plot_paperprofiles(t_list, wf, cb_labels, xlim=249, ylim=(2.4,7.42), z_smooth=4):

#     qtile = 0.341

#     # Plot the summary summs
#     sn_ix = (t_list[0]['time'].values >= wf[0]) & (t_list[0]['time'].values <= wf[1])
#     ed_ix = (t_list[0]['time'].values >= wf[1]) & (t_list[0]['time'].values <= wf[2])

#     s_dist = t_list[0].height.sel(height=slice(ylim[0], ylim[1])).values
#     xlim_adjust = 0 

#     fig, ax = plt.subplots(1, 4, figsize=(8,2.5), gridspec_kw={'wspace':0.07})
#     for x, f_term, cbl in zip(ax, t_list, cb_labels):

#         ti_gf = nan_gauss_xr(f_term.isel(time=sn_ix), z_smooth, axis=[0]).sel(height=slice(ylim[0], ylim[1]))
#         ti_t = ti_gf.integrate('time', datetime_unit='s').median('sample')

#         x.plot(ti_t, s_dist, c='royalblue')
#         ti_05 = ti_gf.integrate('time', datetime_unit='s').quantile(0.5-qtile, dim='sample', skipna=False)
#         ti_95 = ti_gf.integrate('time', datetime_unit='s').quantile(0.5+qtile, dim='sample', skipna=False)
#         x.fill_betweenx(s_dist, x1=ti_05, x2=ti_95, color='royalblue', alpha=0.5)
#         x.text(0.75, 0.9, str(int(ti_t.integrate('height'))), transform=x.transAxes,\
#                c='royalblue')

#         ed_gf = nan_gauss_xr(f_term.isel(time=ed_ix), z_smooth, axis=[0]).sel(height=slice(ylim[0], ylim[1]))  
#         ed_t = ed_gf.integrate('time', datetime_unit='s').median('sample')
#         x.plot(ed_t, s_dist, c='indianred')
#         tl_05 = ed_gf.integrate('time', datetime_unit='s').quantile(0.5-qtile, dim='sample', skipna=False)
#         tl_95 = ed_gf.integrate('time', datetime_unit='s').quantile(0.5+qtile, dim='sample', skipna=False)
#         x.fill_betweenx(s_dist, x1=tl_05, x2=tl_95, color='indianred', alpha=0.5)
#         x.text(0.75, 0.8, str(int(ed_t.integrate('height'))), transform=x.transAxes,\
#                c='indianred')
        
#         xlim_adjust = np.nanmax([xlim_adjust, np.nanmax(np.abs([ti_05, ti_95, tl_05, tl_95]))])
#         # x.set_xlim(-xlim, xlim)
#         # x.set_xticks([-200,0,200])
#         x.set_ylim(ylim[0], ylim[1])
#         x.grid()
#         x.set_title(cbl)
        
#         if x != ax[0]:
#             x.set_yticklabels('')
#         else:
#             x.set_ylabel('m ASB')
#         x.set_xlabel('[g m$^{-3}$]')

#     for x in ax:
#         x.set_xlim(-xlim_adjust, xlim_adjust)
#         # Set ticks to nearest 10 of xlim/2
#         xlim_round = np.round(xlim_adjust/2, -1)
#         x.set_xticks(np.arange(-xlim_round, xlim_round+1, xlim_round))

#     return fig, ax



def plot_paperprofiles_v2(t_list, wf, cb_labels, xlim=249, ylim=(2.4,7.42), z_smooth=4):

    qtile = 0.341

    # Plot the summary summs
    sn_ix = (t_list[0]['time'].values >= wf[0]) & (t_list[0]['time'].values <= wf[1])
    ed_ix = (t_list[0]['time'].values >= wf[1]) & (t_list[0]['time'].values <= wf[2])

    s_dist = t_list[0].height.sel(height=slice(ylim[0], ylim[1])).values
    xlim_adjust = 0 

    fig, ax = plt.subplots(2, 4, figsize=(12,6), gridspec_kw={'wspace':0.07, 'hspace':0.05})
    for x, z, f_term in zip(ax[0], ax[1], t_list):

        ti_gf = nan_gauss_xr(f_term.isel(time=sn_ix), z_smooth, axis=[0]).sel(height=slice(ylim[0], ylim[1]))
        ti_t = ti_gf.integrate('time', datetime_unit='s').median('sample')

        x.plot(ti_t, s_dist, c='royalblue')
        ti_05 = ti_gf.integrate('time', datetime_unit='s').quantile(0.5-qtile, dim='sample', skipna=False)
        ti_95 = ti_gf.integrate('time', datetime_unit='s').quantile(0.5+qtile, dim='sample', skipna=False)
        x.fill_betweenx(s_dist, x1=ti_05, x2=ti_95, color='royalblue', alpha=0.5)
        x.text(0.75, 0.9, str(int(ti_t.integrate('height'))), transform=x.transAxes,\
               c='royalblue')

        ed_gf = nan_gauss_xr(f_term.isel(time=ed_ix), z_smooth, axis=[0]).sel(height=slice(ylim[0], ylim[1]))  
        ed_t = ed_gf.integrate('time', datetime_unit='s').median('sample')
        z.plot(ed_t, s_dist, c='indianred')
        tl_05 = ed_gf.integrate('time', datetime_unit='s').quantile(0.5-qtile, dim='sample', skipna=False)
        tl_95 = ed_gf.integrate('time', datetime_unit='s').quantile(0.5+qtile, dim='sample', skipna=False)
        z.fill_betweenx(s_dist, x1=tl_05, x2=tl_95, color='indianred', alpha=0.5)
        z.text(0.75, 0.8, str(int(ed_t.integrate('height'))), transform=x.transAxes,\
               c='indianred')
        
    for ix, (x, cbl) in enumerate(zip(ax.flatten(), cb_labels)):
        xlim_adjust = np.nanmax([xlim_adjust, np.nanmax(np.abs([ti_05, ti_95, tl_05, tl_95]))])
        # x.set_xlim(-xlim, xlim)
        # x.set_xticks([-200,0,200])
        x.set_ylim(ylim[0], ylim[1])
        x.grid()
        x.set_title(cbl)
        if x != ax[0]:
            x.set_yticklabels('')
        else:
            x.set_ylabel('m ASB')
        if ix < 4:
            x.set_xticklabels('')
        else:
            x.set_xlabel('[g m$^{-3}$]')

    for x in ax:
        x.set_xlim(-xlim_adjust, xlim_adjust)
        # Set ticks to nearest 10 of xlim/2
        xlim_round = np.round(xlim_adjust/2, -1)
        x.set_xticks(np.arange(-xlim_round, xlim_round+1, xlim_round))

    return fig, ax


def plot_appsummary(ds_temp, u_mean, w_mean, c_turb, terms_list, cb_labels, ylim=(2.4,7.4), cbar_min=0.149, pltgf_t=30, pltgf_z=4):
        
    qtile = 0.341

    h_rat = [3,1,2,2,2,1]
    fig, ax = vert_stack(6, hsize=8, vsize=7/6, hspace=0.1, h_ratio=h_rat)
    par = ax[1].twinx()
    axf = np.append(ax, par).flatten()

    ### Plot temperature
    ds_temp.plot.contourf(ax=ax[0], cmap=plt.cm.viridis, levels=np.arange(18,31,1),\
                        cbar_kwargs={'pad':0.01, 'label':'Temperature\n[$^\circ$C]'})
    ds_temp.plot.contour(ax=ax[0], colors='k', linewidths=0.1,\
                        levels=np.arange(18,31,1), add_colorbar=False)
    ax[0].set_ylabel('m ASB')

    ### Plot mean U and W at single height
    u_mean.plot(ax=ax[1], c='k', linewidth=1.5)
    w_mean.plot(ax=par, c='r', linewidth=1.5)
    ax[1].set_ylabel('$\overline{U}$\n[m s$^{-1}$]')
    ax[1].set_ylim(-0.8, 0.8)
    par.set_ylabel('$\overline{W}$\n[m s$^{-1}$]')
    par.yaxis.label.set_color('red')
    par.tick_params(axis='y', colors='red')
    par.set_ylim(-0.04, 0.04)

    ### Plot C
    ax[2].fill_between(c_turb.time, 0.5, 1.5, color='grey', alpha=1)
    c_turb.plot(ax=ax[2], cmap=cm.turbid, vmin=0, vmax=39,\
                cbar_kwargs={'pad': 0.01, 'label': 'C\n[g m$^{-3}$]'})
    ax[2].set_ylim(ylim[0], ylim[1])

    ### Plot WC
    # f_term_plt = nan_gauss_xr(terms_list[1].mean(dim='sample'), [pltgf_z, pltgf_t])
    f_term_plt = terms_list[1].mean(dim='sample')
    cbar_adjust = np.percentile(np.abs(f_term_plt.sel(height=slice(ylim[0], ylim[1])).values), 99.95)
    f_term_plt.plot(cmap='PuOr', vmin=-cbar_adjust, vmax=cbar_adjust,\
                    ax=ax[3], center=0,\
                    cbar_kwargs={'pad':0.01 , 'label':(cb_labels[1] + '\n[g m$^{-3}$ s$^{-1}$]')})
    ax[3].set_ylabel('m ASB')
    ax[3].set_ylim(ylim[0], ylim[1])

    ### Plot w'c'
    # f_term_plt = nan_gauss_xr(terms_list[2].mean(dim='sample'), [pltgf_z, pltgf_t])
    f_term_plt = terms_list[2].mean(dim='sample')
    f_term_plt.plot(cmap='PuOr', vmin=-cbar_adjust, vmax=cbar_adjust,\
                    ax=ax[4], center=0,\
                    cbar_kwargs={'pad':0.01 , 'label':(cb_labels[2] + '\n[g m$^{-3}$ s$^{-1}$]')})
    ax[4].set_ylabel('m ASB')
    ax[4].set_ylim(ylim[0], ylim[1])

    ### Plot term summation
    # dc_dt_int = nan_gauss_xr(terms_list[0], [pltgf_z, pltgf_t], axis=[0,1]).sel(height=slice(ylim[0], ylim[1])).integrate('height')
    dc_dt_int = terms_list[0].sel(height=slice(ylim[0], ylim[1])).integrate('height')
    ax[5].plot(terms_list[0].time.values, dc_dt_int.median(dim='sample'), c='orange', zorder=8)
    ax[5].fill_between(terms_list[0].time.values, dc_dt_int.quantile(0.5-qtile, dim='sample'),\
                        dc_dt_int.quantile(0.5+qtile, dim='sample'), color='orange', alpha=0.5, zorder=7)

    # Plot height integral - sum of other terms
    # dc_oth_int = nan_gauss_xr((terms_list[1] + terms_list[2] + terms_list[3]), [pltgf_z, pltgf_t], axis=[0,1])\
    #                         .sel(height=slice(ylim[0], ylim[1])).integrate('height')
    dc_oth_int = (terms_list[1] + terms_list[2] + terms_list[3]).sel(height=slice(ylim[0], ylim[1])).integrate('height')
    ax[5].plot(terms_list[0].time.values, dc_oth_int.mean(dim='sample'), c='purple', zorder=10)
    ax[5].fill_between(terms_list[0].time.values, dc_oth_int.quantile(0.5-qtile, dim='sample', skipna=False),\
                        dc_oth_int.quantile(0.5+qtile, dim='sample', skipna=False), color='purple', alpha=0.5, zorder=9)
    lim = 1.15 * np.nanmax(np.abs([dc_dt_int, dc_oth_int]))
    ax[5].set_ylim(-lim, lim)
    ax[5].set_ylabel('[g m$^{-2}$ s$^{-1}$]')

    #####################
    plot_align(axf)
    basic_ts(u_mean.time.values, axf)

    for xiax, x in enumerate(ax):
        if (xiax > 1) & (xiax < 5):
            x.set_ylabel('m ASB')
        if (xiax > 1) & (xiax < 5):
            x.set_yticks(np.arange(3.0, 6.01, 3.0))

    myFmt = mdates.DateFormatter('%H:%M')
    ax[-1].xaxis.set_major_formatter(myFmt)
    plot_axislabels(ax, pos='topleft', h_ratios=h_rat)
    ax[1].grid()
    ax[-1].grid()

    return fig, ax



# def plot_fullsetup(ds_sig, tx_sig, ds_temp, tx_temp, ds_land, tx_land):
#     h_rat = [1,3,2,2,2,1]
#     fig, ax = vert_stack(6, hsize=8, vsize=7/6, hspace=0.1, h_ratio=h_rat)

#     #####################

#     # Plot mean U and W at single height
#     ix_sig = np.argmin(np.abs(ds_sig.height.values - 4.0))
#     xr.plot.plot(-1*ds_sig['vel_xyz'].isel(cartesian_axes=1, height=ix_sig, time=tx_sig)\
#                 .rolling(time=300*4, center=True, min_periods=1).mean()[::40],\
#                 ax=ax[0], c='k', linewidth=1.5)

#     par = ax[0].twinx()
#     xr.plot.plot(ds_sig['vel_enu'].isel(cartesian_axes=2, height=ix_sig, time=tx_sig)\
#                 .rolling(time=300*4, center=True, min_periods=1).mean()[::40],\
#                 ax=par, c='r', linewidth=1.5)

#     ax[0].set_ylabel('$\overline{U}$\n[m s$^{-1}$]')
#     ax[0].set_ylim(-0.8, 0.8)
#     ax[0].grid()
#     par.set_title('')

#     par.set_ylabel('$\overline{W}$\n[m s$^{-1}$]')
#     par.yaxis.label.set_color('red')
#     par.tick_params(axis='y', colors='red')
#     par.set_ylim(-0.04, 0.04)

#     #####################

#     # Plot temperature
#     xr_t = ax[1].contourf(ds_temp['time'].values[tx_temp], ds_temp['depth'].values[1:],\
#                         ds_temp['Temperature'].values[1:,tx_temp],\
#                         levels=np.arange(18,31,1), cmap=plt.cm.viridis)
#     xr_tc = ax[1].contour(xr_t, colors='k', linewidths=0.1)
#     cb = fig.colorbar(xr_t, ax=ax[1], pad=0.01, ticks=np.arange(20,29.1,3))
#     mpl.colorbar.ColorbarBase.set_label(cb, 'Temperature\n[$^\circ$C]')

#     #####################

#     xr_sig = xr.plot.plot(-1*ds_sig['vel_xyz'].isel(cartesian_axes=1, time=tx_sig),\
#                             ax=ax[2], cmap='PuOr', vmin=-0.6, vmax=0.6, cbar_kwargs={'pad': 0.01})
#     cb2 = xr_sig.colorbar
#     mpl.colorbar.ColorbarBase.set_label(cb2, '$\overline{U}$ [m s$^{-1}$]')

#     #####################

#     xr_sig = xr.plot.plot(ds_sig['vel_xyz'].isel(cartesian_axes=3, time=tx_sig),\
#                             ax=ax[3], cmap=cmocean.cm.balance, vmin=-0.03, vmax=0.03, cbar_kwargs={'pad': 0.01})
#     cb2 = xr_sig.colorbar
#     mpl.colorbar.ColorbarBase.set_label(cb2, '$\overline{W}$ [m s$^{-1}$]')

#     #####################

#     # Plot ADCP SSC
#     ax[4].fill_between([ds_sig.time[tx_sig][0], ds_sig.time[tx_sig][-1]], [0,0], y2=[9,9], color='gray', zorder=0)
#     xr_im_e = xr.plot.plot(sig_ssc.T, cmap=cmocean.cm.turbid, ax=ax[4],\
#                                     vmin=0, vmax=55, cbar_kwargs={'pad': 0.01})
#     cb=xr_im_e.colorbar
#     mpl.colorbar.ColorbarBase.set_label(cb, 'C\n[g m$^{-3}$]')

#     #####################

#     # Line plots
#     temp_rol = pd.Series(ds_land['Temperature']).rolling(5, center=True).mean()
#     ax[5].plot(ds_land['time'][tx_land][::5], temp_rol[tx_land][::5], c='grey', linewidth=1.5)

#     par2 = ax[5].twinx()
#     # xr.plot.plot((ds_l_ntu['NTU'][obsl_ix]*2.62 - 4.38)/2.33, ax=par2, color='k')
#     # xr.plot.plot((ds_h_ntu['NTU'][obsh_ix]*2.52 - 3.95)/2.33, ax=par2, color='blue')
#     # par2.set_ylim(0, 120)
#     # ax[5].yaxis.label.set_color('grey')
#     # ax[5].tick_params(axis='y', colors='grey')

#     #####################

#     ax_zero = ax[1].get_position().bounds

#     for xiax, x in enumerate(ax):
#         x.set_title('')
#         if xiax < 6:
#             x.set_xlim(ds_sig.time[tx_sig][0], ds_sig.time[tx_sig][-1])
#         if (xiax > 0):
#             x.set_ylabel('m ASB')
#         if xiax < 5:
#             x.set_xlabel('')
#             x.set_xticklabels('')
#         if (x == ax[5]) | (x == ax[0]):
#             ax_one = x.get_position().bounds
#             x.set_position([ax_zero[0], ax_one[1], ax_zero[2], ax_one[3]])

#     ax[5].set_ylabel('T [$^\circ$C]\n0.3 mab')
#     par2.set_ylabel('C\n[g m$^{-3}$]')
#     myFmt = mdates.DateFormatter('%H:%M')
#     ax[5].xaxis.set_major_formatter(myFmt)

#     plot_axislabels(ax, pos='topleft', h_ratios=h_rat)
#     return fig, ax
