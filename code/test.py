
# from datetime import datetime, timedelta


# tgt_time = datetime.strptime('2010010100', "%Y%m%d%H")+timedelta(hours=1*6)

# tgt_time=tgt_time.strftime("%Y-%m-%dT%H:%M:%S")
# print(tgt_time)
import jax
# from jax.lib import xla_bridge
# import torch
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())
print(jax.devices())  # 打印设备列表
# print(xla_bridge.get_backend().platform)  # 获取后端平台信息
import os

l=os.listdir("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/PanGU_infer_ERA5_6hour/00")
l1 = os.listdir("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/PanGU_infer_ERA5_6hour/12")
print(len(l),len(l1))

'''
import os
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
pl_lev = [700, 850, 1000]
apl_sfc = ['tcno2', 'tc_no', 'gtco3', 'pm10', 'pm2p5', 'pm1', 'tcco', 'tcso2']
apl_pl_ = ['co', 'no2', 'no', 'so2', 'go3']
apl_ems = ['nh3_ems', 'co_ems', 'so2_ems', 'nox_ems']
apl_pl = []
for ipl in apl_pl_:
    for ilev in pl_lev:
        apl_pl.append(f"{ipl}_{ilev}")
apl_fac = apl_pl + apl_sfc
def __mkdir__(file_name):
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except:
            pass
import cmaps
cmap_0 = cmaps.MPL_coolwarm
def doraw_conf_1ax(ida, iax, data_type, vmin, vmax):
    ida.plot.contourf(ax=iax, cmap=cmap_0,  vmin=vmin, vmax=vmax)
    iax.add_feature(cfeature.COASTLINE, edgecolor='k', linewidth=0.5)
    mean = np.round(ida.mean().values, )
    iax.set_title(f'{data_type} Data ||| mean {mean}')
def doraw_pcmesh_1ax(ida, iax, data_type, vmin, vmax):
    ida.plot.contourf(ax=iax, cmap=cmap_0,  vmin=vmin, vmax=vmax)
    iax.add_feature(cfeature.COASTLINE, edgecolor='k', linewidth=0.5)
    mean = np.round(ida.mean().values, )
    iax.set_title(f'{data_type} Data ||| mean {mean}')
def compute_mae(out, tgt, skipna=True):
    if "normal" in out.dims:
        out = out.isel(normal=0, drop=True)
    if "member" in out.dims:
        out = out.mean("member")
    weights = np.cos(np.deg2rad(np.abs(tgt.lat)))
    error = np.abs(out - tgt)
    error = error.where(weights > 0, 0)
    if "time" in tgt.dims:
        mae = error.weighted(weights).mean(("time", "lat", "lon"), skipna=skipna)
    else:
        mae = error.weighted(weights).mean(("lat", "lon"), skipna=skipna)
    return mae
def compute_rmse(out, tgt, skipna=True):
    if "normal" in out.dims:
        out = out.isel(normal=0, drop=True)
    if "member" in out.dims:
        out = out.mean("member")
    weights = np.cos(np.deg2rad(np.abs(tgt.lat)))
    error = (out - tgt) ** 2 
    error = error.where(weights > 0, 0)
    if "time" in tgt.dims:
        rmse = np.sqrt(error.weighted(weights).mean(("time", "lat", "lon"), skipna=skipna))
    else:
        rmse = np.sqrt(error.weighted(weights).mean(("lat", "lon"), skipna=skipna))
    return rmse
def compute_mae(out, tgt, skipna=True):
    if "normal" in out.dims:
        out = out.isel(normal=0, drop=True)
    if "member" in out.dims:
        out = out.mean("member")
    weights = np.cos(np.deg2rad(np.abs(tgt.lat)))
    error = np.abs(out - tgt)
    error = error.where(weights > 0, 0)
    if "time" in tgt.dims:
        mae = error.weighted(weights).mean(("time", "lat", "lon"), skipna=skipna)
    else:
        mae = error.weighted(weights).mean(("lat", "lon"), skipna=skipna)
    return mae
def plot_res_vars(da_ls, vars, save_path, itime_str, istep):
    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(3, 2, figsize=(8, 5), subplot_kw={"projection": projection})
    dty_ls = ['lab', 'out', 'base']
    vmin = da_ls[0].min().values
    vmax = da_ls[0].max().values
    for i in range(3):
        # for j in range(2):
        if i ==2:
            vmin = da_ls[2].min().values
            vmax = da_ls[2].max().values
        doraw_conf_1ax(da_ls[i], axes[i, 0], dty_ls[i]+f'--{vars}', vmin, vmax)    
        doraw_pcmesh_1ax(da_ls[i], axes[i, 1], dty_ls[i]+f'--{vars}', vmin, vmax) 
    mae = compute_mae(da_ls[0], da_ls[1])
    mae = mae.values.item()
    title = "{} {} MAE {:.6f}".format(itime_str, vars, mae)
    fig.suptitle(title)
    save_fn = os.path.join(save_path, itime_str, f'{vars}_step{istep}_map.png')
    __mkdir__(save_fn)
    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    fig.savefig(save_fn)
    plt.close()
# def out_mae(o_ds, l_ds, step):
#     out_ls = [
#         ['tcno2', 'tc_no', 'gtco3', 'pm10', 'pm2p5', 'pm1', 'tcco', 'tcso2'],
#         [f'co_{ipl}' for ipl in [100, 700, 850, 1000]] + [f'no_{ipl}' for ipl in [100, 700, 850, 1000]],
#         [f'so2_{ipl}' for ipl in [100, 700, 850, 1000]] + [f'no2_{ipl}' for ipl in [100, 700, 850, 1000]],
#         [f'go3_{ipl}' for ipl in [100, 300, 500, 700, 850, 1000]],
#     ]
#     mae = compute_mae(o_ds, l_ds)
#     # df = mae.to_dataframe().reset_index()
#     values = {var: mae[var].values.flatten()[0] for var in mae.data_vars}
#     # 将字典转换为 DataFrame
#     df_single_value = pd.DataFrame([values])
#     print('{:#^50}'.format(f" step {step} MAE  "))
#     for iapl_ls in out_ls:
#         print(df_single_value[iapl_ls])
#     rmse = compute_rmse(o_ds, l_ds)
#     values = {var: rmse[var].values.flatten()[0] for var in rmse.data_vars}
#     # 将字典转换为 DataFrame
#     print('{:#^50}'.format(f" step {step} RMSE  "))
#     df_single_value = pd.DataFrame([values])
#     for iapl_ls in out_ls:
#         print(df_single_value[iapl_ls])
def out_mae(o_ds, l_ds):
    out_ls = ['tcno2', 'tc_no', 'gtco3', 'pm10', 'pm2p5', 'pm1', 'tcco', 'tcso2']+ \
        [f'co_{ipl}' for ipl in [100, 700, 850, 1000]] + [f'no_{ipl}' for ipl in [100, 700, 850, 1000]]+ \
        [f'so2_{ipl}' for ipl in [100, 700, 850, 1000]] + [f'no2_{ipl}' for ipl in [100, 700, 850, 1000]]+ \
        [f'go3_{ipl}' for ipl in [100, 300, 500, 700, 850, 1000]]
    stic_dict = {'Variable': out_ls}
    for istep in o_ds.step.values:
        mae_ = compute_mae(o_ds.sel(step=istep), l_ds.sel(step=istep))
        rmse_ = compute_rmse(o_ds.sel(step=istep), l_ds.sel(step=istep))
        idict = {f'Step {istep} MAE': [mae_[var].values.item() for var in out_ls],
                 f'Step {istep} RMSE': [rmse_[var].values.item()  for var in out_ls]}
        stic_dict.update(idict)
    df = pd.DataFrame(stic_dict)
    # for iapl_ls in out_ls:
    #     df_subset = df[df['Variable'].isin(iapl_ls)]
    print('{:#^100}'.format(f" Variables "))
    print(df.to_string(index=False))
if __name__ == '__main__':
    # res_dir = 'work_dir/experiments_Unet_model_Micro/Unet_model_0_11_debug_results/eval/epoch_20/eval/origin_ds'
    # res_dir = 'work_dir/experiments_Unet_model_Micro/Unet_model_0_out_scaler_results/eval/epoch_11/eval/origin_ds'
    # res_dir = 'work_dir/experiments_Unet_model_micor/Unet_model_0721_out_scaler_0_results/eval/epoch_35/eval/origin_ds'
    # res_dir = './work_dir/experiments_Unet_model_micor_result/Unet_model_0722_out_scaler_results_0/eval/epoch_15/eval/origin_ds'
    # res_dir = './work_dir/experiments_Unet_model_micor_result/Unet_model_0722_out_scaler_results_0/eval/epoch_15/eval/origin_ds/'
    # res_dir = 'work_dir/experiments_Unet_model_micor_result/Unet_model_0722_out_scaler_results_0/eval/epoch_15/eval/origin_ds'
    # res_dir = 'work_dir/experiments_Unet_model_micor_result/Unet_model_0722_out_scaler_results_0/eval/epoch_45/eval/origin_ds'
    # res_dir = 'work_dir/experiments_Unet_model_micor_result/Unet_model_0722_out_scaler_results_0/eval/epoch_49/eval/origin_ds'
    scaler_fn = '/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanxu/data/CMAS/hourly/version_1/statistic/apl_max_mean.nc'
    scaler = xr.open_dataset(scaler_fn)
    save_path = os.path.join(res_dir, 'res_imgs')
    o_ds = xr.open_dataset(os.path.join(res_dir, 'outs.nc'))
    l_ds = xr.open_dataset(os.path.join(res_dir, 'labs.nc'))
    b_ds = xr.open_dataset(os.path.join(res_dir, 'base.nc'))
    o_ds = o_ds * scaler/2*1000*1000
    l_ds = l_ds * scaler/2*1000*1000
    # for istep in range(4):
    #     out_mae(o_ds.sel(step=istep), l_ds.sel(step=istep), istep)
    out_mae(o_ds, l_ds)
    da_ls = [l_ds, o_ds, b_ds]
    apl_fac = ['tcno2', 'tc_no', 'gtco3', 'pm10', 'pm2p5'] + apl_pl
    for itime in l_ds.time.values:
        bs_time = pd.to_datetime(itime) - pd.Timedelta('12H')
        time_str = pd.to_datetime(itime).strftime('%Y%m%d%H')
        for ivar in apl_fac:
            for istep in range(4):
                da_ls = [l_ds.sel(time=itime, step=istep)[ivar], 
                         o_ds.sel(time=itime, step=istep)[ivar], 
                         b_ds.sel(time=itime, step=istep)[ivar]]
                plot_res_vars(da_ls, ivar, save_path, time_str, istep)


'''