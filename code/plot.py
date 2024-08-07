import os
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
import time
# vals=[ 'z50', 'z100', 'z150', 'z200', 'z250', 'z300', 'z400', 'z500', 'z600',
#        'z700', 'z850', 'z925', 'z1000', 't50', 't100', 't150', 't200', 't250',
#        't300', 't400', 't500', 't600', 't700', 't850', 't925', 't1000', 'u50',
#        'u100', 'u150', 'u200', 'u250', 'u300', 'u400', 'u500', 'u600', 'u700',
#        'u850', 'u925', 'u1000', 'v50', 'v100', 'v150', 'v200', 'v250', 'v300',
#        'v400', 'v500', 'v600', 'v700', 'v850', 'v925', 'v1000', 'q50', 'q100',
#        'q150', 'q200', 'q250', 'q300', 'q400', 'q500', 'q600', 'q700', 'q850',
#        'q925', 'q1000', 'clwc50', 'clwc100', 'clwc150', 'clwc200', 'clwc250',
#        'clwc300', 'clwc400', 'clwc500', 'clwc600', 'clwc700', 'clwc850',
#        'clwc925', 'clwc1000', 't2m', 'u10m', 'v10m','msl', 'tp']
# fn = '/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanxu/data/ERA5/zarr/6hourly/fuxi_20_all/2002_2022.c92.p25'
# data = xr.open_zarr(fn)

# # start_time = '2020'
# # end_time = '2020'
# data = data.sel(time='2018-01-01',channel ='t2m')
# print()
# level = [50, 100, 150,200, 250, 300, 400, 500, 600,700, 850, 925, 1000]
# def get_level_name(name):
#     return [f'{name}{i}' for i in level]
# '''
# # z t u v q
# '''
# z = data.sel(channel = get_level_name('z'))   # temperature
# t = data.sel(channel = get_level_name('t'))  # geopotential
# ERA5 = xr.merge([z,t])
# u = data.sel(channel = get_level_name('u'))   # u_component_of_wind
# ERA5 = xr.merge([ERA5,u])
# v = data.sel(channel = get_level_name('v'))   # v_component_of_wind
# ERA5 = xr.merge([ERA5,v])
# q = data.sel(channel = get_level_name('q'))   # specific_humidity 
# ERA5 = xr.merge([ERA5,q])

# print(ERA5.info())
# # 't2m', 'u10m', 'v10m','msl', 'tp','toa_incident_solar_radiation'
# # 10m_u_component_of_wind   ['time','channel','lat','lon']
# u10m = data.sel(channel=['u10m']) # (1460, 1, 721, 1440)  
# msl = data.sel(channel=['msl'])  # mean_sea_level_pressure
# t2m = data.sel(channel=['t2m'])   # 2meter_temperature
# v10m = data.sel(channel=['v10m'])  # 10m_v_component_of_wind
# tp = data.sel(channel=['tp'])  # total_precipitation_6hr



# ds = xr.Dataset({
#     'temperature': (['time', 'level','lat','lon'], t),
#     'geopotential': (['time', 'level','lat','lon'], z),
#     'u_component_of_wind ': (['time', 'level','lat','lon'], u),
#     'v_component_of_wind': (['time', 'level','lat','lon'],v),
#     'specific_humidityl': (['time', 'level','lat','lon'],q),
#     '10m_u_component_of_wind':(['time','lat','lon'],u10m),
#     'mean_sea_level_pressure':(['time','lat','lon'],msl),
#     'total_precipitation_6hr':(['time','lat','lon'], tp),
#     '10m_v_component_of_wind':(['time','lat','lon'], v10m),
#     '2meter_temperature':(['time','lat','lon'], t2m),
#     },
#     coords={
#         'time': time,  
#         'lat': lat, 
#         'lon': lon, 
#         'level':level
#     })

# # ds1 = other_data()
# # ds = xr.merge([ds, ds1])
# print(ds.info())


import numpy as np
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
def inv_normalize(ds, mean, std):
    #print(mean.data,std.data)
    ds = ds * std + mean
    return ds
def get_level_name(name,level):
    return [f'{name}{i}' for i in level]
def generate_daily_dates(start_time, end_time):
    """
    Parameters:
    - start_time (str): Start date in 'YYYYMMDD' format.
    - end_time (str): End date in 'YYYYMMDD' format.
    - interval (str): Interval for generating dates. Options: '12h' or '1 day'.
    Returns:
    - list: List of dates in 'YYYYMMDDHH' format.
    """
    start_datetime = datetime.strptime(start_time, '%Y%m%d')
    end_datetime = datetime.strptime(end_time, '%Y%m%d')

    date_sequence = []
    current_datetime = start_datetime 
    while current_datetime <= end_datetime:
        date_sequence.append(current_datetime.strftime('%Y%m%d'))
        current_datetime += timedelta(days=1)

    return date_sequence

def compute_rmse(out, tgt, skipna=True):
    # print(out.mean(dim='z500'))
    # print(tgt.mean(dim='z500'))
    if "normal" in out.dims:
        out = out.isel(normal=0, drop=True)
    if "member" in out.dims:
        out = out.mean("member")
    weights = np.cos(np.deg2rad(np.abs(tgt.lat))) 
    weights = weights / weights.sum() * len(tgt['lat'])
    error = (out - tgt) ** 2 
    error = error.where(weights > 0, 0)
    rmse = np.sqrt(error.weighted(weights).mean(("lat", "lon"), skipna=skipna))
    #ds = xr.Dataset(dict(rmse=rmse))
    return rmse.values

def compute_acc(tgt, out, mean,skipna=True):
    w = np.cos(np.deg2rad(np.abs(tgt.lat))) 
    w = w / w.sum() * len(tgt['lat'])
    tgt = tgt -mean
    out = out-mean
    A = (w * out * tgt).sum(("lat", "lon"), skipna=skipna)
    B = (w * out**2).sum(("lat", "lon"), skipna=skipna)
    C = (w * tgt**2).sum(("lat", "lon"), skipna=skipna)
    acc = A / np.sqrt(B * C + 1e-12)
    # acc = acc.mean("time", skipna=skipna)
    return acc.values
def plot_fig(eva,var,path,f='RMSE'):
    # 绘制折线图
    steps = eva['step']
    plt.figure(figsize=(10, 6))
    plt.plot(steps, eva[var+'_'+f].mean(dim='time'), marker='o', linestyle='-', color='r')
    plt.xlabel('Step')
    plt.ylabel(f)
    plt.title(f'{var} {f} for each Step')
    plt.grid(True)
    plt.xticks(steps)
    plt.savefig(os.path.join(path,f"{var}_{f}.jpg"))
def evalute(Tgt,time_list,var,daily_forecast_start_list=['00','12'],forecast_range = [i for i in range(6,144+1,6)]):
    fn = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanxu/data/ERA5/zarr/6hourly/fuxi_20_all/2002_2022.c92.p25"
    mean = xr.open_dataset(os.path.join(fn, 'mean.nc'))
    std = xr.open_dataset(os.path.join(fn, 'std.nc'))
    ACC = []
    RMSE = []
    cur_time = []
    mean = mean['data'].sel(channel=[var]).data
    std = std['data'].sel(channel=[var]).data
    Tgt = Tgt['data'].sel(channel=var)
    print(var)
    for time in tqdm(time_list,desc="Processing", ncols=80, ascii=True, unit="time"):
        for start in daily_forecast_start_list:
            ds = xr.open_zarr(f"/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/PanGU_infer_ERA5_6hour/{start}/PanGu_{time}")
            tgt_time = datetime.strptime(time+start, "%Y%m%d%H")
            forecast_start = tgt_time+timedelta(hours=forecast_range[0])
            forecast_end =   tgt_time+timedelta(hours=forecast_range[-1])
            tgt = Tgt.sel(time=slice(forecast_start,forecast_end))
            out = ds[var].sel(step=forecast_range,time = ds['time'].values[0])
            tgt = inv_normalize(tgt,mean,std)
            tgt = tgt.rename({'time': 'step'})
            tgt = tgt.assign_coords(step=out['step'].values)
            if 'q' in var:
                out = out*1000
            rmse = compute_rmse(out, tgt)
            acc = compute_acc(tgt, out,mean)
            # rmse =[]
            # acc =[]
            # for time_step in forecast_range:
            #     #d = tgt_time+timedelta(hours=time_step)
            #     tgt = Tgt.sel(time=tgt_time+timedelta(hours=time_step))
            #     out = ds[var].sel(step=time_step, time=time)
            #     tgt = inv_normalize(tgt,mean,std)
            #     # out = inv_normalize(out,mean,std)
            #     rmse.append(compute_rmse(out, tgt).item())
            #     acc.append(compute_acc(tgt, out,mean).item())
            RMSE.append(rmse)
            ACC.append(acc)
            cur_time.append(time+start)

    coords={'time':cur_time,
            'step':forecast_range}
    ds = xr.Dataset({
        f'{var}_RMSE': (['time', 'step'], RMSE),
        f'{var}_ACC':(['time', 'step'], ACC),},
        coords=coords)
    
    # save
    fn = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/PanGU_infer_ERA5_6hour/data_rmse_acc"
    ds.to_zarr(os.path.join(fn,f"{var}.zarr"))
    path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/picture/PanGu"
    plot_fig(ds,var,path,f='RMSE')
    plot_fig(ds,var,path,f='ACC')
    print(f'Finish--{var}')
    return ds


# var = 'q500'
# eva= evalute(tgt,time_list,var=var)
# plot_fig(eva,var,path,f='RMSE')
# plot_fig(eva,var,path,f='ACC')
# var = 'v10m'
# eva = evalute(tgt,time_list,var=var)
# plot_fig(eva,var,path,f='RMSE')
# plot_fig(eva,var,path,f='ACC')
# for var in sur_var:
#     eva= evalute(tgt,time_list,var=var)
#     plot_fig(eva,var,path,f='RMSE')
#     plot_fig(eva,var,path,f='ACC')
# for name in lev_var:
#     for var in get_level_name(name,level):
#         eva= evalute(tgt,time_list,var=var)
#         plot_fig(eva,var,path,f='RMSE')
#         plot_fig(eva,var,path,f='ACC')

import argparse
def main():
    #path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanxu/data/Project/FuXi_eval/PanGu_Inference/data/debug/2018_selfdata_6hour/12/PanGu_20180927"
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", default='t2m')
    args = parser.parse_args()
    var = args.var
    fn = '/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanxu/data/ERA5/zarr/6hourly/fuxi_20_all/2002_2022.c92.p25'
    tgt =  xr.open_zarr(fn)
    start_time = '20180101'
    end_time =  '20191231'
    time_list = generate_daily_dates(start_time, end_time)
    print(f"datetime:{time_list[0]} - {time_list[-1]},{len(time_list)}")

    sur_var = ['t2m','u10m','v10m']
    #level = [50, 100, 150,200, 250, 300, 400, 500, 600,700, 850, 925, 1000]
    level = [500]
    lev_var = ['z','t','u','v','q'] 
    path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/picture/PanGu"
    eva = evalute(tgt,time_list,var=var)
if __name__ == '__main__':
    main()