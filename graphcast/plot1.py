import os
from matplotlib import pyplot as plt

import gc
# 设置垃圾回收的阈值
gc.set_threshold(100, 2, 2)
import numpy as np
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from data_io import *
target_variables= ['2m_temperature', 'mean_sea_level_pressure', 
                  '10m_v_component_of_wind', '10m_u_component_of_wind', 'total_precipitation_6hr',
                  'temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind',
                  'vertical_velocity', 'specific_humidity']
sur = {'2m_temperature':'t2m', 'mean_sea_level_pressure':'msl', 
      '10m_v_component_of_wind':'v10m', '10m_u_component_of_wind':'u10m', 
      'total_precipitation_6hr':'tp'}
lev = {'temperature':'t', 'geopotential':'z', 
       'u_component_of_wind':'u', 'v_component_of_wind':'v',
      'vertical_velocity':'w', 'specific_humidity':'q'}
levels = [50, 100, 150,200, 250, 300, 400, 500, 600,700, 850, 925, 1000]
def get_level_name(name,level):
    return [f'{name}{i}' for i in level]
def generate_daily_dates(start_time, end_time):
    """
    Parameters:
    - start_time (str): Start date in 'YYYYMMDDHH' format.
    - end_time (str): End date in 'YYYYMMDDHH' format.
    - interval (str): Interval for generating dates. Options: '12h' or '1 day'.
    Returns:
    - list: List of dates in 'YYYYMMDDHH' format.
    """
    start_datetime = datetime.strptime(start_time, '%Y%m%d%H')
    end_datetime = datetime.strptime(end_time, '%Y%m%d%H')

    date_sequence = []
    current_datetime = start_datetime 
    while current_datetime <= end_datetime:
        date_sequence.append(current_datetime)
        current_datetime += timedelta(hours=12)

    return date_sequence
def chunk(ds, time=1, **kwargs):
    if isinstance(ds, xr.DataArray):
        dims = {k: v for k, v in zip(ds.dims, ds.shape)}
    else:
        dims = {k: v for k, v in ds.sizes.items()}
    if "time" in dims:
        dims["time"] = time
    if 'steps' in dims:
        dims["steps"] = 1
    dims.update(**kwargs)
    ds = ds.chunk(dims)
    return ds
def compute_rmse(out, tgt, curtime,skipna=True):
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
    # 增加 time 维度
    for var in list(rmse.data_vars):
        rmse[var] = rmse[var].expand_dims(dim={'time': [curtime]})
    del error,weights,out, tgt
    gc.collect()
    return rmse

def compute_acc(tgt, out,curtime,skipna=True):
    stats_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/graphcast/stats"
    with open(f"{stats_path}/mean_by_level.nc",'rb') as f:
        mean_by_level =  xarray.load_dataset(f).compute()
    mean = mean_by_level[target_variables]
    w = np.cos(np.deg2rad(tgt.lat))
    w /= w.mean()
    # w = w / w.sum() * len(tgt['lat'])
    tgt = tgt -mean
    out = out-mean
    A = (w * out * tgt).sum(("lat", "lon"), skipna=skipna)
    B = (w * out**2).sum(("lat", "lon"), skipna=skipna)
    C = (w * tgt**2).sum(("lat", "lon"), skipna=skipna)
    acc = A / np.sqrt(B * C + 1e-12)
    # acc = acc.mean("time", skipna=skipna)
    for var in list(acc.data_vars):
        acc[var] = acc[var].expand_dims(dim={'time': [curtime]})
    return acc
def plot_fig(eva,path,f):
    # 绘制
    steps = eva['step']
    for var in list(eva.data_vars):
        values = eva[var]
        if var in sur.keys():
            var = sur[var]  # 'temperature' -> 't'
            plt.figure(figsize=(10, 6))
            plt.plot(steps,values, marker='o', linestyle='-', color='r')
            plt.xlabel('Step')
            plt.ylabel(f)
            plt.title(f'{var}-{f} for each Step(6h)')
            plt.grid(True)
            plt.xticks(steps)
            plt.savefig(os.path.join(path,f"{var}_{f}.jpg"))
        elif var in lev.keys():
            var =lev[var]
            values =values.T   # (step, level) - > (level,step)
            L =[7,9,10] # 500 700 850  # [50, 100, 150,200, 250, 300, 400, 500, 600,700, 850, 925, 1000]
            for i in L:
                value = values[i]
                if var == 'q':
                    value = value * 1000 
                plt.figure(figsize=(10, 6))
                plt.plot(steps,value, marker='o', linestyle='-', color='r')
                plt.xlabel('Step')
                plt.ylabel(f)
                plt.title(f'{var}-{f} for each Step(6h)')
                plt.grid(True)
                plt.xticks(steps)
                plt.savefig(os.path.join(path,f"{var}{levels[i]}_{f}.jpg"))
                    
            
       
        
def evalute(time_list,forecast_range = range(6,6*24+1,6)):
    # fn = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanxu/data/ERA5/zarr/6hourly/fuxi_20_all/2002_2022.c92.p25"
    # mean = xr.open_dataset(os.path.join(fn, 'mean.nc'))
    # std = xr.open_dataset(os.path.join(fn, 'std.nc'))
    ACC = None
    RMSE = None
    fn = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanxu/data/ERA5/zarr/6hourly/fuxi_20_all/2002_2022.c92.p25"
    # data = xarray.open_zarr(fn)  # 主要部分数据
    # data = data.sel(time = slice('2018-01-01','2019-12-31'))
    Dataload  = DataLoader()
    # for time in time_list:
    #     print(time)  
    for time in tqdm(time_list,desc="Processing", ncols=80, ascii=True, unit="time"):
        Hour = datetime.strftime(time,"%H")
        Day =  datetime.strftime(time,"%Y%m%d")
        save_path= f"/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/Graphcast_infer_ERA5_6hour/RMSE/{Hour}{Day}"
        if os.path.exists(save_path):
            continue
        out = xr.open_zarr(f"/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/Graphcast_infer_ERA5_6hour/{Hour}/{Day}")
        tgt = Dataload.get_all_data([time + timedelta(hours= h) for h in forecast_range])
        out = chunk(out)
        tgt = chunk(tgt)
        tgt = tgt[target_variables]
        tgt['time'] = np.array(forecast_range)
        tgt = tgt.rename({'time':'step'})
        rmse = compute_rmse(out, tgt,time)
        # acc = compute_acc(tgt, out,time)
        # if RMSE is None:
        #     RMSE = rmse
        # else:
        #     RMSE = xarray.concat((RMSE,rmse),dim = 'time')
        # if ACC is None:
        #     ACC = acc
        # else:
        #     ACC = xarray.concat((ACC,acc),dim = 'time')
        
        
        rmse.to_zarr(save_path)
        del out,tgt,rmse
        gc.collect()
    # save
    # path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/picture/Graphcast"
    # RMSE = RMSE.mean(('time'))
    # start_time = datetime(time_list[0],"%Y%m%d").strftime
    # end_time = datetime(time_list[-1],"%Y%m%d").strftime
    # RMSE.to_zarr(os.path.join(path,f"Graphcast-rmse-{start_time}-{end_time}"))
    #plot_fig(RMSE,path,f='RMSE')

    # ACC = ACC.mean(('time'))
    # ACC.to_zarr(os.path.join(path,"Graphcast-acc"))
    #plot_fig(ACC,path,f='ACC')



import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time", type=str, default='2019010100')
    parser.add_argument("--end_time", type=str, default='2019123112') 
    args = parser.parse_args()
    start_time = args.start_time
    end_time = args.end_time
    forecast_range = range(6,6*24+1,6)
    time_list = generate_daily_dates(start_time, end_time)
    #evalute(time_list)
    RMSE = None
    path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/Graphcast_infer_ERA5_6hour/RMSE"
    files = os.listdir(path)
    for file in  tqdm( files,desc="Processing", ncols=80, ascii=True):
        rmse = xarray.open_zarr(os.path.join(path,file))
        if RMSE is None:
            RMSE = rmse
        else:
            RMSE = xarray.concat((RMSE,rmse),dim = 'time')
    save_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/picture/Graphcast"
    RMSE = RMSE.mean(('time'))
    RMSE = RMSE.compute()
    plot_fig(RMSE,save_path,'rmse')


# acc = xarray.open_zarr("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/Graphcast-acc-2019010200-infer")
# rmse = xarray.open_zarr("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/picture/Graphcast/Graphcast-rmse")
# save_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/picture/Graphcast"
# plot_fig(rmse,save_path,"rmse")
# plot_fig(acc,"/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/picture/Graphcast","acc")
# nohup python graphcast/plot1.py --start_time 2018010100 --end_time 2018123112 > graphcast_plot.log 2>&1 &
# 28715




