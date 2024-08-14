import os
from matplotlib import pyplot as plt



import numpy as np
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import xarray
surf_vars = ['msl', 'u10m', 'v10m', 't2m']
lev_vars = ['z', 'q', 't', 'u', 'v']
levels = [50, 100, 150,200, 250, 300, 400, 500, 600,700, 850, 925, 1000]

def get_level_name(name,level):
    return [f'{name}{i}' for i in level]
level_vars = []
for i in lev_vars:
    for j in get_level_name(i,levels):
        level_vars.append(j)
def generate_daily_dates(start_time, end_time):
    start_datetime = datetime.strptime(start_time, '%Y%m%d%H')
    end_datetime = datetime.strptime(end_time, '%Y%m%d%H')

    date_sequence = []
    current_datetime = start_datetime 
    while current_datetime <= end_datetime:
        date_sequence.append(current_datetime)
        current_datetime += timedelta(hours=12)

    return date_sequence

def compute_rmse(out, tgt,skipna=True):
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
    return rmse.values

def compute_acc(tgt, out,mean,skipna=True):
    w = np.cos(np.deg2rad(tgt.lat))
    w /= w.mean()
    # w = w / w.sum() * len(tgt['lat'])
    tgt = tgt - mean
    out = out - mean
    tgt = tgt -tgt.mean()
    out = out -out.mean()
    A = (w * out * tgt).sum(("lat", "lon"), skipna=skipna)
    B = (w * out**2).sum(("lat", "lon"), skipna=skipna)
    C = (w * tgt**2).sum(("lat", "lon"), skipna=skipna)
    acc = A / np.sqrt(B * C + 1e-12)
    # acc = acc.mean("time", skipna=skipna)
    # for var in list(acc.data_vars):
    #     acc[var] = acc[var].expand_dims(dim={'time': [curtime]})
    return acc.values
def plot_fig(values,path,var,f):
    # 绘制
    step = range(6,6*24+1,6)
    plt.figure(figsize=(10, 6))
    plt.plot(step,values, marker='o', linestyle='-', color='r')
    plt.xlabel('Step')
    plt.ylabel(f)
    plt.title(f'{var}-{f} for each Step(6h)')
    plt.grid(True)
    plt.xticks(step)
    plt.savefig(os.path.join(path,f"{var}_{f}.jpg"))
        
            
    
def inv_normalize(ds, mean, std):
    ds = ds.astype(np.float32)
    ds = ds * std + mean
    return ds
    
        
def evalute(time_list,var,forecast_range = range(6,6*24+1,6)):
    # fn = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanxu/data/ERA5/zarr/6hourly/fuxi_20_all/2002_2022.c92.p25"
    # mean = xr.open_dataset(os.path.join(fn, 'mean.nc'))
    # std = xr.open_dataset(os.path.join(fn, 'std.nc'))
    ACC = None
    RMSE = None
    fn = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanxu/data/ERA5/zarr/6hourly/fuxi_20_all/2002_2022.c92.p25"
    mean_file = os.path.join(fn, 'mean.nc')
    std_file = os.path.join(fn, 'std.nc')
    mean = xr.open_dataset(mean_file)
    std = xr.open_dataset(std_file)
    data = xarray.open_zarr(fn)  # 主要部分数据
    wb  = inv_normalize(data, mean, std)
    # 逆归一化
    data = wb['data'].sel(channel = var)
    for time in tqdm(time_list,desc="Processing", ncols=80, ascii=True, unit="time"):
        Hour = datetime.strftime(time,"%H")
        Day =  datetime.strftime(time,"%Y%m%d")
        out = xr.open_zarr(f"/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/PanGU_infer_ERA5_6hour/{Hour}/PanGu_{Day}")
        out = out.compute()
        out = out.sel(time = time)
        tgt = data.sel(time = [time + timedelta(hours= h) for h in forecast_range])
        out = out[var]
        tgt['time'] = np.array(forecast_range)
        tgt = tgt.rename({'time':'step'})
        rmse = compute_rmse(out, tgt)
        mean_var = mean['data'].sel(channel  = var)
        acc = compute_acc(tgt, out,mean_var)
        if RMSE is None:
            RMSE = rmse
        else:
            RMSE = np.concatenate((RMSE, rmse), axis=0)
        if ACC is None:
            ACC = acc
        else:
            ACC = np.concatenate((ACC,acc),axis =0)

    # save
    path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/picture/PanGu1"
    RMSE = np.mean(RMSE,axis = 0)
    #RMSE.to_zarr("Graphcast-rmse-2019010200-infer")
    plot_fig(RMSE,path,var,f='RMSE')

    ACC = np.mean(ACC,axis = 0)
    #ACC.to_zarr("Graphcast-acc-2019010200-infer")
    plot_fig(ACC,path,var,f='ACC')





start_time = '2019010100'
end_time = '2019013112'
forecast_range = range(6,6*24+1,6)
time_list = generate_daily_dates(start_time, end_time)
evalute(time_list,'t2m')