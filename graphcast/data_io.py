import os
import numpy as np
import pandas as pd
import xarray 
from datetime import datetime, timedelta
import gc
# 设置垃圾回收的阈值
gc.set_threshold(100, 2, 2)
from graphcast import checkpoint
from graphcast import graphcast
from graphcast import data_utils
input_variables=['2m_temperature', 'mean_sea_level_pressure', '10m_v_component_of_wind',
                  '10m_u_component_of_wind', 'temperature', 'geopotential',
                  'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
                  'specific_humidity', 'toa_incident_solar_radiation', 
                  'year_progress_sin', 'year_progress_cos', 'day_progress_sin', 
                  'day_progress_cos', 'geopotential_at_surface', 'land_sea_mask']


forcing_variables=['toa_incident_solar_radiation', 'year_progress_sin', 
                  'year_progress_cos', 'day_progress_sin', 'day_progress_cos']

target_variables= ['2m_temperature', 'mean_sea_level_pressure', 
                  '10m_v_component_of_wind', '10m_u_component_of_wind', 'total_precipitation_6hr',
                  'temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind',
                  'vertical_velocity', 'specific_humidity']

pressure_levels = [50, 100, 150,200, 250, 300, 400, 500, 600,700, 850, 925, 1000]
def inv_normalize(ds):
    '''
    反归一化
    '''
    fn = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanxu/data/ERA5/zarr/6hourly/fuxi_20_all/2002_2022.c92.p25"
    mean = xarray.open_dataset(os.path.join(fn, 'mean.nc'))
    std = xarray.open_dataset(os.path.join(fn, 'std.nc'))
    ds = ds * std + mean
    return ds

def load_model():
    checkpoint_path="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/graphcast/checkpoint/GraphCast_operational.npz"
    with open(checkpoint_path,'rb') as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    params = ckpt.params
    print("Model description:\n", ckpt.description, "\n")
    return model_config,task_config,params

def load_normalization_data():
    # Load normalization data
    stats_path="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/graphcast/stats"
    with open(f"{stats_path}/diffs_stddev_by_level.nc",'rb') as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with open(f"{stats_path}/mean_by_level.nc",'rb') as f:
        mean_by_level =  xarray.load_dataset(f).compute()
    with open(f"{stats_path}/stddev_by_level.nc",'rb') as f:
        stddev_by_level =  xarray.load_dataset(f).compute()
    return diffs_stddev_by_level , mean_by_level ,stddev_by_level
def get_level_name(name):
    level = [50, 100, 150,200, 250, 300, 400, 500, 600,700, 850, 925, 1000] 
    return [f'{name}{i}' for i in level]
def concat_lev_sur(data,data1): # 拼接surface,level变量
    '''
    data1是补充数据
    '''
    u10m = data['data'].sel(channel='u10m').values.astype(np.float32)#  
    msl = data['data'].sel(channel='msl').values.astype(np.float32) # mean_sea_level_pressure
    t2m = data['data'].sel(channel='t2m').values.astype(np.float32)  # 2meter_temperature
    v10m = data['data'].sel(channel='v10m').values.astype(np.float32)  # 10m_v_component_of_wind
    tp = data['data'].sel(channel='tp').values.astype(np.float32) # total_precipitation_6hr
    data1['10m_u_component_of_wind'] = xarray.DataArray(u10m, dims=['time','lat','lon']) 
    data1['mean_sea_level_pressure'] = xarray.DataArray(msl, dims=['time','lat','lon']) 
    data1['total_precipitation_6hr'] = xarray.DataArray(tp, dims=['time','lat','lon']) 
    data1['10m_v_component_of_wind'] = xarray.DataArray(v10m, dims=['time','lat','lon']) 
    data1['2m_temperature'] = xarray.DataArray(t2m, dims=['time','lat','lon']) 
    z = data['data'].sel(channel = get_level_name('z')).values.astype(np.float32)   # geopotential
    t = data['data'].sel(channel = get_level_name('t')).values.astype(np.float32)   # temperature
    u = data['data'].sel(channel = get_level_name('u')).values.astype(np.float32)   # u_component_of_wind
    v = data['data'].sel(channel = get_level_name('v')).values.astype(np.float32)  # v_component_of_wind
    q = data['data'].sel(channel = get_level_name('q')).values.astype(np.float32)/1000  # g / kg  -> kg /kg  # specific_humidity
    data1['temperature'] = xarray.DataArray(t, dims=['time','level','lat','lon'],) 
    data1['geopotential'] = xarray.DataArray(z, dims=['time','level','lat','lon'])
    data1['u_component_of_wind'] = xarray.DataArray(u, dims=['time','level','lat','lon'])
    data1['v_component_of_wind'] = xarray.DataArray(v,dims=['time','level','lat','lon']) 
    data1['specific_humidity'] = xarray.DataArray(q, dims=['time','level','lat','lon']) 
    del z,t,u,v,q,u10m,v10m,t2m,tp,msl,data
    gc.collect()
    return data1

class DataLoader:
    def __init__(self,orig_path = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanxu/data/ERA5/zarr/6hourly/fuxi_20_all/2002_2022.c92.p25",
                      path="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/data"):
        self.path = path
        self.orig_path = orig_path
    def add_geopotential_mask(self,data):
        '''
        geopotential_at_surface(常量) ,land_sea_mask(常量)
        '''
        fn = os.path.join(self.path,"2018-01")
        data1 = xarray.open_zarr(fn)  # 补充的数据
        data1 = data1.rename({'latitude':'lat','longitude':'lon'})
        geopotential_at_surface = data1['geopotential_at_surface'].values.astype(np.float32)
        land_sea_mask = data1['land_sea_mask'].values.astype(np.float32)
        data['geopotential_at_surface'] = xarray.DataArray(geopotential_at_surface,dims=['lat','lon']) 
        data['land_sea_mask'] = xarray.DataArray(land_sea_mask,dims=['lat','lon'])
        del data1,geopotential_at_surface,land_sea_mask
        gc.collect()
        return data

    def get_addition_data(self,cur_time):
        '''
        获取补充数据，vertical_velocity','toa_incident_solar_radiation
        '''
        var = ['vertical_velocity','toa_incident_solar_radiation']
        #cur_time = datetime.strptime(cur_time, '%Y%m%d%H')
        cur_month= cur_time[0].strftime('%Y-%m')
        fn = os.path.join(self.path,cur_month)
        data1 = xarray.open_zarr(fn)
        data1 = data1[var]
        for idx in range(1,len(cur_time)):  # 日期可能跨月
            if cur_time[idx].strftime('%Y-%m')!= cur_month:
                fn = os.path.join(self.path,cur_time[idx].strftime('%Y-%m'))
                data2 = xarray.open_zarr(fn)  # 补充的数据
                data2 = data2[var]
                data1 = xarray.concat((data1,data2),dim='time')
                break
        data1 = data1.rename({'latitude':'lat','longitude':'lon'})
        data1 = data1.sel(time = cur_time)
        return data1.compute()
    
    def get_all_data(self,cur_time):
        '''
        获取输入数据，将主要数据与补充数据拼接
        cur_time ->list
        ''' 
        data = xarray.open_zarr(self.orig_path)
        data = data.sel(time=cur_time)
        data = inv_normalize(data)
        data_add = self.get_addition_data(cur_time)
        input = concat_lev_sur(data,data_add)
        del data,data_add
        gc.collect()
        return input
    def add_addition_coord_var(self,input):  
        '''
        修改坐标名字，增加datetime，修改time
        '''
        input = input.assign(datetime = np.array(input['time'].values))
        input['time'] = np.array(input['time'].values,dtype = 'timedelta64[ns]')
        input['datetime']=input['datetime'].expand_dims('batch')
        for var in list(input.data_vars):# 增加 batch
            if var not in ['geopotential_at_surface','land_sea_mask']:
                input[var]=input[var].expand_dims('batch')
        timelist = input['time'].values
        input['time'] = np.array(timelist - timelist[0],dtype = 'timedelta64[ns]')  # 修改time
        return input
    def get_input_target_foceing(self,cur_time):
        dataset = self.get_all_data(cur_time)
        dataset = self.add_addition_coord_var(dataset)  # 修改坐标
        dataset = self.add_geopotential_mask(dataset)  # 增加常量
        inputs, targets, forcings =data_utils.extract_inputs_targets_forcings(
                                              dataset,
                                              input_variables = input_variables,
                                              target_variables = target_variables,
                                              forcing_variables = forcing_variables,
                                              pressure_levels = pressure_levels,
                                              input_duration = "12h",
                                              target_lead_times = slice(f"{6}h",f"{144}h")
                                              )

        return inputs, targets, forcings


  






'''
  inputs = inputs[list(input_variables)]
  # The forcing uses the same time coordinates as the target.
  forcings = targets[list(forcing_variables)]
  targets_template = xr.zeros_like(targets[list(target_variables)]) 

  return inputs, targets_template, forcings
input_variables=('2m_temperature', 'mean_sea_level_pressure', '10m_v_component_of_wind',
'10m_u_component_of_wind', 'temperature', 'geopotential',
'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
'specific_humidity', 'toa_incident_solar_radiation', 
'year_progress_sin', 'year_progress_cos', 'day_progress_sin', 
'day_progress_cos', 'geopotential_at_surface', 'land_sea_mask')

 target_variables=('2m_temperature', 'mean_sea_level_pressure', 
'10m_v_component_of_wind', '10m_u_component_of_wind', 'total_precipitation_6hr',
'temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind',
'vertical_velocity', 'specific_humidity'), 
forcing_variables=('toa_incident_solar_radiation', 'year_progress_sin', 
'year_progress_cos', 'day_progress_sin', 'day_progress_cos'), 
pressure_levels=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000), 
input_duration='12h')

'''

'''
def bulid_targets_template(time,lat,lon,level,batch=1):
   # 'batch','time', 'level','lat','lon'
   lev_var = np.ones((batch,len(time),len(level),len(lat),len(lon)),
                          dtype=np.float32)
   # 'batch','time','lat','lon'
   sur_var = np.ones((batch,len(time),len(lat),len(lon)),
                          dtype=np.float32)    
   targets_template = xarray.Dataset({
    'temperature': (['batch','time', 'level','lat','lon'], lev_var),    # (batch, time, level, lat, lon)
    'geopotential': (['batch','time', 'level','lat','lon'], lev_var),
    'u_component_of_wind': (['batch','time', 'level','lat','lon'], lev_var),
    'v_component_of_wind': (['batch','time', 'level','lat','lon'],lev_var),
    'specific_humidity': (['batch','time', 'level','lat','lon'],lev_var),
    'vertical_velocity': (['batch','time', 'level','lat','lon'],lev_var),
    '10m_u_component_of_wind':(['batch','time','lat','lon'],sur_var),
    'mean_sea_level_pressure':(['batch','time','lat','lon'],sur_var),
    'total_precipitation_6hr':(['batch','time','lat','lon'], sur_var),
    '10m_v_component_of_wind':(['batch','time','lat','lon'], sur_var),
    '2m_temperature':(['batch','time','lat','lon'], sur_var),
    },
    coords={
        'time': time,  
        'lat': lat, 
        'lon': lon, 
        'level':level
    })
   return targets_template[target_variables]

'''

