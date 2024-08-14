# @title Build jitted functions, and possibly initialize random weights

import functools
import math
import cartopy.crs as ccrs
import haiku as hk
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray
from graphcast import autoregressive
from graphcast import casting
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
import jax
from datetime import datetime, timedelta
from data_io import *
target_variables= ['2m_temperature', 'mean_sea_level_pressure', 
                  '10m_v_component_of_wind', '10m_u_component_of_wind', 'total_precipitation_6hr',
                  'temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind',
                  'vertical_velocity', 'specific_humidity']





# load model
model_config,task_config,params=load_model()
state = {}
diffs_stddev_by_level , mean_by_level ,stddev_by_level = load_normalization_data()

def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig):
  """Constructs and wraps the GraphCast Predictor."""
  # Deeper one-step predictor.
  predictor = graphcast.GraphCast(model_config, task_config)

  # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
  # from/to float32 to/from BFloat16.
  predictor = casting.Bfloat16Cast(predictor)

  # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
  # BFloat16 happens after applying normalization to the inputs/targets.
  predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=diffs_stddev_by_level,
      mean_by_level=mean_by_level,
      stddev_by_level=stddev_by_level)

  # Wraps everything so the one-step model can produce trajectories.
  predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
  return predictor

@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  return predictor(inputs, targets_template=targets_template, forcings=forcings)

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
  return functools.partial(
      fn, model_config=model_config, task_config=task_config)

# Always pass params and state, so the usage below are simpler
def with_params(fn):
  return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
  return lambda **kw: fn(**kw)[0]
  # return lambda **kw: fn(**{k: v.compute() for k, v in kw.items()})[0]



run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
    run_forward.apply))))


def save_zarr(save_name, ds, dtype=np.float32):
    for var in target_variables:
        ds[var] = ds[var].squeeze(dim='batch') # 去除batch
    time = ds['time'].values
    ds['time'] = np.array(time / time[0] * 6,dtype=int)
    ds = ds.rename({'time':'step'})
    from dask.diagnostics import ProgressBar

    ds = ds.astype(dtype)
    obj = ds.to_zarr(save_name, compute=False, mode='w')
    with ProgressBar():
        obj.compute()
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
start_time = '2019122600'

end_time = '2019123112' 
time_list = generate_daily_dates(start_time, end_time)
fn = "/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/fanxu/data/ERA5/zarr/6hourly/fuxi_20_all/2002_2022.c92.p25"
data = xarray.open_zarr(fn)  # 主要部分数据
Dataload  = DataLoader(data)
for curtime in time_list:
  forecast_range = range(6,6*24+1,6)
  force_time = [curtime-timedelta(hours=6),curtime ] + [ curtime + timedelta(hours=step) for step in forecast_range]
  
  inputs, targets, forcings = Dataload.get_input_target_foceing(force_time)
  predictions = rollout.chunked_prediction(
      run_forward_jitted,
      rng=jax.random.PRNGKey(0),
      inputs=inputs,
      targets_template = targets * np.nan,
      forcings=forcings)
  file_name = f"{curtime.strftime('%H')}/{datetime.strftime(curtime,'%Y%m%d')}"
  save_path =os.path.join("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/Graphcast_infer_ERA5_6hour"
                          ,file_name)
  print(save_path)
  save_zarr(save_path, predictions)

  # for var in target_variables:
  #       predictions[var] = predictions[var].squeeze(dim='batch') # 去除batch
  #       targets[var] = targets[var].squeeze(dim='batch') # 去除batch
  # time = predictions['time'].values
  # predictions['time'] = np.array(time / time[0] * 6,dtype=int)
  # targets['time'] = np.array(time / time[0] * 6,dtype=int)
  # predictions = predictions.rename({'time':'step'})
  # targets = targets.rename({'time':'step'})
  # def compute_rmse(out, tgt, skipna=True):
  #   # print(out.mean(dim='z500'))
  #   # print(tgt.mean(dim='z500'))
  #   if "normal" in out.dims:
  #       out = out.isel(normal=0, drop=True)
  #   if "member" in out.dims:
  #       out = out.mean("member")
  #   weights = np.cos(np.deg2rad(np.abs(tgt.lat))) 
  #   weights = weights / weights.sum() * len(tgt['lat'])
  #   error = (out - tgt) ** 2 
  #   error = error.where(weights > 0, 0)
  #   rmse = np.sqrt(error.weighted(weights).mean(("lat", "lon"), skipna=skipna))
  #   #ds = xr.Dataset(dict(rmse=rmse))
  #   return rmse
  # RMSE = compute_rmse(out=predictions, tgt = targets)
  # RMSE.to_zarr(save_path)
  # break
'''



'''

