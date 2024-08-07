import xarray
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
def load_cloud_data(time,
                    vals=['land_sea_mask', 'geopotential_at_surface', 'vertical_velocity',
                          'toa_incident_solar_radiation']):
    era5 = xarray.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2",
        # chunks={'time': 48,'level':13},
        consolidated=True,
    )
    era5 = era5.sel(time=time)
    era5 = era5[vals]
    return era5

def save_zarr(save_name, ds):
    from dask.diagnostics import ProgressBar
    ds = ds.chuck(time=1)
    obj = ds.to_zarr(save_name, compute=False)
    with ProgressBar():
        obj.compute()

def generate_monthly_dates(start_time, end_time):
    """
    Parameters:
    - start_time (str): Start date in 'YYYY-MM' format.
    - end_time (str): End date in 'YYYY-MM' format.
    Returns:
    - list: List of dates in 'YYYY-MM' format.
    """
    start_datetime = datetime.strptime(start_time, '%Y-%m')
    end_datetime = datetime.strptime(end_time, '%Y-%m')
    # Initialize an empty list to store dates
    date_sequence = []
    # Generate monthly dates
    current_datetime = start_datetime
    while current_datetime <= end_datetime:
        date_sequence.append(current_datetime.strftime('%Y-%m'))
        current_datetime += relativedelta(months=1)
    
    return date_sequence

def process_data(cur_time, path):
    save_path = f"{path}/{cur_time}"
    if os.path.exists(save_path):
        print(f"{cur_time} already exists, skipping download.")
        return
    idataSet = load_cloud_data(cur_time)
    print(idataSet)
    save_zarr(save_path, idataSet)
    print(f"{cur_time} is saved")

def download():
    start_time = '2018-01'
    end_time = '2019-12'
    time_list = generate_monthly_dates(start_time, end_time)
    print('months:', time_list)
    path = '/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/data'

    # Use ThreadPoolExecutor to download data concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks to the executor
        futures = [executor.submit(process_data, cur_time, path) for cur_time in time_list]
        # Wait for all futures to complete
        for future in as_completed(futures):
            try:
                future.result()  # Get the result of the future
            except Exception as e:
                print(f"An error occurred: {e}")

    print("All data has been processed and saved.")

def create_log(save_path):
   
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    now = datetime.now().strftime("%Y%m%d%H")
    log_file = f'{save_path}/logs/{now}.log'
    fileinfo = logging.FileHandler(log_file)

    controshow = logging.StreamHandler()
    controshow.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controshow)
    return logger

def check(): 
    log = create_log("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/code")
    start_time = '2019-9'
    end_time = '2019-12'
    time_list = generate_monthly_dates(start_time, end_time)
    path = '/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/data'
    print('months:', time_list)
    for curtime in time_list:
        ds_cloud = load_cloud_data(curtime)
        ds =  xarray.open_zarr(f"/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/zx/data/{curtime}")
        for var in ds.data_vars:
            if np.array_equal(ds[var].data, ds_cloud[var].data):
                log.info(f"{curtime}变量 {var} 一致")
                print(f"{curtime}变量 {var} 一致")
            else:
                print(f"{curtime}变量 {var} 不匹配")
                log.info(f"{curtime}变量 {var} 一致")


    
    
if __name__ == '__main__':
    check()