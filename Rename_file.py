# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 13:48:50 2023

@author: user

使用os.rename函数更改文件名：

"""
# import xarray as xr
import os
        

folder_path = 'Z:/Private/wusj/01-26 Day to Day/02_Result/'  ## 文件路径
old_filename = "tas_dtd-all_interval-1_NorESM2-LM_ssp585_regrid.nc"  ## 原始文件名
new_filename = "tas_dtd-all_interval-1_NorESM2-LM_ssp585_regrid_20150101-21001231.nc"  ## 新文件名

old_path = os.path.join(folder_path, old_filename)
new_path = os.path.join(folder_path, new_filename)
os.rename(old_path, new_path)


# ds = xr.open_dataset(r'Z:\Private\wusj\01-26 Day to Day\02_Result/tas_dtd-all_interval-1_NorESM2-LM_ssp126_regrid_20150101-21001231.nc') 
