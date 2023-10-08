# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 14:49:43 2023

@author: sjwu_
"""

#%% import
import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os.path
import matplotlib.pyplot as plt
import pymannkendall as mk
from scipy import stats
from matplotlib import ticker
import regionmask


def event_select_area(eventList, area_thre = 1e6):
    
    ## Select by threshold(accumulated area)
    area_thre = area_thre
    eventList_sel = eventList[eventList['Accumulated Area'] >= area_thre]
    return eventList_sel

def event_select_continent_area(eventList, continent_name = 'Africa', q = 60):
    '''
    针对每个continent进行面积筛选
    '''
    
    ## Select by threshold(accumulated area)
    eventList_ = eventList[eventList["Continent"] == continent_name] 
    
    # select threshold by area_sum(Top 40%)
    num_threshold = np.nanpercentile(eventList_['Accumulated Area'], q)  ## threshold
    eventList_sel = (eventList_[eventList_['Accumulated Area']>= num_threshold]).reset_index(drop=True)
    
    return eventList_sel


def linear_trend(x, y):  
    '''
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    '''
    
    mask = (~(np.isnan(x))) & (~(np.isnan(y)))
    
    # calculate the trend with 80% valid data
    if len(x[mask]) <= 0.2 * len(x):
        return np.nan, np.nan, np.nan, np.nan
    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
        # result = mk.hamed_rao_modification_test(y[mask])
        result = mk.yue_wang_modification_test(y[mask])
        p_value1 = result.p
        return slope, p_value1, intercept, std_err
    
def scilimit(ax, limit_1 = -1, limit_2 = 3):

    '''
    设置坐标轴的label为科学技术法
    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    limit_1 : TYPE, optional
        DESCRIPTION. The default is -1.
    limit_2 : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    formatter : TYPE
        DESCRIPTION.

    '''
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((limit_1,limit_2)) 
    return formatter


def land_sea_mask_no_Antarctica(data, data_var = 'tmax'):
    # 创建陆地掩膜
    land_mask = regionmask.defined_regions.natural_earth.land_110.mask(data[data_var])
    # 将掩膜应用于数据
    data = data[data_var].where(land_mask == 0)

    # 将纬度在-60到90之间的南极洲区域设置为NaN
    mask = (data['lat'] < -60) | (data['lat'] > 90) # 创建南极洲的掩膜
    # 将南极洲区域设置为NaN
    data = data.where(~mask)
    return data


def region_sel(data, region = 'NH'): 
    if region == 'NH_30_90':
        data_sel = data.sel(time=(data['time.month']== 7)).sel(lat=slice(90, 30))
        annu = data_sel.groupby('time.year').mean(dim='time')
    elif region == 'NH':
        data_sel = data.sel(time=(data['time.month']== 7)).sel(lat=slice(90, 0))
        annu = data_sel.groupby('time.year').mean(dim='time')
    elif region == 'SH':
        data_sel = data.sel(time=(data['time.month']== 1)).sel(lat=slice(0, -60))
        annu = data_sel.groupby('time.year').mean(dim='time')
    elif region == 'Tropics':
        # --Sourthern--DJF----------------------
        data_sel_S = data.sel(time=(data['time.month']== 1)).sel(lat=slice(0, -23.5))
        annu_S = data_sel_S.groupby('time.year').mean(dim='time')
        # --Northern--JJA-------------------
        data_sel_N = data.sel(time=(data['time.month']== 7)).sel(lat=slice(23.5, 0))
        annu_N = data_sel_N.groupby('time.year').mean(dim='time')
        # --concat North and South---------------
        annu = xr.concat([annu_N, annu_S], dim='lat')
    return annu


# regional mean weighted by latitudes
def regional_weight(data):
    weights = np.sqrt(np.cos(np.deg2rad(data['lat'])).clip(0., 1.))
    weights.name = "weights"
    regional_mean = data.weighted(weights).mean(("lon", "lat"))
    return regional_mean

    
#%%[1] CMIP6 yearly plot
'''
According to the result of contiguous heatwaves to plot long-term trend
'''

#%%%[1-1] Input
thre_range = '19812010'
path_figure = r'D:\01PhD\01_09 Three-dimensional events_hot extremes\09_Revise\03_Figure/'

scenario_list =["historical_extend_ssp245","hist-GHG", "hist-nat"]

model_list = ["ACCESS-CM2","ACCESS-ESM1-5", "BCC-CSM2-MR", "CanESM5", "CNRM-CM6-1", 
              "FGOALS-g3", "GFDL-ESM4",  "IPSL-CM6A-LR", "MIROC6", "MRI-ESM2-0", "NorESM2-LM"]

region_list=["Global", 'NH_30_90', 'NH', 'SH', 'Tropics','Africa', 'Eurasian','Australia','North America','South America']

var_list =  ["Frequency","Accumulated Area","Lifetime","Mean Duration",
             "Total Magnitude","Mean Intensity", "Max Intensity","Total Moving Distance","Moving Speed"]


unit_list = ['(event)', '(km$^{2}$)', '(day)', '(day)', '(km$^{2}$×°C)', '(°C)', '(°C)', '(km)', '(km/day)']
rows = ['Slope', 'P_value', 'Intercept', 'Std_err']
years = range(1950, 2101)
ds_cmip6 = xr.Dataset(data_vars=dict(Frequency=(["scenario", "model", "region","year"], np.full([len(scenario_list),11,len(region_list),len(years)],np.nan)),
                                    Accumulated_Area=(["scenario", "model", "region","year"], np.full([len(scenario_list),11,len(region_list),len(years)],np.nan)),
                                    Total_Magnitude=(["scenario", "model", "region","year"], np.full([len(scenario_list),11,len(region_list),len(years)],np.nan)),
                                    Max_Intensity=(["scenario", "model", "region","year"], np.full([len(scenario_list),11,len(region_list),len(years)],np.nan)),
                                    Mean_Intensity=(["scenario", "model", "region","year"], np.full([len(scenario_list),11,len(region_list),len(years)],np.nan)),
                                    Lifetime=(["scenario", "model", "region","year"], np.full([len(scenario_list),11,len(region_list),len(years)],np.nan)),
                                    Mean_Duration=(["scenario", "model", "region","year"], np.full([len(scenario_list),11,len(region_list),len(years)],np.nan)),
                                    Total_Moving_Distance=(["scenario", "model", "region","year"], np.full([len(scenario_list),11,len(region_list),len(years)],np.nan)),
                                    Moving_Speed=(["scenario", "model", "region","year"], np.full([len(scenario_list),11,len(region_list),len(years)],np.nan))),
                     coords=dict(scenario=scenario_list, 
                                 model=model_list, 
                                 region=region_list,
                                 year=(years)))
ds_cmip6 = ds_cmip6.rename({"Accumulated_Area":"Accumulated Area",
                            "Total_Magnitude":"Total Magnitude",
                            "Max_Intensity":"Max Intensity",
                            "Mean_Intensity":"Mean Intensity",
                            "Mean_Duration":"Mean Duration",
                            "Total_Moving_Distance":"Total Moving Distance",
                            "Moving_Speed":"Moving Speed"})


#%%%[1-2] Calculate yearly
'''
Input: Calculate yearly series of summer contiguous heatwaves
'''

for file_scenr in scenario_list:
    if file_scenr in ['historical_extend_ssp245', 'hist-aer', 'hist-GHG', 'hist-nat']:
        year_period = '1950-2020'
        
    for file_model in model_list:
        event_name = file_model+"_"+file_scenr
        # import cmip data
        file_pkl = r'Z:\Private\wusj\3D-Result-revise-0906\02_Result_Processed_01_Continent_02_Summer/tmax_' + event_name +'_eventList_processed_continent_summer_'+year_period+'_thre'+ thre_range+'.pkl'
        if os.path.isfile(file_pkl) == False:
            print(event_name + ' not found!')
            continue;
        else:
            print(event_name)
        ## Read data
        eventList = pd.read_pickle(file_pkl)
        eventList = eventList.rename(columns={'Average Intensity':'Mean Intensity',
                                              'Lifespan':'Lifetime',
                                              'Mean Grid Lifespan':'Mean Duration',
                                              'Moving Distance':'Total Moving Distance',
                                              'Speed':'Moving Speed'})
        for region in region_list:
            if region in ['Global','NH_30_90','NH', 'SH', 'Tropics']:
                eventList_sel = event_select_area(eventList, area_thre = 1e6)  ## 对于全球陆地的研究，选择热浪总面积≤1000000
                if region =='Global':
                    eventList_sel = eventList_sel
                elif region == 'NH_30_90':
                    eventList_sel = eventList_sel[eventList_sel['Mean Latitude']>=30]
                    eventList_sel = eventList_sel[eventList_sel['Mean Latitude']<=90]
                elif region == 'NH':
                    eventList_sel = eventList_sel[eventList_sel['Mean Latitude']>=0]
                    eventList_sel = eventList_sel[eventList_sel['Mean Latitude']<=90]
                elif region == 'SH':
                    eventList_sel = eventList_sel[eventList_sel['Mean Latitude']>=-60]
                    eventList_sel = eventList_sel[eventList_sel['Mean Latitude']<0]
                elif region == 'Tropics':
                    eventList_sel = eventList_sel[eventList_sel['Mean Latitude']>=-23.5]
                    eventList_sel = eventList_sel[eventList_sel['Mean Latitude']<=23.5]
                    
            elif region in ['Africa', 'Eurasian','Australia','North America','South America']:
                eventList_sel = event_select_continent_area(eventList, continent_name = region, q = 60)

            # 按年统计
            eventList_sel_yearly = eventList_sel.groupby(["Mean Year"]).mean()
            eventList_sel_yearly['Frequency'] = eventList_sel.groupby(["Mean Year"]).count()['Segid']
            eventList_sel_yearly['Max Intensity'] = eventList_sel.groupby('Mean Year')['Mean Intensity'].max()
            eventList_sel_yearly['Year'] = eventList_sel_yearly.index
            
            # if file_scenr in ['historical_extend_ssp245', 'hist-aer', 'hist-GHG', 'hist-nat']:
            #     eventList_sel_yearly = eventList_sel_yearly[eventList_sel_yearly['Year']< 2015]
            # elif file_scenr in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
            #     eventList_sel_yearly = eventList_sel_yearly[eventList_sel_yearly['Year']>= 2015]
                
            for var_name in var_list:
                ds_cmip6[var_name][ds_cmip6["scenario"]==file_scenr,
                                   ds_cmip6["model"]==file_model,
                                   ds_cmip6["region"]==region,
                                   eventList_sel_yearly['Year']-years[0]] = eventList_sel_yearly[var_name]
                del var_name
        del event_name, file_pkl


#%%%[1-3] Calculate trend
region = 'Global'
rows = ['Slope', 'P_value', 'Intercept', 'Std_err']
i = 0
for file_scenr in scenario_list:
    print(file_scenr)
    Trend = pd.DataFrame(data = np.full([len(rows), len(var_list)],np.nan), 
                         index = rows, 
                         columns = var_list)
    
    if file_scenr in ['historical_extend_ssp126','historical_extend_ssp245', 
                      'historical_extend_ssp370','historical_extend_ssp585', 
                      'hist-aer', 'hist-GHG', 'hist-nat']:
        ds_cmip6_ = ds_cmip6.sel(year = slice(1979, 2020)) 
        
    elif file_scenr in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
        ds_cmip6_ = ds_cmip6.sel(year = slice(2020, 2100)) 
        
    for var_name in var_list:
        # print(var_name)
        variable_tmp = ds_cmip6_.sel(scenario = file_scenr, region = region)[var_name].mean(dim="model")

        trend = linear_trend(variable_tmp.year,  variable_tmp)
        Trend[var_name] = [trend[0]*10, trend[1], trend[2], trend[3]] 
        Trend['scenario'] = file_scenr   
        del trend 
    
    if i == 0:
        Trend_scenario = Trend
    else:
        Trend_scenario = pd.concat([Trend_scenario, Trend])
    i = i + 1   
    del Trend
    
    
#%%%[1-4] Plot long-term series and shading
scenario_list_hist = ['hist-GHG','historical_extend_ssp245','hist-nat']
# scenario_list_hist = ['historical_extend_ssp245',"hist-GHG", "hist-nat",
#                       'ssp126', 'ssp245', 'ssp370', 'ssp585']
fig = plt.figure(figsize=(15, 8), dpi=300)
# fig = plt.figure(figsize=(20, 15), dpi=300)
i=0
for var_name in var_list:
    print(var_name)
    i+=1
    ax = fig.add_subplot(3, 3, i)
    for file_scenr in scenario_list_hist:
        if file_scenr in ['historical_extend_ssp126','historical_extend_ssp245', 
                          'historical_extend_ssp370','historical_extend_ssp585', 
                          'hist-aer', 'hist-GHG', 'hist-nat']:
            ds_cmip6_ = ds_cmip6.sel(year = slice(1979, 2020)) 
            
        elif file_scenr in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
            ds_cmip6_ = ds_cmip6.sel(year = slice(2020, 2100)) 
        
        variable_tmp = ds_cmip6_.sel(scenario = file_scenr, region = region)[var_name].mean(dim="model")
        variable_tmp_std = ds_cmip6_.sel(scenario = file_scenr, region = region)[var_name].std(dim="model")
        
        _slope = Trend_scenario[Trend_scenario['scenario'] == file_scenr].iloc[0][var_name]
        _pval = Trend_scenario[Trend_scenario['scenario'] == file_scenr].iloc[1][var_name]
        if file_scenr == 'historical_extend_ssp245':
            _color = 'k'
            _label = 'Hist-ALL'
        elif file_scenr == 'hist-aer':
            _color = 'skyblue'
            _label = 'Hist-AER'
        elif file_scenr == 'hist-GHG':
            _color = 'crimson'  ##red
            _label = 'Hist-GHG'
        elif file_scenr == 'hist-nat':
            _color = 'steelblue'  ##seagreen
            _label = 'Hist-NAT'
        elif file_scenr == 'ssp126':
            _color = 'lightcoral'
            _label = 'SSP126'
        elif file_scenr == 'ssp245':
            _color = 'salmon'
            _label = 'SSP245'
        elif file_scenr == 'ssp370':
            _color = 'orangered'
            _label = 'SSP370'
        elif file_scenr == 'ssp585':
            _color = 'firebrick'
            _label = 'SSP585'
            
        ## plot
        data = variable_tmp.to_dataframe()
        data_std = variable_tmp_std.to_dataframe()

        plt.fill_between(data.index,
                  (data[var_name]-1/2*data_std[var_name]),
                  (data[var_name]+1/2*data_std[var_name]),
                  alpha=0.2, facecolor=_color)
        plt.plot(variable_tmp['year'], variable_tmp, 
                 color = _color, alpha=1, linewidth = 1.8,
                 label= _label + '(' + '%+.2f' % (_slope) + ', p=' + '%.2f' % (_pval)+ ')')
    # 设置参数
    plt.xlabel("")
    plt.ylabel(var_name + ' ' + unit_list[i-1], fontsize = 11) 
    ax.tick_params(axis='x',labelsize=10)
    ax.tick_params(axis='y',labelsize=10)
    
    ## y轴设置科学计数法
    formatter = scilimit(ax, limit_1 = -1, limit_2 = 3)
    ax.yaxis.set_major_formatter(formatter)
    
    leg = plt.legend(loc = 'best',fontsize = 9)
    # change the font colors to match the line colors
    for line,text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
plt.tight_layout()    
# plt.savefig(path_figure + 'Figure-R2.'+region+'-CMIP6_historical_GHG_nat_'+thre_range+'_1979-2008.pdf', format = 'pdf')
plt.show()


#%%[2] EKE and Zonal wind
'''
Analysis long-term trend of EKE and Zonal wind

'''

#%%%[2-1] Calculate trend
var = 'EKE'
ds_cmip = xr.open_dataset(r'D:\01PhD\01_09 Three-dimensional events_hot extremes\09_Revise\01_Result\CMIP6_multimodel_'+var+'_500hPa'+'_1979-2020_2.5x2.5.nc')

region = 'NH_30_90'
rows = ['Slope', 'P_value']
scenario_list =["historical_extend_ssp245", "hist-GHG", "hist-nat"]

i = 0
for file_scenr in scenario_list:
    print(file_scenr)
    Trend = pd.DataFrame(data = np.full([len(rows), 1],np.nan), 
                         index = rows, 
                         columns = [region+'_' + var])
    
    variable_tmp = ds_cmip.sel(scenario = file_scenr, region = region)[var].mean(dim="model")
    
    trend = linear_trend(variable_tmp.year,  variable_tmp)
    Trend[region+'_' + var] = [trend[0]*10, trend[1]] 
    Trend['scenario'] = file_scenr   
    if var == 'EKE':
        if i == 0:
            Trend_scenario_EKE = Trend
        else:
            Trend_scenario_EKE = pd.concat([Trend_scenario_EKE, Trend])
        i = i + 1   
        
    elif var == 'ua':
        if i == 0:
            Trend_scenario_ua = Trend
        else:
            Trend_scenario_ua = pd.concat([Trend_scenario_ua, Trend])
        i = i + 1   
    
    del Trend, variable_tmp, trend
del i, var, rows, file_scenr, ds_cmip, region, scenario_list


#%%%[2-2] Plot long-term series and shading
scenario_list_hist = ['hist-GHG','historical_extend_ssp245','hist-nat']
unit_list = ['(m$^{2}$/s$^{2}$)', '(m/s)']
level = 500
Lanczos = '2_8'
path_Figure = r'D:\01PhD\01_09 Three-dimensional events_hot extremes\09_Revise\03_Figure/'
region = 'NH_30_90'

fig = plt.figure(figsize=(15, 8), dpi=300)
i=0
for var_name in ['EKE', 'ua']:
    print(var_name)
    if var_name == 'EKE':
        ds_cmip = xr.open_dataset(r'D:\01PhD\01_09 Three-dimensional events_hot extremes\09_Revise\01_Result\CMIP6_multimodel_'+var_name+'_'+str(level)+'hPa'+'_1979-2020_2.5x2.5.nc')
        Trend_scenario = Trend_scenario_EKE
    elif var_name == 'ua' :  
        ds_cmip = xr.open_dataset(r'D:\01PhD\01_09 Three-dimensional events_hot extremes\09_Revise\01_Result\CMIP6_multimodel_'+var_name+'_'+str(level)+'hPa'+'_1979-2020_2.5x2.5.nc')
        Trend_scenario = Trend_scenario_ua
        
    i+=1
    ax = fig.add_subplot(3, 3, i)
    for file_scenr in scenario_list_hist:
        if file_scenr in ['historical_extend_ssp126','historical_extend_ssp245', 
                          'historical_extend_ssp370','historical_extend_ssp585', 
                          'hist-aer', 'hist-GHG', 'hist-nat']:
            ds_cmip_ = ds_cmip.sel(year = slice(1979, 2020)) 
        elif file_scenr in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
            ds_cmip_ = ds_cmip.sel(year = slice(2020, 2100)) 
        
        variable_tmp = ds_cmip_.sel(scenario = file_scenr, region = region)[var_name].mean(dim="model")
        variable_tmp_std = ds_cmip_.sel(scenario = file_scenr, region = region)[var_name].std(dim="model")
        
        _slope = Trend_scenario[Trend_scenario['scenario'] == file_scenr].iloc[0].values[0]
        _pval = Trend_scenario[Trend_scenario['scenario'] == file_scenr].iloc[1].values[0]
        if file_scenr == 'historical_extend_ssp245':
            _color = 'black'
            _label = 'Hist-ALL'
        elif file_scenr == 'hist-aer':
            _color = 'skyblue'
            _label = 'Hist-AER'
        elif file_scenr == 'hist-GHG':
            _color = 'crimson'
            _label = 'Hist-GHG'
        elif file_scenr == 'hist-nat':
            _color = 'steelblue'
            _label = 'Hist-NAT'
        elif file_scenr == 'ssp126':
            _color = 'lightcoral'
            _label = 'SSP126'
        elif file_scenr == 'ssp245':
            _color = 'salmon'
            _label = 'SSP245'
        elif file_scenr == 'ssp370':
            _color = 'orangered'
            _label = 'SSP370'
        elif file_scenr == 'ssp585':
            _color = 'firebrick'
            _label = 'SSP585'
            
        ## plot
        data = variable_tmp.to_dataframe()
        data_std = variable_tmp_std.to_dataframe()

        plt.fill_between(data.index,
                  (data[var_name]-1/2*data_std[var_name]),
                  (data[var_name]+1/2*data_std[var_name]),
                  alpha=0.2, facecolor=_color)
        plt.plot(variable_tmp['year'], variable_tmp, 
                 color = _color, alpha=1, linewidth = 1.5,
                 label= _label + '(' + '%+.2f' % (_slope) + ', p=' + '%.2f' % (_pval)+ ')')
    # 设置参数
    plt.xlabel("")
    plt.ylabel(var_name + ' ' + unit_list[i-1], fontsize = 11) 
    ax.tick_params(axis='x',labelsize=10)
    ax.tick_params(axis='y',labelsize=10)
    
    ## y轴设置科学计数法
    formatter = scilimit(ax, limit_1 = -1, limit_2 = 3)
    ax.yaxis.set_major_formatter(formatter)
    
    leg = plt.legend(loc = 'best',fontsize = 9)
    # change the font colors to match the line colors
    for line,text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
plt.tight_layout()    
# plt.savefig(path_Figure + 'Figure-4.CMIP6-NH_30_90-SH-Tropics_EKE_'+ Lanczos+'_' +'and_U'+'_time_series_'+ str(level)+'hPa.pdf', format = 'pdf') 
plt.show()
