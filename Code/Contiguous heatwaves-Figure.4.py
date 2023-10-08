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

#%%%[0] Input
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


#%%%[1] Calculate yearly
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


#%%%[2] Calculate trend
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
    
    
#%%%[3] Plot long-term series and shading
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

#%%%[2-1] Read and mask and convert U、V data
ds = xr.open_dataset(r'F:\ERA5.Global.2.5\ERA5.2.5.daily\pressure.level\u_component_of_wind\era5.global.daily.u_component_of_wind.pressure.levels.2010.2.5x2.5.nc')
ds = ds.rename({'longitude':'lon', 'latitude':'lat'})

models = ['CNRM-CM6-1',\
          'IPSL-CM6A-LR', 'CanESM5','FGOALS-g3','ACCESS-CM2', 'ACCESS-ESM1-5',  ] 
# models = ['IPSL-CM6A-LR', 'CanESM5','FGOALS-g3', 'MIROC6', 'MRI-ESM2-0']
# models = ['BCC-CSM2-MR', 'NorESM2-LM']
# models = ['ACCESS-ESM1-5']
# expers = ['historical_extend_ssp126', 'historical_extend_ssp245',\
#               'historical_extend_ssp370', 'historical_extend_ssp585']
# expers = ['historical_extend_ssp245']
expers = ['hist-GHG', 'hist-nat']
for model in models:
    for exper in expers:
        print(model, exper)
        if (exper == 'hist-GHG') | (exper == 'hist-nat'):
            if (model == 'ACCESS-ESM1-5')| (model ==  'MRI-ESM2-0'):
                year_period = '19500101-20201231'
            elif (model == 'CNRM-CM6-1')|(model == 'IPSL-CM6A-LR'):
                year_period = '19600101-20201231'
            else:
                year_period = '18500101-20201231'
                
        elif exper == 'historical_extend_ssp245':
            if (model == 'ACCESS-CM2')| (model == 'ACCESS-ESM1-5')| (model ==  'MRI-ESM2-0')| (model =='NorESM2-LM'):
                year_period = '19500101-20201231'
            else:
                year_period = '18500101-20201231'
            
        if (model == 'ACCESS-CM2') | (model == 'FGOALS-g3'):
            if exper == 'hist-GHG':
                rlzn = 'r2i1p1f1_gn'
            else:
                rlzn = 'r1i1p1f1_gn'
        elif (model == 'CNRM-CM6-1'):
            rlzn = 'r1i1p1f2_gr'
        elif (model == 'IPSL-CM6A-LR'):
            rlzn = 'r1i1p1f1_gr'    
        else:
            rlzn = 'r1i1p1f1_gn'
        
        for year in range(1979, 2021):
            file_u = r'Z:/Private/wusj/CMIP6_wind_filtering_500hPa\ua_filtered_'+ model +'_'+ exper +'_'+ rlzn +'_2.5x2.5_500hPa_'+ str (year)+'.nc'
            file_v = r'Z:/Private/wusj/CMIP6_wind_filtering_500hPa\va_filtered_'+ model +'_'+ exper +'_'+ rlzn +'_2.5x2.5_500hPa_'+ str (year)+'.nc'
            file_ref = r'Z:\Private\wusj\CMIP6_wind_plev_50hpa\ua_day_'+ model +'_'+ exper +'_'+ rlzn +'_'+ year_period +'_2.5x2.5_500hPa_'+ str (year)+'.nc'
            
            ds_u = xr.open_dataset(file_u)
            ds_v = xr.open_dataset(file_v)
            ds_ref = xr.open_dataset(file_ref)
            
            ds_u['lon'] = ds['lon'].values
            ds_u['lat'] = ds['lat'].values
            ds_u['time'] = ds_ref['time'].values
            
            ds_v['lon'] = ds['lon'].values
            ds_v['lat'] = ds['lat'].values
            ds_v['time'] = ds_ref['time'].values
            
            EKE_ = (0.5*((ds_u['ua_filtered']**2) + (ds_v['va_filtered']**2))).to_dataset(name='EKE')
            if year == 1979:
                EKE = EKE_
            else:
                EKE = xr.concat([EKE, EKE_], dim='time')
            del EKE_
            gc.collect()
        print(EKE)
        EKE.to_netcdf(path=r'Z:\Private\wusj\CMIP6_EKE\Lanczos_2_8_day\EKE.'+ model +'_'+ exper +'_'+ rlzn + '.Lanczos.2_8.day.level.500.1979_2020.2.5x2.5.nc')
        del file_u, file_v, ds_u, ds_v
        
        ## Landsea mask
        EKE_ = land_sea_mask(EKE['EKE'], time = True)
        EKE_.to_netcdf(path=r'Z:\Private\wusj\CMIP6_EKE\Lanczos_2_8_day\Land_mask\EKE.'+ model +'_'+ exper +'_'+ rlzn + '.Lanczos.2_8.day.level.500.land.2.5x2.5.nc')
        del EKE
        gc.collect()
        
        del year_period, exper, rlzn, EKE_
        gc.collect()
    del model
    gc.collect()
del models, expers, ds
gc.collect()


#%%%[2-2] Calculate yearly for ensemble EKE
'''
逐个读取各model的EKE数据
'''
#切换数据文件夹

# cd Z:\Private\wusj\CMIP6_EKE\Lanczos_2_8_day\Land_mask\
# ls EKE*.nc > file_EKE.txt

# cd Z:\Private\wusj\CMIP6_ua_500hPa\ua\
# ls ua*.nc > file_ua.txt

var = 'EKE'

if var == 'EKE':
    path = r'Z:\Private\wusj\CMIP6_EKE\Lanczos_2_8_day\Land_mask/'
    with open(path + 'file_EKE.txt',"r",encoding="utf-8")  as f:
        fileNames=f.readlines()
elif var == 'ua':
    path = r'Z:\Private\wusj\CMIP6_ua_500hPa\ua/'
    with open(path + 'file_ua.txt',"r",encoding="utf-8")  as f:
        fileNames=f.readlines()

## 读取文件名并对文件进行合并
region_list=['NH_30_90', 'NH', 'SH', 'Tropics']

model_list = ["ACCESS-CM2","ACCESS-ESM1-5", "CanESM5", "CNRM-CM6-1", 
              "FGOALS-g3", "IPSL-CM6A-LR", "MIROC6", "MRI-ESM2-0"]

scenario_list =["historical_extend_ssp245", "hist-GHG", "hist-nat"]

years = range(1979, 2021)

ds_cmip = xr.Dataset(data_vars = dict(EKE=(["scenario", "model", "region","year"], 
                                             np.full((len(scenario_list),
                                                      len(model_list),
                                                      len(region_list),
                                                      len(years)),np.nan))),
                         coords    = dict(scenario=scenario_list,
                                          model=model_list,
                                          region=region_list,
                                          year=(years)))
 

for i in range(0, len(fileNames)):
    fileName = fileNames[i]
    if var == 'EKE':
        file_scenr = fileName[36:-1].split(".")[1].split("_")[1]
        file_model = fileName[36:-1].split(".")[1].split("_")[0]
        if file_scenr == 'historical':
            file_scenr = 'historical_extend_ssp245'

        ds_ = xr.open_dataset(path + fileName[36:-1])   #文件名索引根据要求灵活更改 
        ds = ds_.sel(time = slice('1979-01-01', '2020-12-31'))

    elif var == 'ua':
        file_scenr = fileName[36:-1].split("_")[3]
        file_model = fileName[36:-1].split("_")[2]
        if file_scenr == 'historical':
            file_scenr = 'historical_extend_ssp245'

        ds_ = xr.open_dataset(path + fileName[36:-1])   #文件名索引根据要求灵活更改 
        ds_ = ds_.sel(time = slice('1979-01-01', '2020-12-31'))
        ds = ds_[var].squeeze().to_dataset(name = var)
        
    if file_model in ['CanESM5', 'FGOALS-g3']:
        ds['time'] = ds.indexes['time'].to_datetimeindex()# ###转换时间格式
        ds = ds.sortby('time')  ## 重新排序时间索引
    print(file_scenr, file_model)
    
    ##Compute monthly mean and monthly anomalies
    ds = ds.resample(time = '1m').mean()     # monthly mean
    # moving average-three month
    ds_mean = ds.rolling(center = True, time = 3).mean()
    ds_mean = land_sea_mask_no_Antarctica(ds_mean, data_var = var)
    # Figure test
    # fig = plt.figure(figsize=(5.8, 4.1)) 
    # plot_map_global(ds_mean[323,:,:],
    #                 ax = plt.axes(projection=ccrs.PlateCarree()), 
    #                 levels = 50*np.arange(-1, 1.1, 0.1), 
    #                 shift=False)
    # plt.show()
    
    # plt.plot(ds_mean.sel(time = ds_mean['time.year']==1982)[var].mean(dim = 'lon').mean(dim='lat'))
    # plt.title(label = file_scenr + '_' +file_model)
    # plt.show()

    del ds
    gc.collect()
    
    for region in region_list:
        print(region)
    
        annu_eke = region_sel(ds_mean, region = region)
        # annu_eke = annu_eke.sel(year = annu_eke['year']>1979)
        
        ## regional mean weighted by latitudes     
        regional_annua_mean = regional_weight(annu_eke)
        ds_cmip[var][ds_cmip["scenario"]==file_scenr, 
                           ds_cmip["model"]==file_model, 
                           ds_cmip["region"]==region, 
                           regional_annua_mean.year-years[0]] = regional_annua_mean
    del i

ds_cmip.to_netcdf(r'D:\01PhD\01_09 Three-dimensional events_hot extremes\09_Revise\01_Result\CMIP6_multimodel_'+var+'_500hPa'+'_1979-2020_2.5x2.5.nc')


#%%%[2-3] Calculate trend
var = 'ua'
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


#%%%[2-4] Plot long-term series and shading
scenario_list_hist = ['hist-GHG','historical_extend_ssp245','hist-nat']
unit_list = ['(m$^{2}$/s$^{2}$)', '(m/s)']
level = 500
Lanczos = '2_8'
path_Figure = r'D:\01PhD\01_09 Three-dimensional events_hot extremes\09_Revise\03_Figure/'

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
                 label= _label + '(' + '%+.4f' % (_slope) + ', p=' + '%.2f' % (_pval)+ ')')
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
plt.savefig(path_Figure + 'Figure-4.CMIP6-NH_30_90-SH-Tropics_EKE_'+ Lanczos+'_' +'and_U'+'_time_series_'+ str(level)+'hPa.pdf', format = 'pdf') 
plt.show()
