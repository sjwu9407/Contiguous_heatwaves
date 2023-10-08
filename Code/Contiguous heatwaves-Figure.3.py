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


#%%[1] Reanalysis yearly plot
'''
According to the result of contiguous heatwaves to plot long-term trend
'''


#%%%[0] Input
thre_range = '19812010'
path_figure = r'D:\01PhD\01_09 Three-dimensional events_hot extremes\09_Revise\03_Figure/'

reanalysis_list = ["ERA5","MERRA2", "NCEP2"]

region_list=["Global", 'NH_30_90', 'NH', 'SH', 'Tropics','Africa', 'Eurasian','Australia','North America','South America']

var_list =  ["Frequency","Accumulated Area","Lifetime","Mean Duration",
             "Total Magnitude","Mean Intensity", "Max Intensity","Total Moving Distance","Moving Speed"]

unit_list = ['(event)', '(km$^{2}$)', '(day)', '(day)', '(km$^{2}$×°C)', '(°C)', '(°C)', '(km)', '(km/day)']
rows = ['Slope', 'P_value', 'Intercept', 'Std_err']
years = range(1979, 2021)
ds_reanalysis = xr.Dataset(data_vars=dict(Frequency=(["reanalysis", "region","year"], np.full([len(reanalysis_list), len(region_list),len(years)], np.nan)),
                                          Accumulated_Area=(["reanalysis", "region","year"], np.full([len(reanalysis_list), len(region_list),len(years)], np.nan)),
                                          Total_Magnitude=(["reanalysis", "region","year"], np.full([len(reanalysis_list), len(region_list),len(years)], np.nan)),
                                          Max_Intensity=(["reanalysis", "region","year"], np.full([len(reanalysis_list), len(region_list),len(years)],np.nan)),
                                          Mean_Intensity=(["reanalysis", "region","year"], np.full([len(reanalysis_list), len(region_list),len(years)],np.nan)),
                                          Lifetime=(["reanalysis", "region","year"], np.full([len(reanalysis_list), len(region_list),len(years)],np.nan)),
                                          Mean_Duration=(["reanalysis", "region","year"], np.full([len(reanalysis_list), len(region_list),len(years)],np.nan)),
                                          Total_Moving_Distance=(["reanalysis", "region","year"], np.full([len(reanalysis_list), len(region_list),len(years)],np.nan)),
                                          Moving_Speed=(["reanalysis", "region","year"], np.full([len(reanalysis_list), len(region_list),len(years)],np.nan)),),
                           coords=dict(reanalysis=reanalysis_list,
                                       region=region_list,
                                       year=(years)))
ds_reanalysis = ds_reanalysis.rename({"Accumulated_Area":"Accumulated Area",
                                      "Total_Magnitude":"Total Magnitude",
                                      "Max_Intensity":"Max Intensity",
                                      "Mean_Intensity":"Mean Intensity",
                                      "Mean_Duration":"Mean Duration",
                                      "Total_Moving_Distance":"Total Moving Distance",
                                      "Moving_Speed":"Moving Speed"})


#%%%[1] Calculate yearly
'''
Input: Global summer contiguous heatwaves
'''
file_var = 'tmax_'
summer = True
lifespan_thre = 92

for event_name in reanalysis_list:
    if event_name == 'ERA5':
        year_period = '1950-2020'
    elif event_name == 'MERRA2':
        year_period = '1980-2020'
    elif event_name == 'NCEP2':
        year_period = '1979-2020'
    
    # import cmip data
    if summer:   ## 输入夏季的热浪
        print('summer')
        file_pkl = r'Z:\Private\wusj\3D-Result-revise-0906\02_Result_Processed_01_Continent_02_Summer' +'//' + file_var +  event_name +'_eventList_processed_continent_summer_'+year_period+'_thre'+ thre_range+'.pkl'
    else:  ## 输入全年的热浪
        print('full_calendar_year')
        file_pkl = r'Z:\Private\wusj\3D-Result-revise-0906\02_Result_Processed_01_Continent' +'//' + file_var +  event_name +'_eventList_processed_continent'+year_period+'_thre'+ thre_range+'.pkl'
    
    if os.path.isfile(file_pkl) == False:
        print(event_name + ' not found!')
        continue;
    else:
        print(event_name)
        
    if summer:     
        eventList = pd.read_pickle(file_pkl)
    else:
        eventList = pd.read_pickle(file_pkl)
        eventList = eventList[eventList['Lifespan'] <= lifespan_thre]
        
    eventList = eventList.rename(columns={'Average Intensity':'Mean Intensity',
                                          'Lifespan':'Lifetime',
                                          'Mean Grid Lifespan':'Mean Duration',
                                          'Moving Distance':'Total Moving Distance',
                                          'Speed':'Moving Speed'})
    print(event_name) 
    
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
     
        print(event_name, region, eventList_sel['Accumulated Area'].mean())
        # 按年统计
        eventList_sel_yearly = eventList_sel.groupby(["Mean Year"]).mean()
        eventList_sel_yearly['Frequency'] = eventList_sel.groupby(["Mean Year"]).count()['Segid']
        eventList_sel_yearly['Max Intensity'] = eventList_sel.groupby('Mean Year')['Mean Intensity'].max()
        eventList_sel_yearly['Year'] = eventList_sel_yearly.index
        
        eventList_sel_yearly = eventList_sel_yearly[eventList_sel_yearly['Year']>=1979]
        for var_name in var_list:
            ds_reanalysis[var_name][ds_reanalysis["reanalysis"]==event_name,
                                    ds_reanalysis["region"]==region,
                                    eventList_sel_yearly['Year']-years[0]] = eventList_sel_yearly[var_name]
            del var_name
    del event_name, file_pkl 
    
    
#%%%[2] Calculate trend
region = 'Global'
i = 0
for event_name in reanalysis_list:
    print(event_name)
    if event_name == 'ERA5':
        ds_reanalysis_ = ds_reanalysis.sel(year = ds_reanalysis['year']>= 1979)
    elif event_name == 'MERRA2':
        ds_reanalysis_ = ds_reanalysis.sel(year = ds_reanalysis['year']>= 1980)
    elif event_name == 'NCEP2':
        ds_reanalysis_ = ds_reanalysis.sel(year = ds_reanalysis['year']>= 1979)
    Trend = pd.DataFrame(data = np.full([len(rows), len(var_list)],np.nan), 
                         index = rows, 
                         columns = var_list)
    Trend_part1 = Trend.copy(deep = True)
    Trend_part2 = Trend.copy(deep = True)
    
    for var_name in var_list:
        # print(var_name)
        ## 1979-2020
        variable_tmp = ds_reanalysis_.sel(reanalysis = event_name, region = region)[var_name]
        trend = linear_trend(variable_tmp.year,  variable_tmp)
        Trend[var_name] = [trend[0]*10, trend[1], trend[2], trend[3]] 
        Trend['reanalysis'] = event_name   
        del trend, variable_tmp
        
        ## 1979-1996
        variable_tmp = ds_reanalysis_.sel(year = ds_reanalysis_['year']<=1996).sel(reanalysis = event_name, region = region)[var_name]
        trend = linear_trend(variable_tmp.year,  variable_tmp)
        Trend_part1[var_name] = [trend[0]*10, trend[1], trend[2], trend[3]] 
        Trend_part1['reanalysis'] = event_name
        del trend, variable_tmp
        
        ## 1996-2020
        variable_tmp = ds_reanalysis_.sel(year = ds_reanalysis_['year']>1996).sel(reanalysis = event_name, region = region)[var_name]
        trend = linear_trend(variable_tmp.year,  variable_tmp)
        Trend_part2[var_name] = [trend[0]*10, trend[1], trend[2], trend[3]] 
        Trend_part2['reanalysis'] = event_name
        del trend, variable_tmp 
    
    if i == 0:
        Trend_reanalysis = Trend
        Trend_part1_reanalysis = Trend_part1
        Trend_part2_reanalysis = Trend_part2
    else:
        Trend_reanalysis = pd.concat([Trend_reanalysis, Trend])
        Trend_part1_reanalysis = pd.concat([Trend_part1_reanalysis, Trend_part1])
        Trend_part2_reanalysis = pd.concat([Trend_part2_reanalysis, Trend_part2])
        
    i = i + 1   
    del Trend, Trend_part1, Trend_part2


#%%%[3] Plot long-term trends and shading
fig = plt.figure(figsize=(15, 8), dpi=300)
i=0
for var_name in var_list:
    print(var_name)
    i+=1
    ax = fig.add_subplot(3, 3, i)
    for event_name in reanalysis_list:
        variable_tmp = ds_reanalysis.sel(reanalysis = event_name, region = region)[var_name]
        _slope = Trend_reanalysis[Trend_reanalysis['reanalysis'] == event_name].iloc[0][var_name]
        _pval = Trend_reanalysis[Trend_reanalysis['reanalysis'] == event_name].iloc[1][var_name]
        if event_name == 'ERA5':
            _color = 'crimson'  ###F901F8
            _label = 'ERA5'
        elif event_name == 'MERRA2':
            _color = 'black'  ##brown
            _label = 'MERRA2'
        elif event_name == 'NCEP2':
            _color = 'steelblue'
            _label = 'NCEP2'
        
        ## plot
        data = variable_tmp.to_dataframe()
        sns.regplot(data = data, x= data.index, y = var_name, 
                    ci=95, scatter = None, line_kws={'lw': 0.8, 'color': _color, 'alpha': 0.8,})
        plt.plot(variable_tmp['year'], variable_tmp, 
                 color = _color, alpha=1, linewidth = 1.5,
                 label= _label + '(' + '%+.2f' %_slope+ ', p=' + '%.2f' %_pval+ ')')
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
if summer:   ## 输入夏季的热浪
    print('summer')   
    plt.savefig(path_figure + 'Figure-S19.'+file_var+region+'-ERA5_MERRA2_NCEP2_'+thre_range+'_1979-2020.pdf', format = 'pdf')
else:   ## 输入夏季的热浪
    print('full_calendar_year')   
    plt.savefig(path_figure + 'Figure-S9.'+file_var+region+'-full_calendar_year-ERA5_MERRA2_NCEP2_'+thre_range+'_1979-2020.pdf', format = 'pdf')
plt.show()


#%%[2] EKE and Zonal wind
'''
Analysis long-term trend of EKE and Zonal wind

'''

#%%%[2-1] Compute monthly mean and monthly anomalies
level = 500    
Lanczos = '2_8'

path_data = r'D:\01PhD\01_09 Three-dimensional events_hot extremes\09_Revise\04_Revised_Manuscript_Submit\GitHub/'
path_Figure = r'D:\01PhD\01_09 Three-dimensional events_hot extremes\09_Revise\03_Figure/'

ds = xr.open_dataset(path_data + 'EKE.Lanczos.'+Lanczos+'.day.level.'+str(level)+'.land.1979_2020.2.5x2.5.nc') 
ds = ds.resample(time = '1m').mean()     # monthly mean
# moving average-three month
ds_mean = ds.rolling(center = True, time = 3).mean()  
ds_mean = land_sea_mask_no_Antarctica(ds_mean, data_var = 'EKE').to_dataset(name = 'EKE')


## Read U
ds_u = xr.open_dataset(r'D:\01PhD\01_09 Three-dimensional events_hot extremes\09_Revise\04_Revised_Manuscript_Submit\GitHub\U.level.'+str(level)+'.land.1979_2020.2.5x2.5.nc')
ds_u = ds_u.resample(time = '1m').mean()     # monthly mean
# moving average-three month
ds_u_mean = ds_u.rolling(center = True, time = 3).mean()   
ds_u_mean = land_sea_mask_no_Antarctica(ds_u_mean, data_var = 'u').to_dataset(name = 'u')


#%%%[2-2] Plot Long-term trend
i = 0
fig = plt.figure(figsize=(18, 3), dpi=300)
for region in ['NH_30_90','Tropics', 'SH']:
    print(region)
    i+=1
        
    ax1 = fig.add_subplot(1, 3, i)

    if region in ['SH', 'Tropics']:
        year_sel = 1979
    else:
        year_sel = 1978
        
    var1 = 'EKE'
    annu_eke = region_sel(ds_mean, region = region)
    annu_eke = annu_eke.sel(year = annu_eke['year']>year_sel)
    ## regional mean weighted by latitudes     
    regional_annua_mean = regional_weight(annu_eke)[var1]
    data_EKE = pd.DataFrame(regional_annua_mean)
    data_EKE['year'] = regional_annua_mean.year
    data_EKE = data_EKE.rename(columns = {0:var1})    ##更改列名
    
    var2 = 'u'   
    annu_u = region_sel(ds_u_mean, region = region)
    annu_u = annu_u.sel(year = annu_u['year']>year_sel)
    ## regional mean weighted by latitudes     
    regional_annu_u_mean = regional_weight(annu_u)[var2]
    data_u = pd.DataFrame(regional_annu_u_mean)
    data_u['year'] = regional_annu_u_mean.year
    data_u = data_u.rename(columns = {0:var2})    ##更改列名
    
    _slope, _pval, _inter, _std_err = linear_trend(data_EKE['year'], data_EKE[var1])
     
    sns.regplot(data = data_EKE, 
                x= data_EKE['year'], y = var1, 
                ci=95, scatter = None, 
                line_kws={'lw': 0.8, 'color':'fuchsia', 'alpha': 1,}) 
    ax1.plot(data_EKE['year'], data_EKE[var1],
              color = 'fuchsia', alpha=0.65, linewidth = 1.5,
              label='EKE (' + '%+.2f' % (10*_slope)+ ', p=' + '%.2f' % (_pval)+ ')')
    ax1.tick_params(axis='x',labelsize=11)
    ax1.tick_params(axis='y',labelsize=11)
    ax1.set_xlabel("")
    ax1.set_ylabel(str(level)+'hPa EKE(m$^{2}$/s$^{2}$)',fontsize=11)
    leg1 = ax1.legend(loc = 'upper right',fontsize = 11)
    # change the font colors to match the line colors
    for line,text in zip(leg1.get_lines(), leg1.get_texts()):
        text.set_color(line.get_color())
    del _slope, _pval, _inter, _std_err
    
    ##画负轴的图ax2
    ax2 = ax1.twinx()  
    _slope, _pval, _inter, _std_err = linear_trend(data_u['year'], data_u[var2])
    
    sns.regplot(data = data_u, 
                x= data_u['year'], y = var2, 
                ci=95, scatter = None, 
                line_kws={'lw': 0.8, 'color':'black', 'alpha': 1,}) 
    ax2.plot(data_u['year'], data_u[var2],
              color = 'black', alpha=0.65, linewidth = 1.5,
              label='Zonal Wind (' + '%+.2f' % (10*_slope)+ ', p=' + '%.2f' % (_pval)+ ')')
    ax2.tick_params(axis='x',labelsize=11)
    ax2.tick_params(axis='y',labelsize=11)
    ax2.set_xlabel("")
    ax2.set_ylabel(str(level)+'hPa Zonal Wind(m/s)',fontsize=11)  #设置y轴标签及其字号 
    leg2 = ax2.legend(loc = 'upper left',fontsize = 11)
    # change the font colors to match the line colors
    for line,text in zip(leg2.get_lines(), leg2.get_texts()):
        text.set_color(line.get_color())
    del _slope, _pval, _inter,_std_err
    
    plt.xticks(np.arange(1980,2021,10))
plt.tight_layout() 
plt.savefig(path_Figure + 'Figure-R3.ERA5-NH_30_90-SH-Tropics_EKE_'+ Lanczos+'_' +'and_U'+'_Trend_'+ str(level)+'hPa.pdf', format = 'pdf') 
plt.show() 

