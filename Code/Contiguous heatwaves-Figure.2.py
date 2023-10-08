# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 23:21:54 2022

@author: sjwu
"""
'''
According to the result of contiguous heatwaves to draw Figure 2

'''


#%% import
import xarray as xr
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os.path
import matplotlib.pyplot as plt
from matplotlib import ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from windrose import WindroseAxes
from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm
import pyproj



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


def scatter_map_global(data, var = 'Accumulated Area', legend = True, 
                       levels = np.arange(0.1, 1.1, 0.1), 
                       ticks = np.arange(0.1, 1.1, 0.1),
                       title = 'ERA5',
                       ax = plt.axes(projection=ccrs.PlateCarree())):
    
    bins = pd.IntervalIndex.from_tuples([(0,5000000),(5000000, 10000000),
                                         (10000000, 15000000),
                                         (15000000, 20000000),
                                         (20000000, data[var].max())])  
    
    # 计算切片位置的阈值
    Area_class = pd.cut(data[var].to_list(),bins)
    Area_class.categories = [8,22,42,60,85]
    data[var] = Area_class
    
    x = data['Mean Longitude'] 
    y = data['Mean Latitude']
    c = data['Average Intensity']
    
    ax.set_global()
    ax.add_feature(cfeat.COASTLINE, edgecolor="dimgray",linewidths = 0.75)
    # ax.add_feature(cfeat.LAND, facecolor='0.9')    #将陆地设置为浅灰色，facecolor越大，颜色越浅
    # ax.add_feature(cfeat.BORDERS, edgecolor="grey",linewidths = 0.25,)
    # ax.gridlines()
    
    hc = plt.scatter(x=x, y=y, s= data[var], c= c, 
                     label = ['Group 1','Group 2','Group 3','Group 4'],
                     cmap = 'OrRd', alpha=0.8 , edgecolors='silver', 
                     linewidths = 0.00000001, 
                     transform=ccrs.PlateCarree()) 
    
    cb = plt.colorbar(hc, orientation="horizontal", fraction=0.05, pad=0.09, extendfrac='auto',
                      extend='both', extendrect=True, boundaries=levels, ticks=ticks)  
    cb.mappable.set_clim(min(levels), max(levels))
    cb.set_label(label="(°C)", fontsize=16)
    cb.ax.tick_params(labelsize=16)
    
    # set major and minor ticks
    ax.minorticks_on()
    ax.set_xticks(np.arange(-180, 180, 60), crs=ccrs.PlateCarree())
    ax.set_xticklabels(np.arange(-180, 180, 60), fontsize=14)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(30))
    ax.xaxis.set_major_formatter(LongitudeFormatter())   #x轴设置为经度的格式
    
    ax.set_yticks(np.arange(-60, 90, 20), crs=ccrs.PlateCarree())
    ax.set_yticklabels(np.arange(-60, 90, 20), fontsize=14)
    ax.set_ylim(-60,90)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(20))
    ax.yaxis.set_major_formatter(LatitudeFormatter())  #y轴设置为经度的格式
    ax.tick_params(top=False,bottom=True,left=True, right=False)
    # set figure title
    ax.set_title(title, weight='bold', fontsize=14)

    if legend:
        ## Set legend
        handles, labels = hc.legend_elements(prop="sizes",  alpha=0.8, color= hc.cmap(0.7))
        plt.legend(handles, labels, title="Sizes",loc = "lower right",bbox_to_anchor=(0.11, 0), 
                   prop = {'size':12},markerscale=2) 

    return hc, ax

 
def movement_map_global(data, x, u, y, v,
                        var = 'Lifespan',
                        cmap = 'OrRd',
                        levels = 1*np.arange(1,33,3),
                        ticks = 1*np.arange(1,33,3),
                        ax = plt.axes(projection=ccrs.PlateCarree())):

    x = data['Former Longitude']
    u = data['Latter Longitude']-x
    
    y = data['Former Latitude']
    v = data['Latter Latitude']-y
    ax.set_global()
    ax.add_feature(cfeat.COASTLINE, edgecolor="dimgray",linewidths = 0.75)

    hc = plt.quiver(x,y,u,v,data[var], units='xy', cmap = cmap, width = 0.5, 
                    edgecolor="grey",linewidth=0.15,
                    headwidth=6, headlength=5, headaxislength=4.5,
                    transform=ccrs.PlateCarree())
    cb = plt.colorbar(hc, boundaries=levels, ticks=ticks,
                      orientation="horizontal", fraction=0.05, 
                      pad=0.09, extendfrac='auto',
                      extend='both', extendrect=True)
    cb.mappable.set_clim(min(levels), max(levels))
    cb.set_label(label="(day)", fontsize=16)
    cb.ax.tick_params(labelsize=16) 
    # set major and minor ticks
    ax.minorticks_on()
    ax.set_xticks(np.arange(-180, 180, 60), crs=ccrs.PlateCarree())
    ax.set_xticklabels(np.arange(-180, 180, 60), fontsize=14)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(30))
    ax.xaxis.set_major_formatter(LongitudeFormatter())   #x轴设置为经度的格式
    
    ax.set_yticks(np.arange(-60, 90, 20), crs=ccrs.PlateCarree())
    ax.set_yticklabels(np.arange(-60, 90, 20), fontsize=14)
    ax.set_ylim(-60,90)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(20))
    ax.yaxis.set_major_formatter(LatitudeFormatter())  #y轴设置为经度的格式
    ax.tick_params(top=False,bottom=True,left=True, right=False)
    
    return hc, ax


def windrose_map(data, var = 'Moving Distance',
                 cmap=cm.OrRd,legend = True,y_lim = False,
                 bins = 1e3*np.arange(1, 11, 1), 
                 ax = WindroseAxes.from_ax()):
    # 计算direction
    geodesic = pyproj.Geod(ellps='WGS84')
    _angles, _, _Delta_distance  = geodesic.inv(np.vstack(data['Former Longitude']), 
                                                np.vstack(data['Former Latitude']),
                                                np.vstack(data['Latter Longitude']), 
                                                np.vstack(data['Latter Latitude']))
    
    _angles[_angles<0] = 360 + _angles[_angles<0]
    
    data['Delta Angle'] = _angles
    data['Delta_distance'] = _Delta_distance

    ax.bar(direction=data['Delta Angle'], 
           var=data[var], 
           bins=bins,
           cmap=cmap, lw=0.000001,   
           nsector=16, normed=True, opening=1, 
           edgecolor='w',alpha = 1)  #nsector：扇区的数量，默认值为16(每个扇区22.5°)
    # Moving Distance的bins = np.arange(1, 11, 1)*1000 
    # Delta_distance 的bins = np.arange(1, 11, 1)*100000
    ax.set_xticklabels(['E', 'NE', 'N', 'NW',  'W', 'SW', 'S', 'SE'], fontsize = 14)
    if legend:
        ax.set_legend(loc='lower right', bbox_to_anchor=(1.12, -0.12), ncol=2, 
                      fontsize = 20, labelspacing=0.1, columnspacing=0.8)
    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda s,position:'{:.0f}%'.format(s))) #设置y坐标轴以百分数显示
    
    if y_lim:
        ax.set_yticks(np.arange(0, 25, step=3), fontsize = 10)
        ax.set_ylim([0, 24])
    else:
        ax.set_yticklabels("",fontsize = 10)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda s,position:'{:.0f}%'.format(s))) #设置y坐标轴以百分数显示
    return ax
    

def get_lon_name(ds_):
    for lon_name in ['lon', 'longitude']:
        if lon_name in ds_.coords:
            return lon_name
    raise RuntimeError("Couldn't find a longitude coordinate")
    
def get_lat_name(ds_):
    for lat_name in ['lat', 'latitude']:
        if lat_name in ds_.coords:
            return lat_name
    raise RuntimeError("Couldn't find a latitude coordinate")
    
from matplotlib.colors import LinearSegmentedColormap
def colormap():
    colors = [(1, 1, 1, 0), (0.5, 0.5, 0.5, 0), (1, 1, 1, 1)]
    return LinearSegmentedColormap.from_list('mycmap', colors)

# Plot contour map on the globe
def plot_map_global(data, pvals=None,
                    ax=plt.axes(projection=ccrs.PlateCarree()),
                    levels=np.arange(-1, 1.1, 0.1), ticks=np.arange(-1, 1.1, 0.3), cmap='RdBu_r',
                    clabel="", fraction=0.036, pad=0.1, alpha=0.05, linewidth=0.25,
                    title="", shift=False, grid=False, cb=True, Global=1):

    # shift coordinates from [0, 360] to [-180, 180]
    if shift:
        lon = (((data[get_lon_name(data)] + 180) % 360) - 180)
        idx1 = np.where(lon < 0)[0]
        idx2 = np.where(lon >= 0)[0]
        data = data.assign_coords(lon=xr.concat(
            (lon[idx1], lon[idx2]), dim='lon'))
        data.values = xr.concat((data[:, idx1], data[:, idx2]), dim='lon')

    if False:  # non-filled contour
        hc = data.plot.contour(ax=ax, transform=ccrs.PlateCarree(),
                               x=get_lon_name(data), y=get_lat_name(data),
                               levels=levels, colors='k', linewidth=0.5)
    else:  # filled contour
        hc = data.plot.contourf(ax=ax, transform=ccrs.PlateCarree(),
                                x=get_lon_name(data), y=get_lat_name(data),
                                levels=levels, cmap=cmap, add_colorbar=False, extend='both')
    if cb:
        # colorbar
        cb = plt.colorbar(hc, ticks=levels, extend='both', extendrect=True, extendfrac='auto',
                          orientation="horizontal", fraction=fraction, pad=pad)
        cb.set_label(label=clabel, fontsize=8)
        cb.mappable.set_clim(min(levels), max(levels))
        cb.set_ticks(ticks, fontsize=6)  # 设置colorbar的标注

    # plot signifiance (if have) over the map
    if (pvals is None):
        print('No significance hatched')
    else:
        # hp = pvals.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat',
        #                           levels=[0, alpha], cmap=colormap(),
        #                           add_colorbar=False, extend='both')

        hp = pvals.plot.contourf(ax=ax, transform=ccrs.PlateCarree(),
                                  x=get_lon_name(data), y=get_lat_name(data),
                                  hatches=[None, "..."],
                                  levels=[-1, alpha], colors='r', add_colorbar=False,
                                  extend='both', alpha=0)
        #hatches=[None, "xxx"],

    filename = r'D:\01PhD\00Data\04boundary\global\global.shp'
    countries = cfeat.ShapelyFeature(
        Reader(filename).geometries(), ccrs.PlateCarree(), facecolor=None)
    ax.add_feature(countries, facecolor='none',
                   linewidth=linewidth, edgecolor='k')

    # set major and minor ticks
    ax.set_xticks(np.arange(-180, 360+60, 60), crs=ccrs.PlateCarree())
    ax.set_xticklabels(np.arange(-180, 360+60, 60), fontsize=7)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(25))
    ax.set_yticks(np.arange(-90, 90+30, 30), crs=ccrs.PlateCarree())
    ax.set_yticklabels(np.arange(-90, 90+30, 30), fontsize=7)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(25))
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    # show grid lines
    if grid:
        ax.grid(which='major', linestyle='--',
                linewidth=0.5, color=[0.25, 0.25, 0.25])

    if Global == 0.5:
        # set axis limits
        ax.set_ylim(25, 90)
        ax.set_xlim(-180, 180)
    elif Global == -0.5:
        # set axis limits
        ax.set_ylim(-60, 0)
        ax.set_xlim(-180, 180)
    elif Global == 1:
        # set axis limits
        ax.set_ylim(-80, 80)
        ax.set_xlim(-180, 180)

    # set axis labels
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    # set figure title
    ax.set_title(title, weight='bold', fontsize=14)

    # return
    return hc, ax



# Plot quiver map over globe
def quiver_map_global(x, y, u, v, skip=2, pvals=None, alpha=0.05,
                      ax=plt.axes(projection=ccrs.PlateCarree()),
                      color='k', headwidth=3, headlength=5, headaxislength=4.5,
                      scale_units='inches', scale=3.0, key_unit_lenth=2, unit_str='m/s',
                      title="", grid=False, Global=1):
    x = x[::skip]
    y = y[::skip]
    u = u[::skip, ::skip]
    v = v[::skip, ::skip]

    # plot quiver over a map
    q = ax.quiver(x, y, u, v, color=color,
                  headwidth=headwidth,
                  headlength=headlength,
                  headaxislength=headaxislength,
                  scale_units=scale_units,
                  scale=scale,
                  width=0.002, minlength=0,)
    qk = ax.quiverkey(q, 0.9, -0.26, key_unit_lenth, str(key_unit_lenth)+' ' + unit_str, labelpos='E',
                      coordinates='axes')  ## 1.05  -0.3

    # plot signifiance (if have) over the map
    if (pvals is None):
        print('No significance hatched')
    else:
        hp = pvals.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat',
                                 levels=[0, alpha], colors=['w', 'b'], add_colorbar=False, extend='both', alpha=0.2)
        
    # countries
    filename = r'D:\01PhD\00Data\04boundary\global\global.shp'
    countries = cfeat.ShapelyFeature(
        Reader(filename).geometries(), ccrs.PlateCarree(), facecolor=None)
    ax.add_feature(countries, facecolor='none', linewidth=0.25, edgecolor='k')

    # set major and minor ticks
    ax.set_xticks(np.arange(-180, 360+60, 60), crs=ccrs.PlateCarree())
    ax.xaxis.set_minor_locator(plt.MultipleLocator(25))
    ax.set_yticks(np.arange(-90, 90+30, 30), crs=ccrs.PlateCarree())
    ax.yaxis.set_minor_locator(plt.MultipleLocator(25))
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    # show grid lines
    if grid:
        ax.grid(which='major', linestyle='--',
                linewidth=0.5, color=[0.25, 0.25, 0.25])

    if Global == 0.5:
        # set axis limits
        ax.set_ylim(25, 90)
        ax.set_xlim(-180, 180)
    elif Global == -0.5:
        # set axis limits
        ax.set_ylim(-60, 0)
        ax.set_xlim(-180, 180)
    elif Global == 1:
        # set axis limits
        ax.set_ylim(-80, 80)
        ax.set_xlim(-180, 180)


    # set axis labels
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    # set figure title
    ax.set_title(title, weight='bold')

    # return
    return q, ax


#%% Reanalysis Distribution plot

'''
该步骤绘制时空连续极端高温事件的空间分布图
Input: Reanalysis eventlist

'''

input_path = r'Z:\Private\wusj\3D-Result-revise-0906\02_Result_Processed_01_Continent_02_Summer/'
output_path = r'D:\01PhD\01_09 Three-dimensional events_hot extremes\09_Revise\03_Figure/'
reanalysis_list = ["ERA5","MERRA2", "NCEP2"]
thre_range = '19812010'
file_var = 'tmax_'
figure_num = '2.'


#%%%[0] Scatter plot 
var = 'Accumulated Area'
fig = plt.figure(figsize=(10,18))  
for event_name in reanalysis_list:
    fig = plt.figure(figsize=(10,9))  
    if event_name == 'ERA5': 
        year_period = '1950-2020'
    elif event_name == 'MERRA2':
        year_period = '1980-2020'
    elif event_name == 'NCEP2':
        year_period = '1979-2020'
    
    file_pkl = input_path + file_var + event_name +'_eventList_processed_continent_summer_'+year_period+'_thre'+ thre_range+'.pkl'
    if os.path.exists(file_pkl) ==False:
        continue
    eventList = pd.read_pickle(file_pkl)
    eventList = eventList[eventList['Mean Year']>= 1979]
    eventList_sel = event_select_area(eventList, area_thre = 1e6) 
    print(event_name, eventList_sel[var].max())
    
    ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree()) 
    scatter_map_global(eventList_sel, var = var, 
                       levels = 2*np.arange(0.1, 1.1, 0.01), 
                       ticks = 2*np.arange(0.1, 1.1, 0.1), 
                       legend = False, title = event_name,
                       ax = ax)
    plt.tight_layout() 
    plt.savefig(output_path + 'Figure-'+figure_num+file_var+event_name+'-Scatter_1979-2020.pdf', format = 'pdf')
    plt.show()


#%%%[1] Relation with latitude
var1 = 'Accumulated Area'
var2 = 'Mean Intensity'

for event_name in reanalysis_list:
    if event_name == 'ERA5':
        year_period = '1950-2020'
    elif event_name == 'MERRA2':
        year_period = '1980-2020'
    elif event_name == 'NCEP2':
        year_period = '1979-2020'
    
    file_pkl = input_path + file_var + event_name +'_eventList_processed_continent_summer_'+year_period+'_thre'+ thre_range+'.pkl'
    if os.path.exists(file_pkl) ==False:
        continue
    eventList = pd.read_pickle(file_pkl)
    eventList = eventList[eventList['Mean Year']>= 1979]
    eventList_sel = event_select_area(eventList, area_thre = 1e6)  
    print(event_name, eventList_sel[var].min())
    
    eventList_sel = eventList_sel.rename(columns={'Average Intensity':'Mean Intensity'})
    # 按照纬度的每5°进行切分，设置分组的区间(区间的边界由bins列表来指定)
    bins = np.arange(-60,90+2.5,2.5)
    
    # 切分数据
    Latitude_class = pd.cut(eventList_sel['Mean Latitude'].to_list(),bins, right=True)
    Latitude_class.categories = np.arange(1,len(bins),1)
    eventList_sel['Latitude_class'] = Latitude_class
    eventList_sel = eventList_sel.set_index('Latitude_class')
    eventList_sel_class_mean = eventList_sel.groupby('Latitude_class').mean()  
    eventList_sel_class_std = eventList_sel.groupby('Latitude_class').std()
    
    ## construct data to draw
    data = pd.DataFrame({"Accumulated_Area": eventList_sel_class_mean[var1], 
                         "Mean_Intensity": eventList_sel_class_mean[var2]})
    ## Draw    
    fig, ax1 = plt.subplots(figsize=(2, 6))
    ax1.plot(data.Accumulated_Area,  data.index, color='royalblue',alpha = 0.75,linewidth = 2, label = var1)

    ax1.set_xlabel(var1+ '(km$^{2}$)', fontsize=10, color = 'royalblue')
    ax1.set_ylabel("") 
    
    y_label_str = ['60°S', '40°S', '20°S', '0', '20°N', '40°N', '60°N', '80°N']
    ax1.set_yticks(np.arange(1, 65, 8), y_label_str, fontsize=9)
    ax1.tick_params(labelsize=10)  
    ## y轴设置科学计数法
    formatter = scilimit(ax1, limit_1 = -1, limit_2 = 3)
    ax1.xaxis.set_major_formatter(formatter)
    
    ax2 = ax1.twiny()   
    ax2.plot(data.Mean_Intensity,  data.index, color='r',alpha = 0.65,linewidth = 2, label = var2)

    ax2.set_xlabel(var2 + '(°C)', fontsize=9, color = 'r')
    ax2.tick_params(labelsize=10)   
    plt.rcParams['xtick.direction'] = 'in'
    plt.tight_layout() 
    plt.savefig(output_path + 'Figure-'+figure_num+file_var+event_name+'-With Class latitude_'+ '.pdf', format = 'pdf')
    plt.show()
    
    
#%%%[2] Movement
var = 'Accumulated Area'

for event_name in reanalysis_list:
    fig = plt.figure(figsize=(10,9))   #(21,5):横版
    
    if event_name == 'ERA5':
        year_period = '1950-2020'
    elif event_name == 'MERRA2':
        year_period = '1980-2020'
    elif event_name == 'NCEP2':
        year_period = '1979-2020'
    
    file_pkl = input_path + file_var + event_name +'_eventList_processed_continent_summer_'+year_period+'_thre'+ thre_range+'.pkl'
    if os.path.exists(file_pkl) ==False:
        continue
    eventList = pd.read_pickle(file_pkl)
    eventList = eventList[eventList['Mean Year']>= 1979]
    eventList_sel = event_select_area(eventList, area_thre = 1e6)  ## 对于全球陆地的研究，选择热浪总面积≤1000000
    print(event_name, eventList_sel[var].min())
    
    ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree()) 
    x = eventList_sel['Former Longitude']
    u = eventList_sel['Latter Longitude']-x
    y = eventList_sel['Former Latitude']
    v = eventList_sel['Latter Latitude']-y 
    movement_map_global(eventList_sel, x, u, y, v,
                        var = 'Lifespan',
                        cmap = 'OrRd',
                        levels = 3*np.arange(1,11,0.1),
                        ticks = 3*np.arange(1,11,1),
                        ax = ax)
    plt.tight_layout() 
    plt.savefig(output_path + 'Figure-'+figure_num+ file_var+event_name+'-Movement_1979-2020.pdf', format = 'pdf')
    plt.show()


#%%%[3] Draw Windrose figure
var = 'Moving Distance'
plt.style.use('fast')

for event_name in reanalysis_list:
    if event_name == 'ERA5':
        year_period = '1950-2020'
    elif event_name == 'MERRA2':
        year_period = '1980-2020'
    elif event_name == 'NCEP2':
        year_period = '1979-2020'
    
    file_pkl = input_path + file_var + event_name +'_eventList_processed_continent_summer_'+year_period+'_thre'+ thre_range+'.pkl'
    if os.path.exists(file_pkl) ==False:
        continue
    eventList = pd.read_pickle(file_pkl)
    eventList = eventList[eventList['Mean Year']>= 1979]
    for region in ["Global",'Africa', 'Eurasian','Australia','North America','South America']:
        print(event_name, region)
        if region in ['Global', 'NH', 'SH']:
            eventList_sel = event_select_area(eventList, area_thre = 1e6)  
            if region =='Global':
                eventList_sel = eventList_sel
            elif region == 'NH':
                eventList_sel = eventList_sel[eventList_sel['Mean Latitude']>=0]
                eventList_sel = eventList_sel[eventList_sel['Mean Latitude']<=90]
            elif region == 'SH':
                eventList_sel = eventList_sel[eventList_sel['Mean Latitude']>=-60]
                eventList_sel = eventList_sel[eventList_sel['Mean Latitude']<0]
        elif region in ['Africa', 'Eurasian','Australia','North America','South America']:
            eventList_sel = event_select_continent_area(eventList, continent_name = region, q = 60)
         
        print(region, 'Moving distance:', eventList_sel[var].min(), eventList_sel[var].max())
        
        fig = plt.figure(figsize=(5.827, 8.268/2))
        ax = fig.add_subplot(1,1,1, projection="windrose") 
        windrose_map(eventList_sel, var = var,
                     cmap=cm.OrRd, legend = False,
                     y_lim = False,
                     bins = 1e3*np.arange(1, 11, 1),
                     ax = ax)
        plt.tight_layout() 
        plt.savefig(output_path + 'Figure-'+figure_num+file_var+event_name+ '-' +region+'-Windrose_1979-2020.pdf', format = 'pdf')
        plt.show()