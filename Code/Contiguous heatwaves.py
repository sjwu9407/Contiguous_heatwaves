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
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from windrose import WindroseAxes
from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm
import pyproj
import gc


def detrend(x, y): 
    """ Remove the linear trend of a time series
    
    Parameters
    ----------
    x : Independent variable (e.g., year)
    y : Dependent variable

    Returns
    -------
    detrend_y: detrended data
    """
    
    not_nan_ind = ~np.isnan(y)

    if len(y[not_nan_ind]) < len(y)*0.8:
        return np.full(y.size, np.nan)

    m, b, r_val, p_val, std_err = stats.linregress(x[not_nan_ind], y[not_nan_ind])
    detrend_y = y - (m*x + b)
    
    return detrend_y



def moving_average_v3(data, window=15):
    """ Calculate the sliding percentile at calendar day

    Parameters
    ----------
    data : dataarray that needs to be computed
    window : size of the moving window. The default is 15.

    Returns
    -------
    threshold : moving average

    """

    threshold = xr.DataArray(data=np.full([366, len(data["lat"]), len(data["lon"])], np.nan),
                             dims=["dayofyear", "lat", "lon"],
                             coords=dict(lon=data["lon"].values,
                                         lat=data["lat"].values,
                                         dayofyear=range(1, 367)))
    plusminus = window // 2
    for day in range(1, 367):    
        # For a given 'day' obtain a window around that day.
        window_days = (np.arange(day - plusminus - 1, day + plusminus) % 366) + 1
        window_index = data.time.dt.dayofyear.isin(window_days)
        # Calculate mean values
        threshold[day-1, :] = data[window_index, :].mean(dim='time', skipna=True)

    return threshold

def change_width_horizontal(ax, new_value) :
    
    for patch in ax.patches :
        
        current_height = patch.get_height()
        diff = current_height - new_value

        # we change the bar width
        patch.set_height(new_value)

        # we recenter the bar
        patch.set_y(patch.get_y() + diff * .5)

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
    # filename = '/Users/luoming/Dropbox/Data/Maps/Data_ipynb/country1.shp'
    # filename = 'D:/Dropbox/Research/landareas/landareas.shp'
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


    
#%% CMIP6 yearly plot
#%%%[0] Input
thre_range = '19812010'
path_figure = r'D:\Contiguous_heatwaves\Figure/'

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
    if file_scenr in ['historical_extend_ssp245', 'hist-GHG', 'hist-nat']:
        year_period = '1950-2020'
    elif file_scenr in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
        year_period = '2015-2100'
        
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
        ds_cmip6_ = ds_cmip6.sel(year = slice(1979, 2009)) 
        
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
            ds_cmip6_ = ds_cmip6.sel(year = slice(1979, 2007)) 
            
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
            _color = 'red'
            _label = 'Hist-GHG'
        elif file_scenr == 'hist-nat':
            _color = 'seagreen'
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
plt.savefig(path_figure + 'Figure-R2.'+region+'-CMIP6_historical_GHG_nat_'+thre_range+'_1979-2008.pdf', format = 'pdf')
# plt.savefig(path_figure + 'Figure-R9.'+region+'-CMIP6_future_time_series_'+thre_range+'_1979-2100.pdf', format = 'pdf')
plt.show()


#%%%[3] Plot long-term trends and shading
scenario_hist_list_ = ['historical_extend_ssp245','hist-GHG','hist-nat']
fig = plt.figure(figsize=(15, 11), dpi=300)
i=0
for var_name in var_list:
    print(var_name)
    i+=1
    ax = fig.add_subplot(3, 3, i)
    for file_scenr in scenario_hist_list_:
        variable_tmp = ds_hist.sel(scenario = file_scenr)[var_name].mean(dim="model")
        _slope = Trend_scenario[Trend_scenario['scenario'] == file_scenr].iloc[0][var_name]
        _pval = Trend_scenario[Trend_scenario['scenario'] == file_scenr].iloc[1][var_name]
        if file_scenr == 'historical_extend_ssp245':
            _color = 'black'
            _label = 'Hist-ALL'
        elif file_scenr == 'hist-GHG':
            _color = 'red'
            _label = 'Hist-GHG'
        elif file_scenr == 'hist-nat':
            _color = 'steelblue'
            _label = 'Hist-NAT'
        
        ## plot
        data = variable_tmp.to_dataframe()
        sns.regplot(data = data, x= data.index, y = var_name, 
                    ci=95, scatter = None, line_kws={'lw': 0.8, 'color': _color, 'alpha': 1,})
        
        plt.plot(variable_tmp['year'], variable_tmp, 
                 color = _color, alpha=1, linewidth = 2,
                 label= _label + '(' + '%+.2f' % (_slope)+ ', p=' + '%.2f' % (_pval)+ ')')
    # 设置参数
    plt.xlabel("")
    plt.ylabel(var_name + ' ' + unit_list[i-1], fontsize = 14) 
    ax.tick_params(axis='x',labelsize=12)
    ax.tick_params(axis='y',labelsize=14)
    
    ## y轴设置科学计数法
    formatter = scilimit(ax, limit_1 = -1, limit_2 = 3)
    ax.yaxis.set_major_formatter(formatter)
    
    leg = plt.legend(loc = 'best',fontsize = 10)
    # change the font colors to match the line colors
    for line,text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
plt.tight_layout()    
plt.savefig(path_figure + 'Figure-3.CMIP6_hist-all-GHG-nat_'+thre_range+'_1950-2020.pdf', format = 'pdf')
plt.show()


#%%%[4] Plot long-term trends--ALL
scenario_hist_list_ = ['historical_extend_ssp126', 'historical_extend_ssp245', 
                       'historical_extend_ssp370','historical_extend_ssp585']
# fig = plt.figure(figsize=(15, 11), dpi=300)
fig = plt.figure(figsize=(18, 4), dpi=300)
i=0
# for var_name in var_list:
for var_name in ['Frequency', 'Total Moving Distance', 'Moving Speed']:   
    print(var_name)
    if var_name == 'Frequency':
        ymin = 45
        ymax = 110
    elif var_name == 'Total Moving Distance':
        ymin = 1900
        ymax = 3500
    elif var_name == 'Moving Speed':
        ymin = 260
        ymax = 380
    
    i+=1
    ax = fig.add_subplot(1,3, i)
    for file_scenr in scenario_hist_list_:
        
        if file_scenr in scenario_hist_list_:
            ds_cmip6_ = ds_cmip6.sel(year = slice(1979, 2020)) 
 
        variable_tmp = ds_cmip6_.sel(scenario = file_scenr, region ='Global')[var_name].mean(dim="model")
        
        _slope = Trend_scenario[Trend_scenario['scenario'] == file_scenr].iloc[0][var_name]
        _pval = Trend_scenario[Trend_scenario['scenario'] == file_scenr].iloc[1][var_name]
        if file_scenr == 'historical_extend_ssp126':
            _color = 'indigo'
            _label = 'Hits-ALL-SSP126'
        elif file_scenr == 'historical_extend_ssp245':
            _color = 'seagreen'
            _label = 'Hits-ALL-SSP245'
        elif file_scenr == 'historical_extend_ssp370':
            _color = 'crimson'
            _label = 'Hits-ALL-SSP370'
        elif file_scenr == 'historical_extend_ssp585':
            _color = 'steelblue'
            _label = 'Hits-ALL-SSP585'
        
        ## plot
        data = variable_tmp.to_dataframe()
        sns.regplot(data = data, x= data.index, y = var_name, 
                    ci=95, scatter = None, line_kws={'lw': 0.8, 'color': _color, 'alpha': 1,})
        plt.plot(variable_tmp['year'], variable_tmp, 
                 color = _color, alpha=1, linewidth = 2,
                 label= _label + '(' + '%+.2f' %_slope+ ', p=' + '%.2f' %_pval+ ')')
    # 设置参数
    plt.xlabel("Year", fontsize = 14)
    plt.ylabel(var_name + ' ' + unit_list[i-1], fontsize = 14)
    ax.tick_params(axis='x',labelsize=12)
    ax.tick_params(axis='y',labelsize=12)
    plt.ylim([ymin,ymax])  
    ## y轴设置科学计数法
    formatter = scilimit(ax, limit_1 = -1, limit_2 = 3)
    ax.yaxis.set_major_formatter(formatter)
    
    leg = plt.legend(loc = 'best',fontsize = 11)
    # change the font colors to match the line colors
    for line,text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
plt.tight_layout()    
plt.savefig(path_figure + 'Figure-R1.CMIP6_hist-all_ssp_'+thre_range+'_1979-2020.pdf', format = 'pdf')
plt.show()


### kstest检验
ds_cmip6_ = ds_cmip6.sel(year = slice(2015, 2020)) 

for file_scenr in ['historical_extend_ssp126','historical_extend_ssp370', 'historical_extend_ssp585']:
    for var_name in ['Frequency','Total Moving Distance', 'Moving Speed']:
        variable_tmp_ssp245 = ds_cmip6_.sel(scenario = 'historical_extend_ssp245', region ='Global')[var_name].mean(dim="model")
        variable_tmp = ds_cmip6_.sel(scenario = file_scenr, region ='Global')[var_name].mean(dim="model")

        statistic, pvalue = stats.kstest(variable_tmp_ssp245, variable_tmp)
        # print(file_scenr, var_name,'statistic =',statistic, 'pvalue =', pvalue)
        print(file_scenr, var_name, 'pvalue =', pvalue)
        del var_name, statistic, pvalue


#%% Reanalysis yearly plot
#%%%[0] Input
thre_range = '19812010'
path_figure = r'D:\Contiguous_heatwaves\Figure/'

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
    
        # print(region,  eventList_sel['Mean Latitude'].min(), eventList_sel['Mean Latitude'].max())    
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



#%%% [6] 前5年和后5年均值
reanalysis = 'ERA5'
region = 'Global'
## 1979-1983
ds_reanalysis_first = ds_reanalysis.sel(reanalysis = reanalysis,region = region,year=slice(1979, 1983))
## 2016-2020
ds_reanalysis_last = ds_reanalysis.sel(reanalysis = reanalysis,region = region,year=slice(2016, 2020))

ds_reanalysis_first_mean = ds_reanalysis_first.mean(dim='year')
ds_reanalysis_last_mean = ds_reanalysis_last.mean(dim='year')

print('first_mean：Frequency', ds_reanalysis_first_mean['Frequency'])
print('last_mean：Frequency', ds_reanalysis_last_mean['Frequency'])

print('first_mean:Lifetime', ds_reanalysis_first_mean['Lifetime'])
print('last_mean:Lifetime', ds_reanalysis_last_mean['Lifetime'])

print('first_mean:Mean duration', ds_reanalysis_first_mean['Mean Duration'])
print('last_mean:Mean duration', ds_reanalysis_last_mean['Mean Duration'])

print('first_mean:Total Moving Distance', ds_reanalysis_first_mean['Total Moving Distance'])
print('last_mean：Total Moving Distance', ds_reanalysis_last_mean['Total Moving Distance'])



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
            _color = 'steelblue'  ##
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