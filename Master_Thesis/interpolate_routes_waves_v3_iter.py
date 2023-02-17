# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:01:25 2022

@author: Sia Stroemsnes & Marie Staveland

"""

import pandas as pd
from distance_bearing import haversine, bearing
import numpy as np
import math
import xarray as xa
from raytrace import raytrace
from datetime import datetime,timedelta
import glob
import geopandas as gpd
from shapely.geometry import LineString


eca_polys=gpd.read_file("/Volumes/LaCie/Master Thesis/Thesis_v2/eca_rule_14.geojson")[["area","geometry"]]

###Parameters
speed=12.5 ###knots


###Folder for saving files
sav_fol="/Volumes/LaCie/Master Thesis/Wave_parquet/"

###Frequency waves
wave_dates=pd.DataFrame(pd.date_range(start='2019/1/1', end='2020/1/1',inclusive="left",freq="3H"),columns=["timestamp"]).set_index("timestamp").index

routes=pd.read_parquet("/Volumes/LaCie/Master Thesis/Thesis_v2/thesis_routes_v2")
routes=routes.assign(route_iter_name=routes.apply(lambda x: "{}_pt1".format(x["full_route"]) if x.leg==1 \
                                                  else "{}_pt2".format(x["full_route"]),axis=1))
    
###Interpolate positions based on speed.

exp_df=[]
for ind, group in routes.groupby("route_iter_name"):
    
    group_in=group.copy()
    group_in=group_in.drop_duplicates(subset=["lon","lat"])
    ###Shift positions in same row as previous
    group_in=group_in.assign(lon_f=group_in.lon.shift(),
                          lat_f=group_in.lat.shift())
    
    ##Estimate the distance with haversine function. Imported module.
    group_in=group_in.assign(dis_f=group_in.apply(lambda x: haversine(x.lon_f,x.lat_f,x.lon,x.lat),axis=1))
    
    ##Get the integer division of parts to interpolate positions within existing waypoints every distance/speed
    group_in=group_in.assign(parts_per_hour=group_in.dis_f//speed)
    
    
    ###Convert pairs of points to LineStrings to test with ECA polygons
    group_in=group_in.assign(lon_to=group_in.lon.shift(-1),
                   lat_to=group_in.lat.shift(-1))
    
    group_in=group_in.assign(geometry=group_in.apply(lambda x: LineString(((x["lon"],x["lat"]),(x["lon_to"],x["lat_to"]))),axis=1))
    
    group_in=gpd.GeoDataFrame(group_in,geometry="geometry")
    group_in.crs=eca_polys.crs
    
    group_in=gpd.sjoin(group_in,eca_polys,how="left")
    
    group_in=group_in.assign(eca=np.where(group_in["area"].notnull(),1,0))
    ### Iterate over all parts. Find the positions for interpolating every 3 hours. If waypoints within 3 hours then keep as it is
    for ind_t,row in group_in.iterrows():
        exp=[]
        if math.isnan(row.parts_per_hour):
            continue
        else:
            ###Get interpolation every 3 hours between dataloy routes
            for val in range(0,int(row.parts_per_hour)+1,3):
                new_x=row.lon_f+((val*speed)/row.dis_f)*(row.lon-row.lon_f)
                new_y=row.lat_f+((val*speed)/row.dis_f)*(row.lat-row.lat_f)
                
                exp.append([new_x,new_y,row.route_name,row.full_route,row.leg,row.eca])
            exp.append([row.lon,row.lat,row.route_name,row.full_route,row.leg,row.eca])
        
        ##Concatenate old and new waypoints
        exp=pd.DataFrame(exp,columns=["lon","lat","route_name","full_route","leg","eca"])
        
        exp.drop_duplicates(inplace=True)
        
        
        exp=exp.assign(lon_f=exp.lon.shift(),
                              lat_f=exp.lat.shift())
        
        ##Recaluclate distance between waypoints
        exp=exp.assign(dis_f=exp.apply(lambda x: haversine(x.lon_f,x.lat_f,x.lon,x.lat),axis=1))
        exp=exp.assign(time_f_hours=exp.dis_f/speed,
                       route_iter_name=ind)        
                
        exp_df.append(exp)

##Out of the groupby, concatenate all waypoints for all routes.
exp_df=pd.concat(exp_df).drop(columns=["lon_f","lat_f"]).reset_index(drop=True)
exp_df.drop_duplicates(subset=["lon","lat","route_name"],inplace=True)
exp_df.reset_index(drop=True,inplace=True)

# ###Grids
##Read just as reference to extract the grid coordinates. Recognize the grids coordinates.
xs=xa.open_dataset("/Volumes/LaCie/Master Thesis/Codes/Waves/2006/01/WAVERYS_20060101_R20060101.nc")
x_grid=xs.longitude.values
y_grid=xs.latitude.values

###To start a new timestart time. Relevant only for Panama Canal to Qingdao part 2. Check out line 97 for if condition and line 196.
##As groupby goes alphabeltical, then Panama Qingdao goes first then the part1_timestart is updated with the last time of Panama Qingdao route
# for timestart in pd.date_range(start='2016/1/1', end='2019/10/31',inclusive="left",freq="D"):

##Starting date range iteration. 
for timestart in pd.date_range(start="2019/1/1",end="2020/1/1",inclusive="left",freq="D"):
    
    part1_timestart=timestart
    
    per_route=[]
    ###Iterate per route (6 routes) + 1 (Panama to Qingdao west of International Date Line)
    for route_iter_name,group in exp_df.groupby("route_iter_name"):
        
        route_name=group.route_name.iloc[0]
        full_route=group.full_route.iloc[0]
        
        ##Applicable only when leg 2
        if group["leg"].iloc[0]==2:
            timestart_in=part1_timestart
    
        else:
            timestart_in=timestart
        ##Empty list to append per day weather filtered for route crossing grids 
        export_results=[]
        
        ###Create new columns to have in the same row, the origin-destination positions oroute waypoints
        group=group.assign(lon_to=group.lon.shift(-1),
                            lat_to=group.lat.shift(-1))
        
        ##Clean subset for empty rows after shift
        group=group.dropna(subset="lon_to")
        
        group=group.assign(heading_to=group.apply(lambda x: bearing(x.lon,x.lat,x.lon_to,x.lat_to),axis=1))
        
        ###Recognize the initial grid position by comparing the waypoint positions to all the weather grids.
        ### Argmin returns the index of the closest grid position to the route waypoints.
        
        gr_points_fr_x_wave=(np.abs(x_grid-group["lon"].to_numpy().reshape(-1,1))).argmin(axis=1)[:,np.newaxis]
        
        gr_points_fr_y_wave=(np.abs(y_grid-group["lat"].to_numpy().reshape(-1,1))).argmin(axis=1)[:,np.newaxis]
        
        gr_points_to_x_wave=(np.abs(x_grid-group["lon_to"].to_numpy().reshape(-1,1))).argmin(axis=1)[:,np.newaxis]
        
        gr_points_to_y_wave=(np.abs(y_grid-group["lat_to"].to_numpy().reshape(-1,1))).argmin(axis=1)[:,np.newaxis]
        
        
        ###Concatenate all grids indeces in same row
        ##Dataframe w original data from group.
        hull_test_index=np.concatenate((gr_points_fr_x_wave,gr_points_fr_y_wave,gr_points_to_x_wave,gr_points_to_y_wave,
                                        group["time_f_hours"].to_numpy().reshape(-1,1),
                                        group["lon"].to_numpy().reshape(-1,1),
                                        group["lat"].to_numpy().reshape(-1,1),
                                        group["heading_to"].to_numpy().reshape(-1,1),
                                        group["leg"].to_numpy().reshape(-1,1),
                                        group["eca"].to_numpy().reshape(-1,1)),axis=1)
        
        hull_test_index=pd.DataFrame(hull_test_index,columns=["wave_box_fr_x","wave_box_fr_y","wave_box_to_x",
                                                              "wave_box_to_y","time_f_hours","lon_wp","lat_wp","heading_to","leg","eca"])
        
        hull_test_index=hull_test_index.assign(time_f_hours=hull_test_index.time_f_hours.fillna(0))
        hull_test_index=hull_test_index.assign(cum_time_f_hours=hull_test_index.time_f_hours.cumsum())
        
        hull_test_index=hull_test_index.assign(time_xa=(timestart_in+pd.to_timedelta(hull_test_index.cum_time_f_hours,unit="h")).astype('<M8[s]'))\
                                    .drop(columns=["cum_time_f_hours","time_f_hours"])
        
        ###Transform into indexes
        hull_test=np.concatenate((gr_points_fr_x_wave,gr_points_fr_y_wave,gr_points_to_x_wave,gr_points_to_y_wave),axis=1)
        
        ##Remove duplicates
        test_indices_u=np.unique(hull_test,axis=0)
        
        ### Pass row per row over a raytrace module. Check how raytrace identifies the grids crossed by the route line.
        ### The input in here are positions 
        cell_index=np.concatenate(np.vectorize(raytrace,otypes=[np.ndarray])(test_indices_u[:,0],test_indices_u[:,1],test_indices_u[:,2],test_indices_u[:,3]))
        
        ##Clean dataframe. The result in here are rows of grid longitude, grid latitude, grid index from x, grid index from y,
        #### grid_index_to_x and grid_index_to_y. The last 4 columns repeats as a route from to passes several grids 
        #### with positions declared on the first two columns (longitude and latitude)
        cell_index=np.unique(cell_index,axis=0)
        cell_index=pd.DataFrame(cell_index,columns=["longitude_xa","latitude_xa","wave_box_fr_x","wave_box_fr_y","wave_box_to_x","wave_box_to_y"])
        cell_index=cell_index.assign(longitude_xa=x_grid[cell_index["longitude_xa"]],
                                      latitude_xa=y_grid[cell_index["latitude_xa"]])
        
        # ### Create a column of unique combinations of lats and lons
        cell_index=cell_index.drop_duplicates(subset=["latitude_xa","longitude_xa"])
        cell_index=pd.merge(cell_index,hull_test_index,on=["wave_box_fr_x","wave_box_fr_y","wave_box_to_x","wave_box_to_y"])
        
        cell_index=cell_index.assign(time_wave_xa=wave_dates[wave_dates.get_indexer(cell_index.time_xa,method="nearest")])
        
          
        cell_index=cell_index.assign(year_wa=cell_index.time_wave_xa.dt.year,
                                      month_wa=cell_index.time_wave_xa.dt.month,
                                      day_wa=cell_index.time_wave_xa.dt.day)
        
        
        cell_index=cell_index.assign(file_name_wave=cell_index.apply(lambda x: "/Volumes/LaCie/Master Thesis/Codes/Waves/{}/{:02d}/WAVERYS_{}{:02d}{:02d}_R{}{:02d}{:02d}.nc".\
                                                                      format(x.year_wa,x.month_wa,
                                                                            x.year_wa,x.month_wa,x.day_wa,
                                                                            x.year_wa,x.month_wa,x.day_wa),axis=1))
        
        ##Wave iteration
        per_day=[]
        ##Iterate over valid files only (relevant to the specific route)
        for day, group in cell_index.groupby("file_name_wave"):
            
            xs_in=xa.open_dataset(day)
            
            ##Select only the combination of xa longitude and latitude and specific time of crossing. Cheaper than transforming it to dataframe
            group=group.assign(vhm0=group.apply(lambda x: xs_in.sel(time=x.time_wave_xa,
                                                      longitude=x.longitude_xa,latitude=x.latitude_xa,method="pad").VHM0.values,axis=1),
                                vmdr=group.apply(lambda x: xs_in.sel(time=x.time_wave_xa,
                                                                        longitude=x.longitude_xa,latitude=x.latitude_xa,method="pad").VMDR.values,axis=1),
                                vhm0_sw1=group.apply(lambda x: xs_in.sel(time=x.time_wave_xa,
                                                                          longitude=x.longitude_xa,latitude=x.latitude_xa,method="pad").VHM0_SW1.values,axis=1),
                                vmdr_sw1=group.apply(lambda x: xs_in.sel(time=x.time_wave_xa,
                                                                          longitude=x.longitude_xa,latitude=x.latitude_xa,method="pad").VMDR_SW1.values,axis=1))
            
            group=group.assign(app_wave_dir=np.where((group.vmdr-group.heading_to)>0,group.vmdr-group.heading_to,360+(group.vmdr-group.heading_to)),
                                            app_swell_dir=np.where((group.vmdr_sw1-group.heading_to)>0,group.vmdr_sw1-group.heading_to,360+(group.vmdr_sw1-group.heading_to)))
            
            ###Every trace crossing within a time stamp is averaged.
            group=group.groupby(["time_wave_xa","time_xa",'wave_box_fr_x', 'wave_box_fr_y', 'wave_box_to_x', 'wave_box_to_y',
                                  "lon_wp","lat_wp","heading_to","leg","app_wave_dir","app_swell_dir","eca"]).agg({'vhm0':"mean",'vmdr':"mean", 'vhm0_sw1':"mean", 'vmdr_sw1':"mean"}).reset_index()
            
            
            per_day.append(group)
            
        ##Concatenate all the records for a route per a specific day
        per_day=pd.concat(per_day).reset_index(drop=True)
        
        per_day=per_day.assign(route_name=route_name,starting_date=timestart,
                               full_route=full_route)
        
        per_route.append(per_day)
        
        ##Update part1_timestart only for the Panama Qingdao case. This becomes the starting point of PART 2 Panama Qingdao in the next interation.
        if group["leg"].iloc[0]==1:
            part1_timestart=per_day.loc[per_day.time_wave_xa.last_valid_index(),"time_wave_xa"]+timedelta(days=2)
            
    ##Concatenate all routes
    per_route_g=pd.concat(per_route).reset_index(drop=True)
    
    per_route_g=per_route_g.assign(app_swell_dir=np.where(per_route_g.app_swell_dir.between(0,45),1,
                                                        np.where(per_route_g.app_swell_dir.between(45,90),2,
                                                           np.where(per_route_g.app_swell_dir.between(90,135),3,
                                                                    np.where(per_route_g.app_swell_dir.between(135,180),4,
                                                                             np.where(per_route_g.app_swell_dir.between(180,225),5,
                                                                                      np.where(per_route_g.app_swell_dir.between(225,270),6,
                                                                                               np.where(per_route_g.app_swell_dir.between(270,315),7,
                                                                                                        np.where(per_route_g.app_swell_dir.between(315,360),8,None)))))))),
                                 app_wave_dir=np.where(per_route_g.app_wave_dir.between(0,45),1,
                                                                                     np.where(per_route_g.app_wave_dir.between(45,90),2,
                                                                                        np.where(per_route_g.app_wave_dir.between(90,135),3,
                                                                                                 np.where(per_route_g.app_wave_dir.between(135,180),4,
                                                                                                          np.where(per_route_g.app_wave_dir.between(180,225),5,
                                                                                                                   np.where(per_route_g.app_wave_dir.between(225,270),6,
                                                                                                                            np.where(per_route_g.app_wave_dir.between(270,315),7,
                                                                                                                                     np.where(per_route_g.app_wave_dir.between(315,360),8,None))))))))
                                 )
    
    
    ##Save to a parquet file. YOU need module fastparquet or pyarrow to load and write on this file. 
    per_route_g=per_route_g.sort_values(by=["full_route","leg","time_xa"]).reset_index(drop=True)
    
    ##For me this is the most efficient file format (light and versatile)
    per_route_g.to_parquet(sav_fol+"voyage_started_{}".format(timestart.date()),allow_truncated_timestamps=True)
    
    
       