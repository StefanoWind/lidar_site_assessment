# -*- coding: utf-8 -*-
"""
Plot topogrpahic map around M5

2025-06-06-created
"""

import os
cd=os.getcwd()
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib
import pandas as pd
import utm
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

#%% FUnctions
def plane(X,a,b):
    x,y=X
    return a*x+b*y 

#%% Inputs
source=os.path.join(cd,'data/FC_topo_v2.nc')#source of terrain data
source_nwtc=os.path.join(cd,'data/NWTC.xlsx')#source nwtc sites
source_iec=os.path.join(cd,'data/IEC61400-12-5.xlsx')#source of IEC requirements
H=123#[m] measurement height
elevation=62 #[deg] elevation of lidar
L0=70#[m] distance lidar-met tower

centers=['M5','Site 4.0']#centers of circles
origin='Site 4.0'

#graphics
xlim=[-1000,1000]
ylim=[-1000,1000]

#%% Initialization

#read data
Data=xr.open_dataset(source)
FC=pd.read_excel(source_nwtc).set_index('Site')
IEC=pd.read_excel(source_iec)

#account for lidar probe volume
L=70+H/np.tan(np.radians(elevation))
D=2*H/np.tan(np.radians(elevation))

#locations
xy=utm.from_latlon(FC['Lat'].values,FC['Lon'].values)
FC['x']=xy[0]
FC['y']=xy[1]
        
#grid and select elevation data
x=Data.x.values
y=Data.y.values
Z=Data.z.values.T

X, Y = np.meshgrid(x,y)

#base of lidar
x0=FC['x'][origin]
y0=FC['y'][origin]
z0=np.float64(Data.z.interp(x=x0,y=y0))

#graphics
os.makedirs(os.path.join(cd,'figures'),exist_ok=True)

#%% Main

#loop over centers of cycles
for center in centers:
    fig_2d=plt.figure(figsize=(18,9))

    #select origin
    xc=FC['x'][center]
    yc=FC['y'][center]
    zc=np.float64(Data.z.interp(x=xc,y=yc))

    #loop over requirements
    for i in IEC.index:
        
        #select distance
        dist=((X-xc)**2+(Y-yc)**2)**0.5
        sel_dist=(dist<=IEC['L2'][i]*L)*(dist>=IEC['L1'][i]*L)
        
        #select wind sector
        sect=(90-np.degrees(np.arctan2(Y-yc,X-xc)))%360
        if IEC['sector1'][i]<IEC['sector2'][i]:
            sel_sect=(sect<=IEC['sector2'][i])*(sect>=IEC['sector1'][i])
        else:
            sel_sect=(sect>=IEC['sector1'][i])+(sect<=IEC['sector2'][i])
        sel=sel_dist*sel_sect
        
        #fit plane passing through origin
        popt, pcov = curve_fit(plane, (X[sel]-x0, Y[sel]-y0), Z[sel]-z0,p0=[0,0])
        Z_fit=plane((X-x0,Y-y0),popt[0],popt[1])+z0
        
        #residuals
        res=np.abs(Z_fit[sel]-Z[sel])
        max_res=np.nanmax(res)
        i_max=np.argmax(res)
        if ~np.isnan(IEC['residual'][i]):
            pass_res=max_res<=IEC['residual'][i]*(H-0.5*D)
        else: 
            pass_res='N/A'
        
        #slope from fit
        slope1=(popt[0]**2+popt[1]**2)**0.5*100
        if ~np.isnan(IEC['slope1'][i]):
            pass_slope1=slope1<=IEC['slope1'][i]
        else:
            pass_slope1='N/A'
        
        #slope from individial terrain points
        slope2=np.nanmax(np.abs(Z[sel]-z0)/dist[sel])*100
        if ~np.isnan(IEC['slope2'][i]):
            pass_slope2=slope2<=IEC['slope2'][i]
        else:
            pass_slope2='N/A'
        
        #plot
        fig=plt.figure(figsize=(12,8))
        X_plt=X.copy()
        Y_plt=Y.copy()
        Z_plt=Z.copy()
        Z_plt[~sel]=np.nan
        Z_fit_plt=Z_fit.copy()
        Z_fit_plt[(dist>IEC['L2'][i]*L*1.5)+(~sel_sect)]=np.nan
        ax=fig.add_subplot(111,projection='3d')
        
        ax.plot_surface(X_plt-xc,Y_plt-yc,Z_fit_plt,alpha=0.5,color='g')
        ax.scatter(X[sel]-xc,Y[sel]-yc,Z[sel],c='k',s=0.5,alpha=0.5,zorder=10)
        plt.plot(0,0,zc,'^k')
        plt.plot(x0-xc,y0-yc,z0,'^r')
        plt.xlim([-16*L,4*L])
        plt.ylim([-8*L,8*L])
        ax.set_xlabel('W-E [m]')
        ax.set_ylabel('S-N [m]')
        ax.grid(True)
        ax.set_aspect('equalxy')
        ax.set_box_aspect([1,1,0.5])
        ax.view_init(azim=-70,elev=20)
        
        if ~np.isnan(IEC['slope1'][i]):
            plt.title(f'{IEC["L1"][i]} - {IEC["L2"][i]} L from {center}, {IEC["sector1"][i]}° - {IEC["sector2"][i]}°:\n' +\
                      f'max deviation = {str(np.round(max_res,1))} m: PASS={pass_res},\n'+
                      f'max slope: {str(np.round(slope1,1))}%: PASS={pass_slope1}')
        elif ~np.isnan(IEC['slope2'][i]):
            plt.title(f'{IEC["L1"][i]} - {IEC["L2"][i]} L from {center}, {IEC["sector1"][i]}° - {IEC["sector2"][i]}°:\n' +\
                      f'max deviation = {str(np.round(max_res,1))} m: PASS={pass_res},\n'+
                      f'max slope: {str(np.round(slope2,1))}%: PASS={pass_slope2}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(cd,'figures',f'H{H}.{center}.{str(int(i))}.png'))
        plt.close()
        
        plt.figure(fig_2d)
        cos=np.cos(np.linspace(0,1,100)*2*np.pi)
        sin=np.sin(np.linspace(0,1,100)*2*np.pi)
        vmax=np.nanmax(np.abs(Z[sel]-z0))
        ax=plt.subplot(2,3,i+1)
        plt.pcolor(X_plt-xc,Y_plt-yc,Z_fit-z0,cmap='seismic',alpha=0.5, edgecolors='none', linewidths=0,vmin=-vmax,vmax=vmax)
        plt.pcolor(X_plt-xc,Y_plt-yc,Z_plt-z0,cmap='seismic',edgecolors='none',vmin=-vmax,vmax=vmax)
        plt.plot(IEC['L1'][i]*cos*L,IEC['L1'][i]*sin*L,'k')
        plt.plot(IEC['L2'][i]*cos*L,IEC['L2'][i]*sin*L,'k')
        plt.plot([0,np.cos(np.radians(90-IEC['sector1'][i]))*IEC['L2'][i]*L],
                 [0,np.sin(np.radians(90-IEC['sector1'][i]))*IEC['L2'][i]*L],'k')
        plt.plot([0,np.cos(np.radians(90-IEC['sector2'][i]))*IEC['L2'][i]*L],
                 [0,np.sin(np.radians(90-IEC['sector2'][i]))*IEC['L2'][i]*L],'k')
       
        if ~np.isnan(IEC['residual'][i]):
            plt.plot(X[sel][i_max]-xc,Y[sel][i_max]-yc,'xk',markersize=15)
            plt.text(X[sel][i_max]-xc+IEC['L2'][i]*L/10,Y[sel][i_max]-yc+IEC['L2'][i]*L/5,f'Diff={np.round(max_res,1)} m \n (max={np.round(IEC["residual"][i]*(H-0.5*D),1)} m)',bbox={'facecolor':'w','alpha':0.5})
        if ~np.isnan(IEC['slope1'][i]):
            plt.text(-IEC['L2'][i]*L*1.1,-IEC['L2'][i]*L*1.2,f'Slope={np.round(slope1,1)}% (max={int(IEC["slope1"][i])}%)',bbox={'facecolor':'w','alpha':0.5})
        if ~np.isnan(IEC['slope2'][i]):
            plt.text(-IEC['L2'][i]*L*1.1,-IEC['L2'][i]*L*1.2,f'Slope={np.round(slope2,1)}% (max={int(IEC["slope2"][i])}%)',bbox={'facecolor':'w','alpha':0.5})
        plt.xlim([-IEC['L2'][i]*L*1.3-300,IEC['L2'][i]*L*1.3+300])
        plt.ylim([-IEC['L2'][i]*L*1.3-300,IEC['L2'][i]*L*1.3+300])
        ax.set_aspect('equal')
        plt.colorbar(label='Elevation difference [m]')
        plt.xlabel('W-E [m]')
        plt.ylabel('S-N [m]')
    plt.tight_layout()
    plt.savefig(os.path.join(cd,'figures',f'H{H}.{center}.top.png'))
    plt.close()