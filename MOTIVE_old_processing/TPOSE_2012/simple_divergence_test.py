import xarray as xr
from xmitgcm import open_mdsdataset
import xgcm
import numpy as np
import warnings
import matplotlib.pyplot as plt
import sys
warnings.filterwarnings("ignore")

data_parent_dir = '/data/SO6/TPOSE_diags/tpose6/'
grid_dir = '/data/SO6/TPOSE_diags/tpose6/grid_6/'

filename_state = 'diag_state'

num_diags = 31+29 #
itPerFile = 72 # 1 day
intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
data_dir = data_parent_dir + 'jan2012/diags/'

latMin = float(sys.argv[3])
latMax = float(sys.argv[4])

zMin = -2000
zMax = 0

lonMin = float(sys.argv[1])
lonMax = float(sys.argv[2])

print('lon min: ' + str(lonMin))
print('lon max: ' + str(lonMax))
print('lat min: ' + str(latMin))
print('lat max: ' + str(latMax))

dsState = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=[filename_state])

lats = dsState.YC.data
lons = dsState.XG.data
depths = dsState.Z.data

latli = np.argmin(np.abs(lats - latMin))
latui = np.argmin(np.abs(lats - latMax)) + 1
lonli = np.argmin(np.abs(lons - lonMin))
lonui = np.argmin(np.abs(lons - lonMax)) + 1
depthli = np.argmin(np.abs(lons - zMax))
depthui = np.argmin(np.abs(lons - zMin)) + 1

cropLats = lats[latli:latui]
cropLons = lons[lonli:lonui]

folder_months = ['mar2012/','may2012/','jul2012/','sep2012/']
folder_days = np.array([61,61,62,122])
itPerFile = [72, 72, 72, 72]
i = 0
for month in folder_months:
    num_diags = folder_days[i]
    intervals = range(itPerFile[i],itPerFile[i]*(num_diags+1),itPerFile[i])
    data_dir = data_parent_dir + month + 'diags/'
    dsStateNew = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=[filename_state])
    dsState = xr.concat([dsState, dsStateNew],'time')

    i += 1

N = len(dsState.time)

grid = xgcm.Grid(dsState, periodic=['X','Y'])

u_transport = dsState.UVEL * dsState.dyG * dsState.hFacW * dsState.drF
v_transport = dsState.VVEL * dsState.dxG * dsState.hFacS * dsState.drF

div_uv = -(grid.diff(u_transport, 'X') + grid.diff(v_transport, 'Y')) / dsState.rA
print(div_uv.dims)

fig, ax = plt.subplots()
# all lat, lon (time=0 andd depth=0)
div_uv[0,20,latli:latui,lonli:lonui].plot(ax=ax)
plt.tight_layout()
image_str = 'looking_for_lee_waves.png'
plt.savefig(image_str,format='png')
plt.close()
