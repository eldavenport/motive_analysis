import xarray as xr
from xmitgcm import open_mdsdataset
import xgcm
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
import matplotlib.animation as animation
from fastjmd95 import rho
import sys
sys.path.append('/home/edavenport/analysis/EqMixRemix_Processing/Python/')
from open_tpose import tpose2012
warnings.filterwarnings("ignore")

filename_state = 'diag_state'

dsState = tpose2012(filename_state)

lat = 2.5
zMin = -1500
zMax = 0

lonMin = float(sys.argv[1])
lonMax = float(sys.argv[2])
lats = dsState.YC.data
lons = dsState.XC.data
depths = dsState.Z.data

latidx = np.argmin(np.abs(lats - lat))
lonli = np.argmin(np.abs(lons - lonMin))
lonui = np.argmin(np.abs(lons - lonMax)) + 1
depthli = np.argmin(np.abs(depths - zMax))
depthui = np.argmin(np.abs(depths - zMin)) + 1

N = len(dsState.time)
dsState['time'] = range(0,N,1)

print('done loading')
# grid = xgcm.Grid(dsState, periodic=['X','Y'])
# print(grid)

print('barotropic flow') # depth average
barotropic_w = dsState.WVEL.mean(dim='Zl')
# print(barotropic_w)
baroclinic_w = dsState.WVEL - barotropic_w
# print(baroclinic_w)
print('density')
sigma_0 = (rho(dsState.SALT[:,depthli:depthui], dsState.THETA[:,depthli:depthui], 0)-1000)

# filter each *time series* which is each row
fs = 1/86400
highF = (1/10)*fs # equivalent to 15 days per cycle, filter anything longer than this, 15 days * 24 hours = 360
lowF = (1/30)*fs
order = 4
cutoff = [lowF, highF]
sos = butter(order, cutoff, 'bandpass', fs=fs, output='sos')
baroclinic_w_filt = sosfiltfilt(sos, baroclinic_w.values)

print('filtered')
baroclinic_w.values = baroclinic_w_filt

vmin = -2*(10**-4)
vmax = 2*(10**-4)

startDay = 243 # sept 1

print('plotting')
fig, ax = plt.subplots()
im = baroclinic_w[startDay,depthli:depthui,latidx,lonli:lonui].plot(ax=ax,robust=True,vmin=vmin,vmax=vmax,cmap='RdBu_r')
ax.set_xlabel('Longitude')
ax.set_ylabel('Depth')
ax.set_title("Baroclinic W, Day since Sept 1 2012 = %s"%startDay)
plt.tight_layout()

print('starting animation')
def update(frame):

    day = frame+startDay
    if (day%10 == 0):
        print(day)
    
    data = baroclinic_w[day,depthli:depthui,latidx,lonli:lonui]
    im.set_array(data)
    ax.set_title("Baroclinic W, Day since Sept 1 = %s"%day)

    return (im)

video = animation.FuncAnimation(fig=fig, func=update, frames=len(range(startDay,N)),interval=500)
videostr = "/home/edavenport/analysis/EqMixRemix_Processing/Python/MOTIVE/TPOSE_2012/lee_wave_animation_140W.mp4"
video.save(filename=videostr, writer="ffmpeg")
plt.close()


