import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft,fftfreq
from scipy import signal
import segyio

##Traces

filename = '/data/jgendreau/NNSIS1802/BOUNDARIES/NNSIS-18-02-pstm-filtered_depth.sgy'

with segyio.open(filename,ignore_geometry=True) as f:
    trace = f.trace[100]
    temps = segyio.tools.dt(f)/10**6
    print(temps)
##fft trace

ffttrace=fft(trace)
x=fftfreq(len(trace),temps)
plt.plot(x[0:len(trace)//2], (np.abs(ffttrace[0:len(trace)//2])))
plt.xlim(0,100)
plt.show()

## Ricker wavelet

points = 100
a = 4
vec2 = signal.ricker(points, a)
print(len(vec2))
plt.grid()
plt.plot(vec2)
plt.show()


T=1/20
y=fft(vec2)
x=fftfreq(points,T)
plt.plot(x[0:points//2], (np.abs(y[0:points//2])))
plt.show()