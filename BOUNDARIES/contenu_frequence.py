import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import segyio
from SeisCL import SeisCL
seis = SeisCL()

##Traces

filename = '/data/jgendreau/NNSIS1802/BOUNDARIES/NNSIS-18-02-pstm-filtered_depth.sgy'

with segyio.open(filename,ignore_geometry=True) as f:
    traces = []
    #extraire une trace a chaque 10 traces.
    for x in range(0,4120,1):
        z = f.trace[x]
        traces.append(z)
    temps = segyio.tools.dt(f)/10**6

# fft trace: calculer la somme des amplitudes
somme = np.empty(shape=(len(traces[0]),))
ffttraces = []
for trace in traces:
    ffttrace=fft(trace)
    ffttraces.append(ffttrace)
    somme = somme + np.abs(ffttrace)
moyenne =  np.divide(somme,len(traces))
moyenne = np.divide(moyenne,max(moyenne))

x=fftfreq(len(traces[0]), temps)
plt.plot(x[0:len(moyenne)//2], (np.abs(moyenne[0:len(
    moyenne)//2])))
#plt.plot(x[0:len(ffttraces[40])//2], (np.abs(ffttraces[40]
 #                                            [0:len(moyenne)//2])))
plt.xlim(0, 150)
plt.grid()
plt.show()


## Ricker wavelet

vec2 = seis.ricker_wavelet(f0=25)
plt.grid()
plt.plot(vec2)
plt.show()


T=1/1000
amplitude=fft(vec2)
amplitude = np.divide(np.abs(amplitude),max(np.abs(amplitude)))
freq=fftfreq(len(vec2),T)
#plt.plot(x[0:len(moyenne)//2], (np.abs(np.divide(ffttraces[1000][0:len(
#   moyenne)//2],max(ffttraces[1000])))))
plt.plot(freq[0:len(vec2)//2], (np.abs(amplitude[0:len(vec2)//2])))
plt.plot(x[0:len(moyenne)//2], (np.abs(moyenne[0:len(
    moyenne)//2])))
plt.grid()
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude relative')
plt.legend(['Source','Donnees'])
plt.xlim(0, 150)
plt.savefig('spectres_frequence.png',dpi = 100)
plt.show()