# Ce script permet de calculer le spectre d'amplitude moyen des traces de la
# ligne sismique traitee NNSIS-18-02. Ce spectre permet de selectionner la
# frequence centrale de l'ondelette de Ricker qui correspond le mieux a la
# source utilisee lors du leve.
#
# Importer les modules necessaires

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import segyio
from SeisCL import SeisCL
seis = SeisCL()

# Traces

# Emplacement du fichier

filename = '/data/jgendreau/NNSIS1802/BOUNDARIES/' \
           'NNSIS-18-02-pstm-filtered_depth.sgy'

# Ouvrir le fichier

with segyio.open(filename, ignore_geometry=True) as f:
    traces = []
    # Extraire toutes les traces. On peut en extraire moins en modifiant le
    # "range".
    for x in range(0, 4120, 1):
        # Traces une à une
        z = f.trace[x]
        # Ajouter le trace dans une liste contenant toutes les traces
        traces.append(z)
    # Taille des pas de temps
    temps = segyio.tools.dt(f)/10**6

# fft trace: calculer la somme des amplitudes
somme = np.empty(shape=(len(traces[0])))
ffttraces = []
for trace in traces:
    # Transformee de Fourier de chaque trace
    ffttrace = fft(trace)
    # Liste avec toutes les transformees de Fourier
    ffttraces.append(ffttrace)
    # Sommer les amplitudes pour chaque frequence de chaque trace
    somme = somme + np.abs(ffttrace)
# Diviser la somme des amplitudes par le nombre de traces
moyenne = np.divide(somme, len(traces))
# Normaliser les amplitudes par rapport a l'amplitude maximale
moyenne = np.divide(moyenne, max(moyenne))


# Calcul des frequence selon la longueur des traces et la taill des pas de
# temps
x = fftfreq(len(traces[0]), temps)
# Affichage du spectre d'amplitude moyen
plt.plot(x[0:len(moyenne)//2], (np.abs(moyenne[0:len(
    moyenne)//2])))
plt.xlim(0, 150)
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude relative')
plt.title('Spectre d''amplitude moyen de la ligne NNSIS-18-02')
plt.grid()
plt.show()


# Creer une ondelette de Ricker (ce sera la source utilisee pour les
# simulations

f0 = 26
vec2 = seis.ricker_wavelet(f0=f0)
plt.grid()
plt.plot(vec2)
plt.title('Ondelette de Ricker avec f0 = %i Hz' % f0)
plt.show()

# Transformee de Fourier de l'ondelette de Ricker
temps = 1/1000
amplitude = fft(vec2)
amplitude = np.divide(np.abs(amplitude), max(np.abs(amplitude)))
freq = fftfreq(len(vec2), temps)

# Spectre d'amplitude normalisee d'une trace aleatoire
plt.plot(x[0:len(moyenne)//2], (np.abs(np.divide(np.abs(ffttraces[1000][0:len(
   moyenne)//2]), max(np.abs(ffttraces[1000]))))))

# Spectre d'amplitude de la transformee de Fourier
plt.plot(freq[0:len(vec2)//2], (np.abs(amplitude[0:len(vec2)//2])))

# Spectre d'amplitude moyen
plt.plot(x[0:len(moyenne)//2], (np.abs(moyenne[0:len(
    moyenne)//2])))
plt.grid()
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude relative')
plt.legend(['Trace', 'Source', 'Donnees'])
plt.title('Source:Ricker avec f0 = %i Hz' % f0)
plt.xlim(0, 150)
plt.savefig('spectres_frequence.png', dpi=100)
plt.show()
