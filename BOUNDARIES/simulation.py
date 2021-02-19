from SeisCL import SeisCL
seis = SeisCL()
import numpy as np
import matplotlib.pyplot as plt

#Domaine de la simulation
seis.csts['ND'] = 2                                 # Number of dimension
seis.csts['N'] = np.array([250, 250])               # Grid size [NZ, NX, NY]
seis.csts['dh'] = dh = 1                            # Grid spatial spacing
seis.csts['dt'] = dt = 0.00005                   # Time step size
seis.csts['NT'] = NT = 500                         # Number of time steps

#Parametres du modele. Je prend les 250 premieres valeurs dans les deux axes.

from NNSIS1802 import props2d

model_dict = {"vp":props2d['vp'][:250,:250],"vs":props2d['vs'][:250,:250],"rho":props2d['rho'][:250,:250],}


#Sources

Nx = seis.csts['N'][1]

sx = Nx // 2 * dh
sy = 0
sz = 0
srcid = 0
#Camion avec vibrations verticales force en z
src_type = 0

seis.src_pos_all = np.stack([[sx], [sy], [sz], [srcid], [src_type]], axis=0)

#Receivers (seulement aux points de grilles)

gx = np.arange(25, 225, 5)
gy = gx * 0
gz = gx * 0
gsid = gx*0
recid = np.arange(0, len(gx))
Blank = gx*0

seis.rec_pos_all = np.stack([gx, gy, gz, gsid, recid, Blank, Blank, Blank], axis=0)

#simulation

seis.csts['seisout'] = 1
seis.set_forward(gsid, model_dict, withgrad=False)
#Calcule les données
seis.execute()
#Lecture du fichier de données
datafd = seis.read_data()

#Visualization


clip = 0.1
extent = [min(seis.rec_pos_all[0]), max(seis.rec_pos_all[0]), NT*dt, 0]
vmax = np.max(datafd[1]) * clip
vmin = -vmax
fig, ax = plt.subplots(1, 1, figsize=[4, 6])
ax.imshow(datafd[1], aspect='auto', vmax=vmax, vmin=vmin, extent = extent,
                interpolation='bilinear', cmap=plt.get_cmap('Greys'))
ax.set_title("FD solution to elastic wave equation in 2D \n", fontsize=16, fontweight='bold')
ax.set_xlabel("Receiver position (m)")
ax.set_ylabel("time (s)")
plt.show()