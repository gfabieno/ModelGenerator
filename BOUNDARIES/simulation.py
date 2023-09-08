# Ce script permet d'effectuer une simulation de tir sismique avec un
# vibroseis sur le modele physique issu de l'interpretation de la ligne
# sismique NNSIS-18-02.

import numpy as np
import matplotlib.pyplot as plt
from SeisCL import SeisCL
seis = SeisCL()


# Charger les matrices de proprietes sauvegardees a partir du modele.

model_dict = {"vp": np.load("proprietes2d_vp.npy"),
              "vs": np.load("proprietes2d_vs.npy"),
              "rho": np.load("proprietes2d_rho.npy"),
              "taup": np.load("proprietes2d_taup.npy"),
              "taus": np.load("proprietes2d_taus.npy")}

# Domaine de la simulation
# Nombre de dimensions
seis.csts['ND'] = 2
# Grid spatial spacing
seis.csts['N'] = np.array([np.size(model_dict["vp"], 0),
                           np.size(model_dict["vp"], 1)])
seis.csts['dh'] = dh = 2.5
# Time step size
seis.csts['dt'] = dt = 0.0001870
# Number of time steps
seis.csts['NT'] = NT = 25000
seis.f0 = 26
# Nombre de mécanismes de Maxwell. Pour l'attenuation.
seis.L = 1
seis.FL = np.array([seis.f0])
seis.nab = 40
# Surface libre
seis.csts['freesurf'] = 1
if seis.freesurf == 0:
    z0 = seis.nab
else:
    z0=0

# Ajouter une couche absorbante pour quand il n'y a pas de surface libre.

if seis.freesurf == 0:
    for i in model_dict:
        model_dict[i] = np.pad(model_dict[i], ((seis.nab,seis.nab),(0,0)))

# Sources

# Position de la ou des sources
sx = 8000.0
sy = 0
sz = z0
srcid = 0
# Camion avec vibrations verticales force en z
src_type = 2

seis.src_pos_all = np.stack([[sx], [sy], [sz], [srcid], [src_type]], axis=0)
seis.set_forward([0], model_dict, withgrad=False)
# Ondelette de Ricker
seis.src_all[:, 0] = seis.ricker_wavelet(f0=seis.f0)

# Receveurs: peuvent seulement etre positionnes aux points de grilles

# Geophones espaces de 2.5m de 0 à 10300m.

gx = np.arange(0, 10300, 2.5)
gy = gx * 0
gz = gx * 0 + z0
gsid = gx*0
recid = np.arange(0, len(gx))
Blank = gx*0

seis.rec_pos_all = np.stack([gx, gy, gz, gsid, recid, Blank, Blank, Blank],
                            axis=0)

#Visualiser le domaine

seis.DrawDomain2D(model_dict['vp'], showabs=True, showsrcrec=True)

# Simulation

seis.csts['seisout'] = 1
seis.set_forward(gsid, model_dict, withgrad=False)
# Calculer les données
seis.execute()
# Lecture du fichier de données
datafd = seis.read_data()
np.save("simulation.py",datafd)

# Affichage des resultats

figure = plt.figure()
clip = 0.001
extent = [min(seis.rec_pos_all[0]), max(seis.rec_pos_all[0]), NT*dt, 0]
vmax = np.max(datafd[1]) * clip
vmin = -vmax
fig, ax = plt.subplots(1, 1, figsize=[6, 6])
ax.imshow(datafd[1], aspect='auto', vmax=vmax, vmin=vmin, extent = extent,
                interpolation='bilinear', cmap=plt.get_cmap('Greys'))
ax.set_title("Simulation de tir \n",
             fontsize=16,
             fontweight='bold')
ax.set_xlabel("Position des receveurs (m)")
ax.set_ylabel("Temps (s)")
plt.savefig('simulation.png')
plt.show()
