from SeisCL import SeisCL
seis = SeisCL()
import numpy as np
import matplotlib.pyplot as plt
props2d = np.load("proprietes2d.npy",allow_pickle=True)[()]
#Domaine de la simulation
seis.csts['ND'] = 2                                 # Number of dimension
seis.csts['N'] = np.array([np.size(props2d['vp'],0), np.size(props2d["vp"],
                                                             1)])
seis.csts['dh'] = dh = 2.5                            # Grid spatial spacing
seis.csts['dt'] = dt = 0.0002                     # Time step size
seis.csts['NT'] = NT = 25000                         # Number of time steps

#Parametres du modele.



model_dict = {"vp": props2d['vp'][:np.size(props2d['vp'], 0), :np.size(
    props2d['vp'], 1)], "vs": props2d['vs'][:np.size(props2d['vp'], 0),
                              :np.size(props2d['vp'], 1)],
              "rho": props2d['rho'][:np.size(props2d['vp'], 0), :np.size(
                  props2d['vp'], 1)], }


#Sources

Nx = seis.csts['N'][1]

sx = Nx // 2 * dh
sy = 0
sz = 0
srcid = 0
#Camion avec vibrations verticales force en z
src_type = 2

seis.src_pos_all = np.stack([[sx], [sy], [sz], [srcid], [src_type]], axis=0)
seis.set_forward([0], props2d, withgrad=False)
seis.src_all[:, 0] = seis.ricker_wavelet(f0=25)
#Receivers (seulement aux points de grilles)

gx = np.arange(0, 10300, 2.5)
gy = gx * 0
gz = gx * 0
gsid = gx*0
recid = np.arange(0, len(gx))
Blank = gx*0

seis.rec_pos_all = np.stack([gx, gy, gz, gsid, recid, Blank, Blank, Blank],
                            axis=0)

#simulation

seis.csts['seisout'] = 1
seis.set_forward(gsid, model_dict, withgrad=False)
#Calcule les données
seis.execute()
#Lecture du fichier de données
datafd = seis.read_data()

#Visualization

fig = plt.figure()
clip = 0.001
extent = [min(seis.rec_pos_all[0]), max(seis.rec_pos_all[0]), NT*dt, 0]
vmax = np.max(datafd[1]) * clip
vmin = -vmax
fig, ax = plt.subplots(1, 1, figsize=[6, 6])
ax.imshow(datafd[1], aspect='auto', vmax=vmax, vmin=vmin, extent = extent,
                interpolation='bilinear', cmap=plt.get_cmap('Greys'))
ax.set_title("FD solution to elastic wave equation in 2D \n",
             fontsize=16,
             fontweight='bold')
ax.set_xlabel("Receiver position (m)")
ax.set_ylabel("time (s)")
plt.savefig('simulation.png')
plt.show()
