import ModelGenerator as mg
import numpy as np
import matplotlib.pyplot as plt

directory = "/data/jgendreau/NNSIS1802/BOUNDARIES/"

# Ce code permet de créer un modèle de proprietes 2D de vitesse vp, vs
# et de densite rho a partir des frontieres geologiques interpetees à
# l'aide du logiciel Opendtect et des proprietes mesurees en
# laboratoire.

# Define lithologies

Q_mt = 5
Q_roche = 200

# Supposons un ratio vp/vs = 2

name = "Mort-terrain"
vp = mg.Property("vp", vmin=3700, vmax=3700, texture=200)
vs = mg.Property("vs", vmin=1600, vmax=1600, texture=250)
rho = mg.Property("rho", vmin=1900, vmax=1900, texture=250)
taup = mg.Property("taup", vmin=2/Q_mt, vmax=2/Q_mt, texture=0)
taus = mg.Property("taus", vmin=2/Q_mt, vmax=2/Q_mt, texture=0)
mort_terrain = mg.Lithology(name=name, properties=[vp, vs, rho, taup, taus])

name = "Sediments"
vp = mg.Property("vp", vmin=5278, vmax=5513, texture=250)
vs = mg.Property("vs", vmin=5278 / 2, vmax=5513 / 2, texture=250)
rho = mg.Property("rho", vmin=2811, vmax=2901, texture=250)
taup = mg.Property("taup", vmin=2/Q_roche, vmax=2/Q_roche, texture=0)
taus = mg.Property("taus", vmin=2/Q_roche, vmax=2/Q_roche, texture=0)
Sediments = mg.Lithology(name=name, properties=[vp, vs, rho, taup, taus])

name = "Sediments_graphiteux"
vp = mg.Property("vp", vmin=5341, vmax=5685, texture=250)
vs = mg.Property("vs", vmin=5341 / 2, vmax=5685 / 2, texture=250)
rho = mg.Property("rho", vmin=2773, vmax=22849, texture=250)
Sediments_graphiteux = mg.Lithology(name=name, properties=[vp, vs, rho,
                                                           taup, taus])

name = "Basaltes"
vp = mg.Property("vp", vmin=5899, vmax=6299, texture=70)
vs = mg.Property("vs", vmin=5899 / 2, vmax=6299 / 2, texture=70)
rho = mg.Property("rho", vmin=2855, vmax=3255, texture=70)
basaltes = mg.Lithology(name=name, properties=[vp, vs, rho, taup, taus])

# Basalte intercale de sediments graphiteux du membre central de la Formation
# de Beauparlant.

name = "Basaltes_sed"
vp = mg.Property("vp", vmin=5341, vmax=6299, texture=1000)
vs = mg.Property("vs", vmin=5341 / 2, vmax=6299 / 2, texture=1000)
rho = mg.Property("rho", vmin=2773, vmax=3255, texture=1000)
basaltes_sed = mg.Lithology(name=name, properties=[vp, vs, rho, taup, taus])

# Definition des sequences

sequence_beauparlant_superieur = mg.Sequence(lithologies=[basaltes,
                                                          Sediments_graphiteux]
                                             )

sequence = mg.Sequence(lithologies=[mort_terrain,
                                    Sediments,
                                    Sediments,
                                    basaltes,
                                    basaltes_sed,
                                    basaltes,
                                    Sediments,
                                    Sediments,
                                    Sediments,
                                    basaltes,
                                    basaltes_sed,
                                    basaltes],
                       ordered=True)

strati = mg.Stratigraphy(sequences=[sequence])

# Generer le modele

gen = mg.ModelGenerator()
gen.NX = 4120
gen.dh = 10000 / 4120
gen.NZ = round(5043 / gen.dh)
x = np.arange(gen.NX)


# Lecture des fichiers exportes de Opendtect et creation des frontieres

def lirefichier(filename):
    with open(directory + filename) as bsp:
        contenu = bsp.read()
    horizon = [[int(i) for i in contenu.split()[3::5]],
               [float(i) / gen.dh for i in contenu.split()[4::5]]]
    bnd = x * 0 + 9999
    for i in range(gen.NX):
        bnd[i] = horizon[1][i]
    return bnd


top = x * 0
debut_leve = x * 0 + 200 / gen.dh
nuvilik_pli_top = lirefichier("NUVILIK_PLI_TOP.dat")
beauparlant_superieur_pli = lirefichier("BEAUPARLANT_SUPERIEUR_PLI_TOP.dat")
beauparlant_centre_pli = lirefichier("BEAUPARLANT_CENTRE_PLI_TOP.dat")
beauparlant_inferieur_pli = lirefichier("BEAUPARLANT_INFERIEUR_PLI_TOP.dat")
nuvilik_top = lirefichier("NUVILIK_TOP.dat")
nuvilik_reflec1 = lirefichier("NUVILIK_CENTRE2.dat")
nuvilik_reflec2 = lirefichier("NUVILIK_CENTRE.dat")
beauparlant_superieur = lirefichier('BEAUPARLANT_SUPERIEUR_TOP.dat')
beauparlant_centre = lirefichier("BEAUPARLANT_CENTRE_TOP.dat")
beauparlant_inferieur = lirefichier("BEAUPARLANT_INFERIEUR_TOP.dat")

# Frontieres

unites = [top,
          debut_leve,
          nuvilik_pli_top,
          beauparlant_superieur_pli,
          beauparlant_centre_pli,
          beauparlant_inferieur_pli,
          nuvilik_top,
          nuvilik_reflec1,
          nuvilik_reflec2,
          beauparlant_superieur,
          beauparlant_centre,
          beauparlant_inferieur
          ]

# Creation du modele

bnds = unites
bnds = [b.astype(int) for b in bnds]

texture_trends = [None,
                  top,
                  nuvilik_pli_top,
                  beauparlant_superieur_pli,
                  beauparlant_centre_pli,
                  beauparlant_inferieur_pli,
                  nuvilik_top,
                  nuvilik_reflec1,
                  nuvilik_reflec2,
                  beauparlant_superieur,
                  beauparlant_centre,
                  beauparlant_inferieur
                  ]
# Plus c'est petit, plus c'est continu
gen.texture_xrange = 3

gen.texture_zrange = 800

props2d, layerids, layers = gen.generate_model(strati,
                                               boundaries=bnds,
                                               texture_trends=texture_trends
                                               )

np.save("proprietes2d_vp", props2d["vp"])
np.save("proprietes2d_vs", props2d["vs"])
np.save("proprietes2d_rho", props2d["rho"])
np.save("proprietes2d_taup", props2d["taup"])
np.save("proprietes2d_taus", props2d["taus"])

gen.plot_model(props2d, layers)
plt.savefig("props2dfig")
plt.show()
