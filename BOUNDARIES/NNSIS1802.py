import ModelGenerator as mg
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
## Define lithologies

#name = "mt"
#vp = mg.Property("vp", vmin = 0, vmax = 0)
#rho = mg.Property("rho", vmin = 0, vmax = 0)
#mt = mg.Lithology(name = name, properties=[vp, rho])
#Supposons un ratio vp/vs = 2

"""
Gabriel: Pycharm te dit quand il y a des erreurs de style (souligné orange).
 Regarde-les, ça apprend à respecter les conventions python. Dans ce qui suit,
 tu devrais par exemple enlever les espaces vmin = 5278 -> vmin=5278"""

name = "Sediments"
vp = mg.Property("vp", vmin = 5278, vmax = 5513,texture = 150)
vs = mg.Property("vs",vmin = 5278/2, vmax = 5513/2,texture = 150)
rho = mg.Property("rho", vmin = 2811, vmax = 2901, texture = 150)
Sediments = mg.Lithology(name = name, properties=[vp, vs, rho])

name = "Sediments_graphiteux"
vp = mg.Property("vp", vmin = 5341, vmax = 5685,texture = 150)
vs = mg.Property("vs", vmin = 5341/2, vmax = 5685/2,texture = 150)
rho = mg.Property("rho", vmin = 2773, vmax = 22849, texture = 150)
Sediments_graphiteux = mg.Lithology(name = name, properties=[vp, vs, rho])

name = "Basaltes"
vp = mg.Property("vp", vmin = 5899 , vmax = 6299,texture = 150)
vs = mg.Property("vs", vmin = 5899/2 , vmax = 6299/2,texture = 150)
rho = mg.Property("rho", vmin = 2855, vmax = 3255, texture = 150)
basaltes = mg.Lithology(name = name, properties=[vp, vs, rho])

#Definition des sequences
"""
Gabriel: Une convention importante est de ne pas dépasser 80 caractères par ligne.
Tu peux morceler une ligne de code sur plusieurs ligne pour rendre le tout pus
lisible. Exemple à la ligne suivante.
"""
sequence_beauparlant_superieur = mg.Sequence(lithologies=[basaltes,
                                                          Sediments_graphiteux])
"""
Gabriel: Une convention importante est de ne pas dépasser 80 caractères par lignes.
Tu peux morceler une ligne de code sur plusieurs ligne pour rendre le tout pus
lisible. Exemple à la ligne suivante.
Pycharm peux afficher une ligne à 80 caractaires 
https://stackoverflow.com/questions/17319422/how-do-i-set-the-maximum-line-length-in-pycharm
"""
sequence = mg.Sequence(lithologies=[Sediments,
                                    Sediments,
                                    basaltes,
                                    basaltes,
                                    basaltes,
                                    Sediments,
                                    Sediments,
                                    Sediments,
                                    basaltes,
                                    basaltes,
                                    basaltes],
                       ordered=True)

strati = mg.Stratigraphy(sequences=[sequence])

# Generer le modele

gen = mg.ModelGenerator()
gen.NX = 4120
gen.dh = 10000/4120
gen.NZ = round(5043/gen.dh)
x = np.arange(gen.NX)



#Importer les frontieres exportees de Opendtect

directory = "/data/jgendreau/NNSIS1802/BOUNDARIES/"

#Lecture des fichiers exportes de Opendtect et creation des frontieres

def lirefichier(filename):
    with open(directory+filename) as bsp:
        contenu = bsp.read()
    horizon =  [[int(i) for i in contenu.split()[3::5]],[float(i) / gen.dh for i in contenu.split()[4::5]]]
    bnd = x * 0 + 9999
    for i in range(gen.NX):
        bnd[i] = horizon[1][i]
    return bnd

debut_leve = x * 0 + 200 / gen.dh
nuvilik_pli_top = lirefichier("NUVILIK_PLI_TOP.dat")
beauparlant_superieur_pli = lirefichier("BEAUPARLANT_SUPERIEUR_PLI_TOP.dat")
beauparlant_centre_pli = lirefichier("BEAUPARLANT_CENTRE_PLI_TOP.dat")
beauparlant_inferieur_pli = lirefichier("BEAUPARLANT_INFERIEUR_PLI_TOP.dat")
nuvilik_top = lirefichier("NUVILIK_TOP.dat")
nuvilik_reflec1 = lirefichier("NUVILIK_CENTRE2.dat")
nuvilik_reflec2 = lirefichier("NUVILIK_CENTRE.dat")
beauparlant_superieur = lirefichier('BEAUPARLANT_SUPERIEUR_TOP.dat')
beauparlant_centre= lirefichier("BEAUPARLANT_CENTRE_TOP.dat")
beauparlant_inferieur = lirefichier("BEAUPARLANT_INFERIEUR_TOP.dat")

unites = [debut_leve,nuvilik_pli_top,beauparlant_superieur_pli,beauparlant_centre_pli,beauparlant_inferieur_pli
          ,nuvilik_top,nuvilik_reflec1,nuvilik_reflec2,beauparlant_superieur,beauparlant_centre,beauparlant_inferieur]

#Creation du modele

bnds = unites
bnds = [b.astype(int) for b in bnds]

texture_trends = [None,nuvilik_pli_top,beauparlant_superieur_pli,beauparlant_centre_pli,beauparlant_inferieur_pli
          ,nuvilik_top,nuvilik_reflec1,nuvilik_reflec2,beauparlant_superieur,beauparlant_centre,beauparlant_inferieur]
gen.texture_xrange = 3
gen.texture_zrange =0.25*gen.NZ

props2d, layerids, layers = gen.generate_model(strati, boundaries=bnds,texture_trends=texture_trends)
gen.plot_model(props2d, layers)
plt.show()
"""
Gabriel: tu peux sauvegarder la figure en image avec plt.savefig
"""