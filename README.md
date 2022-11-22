# generateACP

![Badge KoPaTiK](https://img.shields.io/badge/KoPaTiK-Agency-blue "Badge KoPaTiK") [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/FredGainza/generateACP.git/HEAD)

*** 

## <p style="color:#00FFFF;">Fonction générant les calculs et les graphiques d'une ACP.</p>

***

### Paramètre principal à renseigner => dataframe avec :

- en index : la variable d'observation
- en colonnes : les variables explicatives quantitatives
- en ligne : les individus / observations


### Structure de l'analyse réalisée :

#### Information sur les données
        * données initiales
        * données centrées réduites
#### Recherche du nombre de facteurs à retenir
        * graphique eboulis des valeurs propres
        * calcul de la proportion de variance expliquée
        * test des bâtons brisés
#### Représentation des individus
        * coordonnées factorielles des individus
        * qualité de la représentation des individus (cos² de chaque individu par axe)
        * contribution des individus aux axes
#### Représentation des variables
        * les vecteurs propres
        * corrélations par facteur
        * qualité de la représentation des variables (cos² de chaque variable par axe)
        * contribution des variables aux axes
#### Traitement des variables supplémentaires
        * variables illustratives quantitatives
        * variables illustratives qualitatives
#### Représentation graphique (pour chaque plan factoriel)
        * Projection des individus
        * Cercle des corrélations


## 2. Installation

### Exemple avec Anaconda

```bash
    ### Créer un dossier pour le projet
    $ cd /vers/dossier/testACP
	
    ### Créer un nouvel environnement
    $ conda create -n envAcp python=3.9
    ### Activer le nouvel environnement
    $ conda activate envAcp
	
    ### Installer le module generateAcp
    $ pip install git+https://github.com/FredGainza/generateACP.git
```

Pour utiliser le module dans un notebook, il faut importer la fonction **acp_global()** :

```python
    from generateACP import acp_global
```

Les modules suivants seront automatiquement installés :

- pandas
- numpy
- scikit-learn
- matplotlib
- jupyter
- adjustText
- pdfservices-sdk
- openpyxl

## Documentation

RDV [ICI](https://FredGainza.github.io/generateACP/)

## Exemple

* Notebook disponible au format [html](https://kopadata.fr/data/analyse_acp_exemple.html)
* Notebook exécuté *analyse_acp_exemple.ipynb* présent dans le dossier [/docs/](https://github.com/FredGainza/generateACP/blob/7d11265565799562513f6fc94815bef7ce1f9448/docs/analyse_acp_exemple.ipynb)
* Notebook dispo sur [binder](https://mybinder.org/v2/gh/FredGainza/generateACP.git/HEAD) (exécutez le notebook *binder_exemple_module_acp.ipynb* présent dans le dossier /docs/)
