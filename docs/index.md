# generateACP

## 1. Présentation

Réalisation d'une ACP à partir d'un fichier .csv ou .xlsx de données.

![GitHub](https://kopadata.fr/imgs/logo-github_ok.png)   [Dépôt GitHub](https://github.com/FredGainza/generateACP)

Structure de l'analyse réalisée :

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

## 3. Détail de la fonction

*** 

### Préambule

***

le seul paramètre obligatoire de la fonction est le df des données.

Veillez à le formater correctement :

Pour les variables servant aux calculs de l'ACP (ne concerne pas les variables illustratives) :

    - mettre en index la variable d'observation (variable qualitative générallement)
    - les colonnes du df ne doivent contenir UNIQUEMENT DES VARIABLES QUANTITATIVES
    - les variables qualitatives doivent donc être transformées en variables quantitatives

Si votre jeu de données contient les variables explicatives et illustratives, vous devez créer plusieurs df (avec toujours la variable d'observation en index), soit :

    - un df avec les variables quantitaives explicatives
    - un df avec les variables quantitatives illustratives
    - un df avec les variables qualitatives explicatives

Se référer à l'[exemple](https://github.com/FredGainza/generateACP/docs/analyse_exemple.html) pour plus de détails.
<br>

*** 

### Fonction acp_global

***

##### ::: generateACP

## 4. Exemple

Exemple disponible dans le dossier *docs* du module sur [github](https://github.com/FredGainza/generateACP/docs/analyse_exemple.html)

## 5. Reference

* Ricco Rakotomalala - [Pratique des Méthodes Factorielles avec Python](https://tutoriels-data-mining.blogspot.com/2020/07/pratique-des-methodes-factorielles-avec.html)
* François Husson - [Rappels d’analyse factorielle](https://husson.github.io/img/cours_AnaDo_M2_intro.pdf)
* Nicolas Rangeon - [Réalisez une analyse exploratoire de données](https://openclassrooms.com/fr/courses/4525281-realisez-une-analyse-exploratoire-de-donnees)

## 6. About

* Created by [FredGainza](mailto:kopatiktak@gmail.com) - [GitHub](https://github.com/FredGainza) - [WebSite](https://fgainza.fr)
