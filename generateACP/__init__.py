import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from adjustText import adjust_text
from IPython.core.display import display, HTML
from IPython.display import display
from PIL import Image
from math import floor
import os


def acp_global(
    df,
    axis_ranks=[(0, 1)],
    group=None,
    group_special=None,
    varSupp=None,
    varSuppQual=None,
    labels=True,
    labels_ind=None,
    legend_label=None,
    version_name=None,
    graph_only=None,
    graph_only_acp=None,
    data_only=None,
    color_titles=None,
    palette_color=None
):

    """Générer les calculs et les graphs d'une ACP

        Parameters:

            df (DataFrame): df initial structuré pour une ACP:

                - en ligne: des individus / observations
                - en colonne: les variables explicatives quantitatives 
                - en index: la variable des observations

            axis_ranks (list): les dimensions du / des plan(s) factoriels à étudier

                - vérifier que les degrés factortiels ne soient pas supérieurs aux nombres de variables
                - possible de lancer la fonction acp_global() pour connaitre le résultat des tests de recherche du nombre
                de facteurs à retenir (eboulis des valeurs propres avec proportion cumulée de la variance expliquée, test
                des bâtons brisés) en laissant la valeur des paramètres par défaut


                    Exemple: en supposant que:

                    - df soit le nom du DataFrame des données initiales
                    - resultat_acp soit le nom donné au résultat de la fonction :

                            resultat_acp = acp_globale(df)

            group (str): clustering

                - valeur acceptée : nom de la colonne de groupe
                - il faut donc que le df initial comporte cette colonne

            group_special (list): liste d'individus constituant un groupe particulier (individus particuliers de la variable en index)

                - valeur acceptée : liste d'individus

            varSupp (DataFrame): variables illustratives quantitatives

                - valeur acceptée : df avec

                    - même nb de lignes que le df initial (les mêmes individus)
                    - en colonnes les variables illustratives quantitatives

            varSuppQual (DataFrame): variables illustratives qualitatives

                - valeur acceptée : df avec

                    - même nb de lignes que le df initial (les mêmes individus)
                    - en colonnes les variables illustratives qualitatives

            labels (boolean): affichage des valeurs de la variable des observations (variable en index)

                - valeur acceptée : True or None
                - NB : si labels = True, alors labels_ind = False  (ne choisir que 1 de ces 2 possibilités)

            labels_ind (boolean): affichage de l'indice des valeurs de la variable des observations (variable en index)

                - valeur acceptée : True or None
                - NB : si labels_ind = True, alors labels = False  (ne choisir que 1 de ces 2 possibilités)
                - intérêt de cette option: facilite la lecture du graphique "Projection des individus", notamment si le
                nombre d'oservations (nombre de lignes) est important

            legend_label (boolean): affichage des correspondances indice et valeur de la variable d'observation (valable ssi labels_ind=True)

                - valeur acceptée : True or None
                - si True, une légende de correspondance entre index affiché sur le graphique et valeur réelle de la 
                variable d'observationb (en index) sera affichée sous le graphique

            version_name (str): prefixer les graphiques par un nom particulier

                - valeur acceptée : string qui prefixera le nom des graphiques 

                        ex : si version_name = "test1" alors le graphique "Eboulis des valeurs propres se nommera 
                        "test1_recherche_nb_facteurs_optimal.png"

                - intérêt: enregistrer tous les graphiques de différentes versions d'une même ACP
                
            graph_only (boolean): afficher seulement les graphiques (cercle de corrélation et projection des individus)

                - valeur acceptée : True or None

            data_only (boolean): ne rien renvoyer lors de l'appel de la fonction

                - valeur acceptée : True or None
                - pour print un élément en particulier:
                    * donner un nom à la fonction lors de l'appel
                        ex : resultat_acp = acp_perso(df, data_only=True)
                    * faire print(*resultat_acp)

            color_titles (list): liste de 3 couleurs au format hexadécimale (ex "#FFFFFF") pour les titres h2, h3, h4

                - couleurs des titres par défaut : ["#b08c20", "#3aa237", "#29858a"]

            palette_color (list): liste de couleurs au format hexadécimale (ex "#FFFFFF")pour les graphiques de l'ACP

                - couleurs par défaut : ["#b30f1c", "#0e6667", "#3353e1", "#4ed147", "#d17e1a", "#2bb6b6",
                "#514a39", "#4ed147", "#acb073", "#8b3afd", "#dec11b", "#58f3e6", "#728e9d"]

        Returns:

            val_centre_reduit (DataFrame): df centré-réduit
            valeurs_propres (list): valeurs propres
            pct_inertie_par_facteur           (list): pourcentage d'inertie par facteur
            nb_groupes_opt                    (str): path du graph "Eboulis des valeurs propres"
            test_batons_brises                (DataFrame): df valeurs propres / valeurs de seuil par facteur
            coord_fact_ind                    (DataFrame): coordonnées factorielles des individus
            cos2_ind                          (DataFrame): cos² des individus pour chaque axe factoriel
            contribution_ind                  (DataFrame): df globale: contribution des individus aux axes
            ctr_ind_sorted_facteur_i          (DataFrame): df par facteur: les plus gros contributeurs à l'axe i

                - pour tous les facteurs i

            vecteurs_propres                  (DataFrame): vecteurs propres des variables par facteur
            matrice_cor                       (array): matrice des corrélations des variables par facteur
            cor_par_facteur                   (DataFrame): df de la matrice des corrélations                 
            cos2_var                          (DataFrame): cos² des variables par axe factoriel
            contribution_var                  (DataFrame): contribution des variables aux axes              
            cor_par_facteur_VarQttiveIllus    (DataFrame): corrélations des variables quantitatives illustratives par axe
            resultats_tests                   (list): résultats des différents tests interes (1 if ok else 0)
            centroides_a_b                    (array): si clusters ou variable qualitative illustrative: coordonnées des centroïdes

                - pour chaque plan factoriel [a,b]

            graph_proj_ind_a_b                (str): path du graph de projection des individus
                
                - pour chaque plan factoriel [a,b]

            cercle_cor_a_b                    (str): path du graph du cercle des corrélations

                - pour chaque plan factoriel [a,b]

            graph_combo_a_b                   (str): path du graph composé (cf. les 2 graphs précédents)

                - pour chaque plan factoriel [a,b]

            select_obs                        (dict): dictionnaire :

                * key:
                    - si labels=True        alors valeurs de la variable d'observation
                    - si labels_ind = True  alors index des valeurs de la variable d'observation
                        
                * value:
                    - tuple composé de 2 df:
                        - df données initiales de l'observation
                        - df données calculées pour cette observation
                            - ses coordonnées factorielles
                            - sa qualité de représentation sur les différents axes (cf. COS²)
                            - sa contribution aux différents axes

    """

    def C(k, n):
        """Nombre de combinaisons sans répétition de k objets pris parmi n"""
        if k > n // 2:
            k = n - k
        x = 1
        y = 1
        i = n - k + 1
        while i <= n:
            x = (x * i) // y
            y += 1
            i += 1
        return x


    def choix_coord(a, b):
        """Calcul des valeurs absolues des coordonnées pour définir xlim et ylim du graph de projection des individus"""
        if a < 0:
            if (-a) > b:
                lim = (a, -a)
            else:
                lim = (-b, b)
        else:
            lim = (a, b)
        return lim


    def multi_table(table_list, marg_r=None):
        """Afficher des df côte à côte"""
        return HTML(
            '<table><tr style="text-align:right;">'
            + "".join(
                [
                    "<td "
                    + (
                        ' style="text-align:right;padding-right:'
                        + marg_r
                        + 'rem!important;"'
                        if marg_r is not None
                        else ""
                    )
                    + ">"
                    + table._repr_html_()
                    + "</td>"
                    for table in table_list
                ]
            )
            + "</tr></table>"
        )


    def alert_verif(fail, success, valid):
        """Afficher alertes error ou success des tests lors de l'ACP"""
        if valid == False:
            display(
                HTML(
                    '<div class="alert alert-block alert-danger" style="width:fit-content; margin:8px auto 8px 0; border-radius:5px; padding:7px 50px 7px 15px;">'
                    + fail
                    + "</div>"
                )
            )
        else:
            display(
                HTML(
                    '<div class="alert alert-block alert-success" style="width:fit-content; margin:8px auto 8px 0; border-radius:5px; padding:7px 50px 7px 15px;">'
                    + success
                    + "</div>"
                )
            )


    def dic_modalite(df, group):
        """Si modalités de colonne groupe ne sont pas de type integer => ajout d'une colonne avec correspondance modalité <=> integer"""
        modalites = df[group].unique().tolist()
        modalites = sorted(modalites)
        dic_mod = {}
        for i in range(len(modalites)):
            dic_mod[modalites[i]] = str(i + 1)
        df["i_" + group] = df[group].apply(lambda x: dic_mod[x])
        df = df.drop(columns=[group])
        df = df.rename(columns={"i_" + group: group})
        return [df, group, dic_mod]


    # test cohérence des paramètres
    msg_array=[]
    if labels is None and labels_ind is None:
        labels = True
        msg_array.append("ATTENTION: les paramètres 'labels' et 'labels_int' avaient pour valeur None.<br>"+
        "Afin de pouvoir exécuter la fonction, la valeur de 'labels' a été passée à True")
    if labels is not None and labels_ind is not None:
        labels_ind = None
        msg_array.append("ATTENTION: les paramètres 'labels' et 'labels_int' avaient pour valeur True.<br>"+
        "Afin de pouvoir exécuter la fonction, la valeur de 'labels_ind' a été passée à None")
    if labels_ind is None and legend_label is not None:
        legend_label = None
        msg_array.append("ATTENTION: incompatibilité entre 'labels_ind' = None et legend_label != None.<br>"+
        "Afin de pouvoir exécuter la fonction, la valeur de 'legend_label' a été passée à None")
        

    # Création du dossier de sauvegarde des graphs
    os.makedirs("apc_graphs", exist_ok=True)

    # Gestion des couleurs des fonts
    cc_titles = color_titles if color_titles is not None else ["#b08c20", "#3aa237", "#29858a"]
    h2s = "color:"+cc_titles[0]+"!important;padding-left:.5rem!important;line-height:150%!important;"
    h3s = "color:"+cc_titles[1]+"!important;padding-left:1rem !important;line-height:150%!important;font-size:120%!important;"
    h4s = "color:"+cc_titles[2]+"!important;padding-left:2rem !important;font-size:110% !important;"

    if palette_color == None:
        palette_perso = [
            "#b30f1c",
            "#0e6667",
            "#3353e1",
            "#4ed147",
            "#d17e1a",
            "#2bb6b6",
            "#514a39",
            "#4ed147",
            "#acb073",
            "#8b3afd",
            "#dec11b",
            "#58f3e6",
            "#728e9d",
        ]
    else:
        palette_perso = palette_color

    # Afficher 4 décimales pour les floats
    pd.options.display.float_format = "{:.4f}".format

    # Si data_only, alors on n'affiche rien
    if data_only is True:
        graph_only = True

    ############################
    # Préparation des données
    ############################

    # initialisation de variables
    all_values = {}
    df_liste = []
    df_liste_data = []
    resultats_test = []

    # creation d'un df_global avec toutes les variables (actives + illus quant + illus qual)
    df_liste.append(df)
    df_liste_data.append(df)
    if varSupp is not None:
        df_liste.append(varSupp)
        df_liste_data.append(varSupp)
    if varSuppQual is not None:
        df_liste.append(varSuppQual)
    if group is not None:
        df_c = df.copy()
        df = df_c.drop(columns=(group))
    df_global_data = pd.concat(df_liste_data, axis=1)

    if group_special is not None:
        ind_spe = df[df.index.isin(group_special)]

    # df pour l'ACP
    X = df
    # Nombre d'individus
    n = int(X.shape[0])
    # Nombres de Variables
    p = int(X.shape[1])
    # instanciation ACP normée
    sc = StandardScaler()
    # transformation
    Z = sc.fit_transform(X)
    # moyenne
    np.mean(Z, axis=0)
    # ecart-type
    np.std(Z, axis=0, ddof=0)

    # df centré-réduit
    x1 = pd.DataFrame({"id": X.index})
    for j in range(p):
        x1[X.columns[j]] = Z[:, j]

    # df enregistré dans notre dictionnaire qui sera return
    all_values["val_centre_reduit"] = x1

    # Nombre de dimensions à étudier
    n_comp = p
    dd = []
    for i in range(p):
        dd.append(i)
    nb_dim = len(axis_ranks)
    max_combi = C(2, n_comp)

    # message d'erreur possible
    if nb_dim > max_combi:
        print(
            f"Vous avez indiqué un nombre trop important de plan factoriels. Il y en a {max_combi} de possibles au maximum (votre requête a été modifiée en ce sens.)"
        )
    for d1, d2 in axis_ranks:
        if d1 == d2 or d1 < 0 or d2 < 0 or d1 > n_comp or d2 > n_comp or n_comp > p:
            print(
                'Il y a un problème avec les dimensions que vous avez saisis, merci de vérifier les variables "axis_ranks" et "n_comp"'
            )
            break
        else:
            if d1 > d2:
                d1, d2 = d2, d1

    ############################################################################
    #                      Analyse en composantes principale                   #
    ############################################################################

    ###########################################################################
    #       1. Informations sur les données
    ###########################################################################

    if graph_only is None:
        display(HTML("<h2 style=" + h2s + ">Analyse en composantes principales</h2>"))

        # si problème de cohérence dans les paramètres => message d'alerte
        if len(msg_array) != 0:
            msg_coherence_fail = "\n\n".join(msg_array)
            alert_verif(
                fail=msg_coherence_fail,
                success="",
                valid=False,
            )
            print()

        # affichage d'infos générales sur les données initiales
        display(HTML("<h3 style=" + h3s + ">1. Informations sur les données</h3>"))
        print()
        display(HTML("<h4 style=" + h4s + ">1.1 Données initiales</h4>"))
        print("Nombre d'individus : " + str(n))
        print("Nombre de variables : " + str(p))
        display(df.describe())
        print()


    ######### Instanciation et lancement des calculs

    # instanciation des classes
    acp = PCA(svd_solver="full")

    #########################################
    # calculs des coordonnées factorielles
    #########################################
    coord = acp.fit_transform(Z)

    #########################################
    # calculs des valeurs propres
    #########################################
    # valeur corrigée
    eigval = (n - 1) / n * acp.explained_variance_

    # valeurs propres enregistrées pour le return
    all_values["valeurs_propres"] = eigval

    # pourcentage d'inertie par facteur pour le return
    all_values["pct_inertie_par_facteur"] = 100 * acp.explained_variance_ratio_
    # infos sur les données centrées-réduites
    x1 = pd.DataFrame(
        {
            "Variables": df.columns,
            "Moyennes": np.mean(Z, axis=0),
            "Ecarts_types": np.std(Z, axis=0, ddof=0),
            "Var_Exp": eigval,
            "Prop_Var_Exp": acp.explained_variance_ratio_,
        }
    )
    x1.iloc[:, [1, 2, 3]] = x1.iloc[:, [1, 2, 3]].apply(lambda x: round(x, 4))

    if graph_only is None:
        display(HTML("<h4 style=" + h4s + ">1.2 Données centrées réduites</h4>"))
        # nombre de composantes calculées
        print("nombre de composantes calculées : " + str(acp.n_components_))
        # affichage d'infos sur les données centrées-réduites
        display(x1)

    # vérification que valeurs propres = p
    valid = True
    vp_test = x1.Var_Exp.sum()
    if vp_test < p - 0.05 or vp_test > p + 0.05:
        valid = False
        resultats_test.append(0)
    else:
        resultats_test.append(1)

    if graph_only is None:
        alert_verif(
            fail="La vérifiction de la somme des valeurs propres égale au nombre de variables actives a échoué",
            success="La vérifiction de la somme des valeurs propres égale au nombre de variables s'est correctement déroulée",
            valid=valid,
        )
        print()

        ###########################################################################
        #       2. Recherche du nombre de facteurs à retenir
        ###########################################################################

        display(
            HTML(
                "<h3 style="
                + h3s
                + ">2. Recherche du nombre de facteurs à retenir</h3>"
            )
        )

    ###################################################
    # graphique d'Eboulis des valeurs propres
    ###################################################

    # initialisation de la figure
    if graph_only_acp is not True:
        nb_gr_op = plt.figure(figsize=(8, 6))
        plt.plot(
            np.arange(1, p + 1),
            np.cumsum(100 * acp.explained_variance_ratio_),
            "o-",
            c="teal",
            mfc="tomato",
            mec="tomato",
        )
        plt.bar(
            np.arange(1, p + 1),
            100 * acp.explained_variance_ratio_,
            color=(0.2, 0.4, 0.6, 0.6),
        )
        plt.title("Eboulis des valeurs propres")
        plt.ylabel("Pourcentage d'inertie")
        plt.xlabel("Rang de l'axe d'inertie")
        y_pos = [0, 20, 40, 60, 80, 100]
        y_label = []
        for i in range(len(y_pos)):
            y_label.append(str(y_pos[i]) + "%")
        plt.yticks(y_pos, y_label)
        plt.xticks(np.arange(1, len(eigval) + 1, 1))
        plt.grid(linestyle="--", alpha=0.3)

        # affichage du pourcentage cumulé de variance expliquée
        part_inertie = acp.explained_variance_ratio_.cumsum()
        texte = "Variance Cumulée  :  "
        s = 0
        for i in range(len(part_inertie)):
            s = part_inertie[i] * 100
            if i != len(part_inertie) - 1:
                texte += str(round(s, 2)) + "%,  "
                if i == 6 or i == 14 or i == 20 or (i > 16 and (i % 8) - 5 == 0):
                    texte += "\n"
            else:
                texte += str(round(s, 2)) + "%"
        plt.text(0.5, -20, texte, color="navy", size=10)

        plt.tight_layout()
        nb_groupes_opt = (
            "apc_graphs/"
            + (version_name + "_" if version_name is not None else "")
            + "recherche_nb_facteurs_optimal"
        )
        nb_gr_op.savefig(nb_groupes_opt)
        all_values["nb_groupes_opt"] = nb_groupes_opt

        if graph_only is None:
            plt.tight_layout()
            plt.show(nb_gr_op)
            plt.close()

    # Détermination du nombre de facteurs à retenir
    ################################################

    # proportion de variance expliquée
    part_var_exp = acp.explained_variance_ratio_

    # seuils pour test des bâtons brisés
    bs = 1 / np.arange(p, 0, -1)
    bs = np.cumsum(bs)
    bs = bs[::-1]

    # test des bâtons brisés
    bb = pd.DataFrame({"Val.Propre": eigval, "Seuils": bs})
    for z in range(len(bb)):
        if bb.loc[z, "Val.Propre"] <= bb.loc[z, "Seuils"]:
            break

    all_values["test_batons_brises"] = bb

    # coordonnées factorielles des individus
    cf = pd.DataFrame(coord, index=X.index, columns=np.arange(1, p + 1))

    if graph_only is None:
        print("--------------------------------------------\n"+
        "|     Proportion de variance expliquée     |\n"+
        "--------------------------------------------")

        df_var_e = pd.DataFrame(part_var_exp.cumsum())
        df_var_e.insert(0, "Facteurs", "")
        for j in range(len(df_var_e)):
            if j == 0:
                df_var_e.loc[j, "Facteurs"] = "le 1er facteur"
            else:
                df_var_e.loc[j, "Facteurs"] = "les " + str(j + 1) + " 1ers facteurs"
        df_var_e.columns = ["Facteurs", "part_var"]

        nb_f_var_e = 0
        for jj, pp in enumerate(list(df_var_e["part_var"])):
            if pp >= 0.8:
                nb_f_var_e = jj + 1
                break

        df_var_e["Part variance expliquée"] = round(df_var_e["part_var"]*100, 4)
        df_var_e["Part variance expliquée"] = df_var_e["Part variance expliquée"].astype(str)
        df_var_e["Part variance expliquée"] = df_var_e["Part variance expliquée"].apply(lambda x: x+"%")
        df_var_e = df_var_e.drop(colums=['part_var'])
        display(df_var_e)
        print(
            "Si on recherche à expliquer au minimum 80% de la variance, il faut retenir "
            + (
                "uniquement le 1er facteur."
                if nb_f_var_e == 1
                else "les " + str(nb_f_var_e) + " 1ers facteurs."
            )
        )
        print()

        print("----------------------------------")
        print("|     Test des bâtons brisés     |")
        print("----------------------------------")
        display(bb)
        print(
            "Selon le \"test des bâtons brisés\", "
            + (
                "seul le premier facteur est valide"
                if z == 0
                else "les " + str(z + 1) + " premiers facteurs sont valides"
            )
            + " (car pour le facteur "
            + str(z + 2)
            + ", la valeur du seuil est supérieure à la valeur propre)."
        )
        print("--------------------------------------------------")

        eigval_ok = [x for x in eigval if x > 1]
        len_v_ok = len(eigval_ok)
        print(
            "Autre interprétation possible : \n"+
            "plus la valeur propre initiale est élevée, plus le facteur explique une portion significative de la variance totale."+
            "Par convention, tout facteur avec une valeur propre initiale supérieure à 1 est considéré comme facteur significatif.\n"
            + "Ainsi, selon cette interprétation, "
            + (
                "seul le premier facteur est valide."
                if len_v_ok == 1
                else "les " + str(len_v_ok) + " premiers facteurs sont valides."
            )
        )
        print()

        if nb_f_var_e == 1 or len_v_ok == 1 or i == 1:
            print(
                "Remarque : <br>Selon certaines procédures, il faudrait limiter le nombre de facteurs à retenir à 1.\n"
                + "Mais pour pouvoir effectuer cette analyse, nous étudierons au minimumm 2 facteurs."
            )
            print()

        ###########################################################################
        #       3. Représentation des individus
        ###########################################################################

        display(HTML("<h3 style=" + h3s + ">3. Représentation des individus</h3>"))

        print()
        print(
            "################################################################################\n"+
            "####               Coordonnées factorielles des individus                   ####\n"+
            "################################################################################"
        )

        print("On affiche les coordonnées factorielles de 3 individus (random)")
        cff = cf.copy()
        old_cols = list(cff.columns)
        new_cols = ["F"+str(x) for x in old_cols]
        cff.columns = new_cols
        display(cff.sample(3))

    # coord. factorielles des individus enregistrées pour le return
    all_values["coord_fact_ind"] = cff

    if graph_only is None:
        print()
        print(
            "################################################################################\n"+
            "####             Qualité de la représentation des individus                 ####\n"+
            "################################################################################"
        )

    di = np.sum(Z**2, axis=1)
    x = pd.DataFrame({"ID": X.index, "d_i": di})

    if graph_only is None:
        print(
            "La qualité de représentation des individus sur les axes du plan factoriel.\n"+
            "Il s'agit de calculer le COS² de chaque individu pour chaque axe.\n"+
            "On affiche un random sur 3 individus de la mesure des COS²"
        )

    # calcul des cos² des individus
    cos2 = coord**2
    cos2_ind = pd.DataFrame({"id": X.index})
    for j in range(p):
        cos2_ind["COS²_F" + str(j + 1)] = cos2[:, j] / di
    for j in range(p):
        cos2_ind["COS²_F" + str(j + 1)] = cos2_ind["COS²_F" + str(j + 1)].apply(
            lambda x: round(x, 4)
        )

    if graph_only is None:
        display(cos2_ind.sample(3))

    # cos² des individus enregistrés pour le return
    all_values["cos2_ind"] = cos2_ind

    # vérifions la théorie - somme en ligne des cos2 = 1
    x = cos2_ind.drop(columns="id")
    somme = x.sum(axis=1)
    valid = True
    for i in range(len(somme)):
        if somme[i] < 0.95 or somme[i] > 1.05:
            valid = False
            break

    if valid is not True:
        resultats_test.append(0)
    else:
        resultats_test.append(1)

    if graph_only is None:
        alert_verif(
            fail="La vérifiction de la somme des COS² égale à 1 pour tous les individus a echoué",
            success="La vérifiction de la somme des COS² égale à 1 pour tous les individus s'est correctement déroulée",
            valid=valid,
        )
        print()
        # Contributions des individus aux axes
        print(
            "################################################################################\n"+
            "####                  Contributions des individus aux axes                  ####\n"+
            "################################################################################"
        )
        print(
            "Elles permettent de déterminer les individus qui pèsent le plus dans la définition de chaque facteur.\n"+
            "On regarde quels sont les individus qui sont les plus contributifs et ce, pour les différents axes.\n"+
            "Pour chaque axe, on affiche les 10 individus qui contribuent le plus."
        )

    # calcul des contributions des individus aux axes
    ctr = coord**2
    ctr_ind = pd.DataFrame({"ind": X.index})
    for j in range(p):
        ctr_ind["CTR_F" + str(j + 1)] = ctr[:, j] / (n * eigval[j])
        ctr_ind["CTR_F" + str(j + 1)] = round(ctr_ind["CTR_F" + str(j + 1)], 4)

    # contribution des individus enregistrées pour le return
    all_values["contribution_ind"] = ctr_ind

    # préparation de l'affichage des plus gros contributeurs par axe
    ctr_list = []
    ctr_list_aff = []
    rang_liste = (np.arange(1, n + 1, 1)).tolist()
    for j in range(p):
        ctr_list.append(
            ctr_ind[["ind", "CTR_F" + str(j + 1)]].sort_values(
                by="CTR_F" + str(j + 1), ascending=False
            )
        )
        ctr_list_aff.append(ctr_list[j].copy())
        ctr_list[j]["rang"] = rang_liste
        ctr_list_aff[j]["CTR_F" + str(j + 1)] = (
            ctr_list_aff[j]["CTR_F" + str(j + 1)] * 100
        ).round(5).astype(str) + "%"
        all_values["ctr_ind_sorted_facteur_" + str(j + 1)] = ctr_list_aff[j]
        ctr_list_aff[j] = ctr_list_aff[j].head(10)

    if graph_only is None:
        len_multi = len(ctr_list_aff)
        if len_multi > 4:
            reste = len_multi % 4
            nb_multi = floor(len_multi / 4) + 1 if reste != 0 else len_multi / 4
            for i in range(nb_multi):
                ar = ctr_list_aff[i*4:(i+1)*4]
                display(multi_table(ar))
        else:
            display(multi_table(ctr_list_aff))

    # vérifions la théorie - somme en colonnes = 1
    x = ctr_ind.drop(columns="ind")
    somme = x.sum(axis=0)
    valid = True
    for i in range(len(somme)):
        if somme[i] < 0.95 or somme[i] > 1.05:
            valid = False
            break

    if valid is not True:
        resultats_test.append(0)
    else:
        resultats_test.append(1)

    if graph_only is None:
        alert_verif(
            fail="La vérifiction de la somme des contributeurs égale à 1 pour tous les axes factoriels a échoué",
            success="La vérifiction de la somme des contributeurs égale à 1 pour tous les axes factoriels s'est correctement déroulée",
            valid=valid,
        )
        print()

        ###########################################################################
        #       4. Représentation des variables
        ###########################################################################

        display(HTML("<h3 style=" + h3s + ">4. Représentation des variables</h3>"))
        print(
            "Le champ components_ de l'objet ACP correspond aux valeurs propres.\n" +
            "On peut alors calculer la matrice des corrélations variables * facteurs"
        )
        print()

    # calcul des vecteurs propres
    vp = acp.components_
    x1 = pd.DataFrame({"id": X.columns})
    for j in range(p):
        x1["F" + str(j + 1)] = vp[j, :]
        x1["F" + str(j + 1)] = round(x1["F" + str(j + 1)], 4)

    # vecteurs propres enregistrées pour le return
    all_values["vecteurs_propres"] = x1

    # racine carrée des valeurs propres
    sqrt_eigval = np.sqrt(eigval)

    # corrélation des variables avec les axes
    corvar = np.zeros((p, p))

    for k in range(p):
        corvar[:, k] = acp.components_[k, :] * sqrt_eigval[k]

    # afficher la matrice des corrélations variables x facteurs
    x2 = corvar.round(4)

    all_values["matrice_cor"] = x2

    x3 = pd.DataFrame({"id": X.columns})
    for j in range(p):
        x3["COR_F" + str(j + 1)] = corvar[:, j]
        x3["COR_F" + str(j + 1)] = round(x3["COR_F" + str(j + 1)], 4)

    all_values["cor_par_facteur"] = x3

    # Qualité de représentation des variables : COS²
    cos2var = corvar**2
    x = pd.DataFrame({"id": X.columns})
    for j in range(p):
        x["COS²_var_F" + str(j + 1)] = cos2var[:, j]
        x["COS²_var_F" + str(j + 1)] = round(x["COS²_var_F" + str(j + 1)], 4)

    all_values["cos2_var"] = x

    if graph_only is None:
        print(
            "####################################################################################\n"+
            "####                            Les vecteurs propres                            ####\n"+
            "####################################################################################"
        )
        display(x1)
        print()
        print(
            "####################################################################################\n"+
            "####                          Corrélations par facteur                          ####\n"+
            "####################################################################################"
        )
        display(x3)
        print()
        print(
            "####################################################################################\n"+
            "####                             COS² des variables                             ####\n"+
            "####################################################################################"
        )
        display(x)

    # vérifions la théorie - somme COS² en lignes = 1
    x = np.sum(cos2var, axis=1)
    valid = True
    for i in range(len(x)):
        if x[i] < 0.95 or x[i] > 1.05:
            valid = False
            break

    if valid is not True:
        resultats_test.append(0)
    else:
        resultats_test.append(1)

    if graph_only is None:
        alert_verif(
            fail="La vérifiction de la somme des COS² égale à 1 pour tous les facteurs a échoué",
            success="La vérifiction de la somme des COS² égale à 1 pour tous les facteur s'est correctement déroulée",
            valid=valid,
        )
        print()

    # contributions
    ctrvar = cos2var
    for k in range(p):
        ctrvar[:, k] = ctrvar[:, k] / eigval[k]
    x4 = pd.DataFrame({"id": X.columns})
    for j in range(p):
        x4["CTR_var_F" + str(j + 1)] = ctrvar[:, j]
        x4["CTR_var_F" + str(j + 1)] = round(x4["CTR_var_F" + str(j + 1)], 4)
    all_values["contribution_var"] = x4
    if graph_only is None:
        print(
            "####################################################################################\n"+
            "####                        Contribution des variables                          ####\n"+
            "####################################################################################"
        )
        print(
            "La contribution est également basée sur le carré de la corrélation, mais relativisée par l’importance de l’axe."

        )
        cols = [x for x in list(x4.columns) if x != "id"]
        cols_str = [x+'_str' for x in cols]
        for col in cols:
            x4[col+'_str'] = round(x4[col]*100, 4)
            x4[col+'_str'] = x4[col+'_str'].astype(str)
            x4[col+'_str'] = x4[col+'_str'].apply(lambda x: x+"%")
        x4 = x4[cols_str]
        display(x4)

    # vérifions la théorie - somme en colonnes = 1
    x = np.sum(cos2var, axis=0)
    valid = True
    for i in range(len(x)):
        if x[i] < 0.95 or x[i] > 1.05:
            valid = False
            break

    if valid is not True:
        resultats_test.append(0)
    else:
        resultats_test.append(1)

    if graph_only is None:
        alert_verif(
            fail="La vérifiction de la somme des contributeurs égale à 1 pour tous les axes factoriels a échoué",
            success="La vérifiction de la somme des contributeurs égale à 1 pour tous les axes factoriels s'est correctement déroulée",
            valid=valid,
        )
        print()

        ###########################################################################
        #       5. Traitement des variables supplémentaires
        ###########################################################################

        # Traitement des variables supplémentaires
        display(
            HTML(
                "<h3 style=" + h3s + ">5. Traitement des variables supplémentaires</h3>"
            )
        )
        print()

    # variables supplémentaires quantitatives
    if varSupp is not None:
        if graph_only is None:
            print(
                "####################################################################################\n"+
                "####                   Variables illustratives quantitatives                    ####\n"+
                "####################################################################################"
            )

        # corrélation avec les axes factoriels
        vsQuanti = varSupp.values
        corSupp = np.zeros((varSupp.shape[1], p))

        # rappel: p est le nombre total de dimensions générées
        x = pd.DataFrame({"id": varSupp.columns})
        for k in range(p):
            for j in range(vsQuanti.shape[1]):
                corSupp[j, k] = np.corrcoef(vsQuanti[:, j], coord[:, k])[0, 1]

        for j in range(p):
            x["COR_VQI_F" + str(j + 1)] = corSupp[:, j]
            x["COR_VQI_F" + str(j + 1)] = round(
                x["COR_VQI_F" + str(j + 1)], 4
            )
        # affichage des corrélations avec les axes
        all_values["cor_par_facteur_VarQttiveIllus"] = x

        if graph_only is None:
            print("-------------------------------------------------------")
            print("|       Corrélations avec les axes factoriels         |")
            print("-------------------------------------------------------")
            display(x)
            print()

    # variables supplémentaires qualitatives
    vars_qual = []
    if varSuppQual is not None:
        if graph_only is None:
            print(
                "####################################################################################\n"+
                "####                    Variables illustratives qualitatives                    ####\n"+
                "####################################################################################"
            )
            print(
                "Principe : on utilise la variable qualitative pour différencier les individus en fonction des modalités de celle-ci."
            )
            if group is not None:
                print(
                    "Etant donné qu'il existe déjà différents clusters dans ce jeu de données, un graphique supplémentaire sera réalisé, "+
                    "à savoir une projection des individus avec des groupes correspondants aux différentes modalités de la "+
                    "variable qualitative (en faisant abstraction des clusters)."
                )
            else:
                print(
                    "Le jeu de données ne disposant pas de classification des individus, les modalités de la "+
                    "variable qualitative seront utilisées pour différencier les individus "+
                    "dans le graphique de projection."
                )
            print()

            vars_qual = list(varSuppQual.columns)
            len_varq = len(vars_qual)
            pluriel = "s" if len_varq > 1 else ""
            print(str(len_varq) + " variable" +pluriel+ " illustrative" +pluriel+ " qualitative" +pluriel+ " :")
            for vqual in vars_qual:
                name_vqual = "Variable " + vqual
                modalites = varSuppQual[vqual].unique().tolist()
                modalites = sorted(modalites)
                len_modalites = len(modalites)
                mod_str = '", "'.join(modalites)
                print("     ▪ Variable " + vqual + " => " +str(len_modalites)+' modalités : ["'+ mod_str + '"]')
                print()

    all_values["resultats_tests"] = resultats_test

    ###########################################################################
    #       6. Représentation graphique
    ###########################################################################

    if graph_only is None:
        # Représentation graphique
        display(HTML("<h3 style=" + h3s + ">6. Représentation graphique</h3>"))

    # Projection des individus
    if labels is not None:
        labels = df.index
    if labels_ind is not None:
        labels_ind = df.index

    # Nombre de dimensions à étudier
    n_comp = p
    nb_dim = len(axis_ranks)
    max_combi = C(2, n_comp)
    if nb_dim > max_combi:
        print(
            f"Vous avez indiqué un nombre trop important de plan factoriels. "+
            "Il y en a {max_combi} de possibles au maximum (votre requête a été modifiée en ce sens.)"
        )

    ## Détermination du nombre de classifications total
    all_groups = []
    if group is not None or varSuppQual is not None:
        if group is not None:
            # test integer
            test_group_int = True
            modalites = df_c[group].unique().tolist()
            for t in df_c[group].unique().tolist():
                if t.isdigit() is not True:
                    test_group_int = False
                    break
            if test_group_int is not False:
                all_groups.append([df_c, group, {}])
            else:
                ar = dic_modalite(df_c, group)
                if len(ar) == 3:
                    all_groups.append(ar)

        if varSuppQual is not None:
            df = df.sort_index()
            varSuppQual = varSuppQual.sort_index()
            for vqual in vars_qual:
                df_q = df.copy()
                df_q[vqual] = varSuppQual[vqual]
                ar = dic_modalite(df_q, vqual)
                if len(ar) == 3:
                    all_groups.append(ar)
    nb_groups = len(all_groups)

    for i, ar in enumerate(all_groups):
        if nb_groups > 1:
            print(
                "  7."
                + str(i + 1)
                + ". Individus différenciés en fonction de la variable "
                + ar[1]
            )
            print()

        for d1, d2 in axis_ranks:
            if d1 == d2 or d1 < 0 or d2 < 0 or d1 > n_comp or d2 > n_comp or n_comp > p:
                print(
                    'Il y a un problème avec les dimensions que vous avez saisis, merci de vérifier les variables "axis_ranks" et "n_comp"'
                )
                break

            else:
                if d1 > d2:
                    d1, d2 = d2, d1
                if d2 < n_comp:

                    x_min = round(1.15 * (cf.loc[:, d1 + 1].min()), 1)
                    x_max = round(1.15 * (cf.loc[:, d1 + 1].max()), 1)
                    y_min = round(1.15 * (cf.loc[:, d2 + 1].min()), 1)
                    y_max = round(1.15 * (cf.loc[:, d2 + 1].max()), 1)

                    xlim = choix_coord(x_min, x_max)
                    ylim = choix_coord(y_min, y_max)

                    # array des texts sur graphiques pour appliquer module adjust_text
                    texts = []

                    # initialisation de la figure
                    gr = plt.figure(figsize=(20, 9))

                    ###########################################################################
                    # Projection des individus
                    ###########################################################################

                    # Initialisation du graphique de projection des individus
                    ax1 = plt.subplot(1, 2, 1)
                    axe_lim = max([xlim, ylim])
                    ax1.set_xlim(axe_lim)
                    ax1.set_ylim(axe_lim)

                    # ajouter les axes
                    plt.plot(
                        [xlim[0], xlim[1]],
                        [0, 0],
                        color="silver",
                        linestyle="-",
                        linewidth=1,
                    )
                    plt.plot(
                        [0, 0],
                        [ylim[0], ylim[1]],
                        color="silver",
                        linestyle="-",
                        linewidth=1,
                    )
                    legend_patch = []

                    if group is None and varSuppQual is None:
                        if labels is not None:
                            for i in range(n):
                                if group_special is not None and i in (ind_spe):
                                    texts.append(
                                        plt.annotate(
                                            X.index[i],
                                            (coord[i, d1], coord[i, d2]),
                                            c="green",
                                        )
                                    )
                                else:
                                    texts.append(
                                        plt.annotate(
                                            X.index[i],
                                            (coord[i, d1], coord[i, d2]),
                                            c="navy",
                                        )
                                    )
                        if labels_ind is not None:
                            for i in range(n):
                                if group_special is not None and i in (ind_spe):
                                    texts.append(
                                        plt.annotate(
                                            i, (coord[i, d1], coord[i, d2]), c="green"
                                        )
                                    )
                                else:
                                    texts.append(
                                        plt.annotate(
                                            i, (coord[i, d1], coord[i, d2]), c="navy"
                                        )
                                    )

                        adjust_text(
                            texts, 
                            expand_text=(1.05, 1.05),
                            expand_points=(1.05, 1.05),
                            expand_objects=(1.05, 1.05),
                            expand_align=(1.05, 1.05),
                            force_points=(0.1, 0.25)
                        )

                        if labels is None and labels_ind is None:
                            for i in range(n):
                                if group_special is not None and i in (ind_spe):
                                    plt.scatter(coord[i, d1], coord[i, d2], c="red")
                                else:
                                    plt.scatter(coord[i, d1], coord[i, d2], c="teal")

                        m1 = coord[:, d1].mean()
                        m2 = coord[:, d2].mean()

                    else:
                        df_c = ar[0]
                        dic = ar[2]
                        group = ar[1]

                        m1_tous_clusters = []
                        m2_tous_clusters = []
                        df_c_reset = df_c.reset_index()
                        df_c_gb = df_c.groupby(group).mean()
                        df_c_gb = df_c_gb.reset_index()

                        if len(dic) != 0:
                            inv_dic = {v: k for k, v in dic.items()}

                        val = np.array(df_c[group])
                        for v in np.unique(val):
                            z = int(v)
                            selected = np.where(val == v)

                            if labels is None and labels_ind is None:
                                plt.scatter(
                                    coord[selected, d1],
                                    coord[selected, d2],
                                    alpha=0.9,
                                    label=v,
                                    marker="o",
                                    s=70,
                                    c=palette_perso[z],
                                )

                            m1 = coord[selected, d1].mean()
                            m2 = coord[selected, d2].mean()
                            m1_tous_clusters.append(m1)
                            m2_tous_clusters.append(m2)

                            plt.scatter(
                                m1, m2, marker="s", s=30, label=v, c="black", alpha=1
                            )

                            if len(dic) != 0:
                                leg_lab = group + " : " + inv_dic[str(v)]
                            else:
                                leg_lab = group + " : groupe " + str(v)
                            legend_patch.append(
                                Line2D(
                                    [0],
                                    [0],
                                    marker="o",
                                    markersize=10,
                                    markerfacecolor=palette_perso[z],
                                    color="w",
                                    label=leg_lab,
                                )
                            )

                        all_values["centroides_" + str(d1 + 1) + "_" + str(d2 + 1)] = [
                            m1_tous_clusters,
                            m2_tous_clusters,
                        ]

                        # détermination de la palette couleur
                        nb_col = len(list(df_c[group]))
                        palette_perso = palette_perso[:nb_col]

                        if labels is not None:
                            for i, (x, y) in enumerate(coord[:, [d1, d2]]):
                                u = int(df_c_reset.loc[i, group])
                                if group_special is not None and i in (ind_spe):
                                    texts.append(
                                        plt.text(
                                            x,
                                            y,
                                            labels[i],
                                            fontsize="9",
                                            ha="center",
                                            va="center",
                                            c="black",
                                            bbox=dict(
                                                facecolor=palette_perso[-1],
                                                edgecolor="black",
                                                pad=1,
                                            ),
                                        )
                                    )
                                    col_spe = palette_perso[-1]
                                else:
                                    texts.append(
                                        plt.text(
                                            x,
                                            y,
                                            labels[i],
                                            fontsize="9",
                                            ha="center",
                                            va="center",
                                            c=palette_perso[u],
                                        )
                                    )

                        if labels_ind is not None:
                            for i, (x, y) in enumerate(coord[:, [d1, d2]]):
                                u = int(df_c_reset.loc[i, group])
                                if group_special is not None and i in (ind_spe):
                                    texts.append(
                                        plt.text(
                                            x,
                                            y,
                                            i,
                                            fontsize="9",
                                            ha="center",
                                            va="center",
                                            c="black",
                                            bbox=dict(
                                                facecolor=palette_perso[-1],
                                                edgecolor="black",
                                                boxstyle="square,pad=0.2",
                                            ),
                                        )
                                    )
                                    col_spe = palette_perso[-1]
                                else:
                                    texts.append(
                                        plt.text(
                                            x,
                                            y,
                                            i,
                                            fontsize="10",
                                            ha="center",
                                            va="center",
                                            c=palette_perso[u],
                                        )
                                    )
                    adjust_text(
                        texts, 
                        expand_text=(1.05, 1.05),
                        expand_points=(1.05, 1.05),
                        expand_objects=(1.05, 1.05),
                        expand_align=(1.05, 1.05),
                        force_points=(0.1, 0.25)
                    )

                    if group_special is not None or group is not None:
                        if group_special is not None:
                            legend_patch.append(
                                Line2D(
                                    [0],
                                    [0],
                                    marker="o",
                                    markersize=10,
                                    markerfacecolor=col_spe,
                                    markeredgecolor=col_spe,
                                    color="black",
                                    label="selection",
                                )
                            )

                        if group is not None:
                            legend_patch.append(
                                Line2D(
                                    [0],
                                    [0],
                                    marker="s",
                                    markersize=8,
                                    markerfacecolor="black",
                                    color="w",
                                    markeredgecolor="black",
                                    label="Centroïde",
                                )
                            )

                        legend_groups = plt.legend(
                            handles=legend_patch,
                            bbox_to_anchor=(0.01, 0.99),
                            loc="upper left",
                            fontsize=9,
                        )
                        ax1 = plt.gca().add_artist(legend_groups)

                    if legend_label is not None:
                        test_patch = []
                        for i in range(n):
                            if group is not None:
                                u = int(df_c_reset.loc[i, group])
                                test_patch.append(
                                    mpatches.Patch(
                                        color=palette_perso[u],
                                        label=str(i) + " : " + str(X.index[i]),
                                    )
                                )
                            else:
                                test_patch.append(
                                    mpatches.Patch(
                                        color="white",
                                        ec=None,
                                        fc=None,
                                        label=str(i) + " : " + str(X.index[i]),
                                    )
                                )

                        plt.legend(
                            handles=test_patch,
                            fontsize=8,
                            loc="lower left",
                            ncol=6,
                            bbox_to_anchor=(0, -0.2),
                        )

                    # nom des ax1, avec le pourcentage d'inertie expliqué
                    plt.xlabel(
                        "F{} ({}%)".format(
                            d1 + 1, round(100 * acp.explained_variance_ratio_[d1], 2)
                        )
                    )
                    plt.ylabel(
                        "F{} ({}%)".format(
                            d2 + 1, round(100 * acp.explained_variance_ratio_[d2], 2)
                        )
                    )
                    acp.explained_variance_ratio_
                    plt.title(
                        f"Projection des individus sur F{d1+1} et F{d2+1}",
                        color="darkred",
                        fontsize=14,
                        y=1.01,
                    )

                    gr.savefig(
                        "apc_graphs/"
                        + (version_name + "_" if version_name is not None else "")
                        + "proj_ind_sur_{}_et_{}.png".format(d1 + 1, d2 + 1),
                        dpi=200,
                    )
                    im_proj_pop = Image.open(
                        "apc_graphs/"
                        + (version_name + "_" if version_name is not None else "")
                        + "proj_ind_sur_{}_et_{}.png".format(d1 + 1, d2 + 1)
                    )
                    width, height = im_proj_pop.size
                    x_start = 0
                    y_start = 0
                    x_end = width / 2
                    y_end = height
                    im_proj_pop_temp = im_proj_pop.crop(
                        (x_start, y_start, x_end, y_end)
                    )
                    proj_ind = (
                        "apc_graphs/"
                        + (version_name + "_" if version_name is not None else "")
                        + "proj_ind_sur_{}_et_{}.png".format(d1 + 1, d2 + 1)
                    )
                    im_proj_pop_temp.save(proj_ind)
                    all_values[
                        "graph_proj_ind_" + str(d1 + 1) + "_" + str(d2 + 1)
                    ] = proj_ind


                    ###########################################################################
                    # Cercle des corrélations
                    ###########################################################################

                    # initialisation du graphique du cercle des corrélations
                    texts = []
                    ax2 = plt.subplot(1, 2, 2)
                    ax2.set_xlim(-1.1, 1.1)
                    ax2.set_ylim(-1.1, 1.1)

                    for j in range(p):
                        texts.append(
                            plt.annotate(
                                X.columns[j],
                                (corvar[j, d1], corvar[j, d2]),
                                fontsize=12,
                                color="darkred",
                            )
                        )
                        plt.scatter(
                            corvar[j, d1],
                            corvar[j, d2],
                            color="black",
                            s=20,
                            marker="x",
                        )
                        ax2.add_artist(
                            Line2D(
                                [0, corvar[j, d1]],
                                [0, corvar[j, d2]],
                                color="darkred",
                                alpha=0.6,
                                linewidth=2,
                                linestyle="solid",
                                label="variables_active",
                            )
                        )

                    if varSupp is not None:
                        vsQuanti = varSupp.values
                        for j in range(vsQuanti.shape[1]):
                            texts.append(
                                plt.annotate(
                                    varSupp.columns[j],
                                    (corSupp[j, d1], corSupp[j, d2]),
                                    fontsize=12,
                                    color="green",
                                )
                            )
                            plt.scatter(
                                corSupp[j, d1],
                                corSupp[j, d2],
                                color="black",
                                s=20,
                                marker="x",
                            )
                            ax2.add_artist(
                                Line2D(
                                    [0, corSupp[j, d1]],
                                    [0, corSupp[j, d2]],
                                    color="green",
                                    alpha=0.6,
                                    linewidth=2,
                                    linestyle="solid",
                                    label="variables_illustratives",
                                )
                            )

                    # Ajustement du texte
                    adjust_text(texts)

                    # ajouter les axes des abscisses et des ordonnées
                    plt.plot([-1, 1], [0, 0], color="grey", ls="--")
                    plt.plot([0, 0], [-1, 1], color="grey", ls="--")

                    # ajouter un cercle
                    cercle = plt.Circle((0, 0), 1, color="black", fill=False)
                    ax2.add_artist(cercle)

                    # nom des ax2, avec le pourcentage d'inertie expliqué
                    plt.xlabel(
                        "F{} ({}%)".format(
                            d1 + 1, round(100 * acp.explained_variance_ratio_[d1], 2)
                        )
                    )
                    plt.ylabel(
                        "F{} ({}%)".format(
                            d2 + 1, round(100 * acp.explained_variance_ratio_[d2], 2)
                        )
                    )
                    acp.explained_variance_ratio_
                    plt.title(
                        f"Cercle des corrélations de F{d1+1} et F{d2+1}",
                        color="darkred",
                        fontsize=14,
                        y=1.01,
                    )

                    # légende
                    darkred_patch = mpatches.Patch(
                        color="darkred", label="Variables actives"
                    )
                    if varSupp is not None:
                        green_patch = mpatches.Patch(
                            color="green", label="Variables illustratives"
                        )
                        plt.legend(handles=[darkred_patch, green_patch], fontsize=9)
                    else:
                        plt.legend(handles=[darkred_patch], fontsize=9)

                    gr.savefig(
                        "apc_graphs/"
                        + (version_name + "_" if version_name is not None else "")
                        + "graph_combo_{}_et_{}.png".format(d1 + 1, d2 + 1),
                        dpi=100,
                    )
                    combo_name = (
                        "apc_graphs/"
                        + (version_name + "_" if version_name is not None else "")
                        + "graph_combo_{}_et_{}.png".format(d1 + 1, d2 + 1)
                    )
                    all_values[
                        "graph_combo_" + str(d1 + 1) + "_" + str(d2 + 1)
                    ] = combo_name

                    gr.savefig(
                        "apc_graphs/"
                        + (version_name + "_" if version_name is not None else "")
                        + "cercle_cor_{}_et_{}.png".format(d1 + 1, d2 + 1),
                        dpi=200,
                    )
                    im_corr = Image.open(
                        "apc_graphs/"
                        + (version_name + "_" if version_name is not None else "")
                        + "cercle_cor_{}_et_{}.png".format(d1 + 1, d2 + 1)
                    )
                    width, height = im_corr.size
                    x_start = width / 2
                    y_start = 0
                    x_end = width
                    y_end = height
                    im_corr_temp = im_corr.crop((x_start, y_start, x_end, y_end))
                    cercle_cor = (
                        "apc_graphs/"
                        + (version_name + "_" if version_name is not None else "")
                        + "cercle_cor_{}_et_{}.png".format(d1 + 1, d2 + 1)
                    )
                    im_corr_temp.save(cercle_cor)
                    all_values[
                        "cercle_cor_" + str(d1 + 1) + "_" + str(d2 + 1)
                    ] = cercle_cor

                    # affichage
                    if data_only is not True:
                        plt.show()
                    plt.close("all")


    ###########################################################################
    # Afficher les données pour un individu particulier
    ###########################################################################
    wind = df_global_data.copy()
    pd.set_option("display.max_rows", wind.shape[0] + 1)

    cf_reset = cf.reset_index()
    liste_facteurs = ["id"]
    for i in range(p):
        liste_facteurs.append("F" + str(i + 1))
    cf_reset.columns = liste_facteurs
    cos2_ind.columns = liste_facteurs
    ctr_ind.columns = liste_facteurs

    for i in wind.columns:
        wind[i] = wind[i].apply(lambda x: round(x, 2))
    liste_indice = np.arange(0, n, 1)
    wind["indice"] = liste_indice

    # si labels_ind => on remplace l'index par col "indice"
    if labels_ind is not None:
        old_cols = [x for x in list(wind.columns) if x != "indice"]
        ind_name = wind.index.name if wind.index.name is not None else 'ITEM'
        wind[ind_name] = wind.index
        wind = wind.set_index("indice")
        new_cols = [ind_name]
        new_cols.extend(old_cols)
        wind = wind[new_cols]

    def affichage_infos_obs(df1, ind):
        df2 = cf_reset.loc[cf_reset.index == ind]
        df2_c = df2.copy()
        df2_c["id"] = "Coord Fact"
        df2 = df2_c.copy()
        df2 = df2.T

        df3 = cos2_ind.loc[cos2_ind.index == ind]
        df3_c = df3.copy()
        df3_c["id"] = "COS²"
        df3 = df3_c.copy()
        df3 = df3.T

        rang_list = []
        for j in range(p):
            x = ctr_list[j][ctr_list[j].index == ind]
            rang_list.append(x.reset_index().loc[0, "rang"])
        df4 = ctr_ind.loc[ctr_ind.index == ind]
        df4_c = df4.copy()
        df4_c["id"] = "Ctr"
        df4 = df4_c.copy()
        df4 = df4.T

        df5 = pd.concat([df2, df3, df4], axis=1)
        df5.columns = ["coord. fact.", "COS²", "Contribution"]
        df5 = df5.iloc[1:]

        cols_del = ["index", "indice"]
        for c in cols_del:
            if c in list(df1.columns):
                df1 = df1.drop(columns=[c])
        for j in range(p):
            df5.loc[ctr_ind.columns[j + 1], "Contribution"] = (
                str(round(df5.loc[ctr_ind.columns[j + 1], "Contribution"] * 100, 5))
                + "% ("
                + str(rang_list[j])
                + "e contributeur)"
            )

        return (df1, df5)

    def select_element(element):
        if labels_ind is not None:
            df1 = wind.loc[wind["indice"] == element]
            return affichage_infos_obs(df1, element)
        elif labels is not None:
            df1 = wind.loc[wind.index == element]
            ind = df1['indice'].values[0]
            return affichage_infos_obs(df1, ind)

    if labels_ind is not None:
        all_values["select_numero"] = {}
        for i in range(len(wind)):
            all_values["select_numero"][i] = select_element(i)

    elif labels is not None:
        all_values["select_obs"] = {}
        obss = list(wind.index)
        for i in range(len(wind)):
            obs = obss[i]
            all_values["select_obs"][obs] = select_element(obs)

    if graph_only is None:
        print("-------------------------------------------------------")
        print("|          Data d'un individu en particulier            |")
        print("-------------------------------------------------------")
        print()
        print('La key "select_obs" du return renvoie le dictionnaire suivant :')
        print('  - key : '+('les indices des ' if labels_ind is not None else 'les ') + 'valeurs de la variable des observations (cf. la variable en index du df initial)')
        print('  - value : tuple de 2 df :')
        print('    - df des données initiales')
        print('    - df des données calculées pour cet individu :')
        print('        - ses coordonnées factorielles')
        print('        - sa qualité de représentation sur les différents axes (cf. COS²)')
        print('        - sa contribution aux différents axes')
        print()
        serach_str = "l'indice recherché soit 3 :" if labels_ind is not None else "la valeur recherchée soit 'obs3' :"
        vv = "3" if labels_ind is not None else "obs3"
        print('Ex : en supposant que le nom donné à l\'exécution de la fonction soit "resultat_acp" et que '+serach_str)
        display(HTML('  - <b>resultat_acp["select_obs"]["'+vv+'"][0]</b> pour afficher le df de données initiales'))
        display(HTML('  - <b>resultat_acp["select_obs"]["'+vv+'"][1]</b> pour afficher le df de données calculées de l\'individu recherché'))

    if data_only is True:
        print("Traitement terminé")
        if sum(resultats_test) == 5:
            txt = "Les différents tests de vérification sont positifs : pas de problème détecté"
        else:
            txt = "Attention : un ou plusieurs tests de vérification ne se sont pas correctement déroulés : \
            un ou plusieurs problèmes ont été détectés\n(voir la variable 'resultats_tests' pour plus d'infos)"
        print(txt)
    return all_values
