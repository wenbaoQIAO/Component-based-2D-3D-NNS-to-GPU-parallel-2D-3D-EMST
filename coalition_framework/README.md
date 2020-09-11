# Pré-requis 
**Boost 1.55**

*Emplacement d'installation*: C:\boost_1_55_0

\\EXPORT-CIFS3\Li$\GROUPES\LOHR ELECTROMECANIQUE\Librairies et dépendances\Imbricateur_ta_thread\boost_1_55_0-msvc-10.0-32.exe

**Installer Visual Studio C++ Express 2010 (pour le compilateur VC++2010)**

\\EXPORT-CIFS3\Li$\GROUPES\LOHR ELECTROMECANIQUE\Librairies et dépendances\Imbricateur_ta_thread\Visual Studio C++ 2010.zip

*Emplacement d'installation*: c:\Qt\4.8.0vc10\

\\EXPORT-CIFS3\Li$\GROUPES\LOHR ELECTROMECANIQUE\Librairies et dépendances\Imbricateur_ta_thread\qt-win-opensource-4.8.0-vs2010.exe

## Si compilation avec Visual Studio 2015

**Installer Visual Studio 2015 (Connexion internet active nécessaire)**

Version Express: \\EXPORT-CIFS3\Li$\GROUPES\LOHR ELECTROMECANIQUE\Librairies et dépendances\Imbricateur_ta_thread\Visual Studio Express 2015.exe

## Si compilation avec QT

**Installer QT Creator 3.6.1**

\\EXPORT-CIFS3\Li$\GROUPES\LOHR ELECTROMECANIQUE\Librairies et dépendances\Imbricateur_ta_thread\qt-creator-opensource-windows-x86-3.6.1.exe

**Installer Debugging Tools For Windows (Connexion internet active nécessaire)**

\\EXPORT-CIFS3\Li$\GROUPES\LOHR ELECTROMECANIQUE\Librairies et dépendances\Imbricateur_ta_thread\Debugging Tools for Windows.exe

=> Installer uniquement Debugging Tools for Windows lorsque la procédure d'installation le propose

**Configuration du KIT QT**

*Options>Débogueur>Chemin CDB*
* Ajouter C:\temp\symbolchache dans "Chemin de symbole"
* Ajouter srv* dans "Chemin des sources"

*Options>Compiler&Executer>Version de QT*
* Ajouter la version 4.8.0 de QT depuis C:\Qt4.8.0cv10\bin\qmake.exe

*Options>Compiler&Executer>Compilateurs*
* Verifier la présence de Microsoft Visual C++ Compiler 10.0 (x86)

*Options>Compiler&Executer>Debuggers*
* Ajouter CDB depuis C:\Program Files (x86)\Windows Kits\8.1\Debuggers\x64\cdb.exe

*Options>Compiler&Executer>Kits*

**Créer le KIT VC++2010**
* Type de périphérque: Desktop
* Compilateur: Microsoft Visual C++ Compiler 10.0 (x86)
* Débogueur: CDB
* Version de QT: QT 4.8.0 (4.8.0vc10)

# Compilation Visual Studio
1. Ouvrir Imbricateur.sln
2. **Si Visual Studio demande de passer le compilateur en VS2015. Faire "Annuler" - Compiler le projet en mode Visual Studio 2010 uniquement**
3. Executer "imbricateur"

# Compilation dans QT
1. Ouvrir QT
2. Charger le projet test/test.pro et lib/lib.pro et selectionner le KIT "VC++2010"
3. Désactiver "Shadow Build" aux 4 endroits (Lib debug|release + test debug|release)
4. Définir le projet "test" comme projet actif
5. Executer "test"

**Attention à maintenir la synchronisation entre release et debug sur les deux projets**

# Idées de nettoyage/optimisation/refactoring sur l'imbricateur

## Utilisation de QT5 et de VS2015

* Actuellemnt QT4.7 est utilisé ce qui nous oblige à compiler sous VC++2010. QT5 peut être utilisé avec VC++2015, mais la différence principale est l'API de chargement des fichiers XML/DOM qu'il faut récrire.

## Utiliser des interfaces pour les méthodes de gestion de l'arborescence et des pointeurs vers les parents

* Remplacer le pointeur `Systeme *Parent_` et le système d'identifiant manuel `string ID_Tag` par l'héritage d'une interface IHierachie afin d'homogénéiser le code.
* Factoriser les méthodes `ReadXML()` et `WriteXML()` pour avoir un traitement commun de l'arborescence, et uniquement faire une surcharge pour les opérations spécifiques à chaque objet.
* Les appels à `getClosedPolygon()` pour fermer chaque volume à chaque test d'overlap devraient être factorisés et uniquement appelés une seule fois au chargement (normalisation des volumes)

## Améliorations sur l'algo

* Changer le test de non convergence. Aujourd'hui il s'agit d'un nombre de générations (ex: 50), il faudrait essayer de voir si la fonction objective évolue encore (à epsilon près) pour remettre à zéro le compteur de non-convergence.
* Ajouter un seuil de tolérance sur les conflits structure/structure. Actuellement il y a un seuil pour structure/vehicule et vehicule/vehicule.

## Améliorations de la configuration

* Supporter la surcharge des valeurs par défaut en utilisant un jeu de paramètres prédéfinis par type d'imbrication (carbox, ehr, ...). Cela permet de ne mettre en configuration que les paramètres différents du jeu par défaut.

## Tests unitaire de l'imbricateur

* Il est possible d'écrire des tests unitaires en C++ (pour une librairie tiers) via le framework Microsoft Unit Tests : https://msdn.microsoft.com/en-us/library/hh598953.aspx
=> prévoir de mettre en place une série de tests unitaires fonctionnels sur l'imbricateur, notamment pour les méthodes réalisant des opérations mathématiques comme : calculs de plaquage, calculs de transformation, calculs d'overlaps, ...



