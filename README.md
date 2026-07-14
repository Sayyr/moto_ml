# moto-ml

Application desktop de reconnaissance du genre d'une moto (sportive, roadster,
trail, custom, ...) à partir d'une photo, développée en Rust avec Tauri.

Le projet implémente cinq familles de modèles de Machine Learning **from
scratch** (aucune bibliothèque de ML), uniquement à l'aide de
[`nalgebra`](https://nalgebra.org) pour l'algèbre linéaire (utilisation
validée par Mr Vidal) :

- Régression linéaire
- Classification linéaire (régression logistique multiclasse)
- Perceptron multicouche (MLP) avec rétropropagation
- Réseau RBF (k-means + noyau gaussien)
- SVM linéaire multiclasse (one-vs-rest, hinge loss)

L'application reprend le menu du cahier des charges sous forme d'interface
graphique : import/affichage du dataset, un onglet **Entraîner / Utiliser /
Modifier** par famille de modèle, test d'inférence comparatif, export de
modèle.

## Stack technique

| Composant | Choix |
|---|---|
| Backend | Rust |
| Algèbre linéaire | `nalgebra` |
| Application desktop | Tauri 2 |
| Frontend | HTML / CSS / JS (vanilla) |
| Sérialisation modèles | `serde` + `bincode` |
| Traitement d'image | crate `image` |

## Prérequis

- Rust + `cargo`
- Node.js + npm
- Tauri CLI : `cargo install tauri-cli --version "^2"`
- Dépendances système Tauri (webview) : voir
  https://tauri.app/start/prerequisites/ selon l'OS (sur Linux :
  `webkit2gtk`, `libayatana-appindicator3`, etc.)

## Installation

```bash
npm install
```

### Icônes (obligatoire avant le premier build)

`tauri.conf.json` référence des icônes qui ne sont pas versionnées. Les
générer avec :

```bash
cargo tauri icon chemin/vers/un-logo-1024x1024.png
```

## Lancer en développement

```bash
cargo tauri dev
```

Démarre Vite (frontend, port 1420) et compile/lance le backend Rust dans une
fenêtre native.

## Build de production

```bash
cargo tauri build
```

## Tests

Les cas de test de référence (validation des modèles sur des jeux de données
synthétiques : séparabilité linéaire, XOR, Cross, etc.) sont dans
`src-tauri/tests/test_cases.rs`. Le `Cargo.toml` du backend est dans
`src-tauri/`, donc les tests se lancent depuis ce dossier :

```bash
cd src-tauri
cargo test
```

## Structure du projet

```
src-tauri/src/
  data/          extraction de features image (couleur + HOG), chargement du dataset
  models/
    linear.rs      régression logistique multiclasse (classification linéaire)
    mlp.rs          perceptron multicouche + rétropropagation
    rbf.rs          réseau à fonction de base radiale
    svm.rs          SVM linéaire multiclasse (one-vs-rest)
    regression.rs   régression linéaire (sortie continue)
    any_model.rs    enum unifiant les modèles pour le stockage/export générique
    traits.rs       trait Classifier commun + utilitaires (softmax, one-hot)
  commands.rs    commandes Tauri, une par action du menu
  state.rs       état applicatif partagé (dataset + modèles entraînés en mémoire)
src/             interface utilisateur (menu, formulaires, résultats)
```

## Limitations connues / pistes d'amélioration

- **Régression linéaire non branchée dans l'app** : `regression::LinearRegression`
  attend des cibles continues (`&[f64]`), incompatibles avec le trait
  `Classifier` partagé par le reste des modèles (`&[usize]`). Elle est
  validée par les tests (données synthétiques du notebook) mais pas encore
  exposée dans le menu de l'application — le dataset moto n'ayant que des
  labels catégoriels, il n'y a de toute façon pas de cible continue naturelle
  à lui proposer pour l'instant.
- **MLP en mode régression** non implémenté (nécessiterait de retirer le
  softmax de la couche de sortie et d'adapter la rétropropagation).
- **Entraînement synchrone** : `train_model` bloque l'UI le temps de
  l'entraînement. À terme, faire tourner l'entraînement dans un thread
  (`tauri::async_runtime::spawn`) avec des événements de progression.
- **`continue_training`** ne permet réellement d'ajuster les epochs que pour
  la régression logistique (pas encore de setter exposé pour `Mlp`/`RbfNetwork`).
- **Export de modèle** : le dialogue de sauvegarde côté frontend utilise
  actuellement `open()` (sélection de fichier existant) au lieu de `save()`
  (choix d'une destination) — à corriger dans `main.js`.
- **Pas de persistance du dataset entre sessions** (tout est en mémoire).