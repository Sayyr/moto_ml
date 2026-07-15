use crate::data::Dataset;
use crate::models::any_model::{AnyModel, ModelKind};
use std::collections::HashMap;
use std::sync::Mutex;

/// État partagé de l'application, accessible depuis toutes les commandes Tauri
/// via `tauri::State<AppState>`. Un modèle entraîné par catégorie (on peut
/// avoir un MLP ET un RBF entraînés simultanément).
pub struct AppState {
    pub inner: Mutex<AppStateInner>,
}

#[derive(Default)]
pub struct AppStateInner {
    /// Dataset utilisé pour l'entraînement. Si un split train/val/test a été
    /// fourni à l'import, c'est la portion "train" qui est stockée ici — c'est
    /// elle qu'on veut voir passer dans `fit()`, jamais les données de test.
    pub dataset: Option<Dataset>,
    /// Portion "test" du split, mise de côté pour évaluer un modèle déjà
    /// entraîné sans jamais l'avoir vue pendant l'apprentissage. Absente si le
    /// dataset a été importé sans fichier de split (import "simple").
    pub test_dataset: Option<Dataset>,
    pub models: HashMap<ModelKind, AnyModel>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            inner: Mutex::new(AppStateInner::default()),
        }
    }
}
