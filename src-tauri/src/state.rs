use crate::data::{Dataset, FeatureScaler};
use crate::models::any_model::{AnyModel, ModelKind};
use std::collections::HashMap;
use std::sync::Mutex;

/// État partagé de l'application, accessible depuis toutes les commandes Tauri
/// via `tauri::State<AppState>`. Un modèle entraîné par catégorie (on peut
/// avoir un MLP ET un RBF entraînés simultanément en théorie mais en pratique 
/// mon pc est pas assez puissant pour ça mdr).
pub struct AppState {
    pub inner: Mutex<AppStateInner>,
}

#[derive(Default)]
pub struct AppStateInner {
    /// Dataset utilisé pour l'entraînement (portion "train" du split)
    pub dataset: Option<Dataset>,
    /// Portion "test" du split, mise de côté pour évaluer un modèle déjà
    /// entraîné sans jamais l'avoir vue pendant l'apprentissage. Également
    /// déjà standardisée avec les mêmes stats que `dataset`.
    pub test_dataset: Option<Dataset>,
    /// Stats de standardisation (moyenne/écart-type par feature), calculées
    /// une seule fois sur le train à l'import. Réutilisées pour normaliser
    /// toute image soumise en inférence (`run_inference`, `full_test_inference`) cohérence
    pub scaler: Option<FeatureScaler>,
    pub models: HashMap<ModelKind, AnyModel>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            inner: Mutex::new(AppStateInner::default()),
        }
    }
}
