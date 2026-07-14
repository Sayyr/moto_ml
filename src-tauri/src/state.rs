use crate::data::Dataset;
use crate::models::any_model::{AnyModel, ModelKind};
use std::collections::HashMap;
use std::sync::Mutex;

/// État partagé de l'application, accessible depuis toutes les commandes Tauri
/// via `tauri::State<AppState>`. Un seul dataset actif à la fois, mais un modèle
/// entraîné par catégorie (on peut avoir un MLP ET un RBF entraînés simultanément).
pub struct AppState {
    pub inner: Mutex<AppStateInner>,
}

#[derive(Default)]
pub struct AppStateInner {
    pub dataset: Option<Dataset>,
    pub models: HashMap<ModelKind, AnyModel>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            inner: Mutex::new(AppStateInner::default()),
        }
    }
}
