use crate::data::features::extract_features;
use crate::data::loader::load_dataset;
use crate::models;
use crate::models::any_model::{AnyModel, ModelKind, TrainParams};
use crate::state::AppState;
use nalgebra::DMatrix;
use serde::Serialize;
use tauri::State;

// ─────────────────────────────────────────────────────────────
// 0. Importer un dataset
// ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct DatasetInfo {
    pub n_samples: usize,
    pub classes: Vec<String>,
    pub counts: Vec<usize>, // nombre d'images par classe, même ordre que `classes`
}

#[tauri::command]
pub fn import_dataset(state: State<AppState>, dir_path: String) -> Result<DatasetInfo, String> {
    let dataset = load_dataset(&dir_path).map_err(|e| e.to_string())?;
    let info = dataset_info(&dataset);

    let mut inner = state.inner.lock().unwrap();
    inner.dataset = Some(dataset);

    Ok(info)
}

// ─────────────────────────────────────────────────────────────
// 1. Afficher le dataset
// ─────────────────────────────────────────────────────────────

#[tauri::command]
pub fn get_dataset_info(state: State<AppState>) -> Result<DatasetInfo, String> {
    let inner = state.inner.lock().unwrap();
    let dataset = inner.dataset.as_ref().ok_or("Aucun dataset importé")?;
    Ok(dataset_info(dataset))
}

fn dataset_info(dataset: &crate::data::Dataset) -> DatasetInfo {
    let mut counts = vec![0usize; dataset.classes.len()];
    for sample in &dataset.samples {
        counts[sample.label] += 1;
    }
    DatasetInfo {
        n_samples: dataset.samples.len(),
        classes: dataset.classes.clone(),
        counts,
    }
}

// ─────────────────────────────────────────────────────────────
// 2-6. Entraîner / utiliser / modifier un modèle
//      (Régression Linéaire, Classification Linéaire, MLP, SVM, RBF)
// ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct TrainResult {
    pub model_kind: String,
    pub final_loss: f64,
    pub train_accuracy: f64,
}

/// Entraîne un modèle du type demandé sur le dataset actuellement chargé.
/// C'est la fonction appelée par "Entraîner" dans chacune des catégories 2 à 6.
#[tauri::command]
pub fn train_model(
    state: State<AppState>,
    model_kind: ModelKind,
    params: TrainParams,
) -> Result<TrainResult, String> {
    let mut inner = state.inner.lock().unwrap();
    let dataset = inner.dataset.as_ref().ok_or("Aucun dataset importé")?;

    let n_features = dataset.samples[0].features.len();
    let n_classes = dataset.classes.len();

    let (x, y) = dataset_to_arrays(dataset);

    let mut model = AnyModel::new(model_kind, n_features, n_classes, &params).map_err(|e| e.to_string())?;
    model.fit(&x, &y, n_classes);

    let preds = model.predict(&x);
    let train_accuracy = preds.iter().zip(y.iter()).filter(|(p, t)| p == t).count() as f64 / y.len() as f64;

    let result = TrainResult {
        model_kind: model_kind.as_str().to_string(),
        final_loss: 0.0, // TODO : faire remonter la vraie loss finale depuis fit() (actuellement seulement affichée en println!)
        train_accuracy,
    };

    inner.models.insert(model_kind, model);
    Ok(result)
}

fn dataset_to_arrays(dataset: &crate::data::Dataset) -> (DMatrix<f64>, Vec<usize>) {
    let n_features = dataset.samples[0].features.len();
    let flat: Vec<f64> = dataset.samples.iter().flat_map(|s| s.features.clone()).collect();
    let x = DMatrix::from_row_slice(dataset.samples.len(), n_features, &flat);
    let y: Vec<usize> = dataset.samples.iter().map(|s| s.label).collect();
    (x, y)
}

#[derive(Serialize)]
pub struct PredictionResult {
    pub predicted_class: String,
    pub probabilities: Vec<(String, f64)>, // (nom classe, proba), trié décroissant
}

/// Utilise un modèle déjà entraîné (en mémoire) pour prédire sur une image.
/// C'est "Utiliser" dans chaque catégorie 2 à 6, et le cœur de la catégorie 7.
#[tauri::command]
pub fn run_inference(
    state: State<AppState>,
    model_kind: ModelKind,
    image_path: String,
) -> Result<PredictionResult, String> {
    let inner = state.inner.lock().unwrap();
    let dataset = inner.dataset.as_ref().ok_or("Aucun dataset importé (besoin des noms de classes)")?;
    let model = inner.models.get(&model_kind).ok_or("Ce modèle n'a pas encore été entraîné")?;

    let features = extract_features(&image_path).map_err(|e| e.to_string())?;
    let x = DMatrix::from_row_slice(1, features.len(), &features);

    let proba = model.predict_proba(&x);
    let pred = model.predict(&x);

    let mut probabilities: Vec<(String, f64)> = dataset
        .classes
        .iter()
        .zip(proba.row(0).iter())
        .map(|(c, p)| (c.clone(), *p))
        .collect();
    probabilities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    Ok(PredictionResult {
        predicted_class: dataset.classes[pred[0]].clone(),
        probabilities,
    })
}

/// "Modifier" un modèle : recharge un modèle sauvegardé, permet de changer ses
/// hyperparamètres (lr, epochs) et de continuer l'entraînement (fine-tuning)
/// sur le dataset actuellement chargé.
#[tauri::command]
pub fn continue_training(
    state: State<AppState>,
    model_kind: ModelKind,
    extra_epochs: usize,
) -> Result<TrainResult, String> {
    let mut inner = state.inner.lock().unwrap();
    let dataset = inner.dataset.as_ref().ok_or("Aucun dataset importé")?.clone();
    let (x, y) = dataset_to_arrays(&dataset);
    let n_classes = dataset.classes.len();

    let model = inner.models.get_mut(&model_kind).ok_or("Ce modèle n'a pas encore été entraîné")?;

    // NOTE : chaque fit() actuel repart de zéro / réutilise les poids courants selon le modèle.
    // Pour un vrai "continue training", il faut que fit() n'écrase pas les poids déjà appris
    // (actuellement LogisticRegression/Mlp/Rbf réutilisent bien leurs `self.weights` existants,
    // donc rappeler fit() avec `extra_epochs` prolonge effectivement l'entraînement).
    match model {
        AnyModel::LogisticRegression(m) | AnyModel::LinearRegression(m) => m.epochs = extra_epochs,
        AnyModel::Mlp(m) => { /* TODO: exposer un setter epochs sur Mlp */ let _ = m; }
        AnyModel::Rbf(m) => { /* idem */ let _ = m; }
    }
    model.fit(&x, &y, n_classes);

    let preds = model.predict(&x);
    let train_accuracy = preds.iter().zip(y.iter()).filter(|(p, t)| p == t).count() as f64 / y.len() as f64;

    Ok(TrainResult {
        model_kind: model_kind.as_str().to_string(),
        final_loss: 0.0,
        train_accuracy,
    })
}

// ─────────────────────────────────────────────────────────────
// 7. Full test d'une inférence (comparer tous les modèles entraînés)
// ─────────────────────────────────────────────────────────────

#[tauri::command]
pub fn full_test_inference(state: State<AppState>, image_path: String) -> Result<Vec<(String, PredictionResult)>, String> {
    let inner = state.inner.lock().unwrap();
    let dataset = inner.dataset.as_ref().ok_or("Aucun dataset importé")?;
    let features = extract_features(&image_path).map_err(|e| e.to_string())?;
    let x = DMatrix::from_row_slice(1, features.len(), &features);

    let mut results = Vec::new();
    for (kind, model) in inner.models.iter() {
        let proba = model.predict_proba(&x);
        let pred = model.predict(&x);
        let mut probabilities: Vec<(String, f64)> = dataset
            .classes
            .iter()
            .zip(proba.row(0).iter())
            .map(|(c, p)| (c.clone(), *p))
            .collect();
        probabilities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        results.push((
            kind.as_str().to_string(),
            PredictionResult {
                predicted_class: dataset.classes[pred[0]].clone(),
                probabilities,
            },
        ));
    }

    Ok(results)
}

// ─────────────────────────────────────────────────────────────
// 8. Exporter un modèle entraîné
// ─────────────────────────────────────────────────────────────

#[tauri::command]
pub fn export_model(state: State<AppState>, model_kind: ModelKind, output_path: String) -> Result<(), String> {
    let inner = state.inner.lock().unwrap();
    let model = inner.models.get(&model_kind).ok_or("Ce modèle n'a pas encore été entraîné")?;
    models::save_model(model, &output_path).map_err(|e| e.to_string())
}

/// Charge un modèle exporté précédemment (utile pour "Utiliser" un modèle
/// sans avoir eu à le réentraîner dans la session courante).
#[tauri::command]
pub fn import_model(state: State<AppState>, model_kind: ModelKind, input_path: String) -> Result<(), String> {
    let model = models::load_model(&input_path).map_err(|e| e.to_string())?;
    let mut inner = state.inner.lock().unwrap();
    inner.models.insert(model_kind, model);
    Ok(())
}

#[tauri::command]
pub fn list_trained_models(state: State<AppState>) -> Vec<String> {
    let inner = state.inner.lock().unwrap();
    inner.models.keys().map(|k| k.as_str().to_string()).collect()
}
