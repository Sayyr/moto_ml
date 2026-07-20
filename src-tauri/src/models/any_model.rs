use super::linear::LogisticRegression;
use super::mlp::Mlp;
use super::rbf::RbfNetwork;
use super::svm::SvmMulticlass;
use super::traits::Classifier;
use anyhow::Result;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub enum AnyModel {
    LinearRegression(LogisticRegression), // régression linéaire = même struct, loss différente (voir TODO linear.rs)
    LogisticRegression(LogisticRegression),
    Mlp(Mlp),
    Rbf(RbfNetwork),
    Svm(SvmMulticlass),
}

/// Identifiant textuel utilisé côté frontend / commandes (correspond aux entrées du menu)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelKind {
    LinearRegression,
    LogisticRegression,
    Mlp,
    Svm,
    Rbf,
}

impl ModelKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelKind::LinearRegression => "linear_regression",
            ModelKind::LogisticRegression => "logistic_regression",
            ModelKind::Mlp => "mlp",
            ModelKind::Svm => "svm",
            ModelKind::Rbf => "rbf",
        }
    }
}

impl AnyModel {
    pub fn fit(&mut self, x: &DMatrix<f64>, y: &[usize], n_classes: usize) {
        match self {
            AnyModel::LinearRegression(m) => m.fit(x, y, n_classes),
            AnyModel::LogisticRegression(m) => m.fit(x, y, n_classes),
            AnyModel::Mlp(m) => m.fit(x, y, n_classes),
            AnyModel::Rbf(m) => m.fit(x, y, n_classes),
            AnyModel::Svm(m) => m.fit(x, y, n_classes),
        }
    }

    pub fn predict_proba(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        match self {
            AnyModel::LinearRegression(m) => m.predict_proba(x),
            AnyModel::LogisticRegression(m) => m.predict_proba(x),
            AnyModel::Mlp(m) => m.predict_proba(x),
            AnyModel::Rbf(m) => m.predict_proba(x),
            AnyModel::Svm(m) => m.predict_proba(x),
        }
    }

    pub fn predict(&self, x: &DMatrix<f64>) -> Vec<usize> {
        match self {
            AnyModel::LinearRegression(m) => m.predict(x),
            AnyModel::LogisticRegression(m) => m.predict(x),
            AnyModel::Mlp(m) => m.predict(x),
            AnyModel::Rbf(m) => m.predict(x),
            AnyModel::Svm(m) => m.predict(x),
        }
    }

    pub fn kind(&self) -> ModelKind {
        match self {
            AnyModel::LinearRegression(_) => ModelKind::LinearRegression,
            AnyModel::LogisticRegression(_) => ModelKind::LogisticRegression,
            AnyModel::Mlp(_) => ModelKind::Mlp,
            AnyModel::Rbf(_) => ModelKind::Rbf,
            AnyModel::Svm(_) => ModelKind::Svm,
        }
    }

    /// Construit un modèle vierge du bon type avec des hyperparamètres par défaut/fournis.
    pub fn new(kind: ModelKind, n_features: usize, n_classes: usize, params: &TrainParams) -> Result<Self> {
        Ok(match kind {
            ModelKind::LinearRegression | ModelKind::LogisticRegression => {
                let model = LogisticRegression::new(n_features, n_classes, params.lr, params.epochs);
                if kind == ModelKind::LinearRegression {
                    AnyModel::LinearRegression(model)
                } else {
                    AnyModel::LogisticRegression(model)
                }
            }
            ModelKind::Mlp => AnyModel::Mlp(Mlp::new(
                n_features,
                &params.hidden_layers,
                n_classes,
                params.lr,
                params.epochs,
                params.batch_size,
            )),
            ModelKind::Rbf => AnyModel::Rbf(RbfNetwork::new(
                params.n_centers,
                n_classes,
                params.sigma,
                params.lr,
                params.epochs,
            )),
            ModelKind::Svm => AnyModel::Svm(SvmMulticlass::new(
                n_features,
                n_classes,
                params.lr,
                params.lambda,
                params.epochs,
            )),
        })
    }
}

/// Hyperparamètres génériques envoyés depuis le frontend (formulaire "Entraîner").
/// Tous les champs ne sont pas utilisés par tous les modèles (ex: n_centers ignoré pour le MLP).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainParams {
    pub lr: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub hidden_layers: Vec<usize>,
    pub n_centers: usize,
    pub sigma: f64,
    pub lambda: f64, // force de régularisation du SVM (ignoré par les autres modèles)
}

impl Default for TrainParams {
    fn default() -> Self {
        Self {
            lr: 0.05,
            epochs: 200,
            batch_size: 32,
            hidden_layers: vec![64, 32],
            n_centers: 20,
            sigma: 1.0,
            lambda: 0.01,
        }
    }
}
