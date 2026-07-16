pub mod features;
pub mod loader;

/// Une ligne de dataset: vecteur de features + label (index de classe)
#[derive(Debug, Clone)]
pub struct Sample {
    pub features: Vec<f64>,
    pub label: usize, // index dans `classes`
}

#[derive(Debug, Clone)]
pub struct Dataset {
    pub samples: Vec<Sample>,
    pub classes: Vec<String>, // ex: ["sportive", "roadster", "trail", ...]
}

/// Standardisation des features (z-score) : chaque feature est recentrée sur
/// une moyenne de 0 et une variance de 1, calculées UNIQUEMENT sur le train set.
///
/// Pourquoi c'est nécessaire ici : le vecteur de features (3444 valeurs) mélange
/// des blocs à des échelles très différentes (3072 pixels bruts, 48 valeurs
/// d'histogramme, 324 valeurs HOG). Sans standardisation, la descente de
/// gradient sur les modèles linéaires/MLP diverge (loss qui explose), et les
/// distances euclidiennes du RBF sont faussées (dominées par les blocs à plus
/// grande échelle). C'est un problème classique en ML sur features
/// hétérogènes, pas spécifique à cette implémentation.
///
/// Les stats (mean/std) sont calculées une fois sur le train, puis réutilisées
/// telles quelles pour val/test ET pour toute image isolée soumise en
/// inférence — jamais recalculées ailleurs, sinon les échelles deviendraient
/// incohérentes entre entraînement et prédiction.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FeatureScaler {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl FeatureScaler {
    pub fn fit(dataset: &Dataset) -> Self {
        let n = dataset.samples.len() as f64;
        let n_features = dataset.samples[0].features.len();

        let mut mean = vec![0.0; n_features];
        for s in &dataset.samples {
            for (m, v) in mean.iter_mut().zip(s.features.iter()) {
                *m += v / n;
            }
        }

        let mut variance = vec![0.0; n_features];
        for s in &dataset.samples {
            for ((va, m), v) in variance.iter_mut().zip(mean.iter()).zip(s.features.iter()) {
                *va += (v - m).powi(2) / n;
            }
        }
        // .max(1e-8) : évite une division par zéro pour une feature constante
        // sur tout le train set (variance nulle) — rare mais possible.
        let std: Vec<f64> = variance.iter().map(|v| v.sqrt().max(1e-8)).collect();

        Self { mean, std }
    }

    pub fn transform(&self, features: &[f64]) -> Vec<f64> {
        features
            .iter()
            .zip(self.mean.iter())
            .zip(self.std.iter())
            .map(|((f, m), s)| (f - m) / s)
            .collect()
    }

    pub fn transform_dataset(&self, dataset: &Dataset) -> Dataset {
        Dataset {
            samples: dataset
                .samples
                .iter()
                .map(|s| Sample { features: self.transform(&s.features), label: s.label })
                .collect(),
            classes: dataset.classes.clone(),
        }
    }
}
