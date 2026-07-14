use nalgebra::DMatrix;

/// Trait commun aux modèles de classification. X : (n_samples, n_features).
pub trait Classifier {
    fn fit(&mut self, x: &DMatrix<f64>, y: &[usize], n_classes: usize);

    /// Probabilités par classe : (n_samples, n_classes)
    fn predict_proba(&self, x: &DMatrix<f64>) -> DMatrix<f64>;

    /// Classe prédite (argmax des probas) pour chaque échantillon
    fn predict(&self, x: &DMatrix<f64>) -> Vec<usize> {
        let proba = self.predict_proba(x);
        (0..proba.nrows())
            .map(|i| {
                let row = proba.row(i);
                let mut best_idx = 0;
                let mut best_val = f64::MIN;
                for j in 0..row.len() {
                    if row[j] > best_val {
                        best_val = row[j];
                        best_idx = j;
                    }
                }
                best_idx
            })
            .collect()
    }
}

/// Encode les labels entiers en one-hot (n_samples, n_classes)
pub fn one_hot(y: &[usize], n_classes: usize) -> DMatrix<f64> {
    let mut out = DMatrix::<f64>::zeros(y.len(), n_classes);
    for (i, &label) in y.iter().enumerate() {
        out[(i, label)] = 1.0;
    }
    out
}

/// Softmax numériquement stable, ligne par ligne.
/// nalgebra n'a pas de softmax intégré (ce n'est pas une lib de ML) : on boucle
/// sur les lignes à la main.
pub fn softmax(logits: &DMatrix<f64>) -> DMatrix<f64> {
    let mut out = logits.clone();
    for mut row in out.row_iter_mut() {
        let max = row.iter().cloned().fold(f64::MIN, f64::max);
        for v in row.iter_mut() {
            *v = (*v - max).exp();
        }
        let sum: f64 = row.iter().sum();
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
    out
}

/// nalgebra ne fait pas de broadcasting automatique (contrairement à numpy/ndarray) :
/// additionner un vecteur ligne à chaque ligne d'une matrice doit être fait à la main.
/// C'est l'équivalent de l'ajout du biais dans une couche dense.
pub fn add_row_broadcast(m: &DMatrix<f64>, bias: &[f64]) -> DMatrix<f64> {
    let mut out = m.clone();
    for mut row in out.row_iter_mut() {
        for (j, b) in bias.iter().enumerate() {
            row[j] += b;
        }
    }
    out
}
