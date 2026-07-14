use super::traits::Classifier;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// SVM linéaire binaire, optimisé par sous-gradient descent sur la hinge loss.
///
/// Rappel théorique (à remettre dans le rapport sous latex) :
///   - Labels en {-1, +1} (pas 0/1).
///   - Score : f(x) = w·x + b
///   - Hinge loss pour un exemple : max(0, 1 - y_i * f(x_i))
///   - Loss totale (avec régularisation L2) :
///       L(w) = lambda * ||w||^2 + (1/n) * Σ max(0, 1 - y_i * (w·x_i + b))
///   - Sous-gradient par exemple i :
///       si y_i * f(x_i) >= 1 : grad_w = 2*lambda*w         (marge respectée)
///       sinon                : grad_w = 2*lambda*w - y_i*x_i
///       grad_b = 0 si marge respectée, sinon -y_i
#[derive(Serialize, Deserialize)]
pub struct SvmLinear {
    pub weights: Vec<f64>, // (n_features,)
    pub bias: f64,
    pub lr: f64,
    pub lambda: f64,
    pub epochs: usize,
}

impl SvmLinear {
    pub fn new(n_features: usize, lr: f64, lambda: f64, epochs: usize) -> Self {
        Self { weights: vec![0.0; n_features], bias: 0.0, lr, lambda, epochs }
    }

    /// Entraîne le SVM binaire par descente de sous-gradient (style Pegasos).
    /// `y` doit contenir des valeurs -1.0 ou 1.0.
    pub fn fit(&mut self, x: &DMatrix<f64>, y: &[f64]) {
        let n = x.nrows();
        for _epoch in 0..self.epochs {
            for i in 0..n {
                let xi = x.row(i);
                let dot: f64 = xi.iter().zip(self.weights.iter()).map(|(a, w)| a * w).sum();
                let score = dot + self.bias;
                let margin = y[i] * score;

                if margin >= 1.0 {
                    // marge respectée : seule la régularisation continue de tirer w vers 0
                    for w in self.weights.iter_mut() {
                        *w -= self.lr * 2.0 * self.lambda * *w;
                    }
                } else {
                    // violation de marge : on corrige vers l'exemple mal classé (ou trop proche)
                    for (w, xv) in self.weights.iter_mut().zip(xi.iter()) {
                        *w -= self.lr * (2.0 * self.lambda * *w - y[i] * xv);
                    }
                    self.bias += self.lr * y[i];
                }
            }
        }
    }

    /// Score brut avant seuillage : w·x + b, pour chaque ligne de x
    pub fn decision_function(&self, x: &DMatrix<f64>) -> Vec<f64> {
        (0..x.nrows())
            .map(|i| {
                x.row(i).iter().zip(self.weights.iter()).map(|(a, b)| a * b).sum::<f64>() + self.bias
            })
            .collect()
    }

    pub fn predict_sign(&self, x: &DMatrix<f64>) -> Vec<f64> {
        self.decision_function(x).into_iter().map(|v| if v >= 0.0 { 1.0 } else { -1.0 }).collect()
    }
}

/// Wrapper one-vs-rest pour rendre le SVM linéaire binaire utilisable en multiclasse.
#[derive(Serialize, Deserialize)]
pub struct SvmMulticlass {
    pub classifiers: Vec<SvmLinear>,
}

impl SvmMulticlass {
    pub fn new(n_features: usize, n_classes: usize, lr: f64, lambda: f64, epochs: usize) -> Self {
        Self {
            classifiers: (0..n_classes).map(|_| SvmLinear::new(n_features, lr, lambda, epochs)).collect(),
        }
    }
}

impl Classifier for SvmMulticlass {
    /// One-vs-rest : pour chaque classe c, on construit y_c = +1 si label == c
    /// sinon -1, et on entraîne le SVM binaire correspondant dessus.
    fn fit(&mut self, x: &DMatrix<f64>, y: &[usize], _n_classes: usize) {
        for (c, clf) in self.classifiers.iter_mut().enumerate() {
            let y_c: Vec<f64> = y.iter().map(|&label| if label == c { 1.0 } else { -1.0 }).collect();
            clf.fit(x, &y_c);
        }
    }

    /// Un SVM n'a pas de probabilités "natives" (contrairement à softmax) — on
    /// applique quand même softmax aux scores bruts de chaque classifieur pour
    /// rester cohérent avec l'interface `Classifier` du reste du projet. Ce n'est
    /// pas une vraie probabilité calibrée, juste une normalisation pratique.
    fn predict_proba(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let mut scores = DMatrix::<f64>::zeros(x.nrows(), self.classifiers.len());
        for (c, clf) in self.classifiers.iter().enumerate() {
            let s = clf.decision_function(x);
            for (i, v) in s.into_iter().enumerate() {
                scores[(i, c)] = v;
            }
        }
        super::traits::softmax(&scores)
    }
}
