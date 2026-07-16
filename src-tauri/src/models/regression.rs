use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Régression linéaire au sens strict (sortie continue), à ne pas confondre avec
/// `LogisticRegression` (classification).
///
///   ŷ = X·w + b
///   MSE(w, b) = (1/n) * Σ (ŷᵢ - yᵢ)²
///   grad_w = (2/n) * Xᵀ·(ŷ - y)
///   grad_b = (2/n) * Σ(ŷ - y)
#[derive(Serialize, Deserialize)]
pub struct LinearRegression {
    pub weights: Vec<f64>, // (n_features,) — une seule sortie continue
    pub bias: f64,
    pub lr: f64,
    pub epochs: usize,
}

impl LinearRegression {
    pub fn new(n_features: usize, lr: f64, epochs: usize) -> Self {
        Self { weights: vec![0.0; n_features], bias: 0.0, lr, epochs }
    }

    fn forward(&self, x: &DMatrix<f64>) -> Vec<f64> {
        (0..x.nrows())
            .map(|i| x.row(i).iter().zip(self.weights.iter()).map(|(a, b)| a * b).sum::<f64>() + self.bias)
            .collect()
    }

    /// Descente de gradient batch sur la MSE (voir la formule dans le commentaire
    /// de struct plus haut).
    pub fn fit(&mut self, x: &DMatrix<f64>, y: &[f64]) {
        let n = x.nrows() as f64;

        for epoch in 0..self.epochs {
            let preds = self.forward(x);
            let errors: Vec<f64> = preds.iter().zip(y.iter()).map(|(p, t)| p - t).collect();

            let mut grad_w = vec![0.0; self.weights.len()];
            let mut grad_b = 0.0;
            for i in 0..x.nrows() {
                let xi = x.row(i);
                for (g, xv) in grad_w.iter_mut().zip(xi.iter()) {
                    *g += errors[i] * xv;
                }
                grad_b += errors[i];
            }
            for g in grad_w.iter_mut() {
                *g *= 2.0 / n;
            }
            grad_b *= 2.0 / n;

            for (w, g) in self.weights.iter_mut().zip(grad_w.iter()) {
                *w -= self.lr * g;
            }
            self.bias -= self.lr * grad_b;

            if epoch % 50 == 0 {
                let mse: f64 = errors.iter().map(|e| e * e).sum::<f64>() / n;
                println!("epoch {epoch}: mse = {mse:.4}");
            }
        }
    }

    pub fn predict(&self, x: &DMatrix<f64>) -> Vec<f64> {
        self.forward(x)
    }
}

// TODO : MLP en mode régression — retirer le softmax de la dernière couche du
// Mlp (mlp.rs), remplacer le delta initial de la backprop par 2*(ŷ - y)/n au
// lieu de (probs - y_onehot). Duplique `Mlp` en `MlpRegressor` si tu préfères
// garder le code de classification intact (recommandé pour commencer).
