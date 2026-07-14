use super::traits::{add_row_broadcast, one_hot, softmax, Classifier};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Régression logistique multiclasse (= ton "modèle linéaire de classification").
#[derive(Serialize, Deserialize)]
pub struct LogisticRegression {
    pub weights: DMatrix<f64>, // (n_features, n_classes)
    pub bias: Vec<f64>,        // (n_classes,)
    pub lr: f64,
    pub epochs: usize,
}

impl LogisticRegression {
    pub fn new(n_features: usize, n_classes: usize, lr: f64, epochs: usize) -> Self {
        Self {
            weights: DMatrix::zeros(n_features, n_classes),
            bias: vec![0.0; n_classes],
            lr,
            epochs,
        }
    }

    fn forward(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let logits = add_row_broadcast(&(x * &self.weights), &self.bias);
        softmax(&logits)
    }
}

impl Classifier for LogisticRegression {
    fn fit(&mut self, x: &DMatrix<f64>, y: &[usize], n_classes: usize) {
        let y_onehot = one_hot(y, n_classes);
        let n = x.nrows() as f64;

        for epoch in 0..self.epochs {
            let probs = self.forward(x);
            let error = &probs - &y_onehot; // (n, n_classes)

            let grad_w = (x.transpose() * &error).scale(1.0 / n);
            let grad_b: Vec<f64> = (0..error.ncols()).map(|c| error.column(c).sum() / n).collect();

            self.weights = &self.weights - grad_w.scale(self.lr);
            for c in 0..self.bias.len() {
                self.bias[c] -= self.lr * grad_b[c];
            }

            if epoch % 50 == 0 {
                let loss = cross_entropy(&probs, &y_onehot);
                println!("epoch {epoch}: loss = {loss:.4}");
            }
        }
    }

    fn predict_proba(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        self.forward(x)
    }
}

fn cross_entropy(probs: &DMatrix<f64>, y_onehot: &DMatrix<f64>) -> f64 {
    let eps = 1e-12;
    let n = probs.nrows() as f64;
    -y_onehot.component_mul(&probs.map(|p| (p + eps).ln())).sum() / n
}

// TODO : struct LinearRegression pour la régression continue pure — voir
// regression.rs, à ne pas confondre avec cette classe qui fait de la classification.
