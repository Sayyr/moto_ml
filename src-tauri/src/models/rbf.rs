use super::traits::{add_row_broadcast, one_hot, softmax, Classifier};
use nalgebra::DMatrix;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

/// Réseau RBF : k-means -> couche gaussienne -> sortie linéaire + softmax.
#[derive(Serialize, Deserialize)]
pub struct RbfNetwork {
    centers: DMatrix<f64>, // (n_centers, n_features)
    sigma: f64,
    weights: DMatrix<f64>, // (n_centers, n_classes)
    bias: Vec<f64>,
    lr: f64,
    epochs: usize,
}

impl RbfNetwork {
    pub fn new(n_centers: usize, n_classes: usize, sigma: f64, lr: f64, epochs: usize) -> Self {
        Self {
            centers: DMatrix::zeros(n_centers, 1), // redimensionné dans fit()
            sigma,
            weights: DMatrix::zeros(n_centers, n_classes),
            bias: vec![0.0; n_classes],
            lr,
            epochs,
        }
    }

    /// Distance euclidienne au carré entre deux lignes (vues) de matrices.
    /// Générique sur l'itérateur plutôt que de nommer le type exact d'une vue
    /// de ligne nalgebra (qui n'existait pas sous le nom que j'avais utilisé,
    /// et dont le nom exact peut de toute façon varier selon la version) —
    /// on demande juste "quelque chose qu'on peut itérer en &f64".
    fn squared_dist<'a>(a: impl Iterator<Item = &'a f64>, b: impl Iterator<Item = &'a f64>) -> f64 {
        a.zip(b).map(|(x, y)| (x - y).powi(2)).sum()
    }

    /// K-means simplifié (initialisation aléatoire + itérations de Lloyd)
    fn kmeans(x: &DMatrix<f64>, k: usize, iterations: usize) -> DMatrix<f64> {
        let n = x.nrows();
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let mut centers = DMatrix::zeros(k, x.ncols());
        for i in 0..k {
            centers.set_row(i, &x.row(indices[i]));
        }

        for _ in 0..iterations {
            let mut assignments = vec![0usize; n];
            for i in 0..n {
                let row = x.row(i);
                let mut best_dist = f64::MAX;
                for c in 0..k {
                    let dist = Self::squared_dist(row.iter(), centers.row(c).iter());
                    if dist < best_dist {
                        best_dist = dist;
                        assignments[i] = c;
                    }
                }
            }

            let mut sums = vec![vec![0.0; x.ncols()]; k];
            let mut counts = vec![0usize; k];
            for i in 0..n {
                let c = assignments[i];
                let row = x.row(i);
                for j in 0..x.ncols() {
                    sums[c][j] += row[j];
                }
                counts[c] += 1;
            }

            for c in 0..k {
                if counts[c] > 0 {
                    let mean: Vec<f64> = sums[c].iter().map(|v| v / counts[c] as f64).collect();
                    centers.set_row(c, &nalgebra::RowDVector::from_row_slice(&mean));
                } // sinon on garde le centre précédent (évite les centres vides)
            }
        }

        centers
    }

    /// Matrice d'activation gaussienne (n_samples, n_centers)
    fn rbf_activations(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let mut phi = DMatrix::<f64>::zeros(x.nrows(), self.centers.nrows());
        for i in 0..x.nrows() {
            for j in 0..self.centers.nrows() {
                let dist_sq = Self::squared_dist(x.row(i).iter(), self.centers.row(j).iter());
                phi[(i, j)] = (-dist_sq / (2.0 * self.sigma * self.sigma)).exp();
            }
        }
        phi
    }
}

impl Classifier for RbfNetwork {
    fn fit(&mut self, x: &DMatrix<f64>, y: &[usize], n_classes: usize) {
        let k = self.weights.nrows();
        self.centers = Self::kmeans(x, k, 20);

        let phi = self.rbf_activations(x);
        let y_onehot = one_hot(y, n_classes);
        let n = x.nrows() as f64;

        for epoch in 0..self.epochs {
            let logits = add_row_broadcast(&(&phi * &self.weights), &self.bias);
            let probs = softmax(&logits);
            let error = &probs - &y_onehot;

            let grad_w = (phi.transpose() * &error).scale(1.0 / n);
            let grad_b: Vec<f64> = (0..error.ncols()).map(|c| error.column(c).sum() / n).collect();

            self.weights = &self.weights - grad_w.scale(self.lr);
            for c in 0..self.bias.len() {
                self.bias[c] -= self.lr * grad_b[c];
            }

            if epoch % 50 == 0 {
                let eps = 1e-12;
                let loss = -y_onehot.component_mul(&probs.map(|p| (p + eps).ln())).sum() / n;
                println!("epoch {epoch}: loss = {loss:.4}");
            }
        }
    }

    fn predict_proba(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let phi = self.rbf_activations(x);
        let logits = add_row_broadcast(&(&phi * &self.weights), &self.bias);
        softmax(&logits)
    }
}

// NOTE nalgebra : `.set_row(i, &row_vector)` attend un `RowDVector` (ou une vue
// compatible) — si CETTE ligne pose problème à son tour, la doc à consulter est
// la même : `cargo doc --open -p nalgebra`, section `Matrix::set_row`.
