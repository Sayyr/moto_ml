use super::traits::{add_row_broadcast, one_hot, softmax, Classifier};
use nalgebra::DMatrix;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Layer {
    w: DMatrix<f64>, // (n_in, n_out)
    b: Vec<f64>,      // (n_out,)
}

impl Layer {
    fn new(n_in: usize, n_out: usize, rng: &mut impl Rng) -> Self {
        let scale = 1.0 / (n_in as f64).sqrt();
        let data: Vec<f64> = (0..n_in * n_out).map(|_| rng.gen_range(-scale..scale)).collect();
        // from_row_slice attend les données "ligne par ligne" (comme on les a générées),
        // et se charge de la conversion vers le stockage interne column-major de nalgebra.
        // c'est pas claire mais pour moi ça l'est et je suis solo sur le projet
        Self { w: DMatrix::from_row_slice(n_in, n_out, &data), b: vec![0.0; n_out] }
    }
}

fn relu(x: &DMatrix<f64>) -> DMatrix<f64> {
    x.map(|v| v.max(0.0))
}

fn relu_deriv(x: &DMatrix<f64>) -> DMatrix<f64> {
    x.map(|v| if v > 0.0 { 1.0 } else { 0.0 })
}

/// MLP : n_features -> hidden_sizes[...] -> n_classes (softmax en sortie)
#[derive(Serialize, Deserialize)]
pub struct Mlp {
    layers: Vec<Layer>,
    pub lr: f64,
    pub epochs: usize,
    pub batch_size: usize,
}

impl Mlp {
    pub fn new(n_features: usize, hidden_sizes: &[usize], n_classes: usize, lr: f64, epochs: usize, batch_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self::new_with_rng(n_features, hidden_sizes, n_classes, lr, epochs, batch_size, &mut rng)
    }

    pub fn new_seeded(n_features: usize, hidden_sizes: &[usize], n_classes: usize, lr: f64, epochs: usize, batch_size: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self::new_with_rng(n_features, hidden_sizes, n_classes, lr, epochs, batch_size, &mut rng)
    }

    fn new_with_rng(n_features: usize, hidden_sizes: &[usize], n_classes: usize, lr: f64, epochs: usize, batch_size: usize, rng: &mut impl Rng) -> Self {
        let mut sizes = vec![n_features];
        sizes.extend_from_slice(hidden_sizes);
        sizes.push(n_classes);

        let layers = sizes.windows(2).map(|w| Layer::new(w[0], w[1], rng)).collect();
        Self { layers, lr, epochs, batch_size }
    }

    fn forward_full(&self, x: &DMatrix<f64>) -> (Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) {
        let mut activations = vec![x.clone()];
        let mut pre_activations = Vec::new();
        let n_layers = self.layers.len();

        for (i, layer) in self.layers.iter().enumerate() {
            let z = add_row_broadcast(&(activations.last().unwrap() * &layer.w), &layer.b);
            pre_activations.push(z.clone());

            let a = if i == n_layers - 1 { softmax(&z) } else { relu(&z) };
            activations.push(a);
        }

        (activations, pre_activations)
    }

    fn backward(&mut self, activations: &[DMatrix<f64>], pre_activations: &[DMatrix<f64>], y_onehot: &DMatrix<f64>) {
        let n = activations[0].nrows() as f64;
        let n_layers = self.layers.len();

        let mut delta = activations.last().unwrap() - y_onehot;

        for l in (0..n_layers).rev() {
            let a_prev = &activations[l];
            let grad_w = (a_prev.transpose() * &delta).scale(1.0 / n);
            let grad_b: Vec<f64> = (0..delta.ncols()).map(|c| delta.column(c).sum() / n).collect();

            if l > 0 {
                let w = &self.layers[l].w;
                let delta_prev = (&delta * w.transpose()).component_mul(&relu_deriv(&pre_activations[l - 1]));
                delta = delta_prev;
            }

            self.layers[l].w = &self.layers[l].w - grad_w.scale(self.lr);
            for c in 0..self.layers[l].b.len() {
                self.layers[l].b[c] -= self.lr * grad_b[c];
            }
        }
    }
}

impl Classifier for Mlp {
    fn fit(&mut self, x: &DMatrix<f64>, y: &[usize], n_classes: usize) {
        let y_onehot = one_hot(y, n_classes);
        let n = x.nrows();

        for epoch in 0..self.epochs {
            let mut start = 0;
            while start < n {
                let end = (start + self.batch_size).min(n);
                let x_batch = x.rows(start, end - start).into_owned();
                let y_batch = y_onehot.rows(start, end - start).into_owned();

                let (activations, pre_activations) = self.forward_full(&x_batch);
                self.backward(&activations, &pre_activations, &y_batch);

                start = end;
            }

            if epoch % 20 == 0 {
                let (activations, _) = self.forward_full(x);
                let probs = activations.last().unwrap();
                let eps = 1e-12;
                let loss = -y_onehot.component_mul(&probs.map(|p| (p + eps).ln())).sum() / n as f64;
                println!("epoch {epoch}: loss = {loss:.4}");
            }
        }
    }

    fn predict_proba(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let (activations, _) = self.forward_full(x);
        activations.last().unwrap().clone()
    }
}

// TODO : ajouter le shuffle des indices à chaque epoch (important pour la convergence).
// NOTE nalgebra : `.rows(start, count)` retourne une VUE (pas de copie) ; `.into_owned()`
// force la copie en DMatrix indépendante — nécessaire ici car x_batch doit vivre
// au-delà de l'appel (utilisé dans forward_full ET backward).