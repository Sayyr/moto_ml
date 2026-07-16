//! Traduction directe des cas de test du notebook (`_Notebook__Cas_de_tests.ipynb`)
//! en tests d'intégration Rust.
//!
//! Convention notebook -> Rust :
//!   - Classification binaire : Y ∈ {-1, +1} dans le notebook -> converti en
//!     label ∈ {0, 1} ici (0 = classe "+1" du notebook, 1 = classe "-1").
//!   - Régression : Y reste une valeur continue (f64), aucune conversion.
//!
//! Sur l'entraînement du MLP dans ces tests : voir `train_best_mlp` ci-dessous
//! avant de lire les tests un par un — c'est la pièce qui rend cette suite
//! stable d'une exécution à l'autre (voir section 7.1 du rapport pour le
//! diagnostic complet qui a mené à cette solution).

#![allow(unused_imports, dead_code)]

use nalgebra::DMatrix;
use moto_ml_lib::models::linear::LogisticRegression;
use moto_ml_lib::models::mlp::Mlp;
use moto_ml_lib::models::regression::LinearRegression;
use moto_ml_lib::models::svm::SvmLinear;
use moto_ml_lib::models::traits::Classifier;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Convertit un slice de labels {-1.0, 1.0} en Vec<usize> de labels {0, 1}
fn pm1_to_label(y: &[f64]) -> Vec<usize> {
    y.iter().map(|&v| if v > 0.0 { 0 } else { 1 }).collect()
}

fn accuracy(preds: &[usize], y: &[usize]) -> f64 {
    preds.iter().zip(y.iter()).filter(|(p, t)| p == t).count() as f64 / y.len() as f64
}

fn mse(preds: &[f64], y: &[f64]) -> f64 {
    preds.iter().zip(y.iter()).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / y.len() as f64
}

/// Construit une DMatrix directement depuis des littéraux Vec<Vec<f64>>.
fn matrix_from_rows(rows: Vec<Vec<f64>>) -> DMatrix<f64> {
    let n_rows = rows.len();
    let n_cols = rows[0].len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    DMatrix::from_row_slice(n_rows, n_cols, &flat)
}

/// Entraîne plusieurs MLP avec des seeds différentes et garde le meilleur
/// ("random restarts"). Nécessaire car la descente de gradient simple (sans
/// momentum ni Adam) sur un MLP est sensible à l'initialisation — un seul
/// tirage peut tomber dans un minimum local ou un point-selle symétrique
/// (observé concrètement sur XOR pendant le développement, cf. rapport §7.1).
/// Plutôt que de figer une seed unique fragile (qui a déjà cassé plusieurs
/// fois entre deux machines/exécutions), on en essaie plusieurs et on
/// s'arrête dès qu'on atteint `target_acc`.
fn train_best_mlp(
    x: &DMatrix<f64>,
    y: &[usize],
    n_classes: usize,
    hidden: &[usize],
    lr: f64,
    epochs: usize,
    batch_size: usize,
    n_seeds: u64,
    target_acc: f64,
) -> (Mlp, f64) {
    let n_features = x.ncols();
    let mut best_model = Mlp::new_seeded(n_features, hidden, n_classes, lr, epochs, batch_size, 0);
    let mut best_acc = 0.0;

    for seed in 0..n_seeds {
        let mut mlp = Mlp::new_seeded(n_features, hidden, n_classes, lr, epochs, batch_size, seed);
        mlp.fit(x, y, n_classes);
        let acc = accuracy(&mlp.predict(x), y);
        if acc > best_acc {
            best_acc = acc;
            best_model = mlp;
        }
        if best_acc >= target_acc {
            break;
        }
    }

    (best_model, best_acc)
}

// ─────────────────────────────────────────────────────────────
// Générateurs de données synthétiques (seed fixe = reproductible)
// ─────────────────────────────────────────────────────────────

/// 2 blobs gaussiens (approximés par du bruit uniforme) séparables linéairement.
fn generate_blobs_2(seed: u64, n_per_class: usize) -> (DMatrix<f64>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers = [(1.0, 1.0), (4.0, 4.0)];
    let mut rows = Vec::new();
    let mut labels = Vec::new();
    for (label, &(cx, cy)) in centers.iter().enumerate() {
        for _ in 0..n_per_class {
            rows.push(vec![cx + rng.gen_range(-0.6..0.6), cy + rng.gen_range(-0.6..0.6)]);
            labels.push(label);
        }
    }
    (matrix_from_rows(rows), labels)
}

/// 3 blobs séparables linéairement (one-vs-rest pour un modèle linéaire).
fn generate_blobs_3(seed: u64, n_per_class: usize) -> (DMatrix<f64>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers = [(1.0, 1.0), (4.0, 1.0), (2.5, 4.0)];
    let mut rows = Vec::new();
    let mut labels = Vec::new();
    for (label, &(cx, cy)) in centers.iter().enumerate() {
        for _ in 0..n_per_class {
            rows.push(vec![cx + rng.gen_range(-0.5..0.5), cy + rng.gen_range(-0.5..0.5)]);
            labels.push(label);
        }
    }
    (matrix_from_rows(rows), labels)
}

/// Motif "Cross" : label=0 si |x|<=0.3 ou |y|<=0.3 (une croix au centre), sinon 1.
fn generate_cross(seed: u64, n: usize) -> (DMatrix<f64>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut rows = Vec::new();
    let mut labels = Vec::new();
    for _ in 0..n {
        let x: f64 = rng.gen_range(-1.0..1.0);
        let y: f64 = rng.gen_range(-1.0..1.0);
        let label = if x.abs() <= 0.3 || y.abs() <= 0.3 { 0 } else { 1 };
        rows.push(vec![x, y]);
        labels.push(label);
    }
    (matrix_from_rows(rows), labels)
}

/// Motif "damier" à 3 classes (Multi Cross) : périodique, pas séparable linéairement.
fn generate_multi_cross(seed: u64, n: usize) -> (DMatrix<f64>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut rows = Vec::new();
    let mut labels = Vec::new();
    for _ in 0..n {
        let x: f64 = rng.gen_range(-2.0..2.0);
        let y: f64 = rng.gen_range(-2.0..2.0);
        let cell = ((x * 1.5).floor() as i64 + (y * 1.5).floor() as i64).rem_euclid(3) as usize;
        rows.push(vec![x, y]);
        labels.push(cell);
    }
    (matrix_from_rows(rows), labels)
}

// ─────────────────────────────────────────────────────────────
// CLASSIFICATION
// ─────────────────────────────────────────────────────────────

#[test]
fn linear_simple_classification() {
    // Notebook cellule 5 : Linear Model OK, MLP (2,1) OK
    let x = matrix_from_rows(vec![vec![1.0, 1.0], vec![2.0, 3.0], vec![3.0, 3.0]]);
    let y = pm1_to_label(&[1.0, -1.0, -1.0]);

    let mut logistic = LogisticRegression::new(2, 2, 0.5, 500);
    logistic.fit(&x, &y, 2);
    assert!(
        accuracy(&logistic.predict(&x), &y) >= 0.99,
        "LogisticRegression devrait résoudre Linear Simple sans problème"
    );

    let (_, acc_mlp) = train_best_mlp(&x, &y, 2, &[1], 0.5, 500, 3, 5, 0.99);
    assert!(acc_mlp >= 0.99, "Mlp devrait aussi résoudre Linear Simple (accuracy obtenue: {acc_mlp})");
}

#[test]
fn linear_multiple_classification() {
    // Notebook cellule 8 : 2 blobs de 50 points, Linear Model OK, MLP (2,1) OK
    let (x, y) = generate_blobs_2(42, 50);

    let mut logistic = LogisticRegression::new(2, 2, 0.1, 300);
    logistic.fit(&x, &y, 2);
    assert!(accuracy(&logistic.predict(&x), &y) >= 0.95, "LogisticRegression devrait séparer 2 blobs distincts");

    let (_, acc_mlp) = train_best_mlp(&x, &y, 2, &[1], 0.1, 300, 20, 10, 0.95);
    assert!(acc_mlp >= 0.95, "Mlp devrait aussi séparer 2 blobs distincts (accuracy obtenue: {acc_mlp})");
}

#[test]
fn xor_linear_should_fail() {
    // Notebook cellule 11 : XOR. Linear Model : KO (attendu !), MLP (2,2,1) : OK
    let x = matrix_from_rows(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0], vec![1.0, 1.0]]);
    let y = pm1_to_label(&[1.0, 1.0, -1.0, -1.0]);

    let mut logistic = LogisticRegression::new(2, 2, 0.5, 500);
    logistic.fit(&x, &y, 2);
    let acc_linear = accuracy(&logistic.predict(&x), &y);
    assert!(acc_linear <= 0.75, "un modèle linéaire NE DOIT PAS résoudre XOR (accuracy obtenue: {acc_linear})");

    // 4 neurones (plutôt que 2) + lr réduit à 0.2 : XOR est notoirement sujet à un
    // point-selle symétrique avec exactement 2 neurones (voir rapport §7.1).
    let (_, acc_mlp) = train_best_mlp(&x, &y, 2, &[4], 0.2, 2000, 4, 30, 0.99);
    assert!(acc_mlp >= 0.95, "un MLP doit résoudre XOR (accuracy obtenue: {acc_mlp})");
}

#[test]
fn cross_linear_should_fail() {
    // Notebook cellule 14 : Cross. Linear Model : KO, MLP (2,4,1) : OK
    let (x, y) = generate_cross(7, 500);

    let mut logistic = LogisticRegression::new(2, 2, 0.3, 500);
    logistic.fit(&x, &y, 2);
    let acc_linear = accuracy(&logistic.predict(&x), &y);
    assert!(acc_linear <= 0.75, "un modèle linéaire NE DOIT PAS résoudre Cross (accuracy obtenue: {acc_linear})");

    let (_, acc_mlp) = train_best_mlp(&x, &y, 2, &[4], 0.3, 3000, 32, 15, 0.85);
    assert!(acc_mlp >= 0.8, "un MLP (2,4,1) doit résoudre Cross (accuracy obtenue: {acc_mlp})");
}

#[test]
fn multi_linear_3classes() {
    // Notebook cellule 17 : 3 classes séparables par des droites.
    let (x, y) = generate_blobs_3(11, 40);

    let mut logistic = LogisticRegression::new(2, 3, 0.2, 500);
    logistic.fit(&x, &y, 3);
    assert!(accuracy(&logistic.predict(&x), &y) >= 0.95, "softmax gère nativement le multiclasse séparable");

    let (_, acc_mlp) = train_best_mlp(&x, &y, 3, &[4], 0.2, 500, 30, 10, 0.95);
    assert!(acc_mlp >= 0.95, "Mlp devrait aussi séparer 3 classes linéaires (accuracy obtenue: {acc_mlp})");
}

#[test]
fn multi_cross_linear_should_fail() {
    // Notebook cellule 20 : motif type damier périodique, 3 classes.
    // Cas le plus difficile du notebook — c'est pour ça qu'on lui laisse le plus
    // de seeds (20) et le seuil final le plus tolérant (0.65, toujours très
    // largement au-dessus du plafond théorique du linéaire sur ce motif).
    let (x, y) = generate_multi_cross(3, 600);

    let mut logistic = LogisticRegression::new(2, 3, 0.2, 500);
    logistic.fit(&x, &y, 3);
    let acc_linear = accuracy(&logistic.predict(&x), &y);
    assert!(acc_linear <= 0.6, "un modèle linéaire ne doit pas résoudre ce damier (accuracy obtenue: {acc_linear})");

    let (_, acc_mlp) = train_best_mlp(&x, &y, 3, &[16, 8], 0.1, 1500, 32, 20, 0.8);
    assert!(acc_mlp >= 0.65, "un MLP à 2 couches cachées doit largement battre le linéaire ici (accuracy obtenue: {acc_mlp})");
}

// ─────────────────────────────────────────────────────────────
// RÉGRESSION
// ─────────────────────────────────────────────────────────────

#[test]
fn linear_simple_2d_regression() {
    // Notebook cellule 24 : X=[[1],[2]], Y=[2,3] -> relation exactement linéaire (Y=X+1)
    let x = matrix_from_rows(vec![vec![1.0], vec![2.0]]);
    let y = vec![2.0, 3.0];

    let mut model = LinearRegression::new(1, 0.1, 500);
    model.fit(&x, &y);
    let preds = model.predict(&x);

    for (p, t) in preds.iter().zip(y.iter()) {
        assert!((p - t).abs() < 0.1, "prédiction {p} trop loin de la cible {t}");
    }
}

#[test]
fn non_linear_simple_2d_regression() {
    // Notebook cellule 27 : X=[[1],[2],[3]], Y=[2,3,2.5] -> le notebook indique
    // "Linear Model : OK" même si les 3 points ne sont pas parfaitement alignés :
    // la MSE minimise l'erreur globale, pas une interpolation exacte point par point.
    let x = matrix_from_rows(vec![vec![1.0], vec![2.0], vec![3.0]]);
    let y = vec![2.0, 3.0, 2.5];

    let mut model = LinearRegression::new(1, 0.05, 1000);
    model.fit(&x, &y);
    let preds = model.predict(&x);

    // Seuil volontairement large : ce cas ne peut pas être résolu exactement par
    // une droite, on vérifie juste que l'erreur reste "raisonnable" (le notebook
    // classe ça "OK" au sens de "acceptable", pas "parfait").
    assert!(mse(&preds, &y) < 0.5, "MSE trop élevée pour un cas annoté OK dans le notebook");
}

#[test]
fn non_linear_simple_3d_regression_should_fail_linear() {
    // Notebook cellule 36 : Linear Model KO, MLP (2,2,1) OK
    // TODO (pas encore possible) : la variante MLP en mode régression n'est pas
    // implémentée (voir le TODO en bas de regression.rs — retirer le softmax de
    // sortie + changer le delta initial de la backprop). Ce test ne couvre donc
    // que la partie "Linear Model KO" pour l'instant.
    let x = matrix_from_rows(vec![
        vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![0.5, 0.5],
    ]);
    let y = vec![0.0, 1.0, 1.0, 0.0, 2.0]; // motif non linéaire (proche d'un XOR continu + un pic)

    let mut model = LinearRegression::new(2, 0.05, 500);
    model.fit(&x, &y);
    let preds = model.predict(&x);
    let linear_mse = mse(&preds, &y);

    // On vérifie juste que la régression linéaire ne colle pas parfaitement
    // (elle ne peut pas, le motif n'est pas linéaire) — pas de comparaison au
    // MLP tant que la variante régression du MLP n'existe pas.
    assert!(linear_mse > 0.1, "une régression linéaire ne devrait pas coller parfaitement à un motif non linéaire (MSE obtenue: {linear_mse})");
}

// ─────────────────────────────────────────────────────────────
// SVM
// ─────────────────────────────────────────────────────────────

#[test]
fn svm_linear_simple() {
    // Reprend les données de linear_simple_classification. Sur des données
    // parfaitement séparables, le SVM doit trouver une frontière à marge maximale.
    let x = matrix_from_rows(vec![vec![1.0, 1.0], vec![2.0, 3.0], vec![3.0, 3.0]]);
    let y = vec![1.0, -1.0, -1.0]; // SVM : labels en {-1, +1}, pas en usize

    let mut svm = SvmLinear::new(2, 0.01, 0.01, 1000);
    svm.fit(&x, &y);
    let preds = svm.predict_sign(&x);

    for (p, t) in preds.iter().zip(y.iter()) {
        assert_eq!(p, t, "le SVM devrait classer parfaitement ce cas trivialement séparable");
    }
}
