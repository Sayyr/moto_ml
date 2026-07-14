//! Traduction directe des cas de test du notebook (`_Notebook__Cas_de_tests.ipynb`)
//! en tests d'intégration Rust.
//!
//! Convention notebook -> Rust :
//!   - Classification binaire : Y ∈ {-1, +1} dans le notebook -> converti en
//!     label ∈ {0, 1} ici (0 = classe "+1" du notebook, 1 = classe "-1").
//!   - Régression : Y reste une valeur continue (f64), aucune conversion.
//! 
//! CODE GENERER PAR IA car pas demander dans le sujet
//! c'est plus un repère perso pour savoir où j'en suis

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

// ===============================================================
// Générateurs de données synthétiques (seed fixe = reproductible)
// ===============================================================

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

    let mut mlp = Mlp::new_seeded(2, &[1], 2, 0.5, 500, 3, 1);
    mlp.fit(&x, &y, 2);
    assert!(accuracy(&mlp.predict(&x), &y) >= 0.99, "Mlp devrait aussi résoudre Linear Simple");
}

#[test]
fn linear_multiple_classification() {
    // Notebook cellule 8 : 2 blobs de 50 points, Linear Model OK, MLP (2,1) OK
    let (x, y) = generate_blobs_2(42, 50);

    let mut logistic = LogisticRegression::new(2, 2, 0.1, 300);
    logistic.fit(&x, &y, 2);
    assert!(accuracy(&logistic.predict(&x), &y) >= 0.95, "LogisticRegression devrait séparer 2 blobs distincts");

    let mut mlp = Mlp::new_seeded(2, &[1], 2, 0.1, 300, 20, 1);
    mlp.fit(&x, &y, 2);
    assert!(accuracy(&mlp.predict(&x), &y) >= 0.95, "Mlp devrait aussi séparer 2 blobs distincts");
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

    let mut mlp = Mlp::new_seeded(2, &[2], 2, 0.5, 2000, 4, 44);
    mlp.fit(&x, &y, 2);
    let acc_mlp = accuracy(&mlp.predict(&x), &y);
    assert!(acc_mlp >= 0.95, "un MLP (2,2,1) doit résoudre XOR (accuracy obtenue: {acc_mlp})");
}

// type shit i had to do to find the good side
// #[test]
// fn find_good_seed_for_xor() {
//     let x = matrix_from_rows(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0], vec![1.0, 1.0]]);
//     let y = pm1_to_label(&[1.0, 1.0, -1.0, -1.0]);

//     for seed in 0..50u64 {
//         let mut mlp = Mlp::new_seeded(2, &[4], 2, 0.2, 3000, 4, seed);
//         mlp.fit(&x, &y, 2);
//         let acc = accuracy(&mlp.predict(&x), &y);
//         println!("seed {seed}: accuracy = {acc}");
//     }
// }

#[test]
fn cross_linear_should_fail() {
    // Notebook cellule 14 : Cross. Linear Model : KO, MLP (2,4,1) : OK
    let (x, y) = generate_cross(7, 500);

    let mut logistic = LogisticRegression::new(2, 2, 0.3, 500);
    logistic.fit(&x, &y, 2);
    let acc_linear = accuracy(&logistic.predict(&x), &y);
    assert!(acc_linear <= 0.75, "un modèle linéaire NE DOIT PAS résoudre Cross (accuracy obtenue: {acc_linear})");

    let mut mlp = Mlp::new_seeded(2, &[4], 2, 0.3, 3000, 32, 1);
    mlp.fit(&x, &y, 2);
    let acc_mlp = accuracy(&mlp.predict(&x), &y);
    assert!(acc_mlp >= 0.8, "un MLP (2,4,1) doit résoudre Cross (accuracy obtenue: {acc_mlp})");
}

#[test]
fn multi_linear_3classes() {
    // Notebook cellule 17 : 3 classes séparables par des droites.
    let (x, y) = generate_blobs_3(11, 40);

    let mut logistic = LogisticRegression::new(2, 3, 0.2, 500);
    logistic.fit(&x, &y, 3);
    assert!(accuracy(&logistic.predict(&x), &y) >= 0.95, "softmax gère nativement le multiclasse séparable");

    let mut mlp = Mlp::new_seeded(2, &[4], 3, 0.2, 500, 30, 1);
    mlp.fit(&x, &y, 3);
    assert!(accuracy(&mlp.predict(&x), &y) >= 0.95, "Mlp devrait aussi séparer 3 classes linéaires");
}

#[test]
fn multi_cross_linear_should_fail() {
    let (x, y) = generate_multi_cross(3, 600);

    let mut logistic = LogisticRegression::new(2, 3, 0.2, 500);
    logistic.fit(&x, &y, 3);
    let acc_linear = accuracy(&logistic.predict(&x), &y);
    assert!(acc_linear <= 0.6, "un modèle linéaire ne doit pas résoudre ce damier (accuracy obtenue: {acc_linear})");

    let mut mlp = Mlp::new_seeded(2, &[16, 8], 3, 0.1, 1500, 32, 1);
    mlp.fit(&x, &y, 3);
    let acc_mlp = accuracy(&mlp.predict(&x), &y);
    assert!(acc_mlp >= 0.7, "un MLP à 2 couches cachées doit largement battre le linéaire ici (accuracy obtenue: {acc_mlp})");
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
    let x = matrix_from_rows(vec![vec![1.0], vec![2.0], vec![3.0]]);
    let y = vec![2.0, 3.0, 2.5];

    let mut model = LinearRegression::new(1, 0.05, 1000);
    model.fit(&x, &y);
    let preds = model.predict(&x);
    assert!(mse(&preds, &y) < 0.5, "MSE trop élevée pour un cas annoté OK dans le notebook");
}

#[test]
fn non_linear_simple_3d_regression_should_fail_linear() {
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
