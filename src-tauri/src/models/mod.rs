pub mod any_model;
pub mod linear;
pub mod mlp;
pub mod rbf;
pub mod regression;
pub mod svm;
pub mod traits;

use anyhow::Result;
use nalgebra::DMatrix;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

/// Charge un CSV de features (features..., label) avec un header "# classes: a,b,c"
pub fn load_csv(path: &str) -> Result<(DMatrix<f64>, Vec<usize>, Vec<String>)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let header = lines.next().unwrap()?;
    let classes: Vec<String> = header
        .trim_start_matches("# classes: ")
        .split(',')
        .map(String::from)
        .collect();

    let mut rows: Vec<f64> = Vec::new(); // aplati, ligne par ligne
    let mut labels = Vec::new();
    let mut n_features = 0;
    let mut n_rows = 0;

    for line in lines {
        let line = line?;
        let parts: Vec<f64> = line.split(',').map(|s| s.parse().unwrap()).collect();
        let (features, label) = parts.split_at(parts.len() - 1);
        n_features = features.len();
        rows.extend_from_slice(features);
        labels.push(label[0] as usize);
        n_rows += 1;
    }

    // from_row_slice attend des données aplaties "ligne par ligne" (ce qu'on a
    // construit ci-dessus), et gère la conversion vers le stockage column-major interne.
    let x = DMatrix::from_row_slice(n_rows, n_features, &rows);
    Ok((x, labels, classes))
}

pub fn save_model(model: &any_model::AnyModel, path: &str) -> Result<()> {
    let bytes = bincode::serialize(model)?;
    let mut file = File::create(path)?;
    file.write_all(&bytes)?;
    Ok(())
}

pub fn load_model(path: &str) -> Result<any_model::AnyModel> {
    let bytes = std::fs::read(path)?;
    Ok(bincode::deserialize(&bytes)?)
}
