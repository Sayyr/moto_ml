use super::{features::extract_features, Dataset, Sample};
use anyhow::Result;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::fs::File;
use std::io::Write;
use walkdir::WalkDir;

/// Parcourt `input_dir/<classe>/*.jpg` et construit le Dataset.
/// Structure attendue (peu importe la profondeur sous chaque classe, WalkDir
/// est récursif — donc `cruiser/harley-davidson/fat_boy/xxx.jpg` fonctionne
/// aussi bien que `cruiser/xxx.jpg`) :
///   dataset/
///     cruiser/...
///     sport/...
///     trail/...
pub fn load_dataset(input_dir: &str) -> Result<Dataset> {
    let mut classes: Vec<String> = std::fs::read_dir(input_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    classes.sort();

    let mut samples = Vec::new();

    for (label, class_name) in classes.iter().enumerate() {
        let class_dir = format!("{}/{}", input_dir, class_name);
        for entry in WalkDir::new(&class_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path().to_string_lossy().to_string();
            match extract_features(&path) {
                Ok(f) => samples.push(Sample { features: f, label }),
                Err(e) => eprintln!("skip {} ({})", path, e),
            }
        }
    }

    Ok(Dataset { samples, classes })
}

/// Découpe un dataset déjà chargé en train/val/test, de façon STRATIFIÉE :
/// le split est fait indépendamment à l'intérieur de chaque classe, puis
/// recombiné, ce qui garantit que les proportions train/val/test sont
/// respectées classe par classe (et pas juste globalement — important si une
/// classe est plus petite que les autres, pour éviter de la sous-représenter
/// dans le test set par pur hasard).
///
/// `seed` fixe rend le split reproductible : deux imports successifs du même
/// dataset avec la même seed donnent exactement le même découpage.
pub fn split_train_val_test(dataset: Dataset, train_ratio: f64, val_ratio: f64, seed: u64) -> (Dataset, Dataset, Dataset) {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_classes = dataset.classes.len();

    // Regroupe les indices des échantillons par classe
    let mut by_class: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for (i, s) in dataset.samples.iter().enumerate() {
        by_class[s.label].push(i);
    }

    let mut train_idx = Vec::new();
    let mut val_idx = Vec::new();
    let mut test_idx = Vec::new();

    for indices in by_class.iter_mut() {
        indices.shuffle(&mut rng);
        let n = indices.len();
        let n_train = (n as f64 * train_ratio).round() as usize;
        let n_val = (n as f64 * val_ratio).round() as usize;
        let val_end = (n_train + n_val).min(n);

        train_idx.extend_from_slice(&indices[..n_train.min(n)]);
        val_idx.extend_from_slice(&indices[n_train.min(n)..val_end]);
        test_idx.extend_from_slice(&indices[val_end..]);
    }

    let build = |idx: Vec<usize>| Dataset {
        samples: idx.into_iter().map(|i| dataset.samples[i].clone()).collect(),
        classes: dataset.classes.clone(),
    };

    (build(train_idx), build(val_idx), build(test_idx))
}

/// Extrait le dataset et le sauvegarde en CSV : dernière colonne = label,
/// première ligne = header avec les noms de classes en commentaire.
pub fn extract_and_save(input_dir: &str, output: &str) -> Result<()> {
    let dataset = load_dataset(input_dir)?;
    let mut file = File::create(output)?;

    writeln!(file, "# classes: {}", dataset.classes.join(","))?;
    for sample in &dataset.samples {
        let row: Vec<String> = sample.features.iter().map(|f| f.to_string()).collect();
        writeln!(file, "{},{}", row.join(","), sample.label)?;
    }

    println!(
        "{} échantillons extraits, {} classes -> {}",
        dataset.samples.len(),
        dataset.classes.len(),
        output
    );
    Ok(())
}
