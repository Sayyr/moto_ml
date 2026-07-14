use super::{features::extract_features, Dataset, Sample};
use anyhow::Result;
use std::fs::File;
use std::io::Write;
use walkdir::WalkDir;

/// Parcourt `input_dir/<classe>/*.jpg` et construit le Dataset.
/// Structure attendue :
///   dataset/
///     sportive/xxx.jpg
///     roadster/yyy.jpg
///     trail/zzz.jpg
///     ...
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

// TODO : ajouter une fonction split_train_val_test(dataset, 0.7, 0.15, 0.15)
// qui mélange (shuffle) puis découpe le Vec<Sample>.
