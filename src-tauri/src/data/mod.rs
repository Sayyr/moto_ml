pub mod features;
pub mod loader;

/// Une ligne de dataset: vecteur de features + label (index de classe)
#[derive(Debug, Clone)]
pub struct Sample {
    pub features: Vec<f64>,
    pub label: usize, // index dans `classes`
}

#[derive(Debug, Clone)]
pub struct Dataset {
    pub samples: Vec<Sample>,
    pub classes: Vec<String>, // ex: ["sportive", "roadster", "trail", ...]
}
