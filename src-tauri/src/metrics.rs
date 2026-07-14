
/// Affiche une matrice de confusion simple dans le terminal.
pub fn confusion_matrix(preds: &[usize], targets: &[usize], classes: &[String]) {
    let n = classes.len();
    let mut matrix = vec![vec![0usize; n]; n];

    for (p, t) in preds.iter().zip(targets.iter()) {
        matrix[*t][*p] += 1;
    }

    print!("{:>12}", "vrai\\prédit");
    for c in classes {
        print!("{:>10}", &c[..c.len().min(9)]);
    }
    println!();

    for (i, row) in matrix.iter().enumerate() {
        print!("{:>12}", &classes[i][..classes[i].len().min(11)]);
        for v in row {
            print!("{v:>10}");
        }
        println!();
    }
}

/// Precision/recall/F1 par classe
pub fn per_class_metrics(preds: &[usize], targets: &[usize], n_classes: usize) {
    for c in 0..n_classes {
        let tp = preds.iter().zip(targets.iter()).filter(|(&p, &t)| p == c && t == c).count();
        let fp = preds.iter().zip(targets.iter()).filter(|(&p, &t)| p == c && t != c).count();
        let fn_ = preds.iter().zip(targets.iter()).filter(|(&p, &t)| p != c && t == c).count();

        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let recall = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        println!("classe {c}: precision={precision:.2} recall={recall:.2} f1={f1:.2}");
    }
}
