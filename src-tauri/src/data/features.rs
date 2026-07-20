use anyhow::Result;
use image::{imageops::FilterType, DynamicImage};

/// Taille cible pour le resize (garde ça petit pour aller vite : 32x32 ou 64x64)
pub const IMG_SIZE: u32 = 32;

/// Charge une image, la redimensionne, et en extrait un vecteur de features,
/// en combinant trois approches complémentaires :
/// 1. Flatten des pixels RGB normalisés [0,1] en basse résolution (couleur brute)
/// 2. Histogramme de couleur (16 bins par canal RGB) -> robuste aux variations d'échelle/pose
/// 3. HOG (forme/silhouette, voir hog_features plus bas) -> le plus discriminant
///    pour distinguer des genres de moto qui se ressemblent en couleur
pub fn extract_features(path: &str) -> Result<Vec<f64>> {
    let img = image::open(path)?;
    let resized = img.resize_exact(IMG_SIZE, IMG_SIZE, FilterType::Triangle);

    let mut features = flatten_rgb(&resized);
    features.extend(color_histogram(&resized, 16));
    features.extend(hog_features(&resized));

    Ok(features)
}

fn flatten_rgb(img: &DynamicImage) -> Vec<f64> {
    let rgb = img.to_rgb8();
    rgb.pixels()
        .flat_map(|p| [p[0] as f64 / 255.0, p[1] as f64 / 255.0, p[2] as f64 / 255.0])
        .collect()
}

/// Histogramme de couleur normalisé, concaténé sur les 3 canaux RGB.
fn color_histogram(img: &DynamicImage, bins: usize) -> Vec<f64> {
    let rgb = img.to_rgb8();
    let mut hist = vec![0f64; bins * 3];
    let bin_width = 256.0 / bins as f64;

    for p in rgb.pixels() {
        for c in 0..3 {
            let bin = ((p[c] as f64) / bin_width) as usize;
            let bin = bin.min(bins - 1);
            hist[c * bins + bin] += 1.0;
        }
    }

    let total = (rgb.width() * rgb.height()) as f64;
    hist.iter_mut().for_each(|v| *v /= total);
    hist
}

/// HOG (Histogram of Oriented Gradients) — capture la forme/silhouette plutôt
/// que la couleur, souvent plus discriminant pour distinguer des genres de moto
/// que la seule couleur paramètres classiques (Dalal & Triggs, 2005) : cellules
/// de 8x8 px, blocs de 2x2 cellules avec chevauchement (stride = 1 cellule),
/// 9 bins d'orientation sur 0-180° (gradient "non signé" : une direction et son
/// opposée comptent pareil, ce qui rend HOG insensible à l'inversion noir/blanc
/// d'un contour).
///
/// ATTENTION : avec IMG_SIZE=32 et des cellules de 8px, on n'a que 4x4 cellules —
/// c'est un peu grossier pour capturer la silhouette d'une moto. Si les résultats
/// sont décevants, tenter d'augmenter IMG_SIZE (64) plutôt que de réduire la
/// taille des cellules (le ratio image/cellule compte plus que la résolution
/// absolue).
/// 
/// pour l'instant, img32 apportent des résultats plus satisfaisant que 64, 
/// donc on gaarde en 32 pour au moins la démo
pub fn hog_features(img: &DynamicImage) -> Vec<f64> {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();
    let cell_size: u32 = 8;
    let n_bins: usize = 9;
    let cells_x = (width / cell_size) as usize;
    let cells_y = (height / cell_size) as usize;

    let (magnitude, orientation) = compute_gradients(&gray);

    // Étape 1 : histogramme d'orientations pondéré par magnitude, pour chaque cellule.
    let mut cell_histograms = vec![vec![vec![0.0; n_bins]; cells_x]; cells_y];
    for cy in 0..cells_y {
        for cx in 0..cells_x {
            let mut hist = vec![0.0; n_bins];
            let y0 = cy * cell_size as usize;
            let x0 = cx * cell_size as usize;
            for py in y0..y0 + cell_size as usize {
                for px in x0..x0 + cell_size as usize {
                    let angle = orientation[py][px];
                    let mag = magnitude[py][px];
                    let bin = ((angle / 180.0 * n_bins as f64) as usize).min(n_bins - 1);
                    hist[bin] += mag;
                }
            }
            cell_histograms[cy][cx] = hist;
        }
    }

    // Étape 2 : blocs de 2x2 cellules avec chevauchement (stride = 1 cellule),
    // normalisation L2 par bloc, puis concaténation de tous les blocs.
    let eps = 1e-6;
    let mut hog: Vec<f64> = Vec::new();
    if cells_y >= 2 && cells_x >= 2 {
        for by in 0..cells_y - 1 {
            for bx in 0..cells_x - 1 {
                let mut block: Vec<f64> = Vec::with_capacity(4 * n_bins);
                for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    block.extend_from_slice(&cell_histograms[by + dy][bx + dx]);
                }

                let norm = block.iter().map(|v| v * v).sum::<f64>().sqrt();
                for v in block.iter_mut() {
                    *v /= norm + eps;
                }

                hog.extend(block);
            }
        }
    }

    hog
}

/// Calcule la magnitude et l'orientation du gradient à chaque pixel.
/// Retour : deux tableaux de même taille que l'image, indexés [y][x].
fn compute_gradients(gray: &image::GrayImage) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let (width, height) = gray.dimensions();
    let (w, h) = (width as usize, height as usize);
    let mut magnitude = vec![vec![0.0; w]; h];
    let mut orientation = vec![vec![0.0; w]; h];

    // On reste à l'intérieur de l'image (1..w-1, 1..h-1) pour toujours avoir un
    // voisin de chaque côté ; les pixels du bord restent à 0 (magnitude nulle)
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let left = gray.get_pixel((x - 1) as u32, y as u32).0[0] as f64;
            let right = gray.get_pixel((x + 1) as u32, y as u32).0[0] as f64;
            let up = gray.get_pixel(x as u32, (y - 1) as u32).0[0] as f64;
            let down = gray.get_pixel(x as u32, (y + 1) as u32).0[0] as f64;

            let gx = right - left;
            let gy = down - up;

            magnitude[y][x] = (gx * gx + gy * gy).sqrt();

            let angle_degres = gy.atan2(gx).to_degrees(); // entre -180 et 180
            orientation[y][x] = ((angle_degres % 180.0) + 180.0) % 180.0; // ramené sur [0, 180)
        }
    }

    (magnitude, orientation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    /// Reprend exactement l'exemple calculé à la main plus tôt : un contour vertical net
    /// (colonne de gauche à 50, colonne de droite à 200). Le pixel central doit
    /// avoir Gx=150, Gy=0, donc magnitude=150 et orientation=0°.
    #[test]
    fn test_compute_gradients_vertical_edge() {
        let values = [[50u8, 50, 200], [50, 50, 200], [50, 50, 200]];
        let mut img = GrayImage::new(3, 3);
        for y in 0..3 {
            for x in 0..3 {
                img.put_pixel(x as u32, y as u32, Luma([values[y][x]]));
            }
        }

        let (magnitude, orientation) = compute_gradients(&img);

        assert!((magnitude[1][1] - 150.0).abs() < 1e-6, "magnitude attendue 150, obtenue {}", magnitude[1][1]);
        assert!((orientation[1][1] - 0.0).abs() < 1e-6, "orientation attendue 0°, obtenue {}", orientation[1][1]);
    }
}
