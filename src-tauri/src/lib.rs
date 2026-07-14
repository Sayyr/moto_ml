mod commands;
pub mod data;
pub mod metrics;
pub mod models;
mod state;

use state::AppState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            commands::import_dataset,
            commands::get_dataset_info,
            commands::train_model,
            commands::run_inference,
            commands::continue_training,
            commands::full_test_inference,
            commands::export_model,
            commands::import_model,
            commands::list_trained_models,
        ])
        .run(tauri::generate_context!())
        .expect("erreur au lancement de l'application Tauri");
}
