import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

// Correspond exactement au menu ASCII fourni : index -> action
const MENU_ITEMS = [
  { id: 0, label: "Importer un dataset", view: "import-dataset" },
  { id: 1, label: "Afficher le dataset", view: "show-dataset" },
  { id: 2, label: "Régression Linéaire", view: "model", kind: "LinearRegression" },
  { id: 3, label: "Classification Linéaire", view: "model", kind: "LogisticRegression" },
  { id: 4, label: "MLP", view: "model", kind: "Mlp" },
  { id: 5, label: "SVM", view: "model", kind: "Svm" },
  { id: 6, label: "RBF", view: "model", kind: "Rbf" },
  { id: 7, label: "Full test d'une inférence", view: "full-test" },
  { id: 8, label: "Exporter un modèle entraîné", view: "export" },
  { id: 9, label: "Quitter", view: "quit" },
];

const menuEl = document.getElementById("menu");
const contentEl = document.getElementById("content");
const statusEl = document.getElementById("status");

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.className = "status" + (isError ? " error" : "");
}

function renderMenu() {
  menuEl.innerHTML = "";
  MENU_ITEMS.forEach((item) => {
    const btn = document.createElement("button");
    btn.className = "menu-item";
    btn.textContent = `${item.id}. ${item.label}`;
    btn.onclick = () => selectMenuItem(item);
    menuEl.appendChild(btn);
  });
}

function selectMenuItem(item) {
  document.querySelectorAll(".menu-item").forEach((b) => b.classList.remove("active"));
  event?.target?.classList.add("active");

  switch (item.view) {
    case "import-dataset":
      renderImportDataset();
      break;
    case "show-dataset":
      renderShowDataset();
      break;
    case "model":
      renderModelView(item.kind, item.label);
      break;
    case "full-test":
      renderFullTest();
      break;
    case "export":
      renderExport();
      break;
    case "quit":
      window.close();
      break;
  }
}

// ── 0. Importer un dataset ──────────────────────────────────────
function renderImportDataset() {
  contentEl.innerHTML = `
    <h2>Importer un dataset</h2>
    <p>Choisis le dossier racine contenant un sous-dossier par genre de moto
    (ex: <code>dataset/sportive/</code>, <code>dataset/trail/</code>, ...).
    Un split train/validation/test (70/15/15, stratifié par classe) est
    calculé automatiquement à l'import.</p>
    <button id="pick-dir">Choisir un dossier</button>
    <div id="import-result"></div>
  `;
  document.getElementById("pick-dir").onclick = async () => {
    const dir = await open({ directory: true, multiple: false });
    if (!dir) return;
    setStatus("Import, split et extraction des features en cours (peut prendre un moment)...");
    try {
      const info = await invoke("import_dataset", { dirPath: dir });
      setStatus(`Dataset importé : ${info.train.n_samples} train / ${info.val.n_samples} val / ${info.test.n_samples} test.`);
      document.getElementById("import-result").innerHTML = `
        <h3>Train</h3>${renderDatasetInfo(info.train)}
        <h3>Validation</h3>${renderDatasetInfo(info.val)}
        <h3>Test</h3>${renderDatasetInfo(info.test)}
      `;
    } catch (e) {
      setStatus(`Erreur : ${e}`, true);
    }
  };
}

// ── 1. Afficher le dataset ──────────────────────────────────────
async function renderShowDataset() {
  contentEl.innerHTML = `<h2>Dataset actuel</h2><div id="dataset-info">Chargement...</div>`;
  try {
    const info = await invoke("get_dataset_info");
    document.getElementById("dataset-info").innerHTML = renderDatasetInfo(info);
  } catch (e) {
    document.getElementById("dataset-info").innerHTML = `<p class="error">${e}</p>`;
  }
}

function renderDatasetInfo(info) {
  const rows = info.classes
    .map((c, i) => `<tr><td>${c}</td><td>${info.counts[i]}</td></tr>`)
    .join("");
  return `
    <p><strong>${info.n_samples}</strong> images, <strong>${info.classes.length}</strong> classes.</p>
    <table><thead><tr><th>Genre</th><th>Nb images</th></tr></thead><tbody>${rows}</tbody></table>
  `;
}

// ── 2-6. Entraîner / Utiliser / Modifier un modèle ──────────────
function renderModelView(kind, label) {
  contentEl.innerHTML = `
    <h2>${label}</h2>
    <div class="tabs">
      <button class="tab active" data-tab="train">Entraîner</button>
      <button class="tab" data-tab="use">Utiliser</button>
      <button class="tab" data-tab="edit">Modifier</button>
      <button class="tab" data-tab="evaluate">Évaluer</button>
    </div>
    <div id="tab-content"></div>
  `;

  const tabContent = document.getElementById("tab-content");
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.onclick = () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      if (tab.dataset.tab === "train") renderTrainTab(tabContent, kind);
      if (tab.dataset.tab === "use") renderUseTab(tabContent, kind);
      if (tab.dataset.tab === "edit") renderEditTab(tabContent, kind);
      if (tab.dataset.tab === "evaluate") renderEvaluateTab(tabContent, kind);
    };
  });

  renderTrainTab(tabContent, kind);
}

function renderTrainTab(container, kind) {
  const showMlpParams = kind === "Mlp";
  const showRbfParams = kind === "Rbf";
  const showSvmParams = kind === "Svm";

  container.innerHTML = `
    <div class="form">
      <label>Learning rate <input id="p-lr" type="number" step="0.01" value="0.05"></label>
      <label>Epochs <input id="p-epochs" type="number" value="200"></label>
      ${showMlpParams ? `
        <label>Batch size <input id="p-batch" type="number" value="32"></label>
        <label>Couches cachées (ex: 64,32) <input id="p-hidden" type="text" value="64,32"></label>
      ` : ""}
      ${showRbfParams ? `
        <label>Nombre de centres <input id="p-centers" type="number" value="20"></label>
        <label>Sigma <input id="p-sigma" type="number" step="0.1" value="1.0"></label>
      ` : ""}
      ${showSvmParams ? `
        <label>Lambda (régularisation) <input id="p-lambda" type="number" step="0.001" value="0.01"></label>
      ` : ""}
      <button id="train-btn">Lancer l'entraînement</button>
    </div>
    <div id="train-result"></div>
  `;

  document.getElementById("train-btn").onclick = async () => {
    const params = {
      lr: parseFloat(document.getElementById("p-lr").value),
      epochs: parseInt(document.getElementById("p-epochs").value),
      batch_size: showMlpParams ? parseInt(document.getElementById("p-batch").value) : 32,
      hidden_layers: showMlpParams
        ? document.getElementById("p-hidden").value.split(",").map((s) => parseInt(s.trim()))
        : [64, 32],
      n_centers: showRbfParams ? parseInt(document.getElementById("p-centers").value) : 20,
      sigma: showRbfParams ? parseFloat(document.getElementById("p-sigma").value) : 1.0,
      lambda: showSvmParams ? parseFloat(document.getElementById("p-lambda").value) : 0.01,
    };

    setStatus("Entraînement en cours...");
    try {
      const result = await invoke("train_model", { modelKind: kind, params });
      setStatus(`Modèle "${result.model_kind}" entraîné.`);
      document.getElementById("train-result").innerHTML = `
        <p>Accuracy (train) : <strong>${(result.train_accuracy * 100).toFixed(1)}%</strong></p>
      `;
    } catch (e) {
      setStatus(`Erreur : ${e}`, true);
    }
  };
}

function renderUseTab(container, kind) {
  container.innerHTML = `
    <p>Choisis une image de moto pour tester ce modèle.</p>
    <button id="pick-image">Choisir une image</button>
    <div id="predict-result"></div>
  `;
  document.getElementById("pick-image").onclick = async () => {
    const file = await open({ multiple: false, filters: [{ name: "Images", extensions: ["jpg", "jpeg", "png"] }] });
    if (!file) return;
    setStatus("Prédiction en cours...");
    try {
      const result = await invoke("run_inference", { modelKind: kind, imagePath: file });
      setStatus("Prédiction terminée.");
      document.getElementById("predict-result").innerHTML = renderPrediction(result);
    } catch (e) {
      setStatus(`Erreur : ${e}`, true);
    }
  };
}

function renderPrediction(result) {
  const rows = result.probabilities
    .map(([c, p]) => `<tr><td>${c}</td><td>${(p * 100).toFixed(1)}%</td></tr>`)
    .join("");
  return `
    <p>Classe prédite : <strong>${result.predicted_class}</strong></p>
    <table><thead><tr><th>Genre</th><th>Probabilité</th></tr></thead><tbody>${rows}</tbody></table>
  `;
}

function renderEditTab(container, kind) {
  container.innerHTML = `
    <p>Continue l'entraînement du modèle "${kind}" déjà chargé en mémoire
    (entraîne-le d'abord dans l'onglet "Entraîner" si ce n'est pas fait).</p>
    <label>Epochs supplémentaires <input id="p-extra" type="number" value="100"></label>
    <button id="continue-btn">Continuer l'entraînement</button>
    <div id="continue-result"></div>
  `;
  document.getElementById("continue-btn").onclick = async () => {
    setStatus("Ré-entraînement en cours...");
    try {
      const extraEpochs = parseInt(document.getElementById("p-extra").value);
      const result = await invoke("continue_training", { modelKind: kind, extraEpochs });
      setStatus("Ré-entraînement terminé.");
      document.getElementById("continue-result").innerHTML = `
        <p>Nouvelle accuracy (train) : <strong>${(result.train_accuracy * 100).toFixed(1)}%</strong></p>
      `;
    } catch (e) {
      setStatus(`Erreur : ${e}`, true);
    }
  };
}

function renderEvaluateTab(container, kind) {
  container.innerHTML = `
    <p>Évalue le modèle "${kind}" (déjà entraîné) sur le jeu de <strong>test</strong> —
    des images jamais vues pendant l'entraînement, mises de côté automatiquement
    à l'import du dataset (menu 0).</p>
    <button id="eval-btn">Évaluer sur le test set</button>
    <div id="eval-result"></div>
  `;
  document.getElementById("eval-btn").onclick = async () => {
    setStatus("Évaluation en cours...");
    try {
      const result = await invoke("evaluate_model", { modelKind: kind });
      setStatus(`Évaluation terminée : ${(result.test_accuracy * 100).toFixed(1)}% sur ${result.n_test_samples} images de test.`);
      document.getElementById("eval-result").innerHTML = renderEvalResult(result);
    } catch (e) {
      setStatus(`Erreur : ${e}`, true);
    }
  };
}

function renderEvalResult(result) {
  const header = `<th></th>` + result.class_names.map((c) => `<th>préd. ${c}</th>`).join("");
  const rows = result.confusion_matrix
    .map((row, i) => `<tr><td><strong>vrai ${result.class_names[i]}</strong></td>${row.map((v) => `<td>${v}</td>`).join("")}</tr>`)
    .join("");

  return `
    <p>Accuracy (test, ${result.n_test_samples} images) : <strong>${(result.test_accuracy * 100).toFixed(1)}%</strong></p>
    <h3>Matrice de confusion</h3>
    <table><thead><tr>${header}</tr></thead><tbody>${rows}</tbody></table>
  `;
}

// ── 7. Full test d'une inférence ────────────────────────────────
function renderFullTest() {
  contentEl.innerHTML = `
    <h2>Full test d'une inférence</h2>
    <p>Compare les prédictions de tous les modèles actuellement entraînés sur une même image.</p>
    <button id="pick-image-full">Choisir une image</button>
    <div id="full-result"></div>
  `;
  document.getElementById("pick-image-full").onclick = async () => {
    const file = await open({ multiple: false, filters: [{ name: "Images", extensions: ["jpg", "jpeg", "png"] }] });
    if (!file) return;
    setStatus("Comparaison en cours...");
    try {
      const results = await invoke("full_test_inference", { imagePath: file });
      setStatus("Comparaison terminée.");
      document.getElementById("full-result").innerHTML = results
        .map(([kind, res]) => `<h3>${kind}</h3>${renderPrediction(res)}`)
        .join("");
    } catch (e) {
      setStatus(`Erreur : ${e}`, true);
    }
  };
}

// ── 8. Exporter un modèle entraîné ──────────────────────────────
function renderExport() {
  contentEl.innerHTML = `
    <h2>Exporter un modèle entraîné</h2>
    <label>Modèle
      <select id="export-kind">
        <option value="LinearRegression">Régression Linéaire</option>
        <option value="LogisticRegression">Classification Linéaire</option>
        <option value="Mlp">MLP</option>
        <option value="Svm">SVM</option>
        <option value="Rbf">RBF</option>
      </select>
    </label>
    <button id="export-btn">Choisir où enregistrer</button>
  `;
  document.getElementById("export-btn").onclick = async () => {
    const kind = document.getElementById("export-kind").value;
    const path = await open({ directory: false, multiple: false, save: true }).catch(() => null);
    // NOTE : pour un vrai save dialog, utilise `save()` de @tauri-apps/plugin-dialog
    // plutôt que `open()` — à corriger, `open` sert normalement à choisir un fichier existant.
    if (!path) return;
    try {
      await invoke("export_model", { modelKind: kind, outputPath: path });
      setStatus(`Modèle "${kind}" exporté vers ${path}`);
    } catch (e) {
      setStatus(`Erreur : ${e}`, true);
    }
  };
}

renderMenu();
