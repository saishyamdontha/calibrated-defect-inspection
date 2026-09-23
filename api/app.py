"""FastAPI inspection service.

Run from the project root:
  python -m uvicorn api.app:app --port 8000
Then open http://127.0.0.1:8000
"""
import io
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from src.inference import Inspector

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
inspectors = {}


def finite_or_none(x):
    return x if isinstance(x, float) and math.isfinite(x) else None


@asynccontextmanager
async def lifespan(app):
    for path in sorted(MODELS_DIR.glob("*.pt")):
        inspectors[path.stem] = Inspector(path)
        print(f"Loaded model {path.stem} on {inspectors[path.stem].device}")
    if not inspectors:
        print(f"No models found in {MODELS_DIR}. Run src.export_model first.")
    yield


app = FastAPI(title="Calibrated Defect Inspection", lifespan=lifespan)


@app.get("/models")
def list_models():
    return [
        {
            "id": key,
            "category": ins.config["category"],
            "backbone": ins.config["backbone"],
            "image_size": ins.config["image_size"],
            "t_ok": finite_or_none(ins.t_ok),
            "t_defect": finite_or_none(ins.t_defect),
        }
        for key, ins in inspectors.items()
    ]


@app.post("/inspect")
def inspect(model_id: str = Form(...), file: UploadFile = File(...)):
    if model_id not in inspectors:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
    try:
        image = Image.open(io.BytesIO(file.file.read()))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="File is not a readable image.")
    result = inspectors[model_id].inspect(image)
    result["t_ok"] = finite_or_none(result["t_ok"])
    result["t_defect"] = finite_or_none(result["t_defect"])
    result["model_id"] = model_id
    return result


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Calibrated Defect Inspection</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1d1d1f; }
  h1 { font-size: 1.4rem; font-weight: 600; }
  .controls { display: flex; gap: .75rem; flex-wrap: wrap; align-items: center; margin: 1rem 0; }
  select, button, input { font-size: 1rem; padding: .4rem .6rem; }
  button { cursor: pointer; }
  .badge { display: inline-block; padding: .35rem .8rem; border-radius: 999px; font-weight: 600; font-size: 1.1rem; }
  .ok { background: #e3f4e5; color: #1e6b2a; }
  .unsure { background: #fdf1d6; color: #8a5a00; }
  .defect { background: #fbe1e1; color: #9b1c1c; }
  .images { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1rem; }
  .images figure { margin: 0; }
  .images img { width: 320px; max-width: 100%; border: 1px solid #ddd; border-radius: 6px; }
  .meta { color: #555; margin-top: .5rem; font-size: .95rem; }
  .error { color: #9b1c1c; }
</style>
</head>
<body>
<h1>Calibrated Defect Inspection</h1>
<p>Upload a part image. The system answers OK, defect, or unsure (send to human inspection).</p>
<div class="controls">
  <select id="model"></select>
  <input type="file" id="file" accept="image/*">
  <button id="go">Inspect</button>
</div>
<div id="error" class="error"></div>
<div id="result" hidden>
  <span id="badge" class="badge"></span>
  <div class="meta" id="meta"></div>
  <div class="images">
    <figure><img id="orig" alt="Uploaded image"><figcaption>Uploaded</figcaption></figure>
    <figure><img id="heat" alt="Anomaly heatmap"><figcaption>Anomaly heatmap</figcaption></figure>
  </div>
</div>
<script>
const labels = { ok: "OK", unsure: "Unsure: send to human", defect: "Defect" };
async function loadModels() {
  const res = await fetch("/models");
  const models = await res.json();
  const sel = document.getElementById("model");
  if (!models.length) { document.getElementById("error").textContent = "No models loaded on the server."; return; }
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `${m.category} | ${m.backbone} | ${m.image_size}px`;
    sel.appendChild(opt);
  }
}
document.getElementById("go").addEventListener("click", async () => {
  const err = document.getElementById("error");
  err.textContent = "";
  const file = document.getElementById("file").files[0];
  if (!file) { err.textContent = "Choose an image first."; return; }
  const form = new FormData();
  form.append("model_id", document.getElementById("model").value);
  form.append("file", file);
  const res = await fetch("/inspect", { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) { err.textContent = data.detail || "Inspection failed."; return; }
  const badge = document.getElementById("badge");
  badge.className = "badge " + data.decision;
  badge.textContent = labels[data.decision] || data.decision;
  const fmt = v => v === null ? "n/a" : v.toFixed(3);
  document.getElementById("meta").textContent =
    `score ${data.score.toFixed(3)} | OK below ${fmt(data.t_ok)} | defect above ${fmt(data.t_defect)} | ${data.latency_ms.toFixed(1)} ms on ${data.device}`;
  document.getElementById("orig").src = URL.createObjectURL(file);
  document.getElementById("heat").src = "data:image/png;base64," + data.heatmap_png;
  document.getElementById("result").hidden = false;
});
loadModels();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
