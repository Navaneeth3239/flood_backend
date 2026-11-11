"""Cold-start regression using the files actually allowed into the Vercel bundle."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_deployed_bundle_starts_without_optional_ml_packages(tmp_path):
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "vercel.json").read_text())
    patterns = config["functions"]["api/index.py"]["excludeFiles"].strip("{}").split(",")
    patterns += [line.strip() for line in (root / ".vercelignore").read_text().splitlines()
                 if line.strip() and not line.startswith("#")]
    for folder in ("backend", "api"):
        for source in (root / folder).rglob("*"):
            if not source.is_file() or source.suffix not in {".py", ".json", ".gz"}:
                continue
            relative = source.relative_to(root)
            if any(relative.match(pattern) or relative.as_posix().startswith(pattern.rstrip("*"))
                   for pattern in patterns):
                continue
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    script = '''
import importlib.abc, sys
class LeanRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','pandas','shapely','rasterio','scipy','ee','yaml'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, LeanRuntime())
from api.index import app
from fastapi.testclient import TestClient
import json
from services import area_map_store
from services.bundled_area import DIRECTORY
def offline(*args):
    raise AssertionError('Bundled layers must not access Supabase')
area_map_store.load_layers = offline
with TestClient(app) as client:
    response = client.get('/api/health')
    assert response.status_code == 200, response.text
    response = client.get('/api/digital-twin/multi-hazard/model-info')
    assert response.status_code == 200, response.text
    assert response.json()['ready'] is False
    entry = json.loads((DIRECTORY / 'manifest.json').read_text())[0]
    payload = {'area_id': entry['area_id'], 'polygon': entry['polygon']}
    response = client.post('/api/geo/extract-networks', json=payload)
    assert response.status_code == 200, response.text
    for layer, count in entry['counts'].items():
        assert len(response.json()[layer]['geojson']['features']) == count
    response = client.post('/api/geo/extract-buildings', json=payload)
    assert response.status_code == 200
    assert response.json()['buildings']['total_features'] == 2647
'''
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "MULTI_HAZARD_MODEL_CHECKPOINT"}}
    env.update(VERCEL="1", PYTHON_DOTENV_DISABLED="1")
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
