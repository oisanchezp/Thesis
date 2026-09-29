"""Promote the verified notebook/figures with backups and a concurrent-edit check."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

chapter = Path('/home/oisanchezp/Thesis/src/06_Chapter').resolve()
review = chapter / 'revision_diseno_20260924'
ready = review / 'revisado'
backup = review / 'originales'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
main = chapter / 'umbrales_gam_cap6.ipynb'
expected = sys.argv[1]
assert sha(main) == expected, 'Notebook changed since download; stop to preserve concurrent edits.'
assert sha(backup/main.name) == expected, 'Original backup is not the downloaded notebook.'
tables = []
for a in sorted((review/'baseline/salidas').glob('*.csv')):
    pd.testing.assert_frame_equal(pd.read_csv(a), pd.read_csv(ready/'salidas'/a.name), check_exact=True)
    tables.append(a.name)
pd.testing.assert_frame_equal(pd.read_csv(review/'baseline/reliability_check.csv'),
                               pd.read_csv(ready/'reliability_check.csv'), check_exact=True)
layout = json.loads((ready/'layout_review.json').read_text())
assert len(layout) == 4
assert all(not row['off_canvas_text'] for row in layout)
assert all(abs(row['width_in']*2.54-16) < 1e-10 and row['dpi'] == 300 for row in layout)
nb = json.loads((ready/main.name).read_text())
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), f'cell_{i}', 'exec')
        assert cell['execution_count'] is not None
        assert not any(o['output_type'] == 'error' for o in cell['outputs'])
names = ['fig_gam_desempeno', 'fig_gam_efectos', 'fig_gam_roc_calibracion', 'fig_gam_isolineas']
relative = [Path('00Figuras/06Seccion') / f'{name}_excl10d_libre.{ext}'
            for name in names for ext in ['pdf', 'png']]
relative += [Path('salidas')/f'tabla_res_gam_{kind}_excl10d_libre.csv'
             for kind in ['calibracion', 'distribucion_probabilidades']]
relative.append(Path(main.name))
manifest = []
# Preserve every existing destination before overwriting any destination.
for rel in relative:
    dst, original = chapter/rel, backup/rel
    assert (ready/rel).is_file(), f'Missing output {rel}'
    assert dst.resolve().is_relative_to(chapter)
    assert original.resolve().is_relative_to(backup)
    if dst.exists():
        original.parent.mkdir(parents=True, exist_ok=True)
        if original.exists():
            assert sha(original) == sha(dst), f'Backup collision: {original}'
        else:
            shutil.copy2(dst, original)
for rel in relative:
    src, dst = ready/rel, chapter/rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    assert sha(src) == sha(dst)
    manifest.append({'path': str(dst), 'sha256': sha(dst)})
result = {'published_at_utc': datetime.now(timezone.utc).isoformat(),
          'backup': str(backup), 'original_notebook_sha256': expected,
          'unchanged_tables': tables, 'reliability_unchanged': True, 'published': manifest}
(review/'publication_manifest.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
