"""Execute the ordinary-Python notebook cells, saving outputs and layout checks.

No kernel packages are installed or model fitting performed. All input CSVs stay read-only.
Usage: python execute_review.py notebook.ipynb output_directory
"""
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from PIL import Image

src = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
os.chdir(out)
nb = json.loads(src.read_text(encoding='utf-8'))
scope = {'__name__': '__main__'}
layout = []
current_images = []

def capture_show(*args, **kwargs):
    for number in plt.get_fignums():
        fig = plt.figure(number)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        off_canvas = []
        for artist in fig.findobj(match=Text):
            if not artist.get_visible() or not artist.get_text() or artist.get_clip_on():
                continue
            box = artist.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
                off_canvas.append(artist.get_text())
        layout.append({'cell': cell_index, 'width_in': fig.get_figwidth(),
                       'height_in': fig.get_figheight(), 'dpi': fig.dpi,
                       'off_canvas_text': off_canvas})
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches=None)
        current_images.append({'output_type': 'display_data',
                               'data': {'image/png': base64.b64encode(buf.getvalue()).decode('ascii'),
                                        'text/plain': [f'<Figure cell {cell_index}>']},
                               'metadata': {}})
        plt.close(fig)
plt.show = capture_show

execution = 0
for cell_index, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    execution += 1
    current_images = []
    stdout, stderr = io.StringIO(), io.StringIO()
    print(f'Executing cell {cell_index}', flush=True)
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        exec(compile(''.join(cell['source']), f'{src.name}:cell_{cell_index}', 'exec'), scope)
    cell['execution_count'] = execution
    cell['outputs'] = []
    if stdout.getvalue():
        cell['outputs'].append({'output_type': 'stream', 'name': 'stdout',
                                'text': stdout.getvalue().splitlines(keepends=True)})
    if stderr.getvalue():
        cell['outputs'].append({'output_type': 'stream', 'name': 'stderr',
                                'text': stderr.getvalue().splitlines(keepends=True)})
    cell['outputs'].extend(current_images)
    with (out/'execution.log').open('a', encoding='utf-8') as log:
        log.write(f'\nCELL {cell_index}\n{stdout.getvalue()}{stderr.getvalue()}')

nb['metadata']['execution_review'] = {'python': sys.version.split()[0],
                                     'matplotlib': matplotlib.__version__,
                                     'method': 'all Python cells in order; Agg output capture'}
(out / src.name).write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
(out / 'layout_review.json').write_text(json.dumps(layout, indent=2), encoding='utf-8')
scope['rel'].to_csv(out/'reliability_check.csv')
print(json.dumps(layout, indent=2))
for png in sorted((out/'00Figuras/06Seccion').glob('*.png')):
    with Image.open(png) as im:
        print(png.name, im.size, im.info.get('dpi'))
print('COMPLETE', out)
