"""Construye una copia independiente; no escribe sobre el script original."""
from pathlib import Path
import difflib

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "mapas_alerta_cap6.py"
text = original = ORIGINAL.read_text(encoding="utf-8")


def replace_once(old, new):
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Esperaba 1 coincidencia, encontré {count}: {old[:100]!r}")
    text = text.replace(old, new, 1)


start = text.index('"""')
end = text.index('"""', start + 3) + 3
text = text[:start] + '''"""Mapas del capítulo 6 con lluvia interpolada en malla de 100 m.

Copia independiente de mapas_alerta_cap6.py. Acumula por estación con las
funciones originales, interpola componentes temporales no solapados usando
kriging esférico log1p y obtiene medias de área por SU. Conserva GAM/cortes/matriz.
Respeta LTR_DEF del modelo: R30-R1 son 29 días previos, R31-R1 son 30 días.
Los campos R30/RLTR se reconstruyen sumando componentes interpolados en mm.
Incluye comparación retrospectiva con la asignación al pluviómetro más cercano.
El hietograma corresponde a una estación de referencia, no a lluvia media de SU.
Ver LEEME.md para método, dependencias, archivos y límites de las comprobaciones.
"""''' + text[end:]
replace_once('import mc_common as mcc  # noqa: E402',
             'import mc_common as mcc  # noqa: E402\nfrom lluvia_interpolada import ZonalArea, interpolar_evento')
replace_once('OUT_DIR     = _env("CAP6_OUT_MAPAS", f"{CH6_DIR}/Mapas_alerta/")',
             'OUT_DIR     = _env("CAP6_OUT_INTERPOLADO", f"{CH6_DIR}/Mapas_alerta_kriging100m/")')
replace_once('FIGURAS       = ["alerta", "clase_lluvia"]',
             'FIGURAS       = ["alerta", "clase_lluvia", "lluvia"]')
replace_once('def evaluar_hora(ts, su, comp, cortes, cfg, lluvia, fase) -> pd.DataFrame:',
             'def evaluar_hora(ts, su, comp, cortes, cfg, lluvia, fase, por_su=True) -> pd.DataFrame:')
replace_once('    cod = su["Codigo_pluvio"]',
             '    cod = su[COL_ID] if por_su else su["Codigo_pluvio"]')
replace_once('    `lluvia` trae las filas de la hora ts de las tablas R1, R30 y RLTR.',
             '    `lluvia` trae R1, R30 y RLTR: índice ID de SU (o estación si por_su=False).')
replace_once('return pd.DataFrame({"R1": R1, "R30": R30, "p": p, "clase": clase, "nivel": nivel},',
             'return pd.DataFrame({"R1": R1, "R30": R30, "LTR": LTR, "p": p, "clase": clase, "nivel": nivel},')
replace_once('label="No valid rain gauge"', 'label="No valid rainfall estimate"')
replace_once('f"Rain gauge {ref[\'cod\']} ({ref[\'dist_km\']:.1f} km away)"',
             'f"Reference gauge {ref[\'cod\']} ({ref[\'dist_km\']:.1f} km away)"')
replace_once('"Codigo_pluvio": su["Codigo_pluvio"].to_numpy(),',
             '"Codigo_pluvio_referencia": su["Codigo_pluvio"].to_numpy(),')
replace_once('def correr_evento(tag, su, comp, cortes, cfg, cache_series, gpl, fase_oni, capas):',
             'def correr_evento(tag, su, comp, cortes, cfg, cache_series, gpl, fase_oni, capas, zonal):')
replace_once('    # --- fase ENSO de cada hora ---', '''    # La estación más cercana se conserva sólo como referencia/comparación.
    tablas_su = interpolar_evento(
        horas, tablas, validos, gpl, su, cfg, COL_ID,
        os.path.join(OUT_DIR, f"lluvia_{tag}"), zonal=zonal,
        guardar_tif=_env("CAP6_GUARDAR_TIF", "1") == "1")

    # --- fase ENSO de cada hora ---''')
replace_once('                                   {n: t.loc[ts] for n, t in tablas.items()}, fases[ts])',
             '                                   {n: t.loc[ts] for n, t in tablas_su.items()}, fases[ts])')
replace_once('    # --- deslizamientos, zoom y pluviómetro de referencia ---', '''    # Recalcular referencia con el MISMO modelo, horas y estaciones válidas.
    referencia = {ts: evaluar_hora(ts, su, comp, cortes, cfg,
                                  {n: t.loc[ts] for n, t in tablas.items()},
                                  fases[ts], por_su=False) for ts in horas}
    cambios = []
    dom = su["dominio"].to_numpy(bool)
    for ts in horas:
        r, b = resultados[ts], referencia[ts]
        cambios.append({"ts": ts,
                        "SU_dominio": int(dom.sum()),
                        "R1_media_kriging_mm": r.loc[dom, "R1"].mean(),
                        "R1_media_vecino_mm": b.loc[dom, "R1"].mean(),
                        "delta_R1_mediana_mm": (r.loc[dom, "R1"] - b.loc[dom, "R1"]).median(),
                        "SU_cambian_clase_lluvia": int((r.loc[dom, "clase"] != b.loc[dom, "clase"]).sum()),
                        "SU_suben_alerta": int((r.loc[dom, "nivel"] > b.loc[dom, "nivel"]).sum()),
                        "SU_bajan_alerta": int((r.loc[dom, "nivel"] < b.loc[dom, "nivel"]).sum())})
    pd.DataFrame(cambios).to_csv(os.path.join(OUT_DIR, f"comparacion_horaria_{tag}.csv"), index=False)

    # --- deslizamientos, zoom y pluviómetro de referencia ---''')
replace_once('    # --- figuras (en grados, geometría simplificada solo para dibujar) ---', '''    t_ref, fuera_ref, _ = tabla_por_su(horas, referencia, su, ev)
    val_ref = validar(t_ref, ev, fuera_ref, tag)
    val_ref.to_csv(os.path.join(OUT_DIR, f"validacion_vecino_{tag}.csv"), index=False)
    pd.concat([val.assign(asignacion="kriging_log_100m"),
               val_ref.assign(asignacion="vecino_mas_cercano")], ignore_index=True).to_csv(
        os.path.join(OUT_DIR, f"comparacion_validacion_{tag}.csv"), index=False)

    # --- figuras (en grados, geometría simplificada solo para dibujar) ---''')
replace_once('    vals = [correr_evento(tag, su, comp, cortes, cfg, cache_series, gpl, fase_oni, capas)',
'''    print("Preparando intersecciones SU-píxel de 100 m (una sola vez)...", flush=True)
    zonal = ZonalArea(su, cell_size=100, crs="EPSG:32618")
    print(f"Grilla {zonal.shape}; {len(zonal.active_flat)} píxeles con área de SU.", flush=True)
    print(f"mc_common utilizado: {mcc.__file__}")
    vals = [correr_evento(tag, su, comp, cortes, cfg, cache_series, gpl, fase_oni, capas, zonal)''')
destination = HERE / "mapas_alerta_cap6_interpolado.py"
destination.write_text(text, encoding="utf-8")
(HERE / "cambios_vs_original.diff").write_text("".join(difflib.unified_diff(
    original.splitlines(keepends=True), text.splitlines(keepends=True),
    fromfile="mapas_alerta_cap6.py", tofile="mapas_alerta_cap6_interpolado.py")), encoding="utf-8")
print(destination)
