"""Lluvia horaria por SU: acumulados -> kriging log -> píxeles -> media de área.

Sólo prepara predictores; no modifica el GAM, sus cortes ni la matriz.
El alcance es fijo, igual a 0.2 * diámetro de la red válida (processes.py).
No es un ajuste de variograma experimental ni una corrección con radar.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import rasterio
from scipy.linalg import pinvh
from scipy.spatial.distance import cdist, pdist
from scipy.spatial import cKDTree

from zonal_area import ZonalArea


class KrigingLogFijo:
    """Kriging ordinario esférico; pesos fijos para todas las horas/ventanas.

    Con nugget cero, multiplicar el variograma por sill no altera los pesos.
    Se usa sill=1 para evitar reconstruir el mismo sistema para cada campo.
    La inversión logarítmica se realiza ANTES del promedio zonal en mm.
    """

    def __init__(self, xy_m, destinos_m, range_m=None, block_size=8192):
        xy = np.asarray(xy_m, dtype=float)
        dst = np.asarray(destinos_m, dtype=float)
        if (xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3
                or dst.ndim != 2 or dst.shape[1] != 2
                or not np.isfinite(xy).all() or not np.isfinite(dst).all()):
            raise ValueError("Se requieren al menos 3 estaciones y coordenadas finitas (n,2).")
        if len(np.unique(xy, axis=0)) != len(xy):
            raise ValueError("Agrupa primero las estaciones de coordenadas duplicadas.")
        self.range_m = float(range_m if range_m is not None else 0.2 * pdist(xy).max())
        if not np.isfinite(self.range_m) or self.range_m <= 0:
            raise ValueError("El alcance del variograma debe ser positivo y finito.")
        n = len(xy)
        sistema = np.ones((n + 1, n + 1), dtype=float)
        sistema[:n, :n] = self._gamma(cdist(xy, xy))
        sistema[-1, -1] = 0.0
        inversa = pinvh(sistema)
        self.weights = np.empty((len(dst), n), dtype=float)
        for start in range(0, len(dst), block_size):
            end = min(start + block_size, len(dst))
            rhs = np.column_stack((self._gamma(cdist(dst[start:end], xy)),
                                   np.ones(end - start)))
            self.weights[start:end] = (rhs @ inversa)[:, :n]
        if not np.allclose(self.weights.sum(axis=1), 1.0, atol=1e-7, rtol=0):
            raise ValueError("Sistema de kriging mal condicionado: pesos no suman uno.")
        self.nearest_m = cKDTree(xy).query(dst)[0]
        self.last_negative_clipped = 0

    def _gamma(self, distances):
        q = np.minimum(np.asarray(distances) / self.range_m, 1.0)
        return 1.5 * q - 0.5 * q ** 3

    def predict(self, rainfall_mm):
        z = np.asarray(rainfall_mm, dtype=float)
        if z.shape != (self.weights.shape[1],) or not np.isfinite(z).all() or (z < 0).any():
            raise ValueError("Lluvia inválida: se requieren mm finitos no negativos por estación.")
        self.last_negative_clipped = 0
        if np.ptp(z) == 0:
            return np.full(len(self.weights), z[0], dtype=float)
        pred = np.expm1(self.weights @ np.log1p(z))
        if not np.isfinite(pred).all():
            raise ValueError("El kriging generó valores no finitos.")
        self.last_negative_clipped = int((pred < 0).sum())
        return np.maximum(pred, 0.0)


def componentes_temporales(r1, r30, rltr, ltr_def):
    """Intervalos disjuntos: 24h recientes, 696h previas y (opcional) 24h extra.

    La definición se conserva tal como está en el JSON del modelo.
    R30-R1 = 29 días previos; R31-R1 = 30 días previos.
    """
    definition = str(ltr_def).upper().replace(" ", "")
    if definition not in {"R30-R1", "R31-R1"}:
        raise ValueError(f"LTR_DEF no reconocido: {ltr_def!r}")
    r1, r30, rltr = (np.asarray(a, dtype=float) for a in (r1, r30, rltr))
    if not (r1.shape == r30.shape == rltr.shape):
        raise ValueError("Los acumulados deben tener la misma forma.")
    out = {"R1": r1, "PREV29": r30 - r1}
    if definition == "R31-R1":
        out["EXTRA1"] = rltr - r30
    elif not np.allclose(rltr, r30, atol=1e-6, rtol=0):
        raise ValueError("RLTR debe ser R30 cuando LTR_DEF=R30-R1.")
    for name, a in out.items():
        if not np.isfinite(a).all() or (a < -1e-6).any():
            raise ValueError(f"Acumulados inválidos o no anidados para {name}.")
        out[name] = np.maximum(a, 0.0)
    return out


def guardar_raster(path, grid, zonal):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=zonal.shape[0],
                       width=zonal.shape[1], count=1, dtype="float32", crs=zonal.crs,
                       transform=zonal.transform, nodata=-9999.0, compress="deflate") as dst:
        dst.write(np.where(np.isfinite(grid), grid, -9999.0).astype("float32"), 1)
        dst.update_tags(units="mm", method="ordinary_kriging_log1p_spherical")


def interpolar_evento(horas, tablas, validos, gpl, su, cfg, col_id, salida,
                      zonal=None, guardar_tif=True, min_estaciones=3):
    """Devuelve mismas tablas que lluvia_evento, ahora columnas = ID de SU.

    Acumular PRIMERO por estación (funciones originales). Esta función no lee
    las series ni usa observaciones posteriores para calcular los acumulados.
    El conjunto de estaciones válidas durante todo el evento es retrospectivo.
    """
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    if su[col_id].isna().any() or su[col_id].duplicated().any():
        raise ValueError("Los identificadores de SU deben ser únicos y no nulos.")
    zonal = ZonalArea(su, cell_size=100, crs="EPSG:32618") if zonal is None else zonal
    if not zonal.index.equals(su.index):
        raise ValueError("El orden de SU difiere del utilizado en la media zonal.")
    stations = gpl.to_crs(zonal.crs).copy()
    stations["Codigo"] = stations["Codigo"].astype(int)
    if stations["Codigo"].duplicated().any():
        raise ValueError("Hay códigos de estación duplicados en metadatos.")
    stations = stations.set_index("Codigo").reindex(list(validos))
    if stations.geometry.isna().any() or stations.geometry.is_empty.any():
        raise ValueError("Faltan coordenadas para alguna estación válida.")
    xy = np.column_stack((stations.geometry.x, stations.geometry.y))
    unique_xy, group = np.unique(xy, axis=0, return_inverse=True)
    if len(unique_xy) < min_estaciones:
        raise ValueError(f"Sólo {len(unique_xy)} ubicaciones válidas; mínimo {min_estaciones}.")
    count = np.bincount(group)
    active = zonal.active_flat
    rows, cols = np.unravel_index(active, zonal.shape)
    targets = np.column_stack((zonal.x[cols], zonal.y[rows]))
    interpolator = KrigingLogFijo(unique_xy, targets)
    ids = pd.Index(su[col_id].to_numpy(), name=col_id)
    result = {name: pd.DataFrame(index=horas, columns=ids, dtype=float)
              for name in ("R1", "R30", "RLTR", "LTR")}
    diagnostics = []
    definition = cfg["LTR_DEF"]
    for ts in horas:
        values = [tablas[name].loc[ts, validos].to_numpy(float)
                  for name in ("R1", "R30", "RLTR")]
        parts = componentes_temporales(*values, definition)
        grids = {}
        for name, values in parts.items():
            # Estaciones exactamente coincidentes se promedian en mm.
            grouped = np.bincount(group, weights=values) / count
            grid = np.full(zonal.shape, np.nan, dtype=float)
            grid.ravel()[active] = interpolator.predict(grouped)
            grids[name] = grid
            diagnostics.append({"ts": str(ts), "componente": name,
                                "pixeles_negativos_recortados": interpolator.last_negative_clipped})
        fields = {"R1": grids["R1"], "R30": grids["R1"] + grids["PREV29"],
                  "LTR": grids["PREV29"] + grids.get("EXTRA1", 0.0)}
        fields["RLTR"] = fields["R1"] + fields["LTR"]
        stamp = pd.Timestamp(ts).strftime("%Y%m%d_%H%M")
        for name, grid in fields.items():
            result[name].loc[ts] = zonal.mean(grid)
            if guardar_tif:
                guardar_raster(salida / "rasters" / f"{name}_{stamp}.tif", grid, zonal)
        print(f"  Interpolación {ts}: {len(ids)} SU, R1 máximo "
              f"{result['R1'].loc[ts].max():.1f} mm", flush=True)
    pd.DataFrame(diagnostics).to_csv(salida / "diagnostico_kriging.csv", index=False)
    with open(salida / "metodo.json", "w", encoding="utf-8") as fh:
        json.dump({"metodo": "kriging ordinario log1p esférico, alcance fijo, sin radar",
                   "nugget": 0, "rango_m": interpolator.range_m,
                   "regla_rango": "0.2 * distancia máxima entre estaciones válidas",
                   "LTR_DEF": definition, "cell_size_m": abs(zonal.transform.a),
                   "crs": str(zonal.crs), "media_SU": "ponderada por área de intersección en mm",
                   "estaciones": [int(c) for c in validos], "sitios_unicos": len(unique_xy),
                   "max_distancia_pixel_estacion_km": float(interpolator.nearest_m.max() / 1000),
                   "percentil90_distancia_km": float(np.quantile(interpolator.nearest_m, .9) / 1000),
                   "suma_temporal": "componentes disjuntos interpolados; no interpolar totales superpuestos",
                   "correccion_sesgo_log": False}, fh, indent=2, ensure_ascii=False)
    # Formato largo y sin depender de pandas.stack: estable en pandas 1.x a 3.x.
    frames = [pd.DataFrame({"ts": ts, col_id: ids,
                           **{n: t.loc[ts].to_numpy(float) for n, t in result.items()}})
              for ts in horas]
    pd.concat(frames, ignore_index=True).to_csv(salida / "lluvia_por_su.csv.gz", index=False)
    return result
