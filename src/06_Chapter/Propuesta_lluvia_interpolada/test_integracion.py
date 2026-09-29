"""Comprueba la integración sin importar los datos ni módulos de Nazca.

Ejecutar: python test_integracion.py
Extrae por AST las funciones reales de ambos scripts y sus constantes literales.
"""
import ast
from pathlib import Path
import tempfile
import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from lluvia_interpolada import interpolar_evento


HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "mapas_alerta_cap6.py"
PROPUESTA = HERE / "mapas_alerta_cap6_interpolado.py"


def cargar_motor(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    wanted = {"COL_ID", "CLASES_SUSC", "MATRIZ", "FUERA_DOMINIO", "SIN_LLUVIA"}
    namespace = {"np": np, "pd": pd}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {n.id for target in node.targets for n in ast.walk(target)
                     if isinstance(n, ast.Name)}
            if names & wanted:
                ast.literal_eval(node.value)  # Rechazar asignaciones con efectos laterales.
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {"expit", "evaluar_hora"}:
            selected.append(node)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["evaluar_hora"]


def modelo_sintetico():
    return {
        "b0": -2.0,
        "STR": (np.array([0.0, 100.0]), np.array([-1.0, 2.0])),
        "LTR": (np.array([0.0, 500.0]), np.array([-0.5, 1.0])),
        "ENSO": {"Neutro": 0.4},
    }


class IntegracionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = staticmethod(cargar_motor(ORIGINAL))
        cls.propuesta = staticmethod(cargar_motor(PROPUESTA))

    def test_motor_igual_por_estacion_y_por_su_con_indices_reordenados(self):
        rng = np.random.default_rng(20260925)
        n = 72
        codes = np.array([11, 22, 33, 44, 55, 66])
        su = pd.DataFrame({
            "su_id": rng.permutation(np.arange(900, 900 + n)),
            "Codigo_pluvio": np.tile(codes, n // len(codes)),
            "dominio": np.arange(n) % 7 != 0,
            "susc_idx": np.arange(n) % 4,
            "ef_zona": rng.normal(0, 0.7, n),
            "ef_tipo_su": rng.normal(0, 0.4, n),
        }, index=rng.permutation(np.arange(300, 300 + n)))
        r1 = pd.Series([10., 11., 60., 130., np.nan, 0.], index=codes)
        r30 = pd.Series([50., 90., 300., 620., np.nan, 140.], index=codes)
        comp = modelo_sintetico()
        for ltr_def in ("R30-R1", "R31-R1"):
            cfg = {"UMBRAL_1D": 10., "UMBRAL_30D": 50., "LTR_DEF": ltr_def}
            rain = {"R1": r1, "R30": r30,
                    "RLTR": r30 + (20. if ltr_def.startswith("R31") else 0.)}
            rain_su = {name: pd.Series(su["Codigo_pluvio"].map(values).to_numpy(),
                                      index=su["su_id"]).sample(frac=1, random_state=27)
                       for name, values in rain.items()}
            # El campo adicional LTR no es consumido por el motor y no debe romperlo.
            rain_su["LTR"] = rain_su["RLTR"] - rain_su["R1"]
            args = (pd.Timestamp("2025-04-28 00:00"), su, comp, [.3, .5, .7], cfg)
            baseline = self.original(*args, rain, "Neutro")
            nearest = self.propuesta(*args, rain, "Neutro", por_su=False)
            interpolated = self.propuesta(*args, rain_su, "Neutro", por_su=True)
            pd.testing.assert_frame_equal(baseline, nearest[baseline.columns])
            pd.testing.assert_frame_equal(nearest, interpolated)
            self.assertTrue((interpolated.loc[~su["dominio"], "nivel"] == -1).all())
            missing = su["dominio"] & (su["Codigo_pluvio"] == 55)
            self.assertTrue((interpolated.loc[missing, "nivel"] == -2).all())
            low = su["dominio"] & su["Codigo_pluvio"].isin([11, 66])
            self.assertTrue((interpolated.loc[low, "nivel"] == 1).all())

    def test_salida_interpolacion_entra_al_motor(self):
        su = gpd.GeoDataFrame({
            "su_id": [304, 101, 205], "dominio": [True, True, False],
            "susc_idx": [0, 3, 2], "ef_zona": [0., .2, .5],
            "ef_tipo_su": [0., .1, -.1],
        }, geometry=[box(500010, 670010, 500040, 670040),
                     box(500100, 670010, 500250, 670190),
                     box(500050, 670050, 500300, 670300)],
            crs="EPSG:32618", index=[9, 3, 11])
        stations = gpd.GeoDataFrame({"Codigo": [9, 4, 8, 7]},
            geometry=gpd.points_from_xy([500000, 501000, 500000, 501000],
                                        [670000, 670000, 671000, 671000]),
            crs=su.crs)
        times = pd.date_range("2025-04-28 00:00", periods=2, freq="h")
        for definition, extra in [("R30-R1", 0.), ("R31-R1", 20.)]:
            tables = {"R1": pd.DataFrame(15., index=times, columns=[4, 7, 9, 8]),
                      "R30": pd.DataFrame(115., index=times, columns=[4, 7, 9, 8]),
                      "RLTR": pd.DataFrame(115. + extra, index=times, columns=[4, 7, 9, 8])}
            cfg = {"LTR_DEF": definition, "UMBRAL_1D": 10., "UMBRAL_30D": 50.}
            with tempfile.TemporaryDirectory(prefix="cap6_integracion_") as directory:
                result = interpolar_evento(times, tables, [8, 9, 7, 4], stations, su,
                                           cfg, "su_id", directory, guardar_tif=False)
                for name, expected in {"R1": 15., "R30": 115.,
                                       "RLTR": 115. + extra, "LTR": 100. + extra}.items():
                    self.assertEqual(list(result[name].columns), [304, 101, 205])
                    np.testing.assert_allclose(result[name].to_numpy(), expected)
                for ts in times:
                    evaluated = self.propuesta(ts, su, modelo_sintetico(), [.3, .5, .7], cfg,
                                              {n: t.loc[ts] for n, t in result.items()}, "Neutro")
                    self.assertEqual(list(evaluated.index), [9, 3, 11])
                    np.testing.assert_allclose(evaluated["LTR"], 100. + extra)
                    self.assertTrue(np.isfinite(evaluated.loc[[9, 3], "p"]).all())
                    self.assertEqual(evaluated.loc[11, "nivel"], -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
