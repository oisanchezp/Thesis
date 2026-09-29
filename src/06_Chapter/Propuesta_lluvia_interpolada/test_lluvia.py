"""Pruebas sintéticas del promedio de área y de la preparación de lluvia."""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from numpy.testing import assert_allclose
from pykrige.ok import OrdinaryKriging
from shapely.geometry import MultiPolygon, Point, Polygon, box

from lluvia_interpolada import (
    KrigingLogFijo,
    componentes_temporales,
    interpolar_evento,
)
from zonal_area import ZonalArea


E, N = 500000.0, 700000.0


def rect(x0, y0, x1, y1):
    return box(E + x0, N + y0, E + x1, N + y1)


def units(geometries, ids=None, index=None):
    ids = ids if ids is not None else list(range(len(geometries)))
    return gpd.GeoDataFrame(
        {"su_id": ids}, geometry=geometries, crs="EPSG:32618", index=index
    )


def test_dos_su_pequenas_comparten_celda_sin_contener_su_centro():
    su = units([rect(10, 10, 20, 20), rect(30, 30, 40, 40)], index=[9, 9])
    z = ZonalArea(su)
    assert z.shape == (1, 1)
    assert z.index.tolist() == [9, 9]
    assert z.active_flat.tolist() == [0]
    assert z.coverage_count.tolist() == [1, 1]
    assert_allclose(z.area_weights.toarray(), [[100], [100]])
    assert_allclose(z.mean(np.array([[12.0]])), [12, 12])
    assert_allclose(z.last_valid_coverage, [1, 1])


def test_pesos_parciales_segun_area_no_conteo_de_celdas():
    z = ZonalArea(units([rect(25, 0, 125, 100)]))
    assert z.shape == (1, 2)
    assert_allclose(z.area_weights.toarray(), [[7500, 2500]])
    assert_allclose(z.mean(np.array([[10.0, 30.0]])), [15])
    assert_allclose(z.x, [E + 50, E + 150])
    assert_allclose(z.y, [N + 50])
    assert z.transform.a == 100
    assert z.transform.e == -100


def test_huecos_multipoligonos_y_orientacion_norte_arriba():
    outer = rect(0, 0, 200, 200)
    hole = rect(50, 50, 150, 150)
    donut = Polygon(outer.exterior.coords, [hole.exterior.coords])
    multi = MultiPolygon([rect(0, 0, 10, 10), rect(190, 190, 200, 200)])
    z = ZonalArea(units([donut, multi]))
    grid = np.array([[1.0, 2.0], [4.0, 8.0]])
    assert_allclose(z.y, [N + 150, N + 50])
    assert_allclose(z.area_weights.toarray()[0], [7500, 7500, 7500, 7500])
    assert_allclose(z.area_weights.toarray()[1], [0, 100, 100, 0])
    assert_allclose(z.su_area, [30000, 200])
    assert_allclose(z.mean(grid), [3.75, 3.0])


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_faltantes_no_se_reemplazan_por_cero(missing):
    z = ZonalArea(units([rect(25, 0, 125, 100), rect(10, 10, 20, 20)]))
    actual = z.mean(np.array([[10.0, missing]]))
    assert np.isnan(actual[0])
    assert actual[1] == 10
    assert_allclose(z.last_valid_coverage, [0.75, 1])
    masked = np.ma.array([[10.0, 80.0]], mask=[[False, True]])
    assert_allclose(z.mean(masked), [np.nan, 10], equal_nan=True)


def test_cobertura_casi_completa_tiene_tolerancia_numerica():
    # Una fracción inferior a 1e-6 puede faltar sin invalidar toda la SU.
    z = ZonalArea(units([rect(0, 0, 100.00001, 100)]))
    assert_allclose(z.mean(np.array([[5.0, np.nan]])), [5])
    assert 0.999999 < z.last_valid_coverage[0] < 1


def test_reproyeccion_fuente_geografica_y_orden_original():
    source = units([rect(125, 25, 175, 75), rect(25, 25, 75, 75)],
                   ids=[300, 100], index=[42, 7]).to_crs("EPSG:4326")
    z = ZonalArea(source)
    assert z.index.tolist() == [42, 7]
    assert z.crs.to_epsg() == 32618
    assert z.shape == (1, 2)
    assert_allclose(z.mean(np.array([[2.0, 8.0]])), [8, 2], atol=1e-8)
    with pytest.raises(ValueError, match="proyectado en metros"):
        ZonalArea(source, crs="EPSG:4326")


def test_geometrias_y_formas_invalidas_producen_error():
    bowtie = Polygon([(E, N), (E + 20, N + 20), (E, N + 20), (E + 20, N)])
    for bad in [bowtie, Polygon(), None, Point(E, N)]:
        with pytest.raises(ValueError, match="Geometría SU"):
            ZonalArea(units([bad]))
    with pytest.raises(ValueError, match="CRS definido"):
        ZonalArea(units([rect(0, 0, 10, 10)]).set_crs(None, allow_override=True))
    z = ZonalArea(units([rect(0, 0, 10, 10)]))
    with pytest.raises(ValueError, match="grilla"):
        z.mean(np.ones((2, 2)))


def test_kriging_log_fijo_coincide_con_pykrige_incluidas_estaciones():
    rng = np.random.default_rng(219)
    xy = rng.uniform(0, 1000, (11, 2))
    targets = np.vstack([xy, rng.uniform(-100, 1100, (19, 2))])
    rain = rng.uniform(0.1, 90, len(xy))
    range_m = 460.0
    model = KrigingLogFijo(xy, targets, range_m=range_m, block_size=7)
    reference = OrdinaryKriging(
        xy[:, 0], xy[:, 1], np.log1p(rain), variogram_model="spherical",
        variogram_parameters={"sill": 1.0, "range": range_m, "nugget": 0.0},
        coordinates_type="euclidean", exact_values=True, verbose=False,
    )
    predicted_log, _ = reference.execute("points", targets[:, 0], targets[:, 1])
    expected = np.maximum(np.expm1(np.asarray(predicted_log)), 0)
    assert_allclose(model.predict(rain), expected, rtol=1e-10, atol=1e-10)
    assert_allclose(model.predict(rain)[:len(xy)], rain, rtol=1e-10, atol=1e-10)
    assert_allclose(model.weights.sum(axis=1), 1, atol=1e-12)


@pytest.mark.parametrize("constant", [0.0, 12.5])
def test_kriging_campos_cero_y_constantes(constant):
    xy = np.array([[0, 0], [300, 0], [0, 300]], dtype=float)
    model = KrigingLogFijo(xy, np.array([[50, 50], [500, 500]]))
    assert_allclose(model.predict(np.full(3, constant)), constant, atol=0)
    assert model.last_negative_clipped == 0


def test_componentes_temporales_distinguen_29_y_30_dias_previos():
    r1 = np.array([2.0, 5.0])
    r30 = np.array([12.0, 35.0])
    r31 = np.array([15.0, 39.0])
    parts29 = componentes_temporales(r1, r30, r30, "R30-R1")
    assert set(parts29) == {"R1", "PREV29"}
    assert_allclose(parts29["PREV29"], [10, 30])
    parts30 = componentes_temporales(r1, r30, r31, "R31-R1")
    assert_allclose(parts30["PREV29"], [10, 30])
    assert_allclose(parts30["EXTRA1"], [3, 4])
    assert_allclose(parts30["PREV29"] + parts30["EXTRA1"], r31 - r1)
    with pytest.raises(ValueError, match="RLTR"):
        componentes_temporales(r1, r30, r31, "R30-R1")
    with pytest.raises(ValueError, match="no anidados"):
        componentes_temporales(r30, r1, r1, "R30-R1")
    with pytest.raises(ValueError, match="no reconocido"):
        componentes_temporales(r1, r30, r31, "DESCONOCIDO")


@pytest.mark.parametrize("definition", ["R30-R1", "R31-R1"])
def test_evento_completo_orden_id_acumulados_y_geotiff(tmp_path, definition):
    hours = pd.date_range("2024-04-01 00:00", periods=2, freq="h")
    valid = [30, 10, 20]
    columns = [20, 30, 10]
    r1 = pd.DataFrame([[3, 3, 3], [6, 4, 0]], index=hours, columns=columns, dtype=float)
    previous = pd.DataFrame([[10, 15, 20], [30, 25, 40]], index=hours,
                            columns=columns, dtype=float)
    extra = pd.DataFrame([[1, 4, 2], [3, 2, 1]], index=hours, columns=columns, dtype=float)
    tables = {"R1": r1, "R30": r1 + previous,
              "RLTR": r1 + previous + (extra if definition == "R31-R1" else 0)}
    stations = gpd.GeoDataFrame(
        {"Codigo": [20, 10, 30]},
        geometry=[Point(E + 25, N + 175), Point(E + 175, N + 25), Point(E + 25, N + 25)],
        crs="EPSG:32618",
    )
    su = units([rect(10, 10, 45, 45), rect(125, 125, 190, 190)],
               ids=[300, 100], index=[4, 9])
    z = ZonalArea(su)
    results = interpolar_evento(hours, tables, valid, stations, su,
                               {"LTR_DEF": definition}, "su_id", tmp_path, zonal=z)
    assert set(results) == {"R1", "R30", "RLTR", "LTR"}
    for frame in results.values():
        assert frame.columns.tolist() == [300, 100]
        assert frame.index.equals(hours)
        assert np.isfinite(frame.to_numpy()).all()
    assert_allclose(results["R1"].iloc[0], [3, 3])
    assert (results["R30"].to_numpy() >= results["R1"].to_numpy()).all()
    assert (results["RLTR"].to_numpy() >= results["R30"].to_numpy()).all()
    assert_allclose(results["RLTR"], results["R1"] + results["LTR"], atol=1e-12)
    if definition == "R30-R1":
        assert_allclose(results["RLTR"], results["R30"])

    # Comprueba asignación de códigos con un orden distinto al usado en la función.
    destinations = np.array([[E + 50, N + 50], [E + 150, N + 150]])
    xy = np.column_stack((stations.geometry.x, stations.geometry.y))
    expected_r1 = KrigingLogFijo(xy, destinations).predict(
        r1.loc[hours[1], stations["Codigo"].to_list()].to_numpy()
    )
    assert_allclose(results["R1"].iloc[1], expected_r1)

    tif_files = list((tmp_path / "rasters").glob("*.tif"))
    assert len(tif_files) == 8
    for name, frame in results.items():
        for ts in hours:
            path = tmp_path / "rasters" / f"{name}_{ts:%Y%m%d_%H%M}.tif"
            with rasterio.open(path) as src:
                assert src.crs.to_epsg() == 32618
                assert src.res == (100, 100)
                assert src.transform == z.transform
                assert src.nodata == -9999
                assert src.tags()["units"] == "mm"
                grid = src.read(1, masked=True)
                assert grid.mask.sum() == 2
            assert_allclose(z.mean(grid), frame.loc[ts], rtol=1e-6, atol=1e-6)

    saved = pd.read_csv(tmp_path / "lluvia_por_su.csv.gz")
    assert saved["su_id"].tolist() == [300, 100, 300, 100]
    assert len(saved) == 4
    metadata = json.loads((tmp_path / "metodo.json").read_text(encoding="utf-8"))
    assert metadata["LTR_DEF"] == definition
    assert metadata["sitios_unicos"] == 3
    assert metadata["cell_size_m"] == 100
    assert metadata["estaciones"] == valid
