"""Promedios de lluvia por unidad de ladera, ponderados por área exacta.

Cada celda representa una lluvia uniforme dentro de sus 100 × 100 m. La
intersección geométrica reparte su aporte entre las unidades que la cruzan,
incluidas unidades menores que una celda. No se asignan valores por centroides.
Las coordenadas deben transformarse a un CRS proyectado en metros.
"""

import numpy as np
from pyproj import CRS
from rasterio.transform import from_origin
from scipy.sparse import csr_matrix
from shapely.geometry import box

try:
    from shapely import area as _area
    from shapely import box as _boxes
    from shapely import intersection as _intersection
except ImportError:  # Shapely 1.x: se mantiene el mismo cálculo geométrico.
    _area = _boxes = _intersection = None


class ZonalArea:
    """Grilla fija y matriz dispersa de áreas SU × celda, en metros cuadrados.

    ``x`` e ``y`` son vectores de centros; ``y`` decrece de norte a sur.
    ``active_flat`` contiene los índices planos de las celdas que intersectan
    alguna SU. ``coverage_count`` cuenta estas celdas por SU. ``index`` conserva
    el índice y el orden originales, incluso si hay etiquetas repetidas.
    ``mean`` devuelve NaN cuando menos del 99,9999 % del área de una SU tiene
    lluvia finita; ``last_valid_coverage`` guarda esa fracción para diagnóstico.

    Las SU pueden solaparse: cada una conserva su propia contribución de área.
    Los huecos de los polígonos no aportan al promedio. Geometrías inválidas,
    vacías, no poligonales o sin área generan errores, sin reparación implícita.
    """

    min_coverage = 0.999999

    def __init__(self, su_gdf, cell_size=100, crs="EPSG:32618"):
        if not np.isfinite(cell_size) or cell_size <= 0:
            raise ValueError("cell_size debe ser un número positivo y finito.")
        if len(su_gdf) == 0:
            raise ValueError("Se necesita al menos una unidad de ladera.")
        if su_gdf.crs is None:
            raise ValueError("Las unidades de ladera no tienen un CRS definido.")
        target_crs = CRS.from_user_input(crs)
        if not target_crs.is_projected or any(
            axis.unit_name.lower() not in ("metre", "meter")
            for axis in target_crs.axis_info[:2]
        ):
            raise ValueError("El CRS de la grilla debe ser proyectado en metros.")

        self.crs = target_crs
        self.cell_size = float(cell_size)
        self.index = su_gdf.index.copy()
        projected = su_gdf.to_crs(target_crs)
        geometries = list(projected.geometry)
        for position, geometry in enumerate(geometries):
            if (
                geometry is None
                or geometry.is_empty
                or geometry.geom_type not in ("Polygon", "MultiPolygon")
                or not geometry.is_valid
                or not np.isfinite(geometry.area)
                or geometry.area <= 0
            ):
                raise ValueError(
                    "Geometría SU inválida, vacía o no poligonal en posición "
                    f"{position} (índice={self.index[position]!r})."
                )

        bounds = np.asarray(projected.total_bounds, dtype=float)
        if not np.isfinite(bounds).all():
            raise ValueError("Los límites de las unidades contienen valores no finitos.")
        xmin, ymin = np.floor(bounds[:2] / self.cell_size) * self.cell_size
        xmax, ymax = np.ceil(bounds[2:] / self.cell_size) * self.cell_size
        nx = int(round((xmax - xmin) / self.cell_size))
        ny = int(round((ymax - ymin) / self.cell_size))
        if nx <= 0 or ny <= 0:
            raise ValueError("La extensión de las unidades no permite crear la grilla.")
        self.shape = (ny, nx)
        self.transform = from_origin(xmin, ymax, self.cell_size, self.cell_size)
        self.x = xmin + (np.arange(nx) + 0.5) * self.cell_size
        self.y = ymax - (np.arange(ny) + 0.5) * self.cell_size
        self.su_area = np.array([geometry.area for geometry in geometries])

        rows, columns, values = [], [], []
        for su_position, geometry in enumerate(geometries):
            gx0, gy0, gx1, gy1 = geometry.bounds
            c0 = max(0, int(np.floor((gx0 - xmin) / self.cell_size)))
            c1 = min(nx, int(np.ceil((gx1 - xmin) / self.cell_size)))
            r0 = max(0, int(np.floor((ymax - gy1) / self.cell_size)))
            r1 = min(ny, int(np.ceil((ymax - gy0) / self.cell_size)))
            rr, cc = np.meshgrid(
                np.arange(r0, r1), np.arange(c0, c1), indexing="ij"
            )
            rr, cc = rr.ravel(), cc.ravel()
            left = xmin + cc * self.cell_size
            top = ymax - rr * self.cell_size
            right, bottom = left + self.cell_size, top - self.cell_size
            if _boxes is not None:
                cells = _boxes(left, bottom, right, top)
                areas = np.asarray(_area(_intersection(cells, geometry)))
            else:
                areas = np.array([
                    geometry.intersection(box(a, b, c, d)).area
                    for a, b, c, d in zip(left, bottom, right, top)
                ])
            positive = areas > 0
            rows.extend([su_position] * int(positive.sum()))
            columns.extend((rr[positive] * nx + cc[positive]).tolist())
            values.extend(areas[positive].tolist())

        self.area_weights = csr_matrix(
            (values, (rows, columns)),
            shape=(len(geometries), ny * nx),
            dtype=np.float64,
        )
        self.weights = self.area_weights
        self.active_flat = np.unique(self.area_weights.indices)
        self.coverage_count = self.area_weights.getnnz(axis=1)
        self.covered_area = np.asarray(self.area_weights.sum(axis=1)).ravel()
        if not np.allclose(self.covered_area, self.su_area, rtol=1e-7, atol=1e-7):
            raise ValueError("La grilla no representa completamente el área de las SU.")
        self.last_valid_coverage = None

    def mean(self, grid):
        """Promedia una grilla (ny, nx); devuelve un vector en el orden de las SU.

        NaN e infinitos cuentan como lluvia faltante. Una SU sin cobertura casi
        completa devuelve NaN, no cero. Una máscara de NumPy también se respeta.
        La ponderación usa el área de cada intersección SU-celda, no su conteo.
        """
        grid = np.ma.asarray(grid, dtype=np.float64).filled(np.nan)
        if grid.shape != self.shape:
            raise ValueError(f"Se esperaba una grilla {self.shape}; llegó {grid.shape}.")
        flat = grid.ravel()
        finite = np.isfinite(flat)
        valid_area = self.area_weights @ finite.astype(np.float64)
        self.last_valid_coverage = valid_area / self.su_area
        numerator = self.area_weights @ np.where(finite, flat, 0.0)
        accepted = (self.last_valid_coverage >= self.min_coverage) & (valid_area > 0)
        result = np.full(len(self.index), np.nan, dtype=np.float64)
        np.divide(numerator, valid_area, out=result, where=accepted)
        return result
