# Lluvia interpolada para los mapas del capítulo 6

Esta propuesta crea una superficie de lluvia en píxeles de 100 m y asigna a
cada unidad de ladera la media ponderada por el área de sus intersecciones con
los píxeles. Conserva el GAM, sus efectos, sus cortes, el dominio y la matriz.
No se ha conectado a Nazca ni se han modificado los scripts originales.

## Archivos que debes llevar a Nazca

Coloca juntos estos tres archivos en `/home/oisanchezp/Thesis/src/06_Chapter/`
o en otra carpeta de trabajo:

- `mapas_alerta_cap6_interpolado.py`: copia adaptada del script adjunto.
- `lluvia_interpolada.py`: kriging, consistencia de ventanas, exportación y medias.
- `zonal_area.py`: intersecciones exactas de las SU con la grilla.

`cambios_vs_original.diff` permite revisar las modificaciones.
`test_lluvia.py` y `test_integracion.py` contienen las comprobaciones numéricas.
`preparar_copia.py` documenta cómo se generó la copia; NO necesitas ejecutarlo.

Usa el mismo entorno Python con el que ya ejecutas los mapas del capítulo 6.
Además de sus dependencias originales se requieren `scipy`, `rasterio` y
`shapely` (normalmente ya presentes en un entorno geoespacial). El kriging
precalculado no necesita PyKrige en ejecución; las pruebas comparativas sí.
No se incluye el modelo ni sus datos: se leen de tus rutas originales.

```bash
python mapas_alerta_cap6_interpolado.py altavista olivares
```

Por defecto escribe en `.../src/06_Chapter/Mapas_alerta_kriging100m/`.
Para elegir otra salida, usa `CAP6_OUT_INTERPOLADO`. No utiliza `CAP6_OUT_MAPAS`
para evitar sobrescribir la comparación original. Para omitir GeoTIFF:

```bash
CAP6_GUARDAR_TIF=0 python mapas_alerta_cap6_interpolado.py altavista
```

Se mantienen las variables existentes `CAP6_BASE`, `CAP6_CH5_DIR`,
`CAP6_CH6_DIR`, `CAP6_UMB_DIR`, `CAP6_PATH_SU`, `CAP6_RUTA_SERIES`, etc.
`mc_common.py` se importa de `CAP6_CH5_DIR`, como en el script original.

## Método y decisiones

1. **Acumular por estación y hora.** Se conservan `lluvia_evento`, `_lluvia`
   y `mc_common.rain_at`: ventanas móviles, control de cobertura original y
   exclusión de series que terminaron antes del evento. El conjunto de estaciones
   es fijo durante cada evento y común a todas las ventanas. Se emplean todas
   las estaciones válidas con metadatos, incluidas las exteriores al valle.
2. **Conservar la definición temporal entrenada.** El JSON local revisado
   (`revision_graficos_cap6/datos/excl10d_libre/gam_cap6_resumen.json`) dice
   `LTR_DEF = R30-R1`. Son 29 días antecedentes, no 30. La configuración del
   modelo cargado en Nazca es la que gobierna la ejecución.
3. **Interpolar componentes disjuntos.** Para ese modelo son R1 (últimas 24 h)
   y R30-R1 (las 696 h anteriores). Si otro modelo declara `R31-R1`, se agrega
   el día extra `R31-R30`; LTR suma esos 29+1 días. Nunca se cambia la definición
   automáticamente para hacerla coincidir con una descripción en el texto.
4. **Kriging esférico en log1p.** Cada componente se transforma con `log1p`, se
   interpola, vuelve a mm con `expm1` y se recortan predicciones negativas a cero,
   registrando cuántas hubo. Se sigue la rama de SOLO PLUVIÓMETROS de
   `processes.py`: nugget=0 y alcance=0.2 por el diámetro de la red válida.
   El alcance es una regla fija, no un ajuste al variograma experimental.
5. **Sumar campos compatibles.** R30=R1+PREV29 y RLTR=R1+LTR en cada píxel.
   Esto evita R30<R1 y antecedentes negativos. Como log1p no es lineal,
   sumar componentes interpolados NO equivale a interpolar R30 directamente;
   esta adaptación es deliberada y debe documentarse en la tesis.
6. **Promediar en mm por SU.**

   `P_SU = sum(P_pixel * area(SU intersección pixel)) / area(SU)`

   Así no se pierden unidades menores que una celda, no se cuentan áreas
   exteriores a la SU y un píxel puede aportar a varias unidades. Los huecos
   se respetan. No hay asignación por centroide como reemplazo de esta media.
7. **Evaluar el mismo modelo.** Se sustituyen sólo los valores de R1, R30 y LTR.
   Se conservan filtros, términos de ENSO/zona/tipo, cortes y matriz de integración.

La grilla se alinea en EPSG:32618, con centros a media celda y norte arriba.
Las intersecciones se calculan una vez para todos los eventos; los pesos de
kriging, una vez por evento. Con nugget cero, el sill sólo multiplica todas las
semivarianzas y se cancela en los pesos: se normaliza a uno. Así se puede
reutilizar el operador espacial para las horas sucesivas. Los ceros y campos
constantes se manejan explícitamente, sin intentar invertir una matriz de sill=0.

Las estaciones exactamente coincidentes se promedian en mm. Valores de lluvia
faltantes o negativos provocan un error, no se convierten en lluvia cero. Se
requieren al menos tres ubicaciones diferentes; es un mínimo numérico, no una
garantía de cobertura espacial. La red se extrapola hasta todo el dominio y
`metodo.json` registra distancias de los píxeles a las estaciones.

## Salidas para revisar el cambio

- Mapas y animaciones habituales con los nuevos predictores.
- `lluvia_<evento>/rasters/`: GeoTIFF R1, R30, LTR y RLTR para cada hora.
- `lluvia_<evento>/lluvia_por_su.csv.gz`: lluvia por hora e ID de SU.
- `lluvia_<evento>/metodo.json`: estaciones, resolución, alcance y definición LTR.
- `lluvia_<evento>/diagnostico_kriging.csv`: recortes de predicciones negativas.
- `comparacion_horaria_<evento>.csv`: diferencias de lluvia y laderas que suben
  o bajan de alerta respecto al vecino más cercano. Las medias indicadas allí
  son medias entre SU, no un promedio territorial ponderado por superficie.
- `comparacion_validacion_<evento>.csv`: detección, falsas alarmas y HK para
  ambas asignaciones, calculados con las mismas horas, estaciones, GAM y matriz.
- `validacion_vecino_<evento>.csv`: referencia recalculada en la misma ejecución.

El hietograma sigue siendo una estación observada, ahora rotulada como
`Reference gauge`. `Codigo_pluvio_referencia` en la tabla por SU identifica
una referencia espacial: no es la fuente única de la lluvia interpolada.

## Interpretación

La interpolación suaviza los límites impuestos por la asignación al vecino.
Puede bajar lluvia en unas SU y subirla en otras; no garantiza disminuirla en
todas. La resolución de 100 m es de cálculo, no evidencia de observaciones de
lluvia independientes a esa escala. La inversión `expm1` sin corrección de sesgo
es coherente con el ejemplo SIATA, pero no garantiza una estimación insesgada
de la media aritmética de la lluvia. Después sí se calcula una media aritmética
ponderada por área de los píxeles estimados, en mm.

El campo puede verse continuo y las alertas conservar límites por zona, tipo
de ladera, susceptibilidad y cortes discretos. Un mapa más suave no demuestra
mejor predicción. El cambio de lluvia puntual a media de área cambia el soporte
del predictor: los índices de validación anteriores no se heredan. Para esta
prueba se mantiene el modelo; para adoptarlo como versión definitiva conviene
comparar ambas entradas y entrenar/validar con el mismo método de asignación.

Los filtros del JSON local revisado son R1>10 mm y R30>50 mm. Por ello incluso
con cortes probabilísticos bajos una reducción de lluvia puede mantener una
SU en Low antes de evaluar el GAM.

Se conserva el 90 % de cobertura de las ventanas originales: los huecos
aceptados siguen sin escalarse al 100 %. La cobertura de un intervalo obtenido
por resta no se controla independientemente; en particular, R31 y R30 válidos
no garantizan cobertura del día extra. El conjunto válido en *todo* el evento
usa información sobre disponibilidad futura, apropiado para la reconstrucción
retrospectiva; una implementación en tiempo real necesitaría otra regla.

## Comprobaciones y referencias

Las pruebas se ejecutan con `python -m pytest test_lluvia.py test_integracion.py`.
Comprueban el núcleo numérico contra PyKrige, ventanas temporales, geometrías
pequeñas/compartidas, orden de IDs, exportación raster y conservación del motor.
No sustituyen una ejecución con todos los registros históricos de Nazca.

- [PyKrige OrdinaryKriging](https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/generated/pykrige.ok.OrdinaryKriging.html)
- [Rasterio: selección de píxeles](https://rasterio.readthedocs.io/en/stable/topics/features.html)
