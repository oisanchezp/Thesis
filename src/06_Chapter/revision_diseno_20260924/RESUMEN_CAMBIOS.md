# Figuras de umbrales GAM, capítulo 6

Revisión del 24 de septiembre de 2026 del notebook de Nazca:
`/home/oisanchezp/Thesis/src/06_Chapter/umbrales_gam_cap6.ipynb`.

Se utiliza el diseño de ausencias seleccionado en el notebook: `excl10d_libre`.

## Tamaño y exportación

El usuario confirmó un ancho de texto de **16 cm**. Las figuras se generan con ese ancho
real, etiquetas de ejes de 9 pt, marcas y leyendas de 8,5 pt y etiquetas de contorno de 8 pt.
Los PNG y las figuras se generan a 300 dpi. Los PDF conservan texto y líneas vectoriales;
la resolución de 300 dpi también se aplica a los elementos rasterizados del PDF.
Se evita `bbox_inches='tight'` para conservar el ancho físico del lienzo.

| Figura | Tamaño (cm) | Cambio principal |
|---|---:|---|
| Desempeño | 16 × 12,19 | 2 × 2; color GAM del capítulo 5; sin mediana numérica en esquina |
| Efectos parciales | 16 × 16,51 | Cinco paneles más legibles; sin franja de exclusión ni conteos en categorías |
| ROC y calibración | 16 × 18,42 | ROC arriba; calibración e histograma de soporte abajo |
| Plano LTR–STR | 16 × 18,80 | Cinco zonas × tres fases ENSO; escala y simbología del capítulo 5 |

Nombres de exportación: `fig_gam_desempeno_excl10d_libre`,
`fig_gam_efectos_excl10d_libre`, `fig_gam_roc_calibracion_excl10d_libre` y
`fig_gam_isolineas_excl10d_libre`, cada uno en `.pdf` y `.png`.

En Nazca se guardan en `00Figuras/06Seccion/`, relativo a `src/06_Chapter/`.
Las copias locales están en `revisado/00Figuras/06Seccion/`.

## Qué significaba «Obs. per run»

Era el **número medio de observaciones de prueba por corrida en cada intervalo de
probabilidad**. No representaba la probabilidad de ocurrencia de un deslizamiento.
Las 100 corridas tienen 200 observaciones de prueba cada una. En el histograma original:

- El intervalo `p < 0,04` contiene 86,19 no-deslizamientos por corrida.
- El intervalo `p ≥ 0,96` contiene 24,54 deslizamientos por corrida.
- Muchos intervalos intermedios contienen entre 0,5 y 3 observaciones por corrida.

El eje llegaba aproximadamente a 151 y el panel era muy bajo: los valores pequeños
parecían cero. El conteo era correcto.

La nueva versión muestra diez intervalos, iguales a los utilizados en la calibración,
y el **porcentaje dentro de cada grupo**, promediado entre corridas. Cada grupo suma 100%.
Así, el 77,91% de los no-deslizamientos está en `p < 0,1` y el 51,51% de los deslizamientos
en `p ≥ 0,9`. La tabla nueva de distribución conserva también los conteos medios.

La curva de calibración compara probabilidad predicha (x) con frecuencia observada de
deslizamientos (y). La diagonal representa acuerdo perfecto. Los puntos y barras conservan
la mediana y percentiles 5–95 entre corridas. Se mantienen los filtros originales:
al menos cinco observaciones por intervalo en cada corrida y soporte en al menos la
mitad de las corridas. Esas barras representan variación entre corridas, no un IC.

## Coherencia con el capítulo 5

El plano utiliza fondo continuo `RdYlBu_r`, contornos amarillo/naranja/rojo, círculos
blancos con borde rojo para deslizamientos y cruces grises para no-deslizamientos.
El fondo representa **probabilidad continua**, y las líneas delimitan las clases.
La superficie fija el tipo de unidad de ladera 1; los puntos corresponden a todos los
tipos de cada combinación de zona y ENSO, igual que en el notebook original.

En los rótulos se usa `Medium` como en el capítulo 5; la clave de datos `Moderate`
se conserva. La trama marca condiciones fuera del muestreo, con la relación exacta
LTR = R30 − R1 de este diseño. Las anotaciones del histograma y los captions se actualizaron.

## Inserción en Overleaf

```latex
\begin{figure}[p]
  \centering
  \includegraphics[width=\textwidth]{06Seccion/fig_gam_isolineas_excl10d_libre.pdf}
  \caption{...}
  \label{fig:res-gam-isolines}
\end{figure}
```

Conviene dedicar una página a la figura de 15 paneles y usar un caption breve para
mantener las letras a su tamaño previsto. Los captions completos sugeridos están en
la sección 8 del notebook; incluyen la descripción correcta del fondo continuo.

## Verificación

Se ejecutaron todas las celdas Python en orden en Nazca con el entorno `geo_env`.
Se compararon exactamente las siete tablas originales y la tabla de calibración:
no cambiaron los resultados. También se comprobó que los porcentajes del histograma
suman 100% para cada grupo, que los cuatro PDF miden 16 cm de ancho y que los cuatro
PNG tienen 300 dpi. Se revisaron visualmente las cuatro figuras y su texto dentro
del lienzo. Los datos y los modelos guardados no se modificaron ni se reentrenaron.

Los archivos originales se respaldan en
`/home/oisanchezp/Thesis/src/06_Chapter/revision_diseno_20260924/originales/`.
