# Master's thesis in Environment and Development

Esta plantilla proporciona una estructura general para proyectos de investigación. Las siguientes secciones explicarán cómo y por qué se define esta estructura, además de proporcionar las mejores prácticas para mantener los proyectos organizados. Siguiendo esta estructura, los investigadores podrán familiarizarse mejor con los proyectos en curso, y los proyectos terminados se archivarán de manera que garantice el acceso futuro a los datos y métodos. Esta plantilla se desarrolló como una herramienta para ayudar a mantener la continuidad dentro de un contexto de investigación organizacional en Geociencias SIATA. Se basa en las estructuras proporcionadas en:
- https://github.com/EthanJantz/Research-Project-Template
- https://github.com/coderefinery/reproducible-research

## Repository structure

Esta plantilla se basa en 4 carpetas: Información (info), Datos (data), Scripts (src) y Resultados (results). Adicionalmente, tiene 2 carpetas para la documentación del código (doc) y el manuscripo derivado (manuscript).

El proyecto tiene la siguiente estructura:
```shell
project_name/
├── README.md
├── info/
├── data/
│   ├── README.md
│   ├── raw/ 
│   ├── processed/     
│   └── metadata/ 
├── manuscript/
│   ├── plantilla
│   ├── figs
│   ├── manuscrito.tex
│   └── referencias.bib
├── results/
├── src/
│   ├── LICENSE
│   └── requirements.txt
└── doc/
```

### Carpeta root
La carpeta root es donde se encuentran todos los archivos y carpetas del proyecto. Aquí van los principales archivos del proyecto, que incluyen:

- Un archivo README.md que contiene información importante sobre el proyecto. Esto incluye instrucciones para replicar cualquier hallazgo, información detallada sobre dónde se obtuvieron los datos y algo de información narrativa explicando los orígenes y propósito del proyecto. Si este proyecto está alojado en un repositorio de GitHub, este archivo README será lo primero que una persona vea si mira la página del repositorio. En esta carpeta también se encontrarían los proyectos de ArcGIS o QGIS que sean utilizados en el desarrollo del análisis.
- Un archivo .gitignore que es una herramienta fundamental en el control de versiones con Git. Su propósito principal es indicar a Git qué archivos o carpetas debe ignorar y no incluir en el repositorio. Esto es especialmente útil para evitar subir archivos innecesarios, como archivos de configuración local, archivos temporales generados por el sistema operativo, archivos de compilación, o cualquier otro tipo de archivo que no sea relevante para el proyecto en sí. Se utiliza colocando nombres de archivos, extensiones o patrones de nombres de archivos en el archivo .gitignore, lo que permite a Git filtrar estos archivos automáticamente. Esto garantiza que el repositorio se mantenga limpio y ordenado, evitando la sobrecarga de archivos no deseados y facilitando el trabajo colaborativo entre los miembros del equipo.

### Carpeta de Información
La carpeta info está ubicada dentro de la carpeta root. Contiene notas y narrativa que no están directamente relacionadas con los procesos técnicos realizados en la investigación. Esto podría incluir:

- Documentos que contienen bibliografías anotadas o revisiones de literatura.
- Documentos narrativos que detallan información importante de antecedentes sobre el proyecto, como cómo se tomaron decisiones de investigación o notas sobre quién ha participado en el proyecto y cuál fue su papel.
- Registros de acciones realizadas dentro del proyecto y listas de tareas pendientes para que los recién llegados puedan ver qué se ha hecho y qué aún necesita hacerse.

Por lo general, esta carpeta será la primera en llenarse a medida que los investigadores recopilen información de antecedentes y notas sobre cómo llevarán a cabo su investigación.

### Carpeta de Datos
La carpeta data está ubicada dentro de la carpeta root. Todos los datos relacionados con un proyecto se guardarán dentro de las subcarpetas de la carpeta data. Hay múltiples subcarpetas dentro de la carpeta data:

- La carpeta raw contiene todos los datos de investigación crudos y sin procesar. Esto podría ser series de tiempo capturadas por sensores, datos espaciales extraídos de imágenes satelitales, o hojas de cálculo proporcionadas por alguna entidad. Por lo general, los datos crudos se dejan tal cual después de ser guardados en esta carpeta. Cualquier cambio o resultados utilizando datos crudos se guardan en la carpeta processed.
- La carpeta processed contiene archivos de datos que han sido alterados (o manipulados) como parte del proceso de investigación. Por ejemplo, una tabla de nuevas variables creadas utilizando datos brutos de los sensores se guardaría en esta carpeta. Mantener los datos crudos y procesados separados facilita a los investigadores mantenerse organizados y evitar errores costosos resultantes de nombres de archivo ambiguos.
- La carpeta metadata contiene cualquier dato que se haya creado como parte del proceso de investigación o que describa las relaciones entre datos en las otras dos subcarpetas de data.

Es una buena práctica mantener el número de subcarpetas por debajo de 4. Esta estructura fue diseñada para permitir a los investigadores libertad en cómo organizan los datos dentro de las subcarpetas data, al mismo tiempo que proporciona una estructura predecible para futuras referencias. Por razones organizativas, es una buena práctica mantener los archivos de datos espaciales como archivos .shp y archivos relacionados separados de los archivos de datos tabulares como archivos .csv o .xlsx utilizando subcarpetas. También se recomienda mantener los datos separados por fuente, por lo que tener carpetas dentro de la carpeta raw, por ejemplo, con etiquetas como "Pluviómetro" o "Sentinel2" sería apropiado.

### Carpeta de Scripts
La carpeta src está ubicada dentro de la carpeta root. Esta carpeta contiene scripts auxiliares como archivos .R, .py o .SQL que se usarían para manipular, transformar y analizar los datos pero no son el archivo principal del proyecto. En un proyecto de python, esta carpeta podría contener un script funciones.py para mantener funciones personalizadas utilizadas en la investigación. Al finalizar la investigación también se debe generar un archivo requirements.txt que contenga los paquetes y versiones compatibles con los códigos utilizados, se puede generar con:
   ```console
   $ pip freeze > requirements.txt
   ```

### Carpeta de Resultados
La carpeta results está ubicada dentro de la carpeta root. Todos los resultados finales de la investigación se guardan en esta carpeta. En general, un proyecto se puede considerar terminado una vez que se ha guardado un resultado externo, como un informe final, artículos científicos, presentaciones en conferencias, infografías o mapas en formatos .pdf o de imagen. Los archivos guardados aquí están destinados explícitamente a ser compartidos con el público.

### Carpeta Manuscrito
La carpeta manuscript alberga el manuscrito, bitácora de investigación o informe de investigación que describe los resultados obtenidos. Este espacio está destinado exclusivamente para la redacción del documento final, donde se detallan los hallazgos, análisis y conclusiones del estudio. Se recomienda llevar una estructura tipo artículo científico. Esta carpeta contiene 3 elementos vitales:
- La subcarpeta plantilla que contiene todos los elementos gráficos y archivos tex auxiliares para la compilación correcta del documento. Se encuentra actualizada con el concepto gráfico del informe mensual del contrato 106 de 2024.
- El archivo manuscrito.tex que contiene la estructura básica y es el documento principal a ser llenado.
- El arvhico referencias.bib que debe ser llenado con las referencias utilizadas en formato BibTeX.

### Carpeta de Documentación
La carpeta doc contiene la documentación relacionada con el proyecto. Aquí se pueden encontrar los archivos que sirven como punto de partida para la generación de documentación técnica o instructiva sobre el proyecto. Se recomienda trabajar con [Sphinx](https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html).

## Buenas prácticas:
Las siguientes son recomendaciones sobre las mejores prácticas al administrar y organizar un proyecto de investigación.
- Evite los espacios en los nombres de directorios y archivos: es más feo para los humanos pero útil para las computadoras.
- Mantén las carpetas lo más superficiales posible. ; si alguien que accede a tus datos necesita recorrer 10 subcarpetas para encontrar un archivo, puede que quieras considerar reorganizar tus carpetas.
- Utiliza nombres de archivos concisos y descriptivos. Evita crear una serie de actualizaciones de modo que tu archivo final se llame output_final_FINALFINAL_FINALFORREAL.jpeg.
- Si necesita separar público/secreto, use `.gitignore` o una carpeta separada que no esté en Git
- Use el **archivo README** para describir el proyecto e instrucciones sobre cómo reproducir los resultados.
- Si un código se reutiliza en varios proyectos, puede tener sentido colocarlo en el repositorio principal del grupo.
- Todo el código está controlado por versión y va en el directorio `src/`
- También puede controlar la versión de archivos de datos o archivos de entrada en `datos/`
- Si los archivos de datos son demasiado grandes (o sensibles) para rastrearlos, elimine el seguimiento usando `.gitignore`
- Los archivos intermedios del análisis se guardan en `processed_data/`
- Considere el uso de etiquetas Git para marcar versiones específicas de los resultados (versión presentado a una revista, versión de tesis, versión de póster, etc.):
   ```console
   $ git tag -a thesis-submitted -m "Esta es la versión enviada de mi tesis"
   ```

## Usando esta Plantilla
Cuando comiences un proyecto por primera vez, copia esta carpeta a una nueva ubicación y renombra la plantilla a algo apropiado para el proyecto. Antes de hacer cualquier otra cosa, asegúrate de haber leído la sección anterior que explica la estructura del proyecto y llena el archivo README.md con la información básica. A medida que trabajas en el proyecto, asegúrate de actualizar este archivo, es mucho más fácil hacerlo durante el proceso que hacerlo todo una vez que termines el proyecto.
