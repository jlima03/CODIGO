README
======

Trabajo Fin de Máster
Comparativa del estudio de la carrera con distintas cámaras y puntos de vista

Autor: João Lima Duarte
Universidad Politécnica de Madrid


1. DESCRIPCIÓN GENERAL
----------------------

El archivo CODIGO_MP_YOLO.py contiene el código utilizado para procesar las
grabaciones de carrera mediante MediaPipe Pose y YOLO11 Pose y comparar los
resultados obtenidos con los datos registrados mediante VICON.

El código permite obtener los puntos anatómicos del corredor, calcular los
ángulos articulares, filtrar las señales, detectar y separar los ciclos de
carrera, normalizarlos temporalmente y comparar las curvas resultantes.


2. LIBRERÍAS UTILIZADAS
-----------------------

El código utiliza principalmente las siguientes librerías:

- OpenCV: lectura y procesamiento de los vídeos.
- MediaPipe: estimación de los puntos anatómicos mediante MediaPipe Pose.
- Ultralytics: utilización del modelo YOLO11 Pose.
- NumPy: operaciones numéricas.
- Pandas: lectura y organización de los datos.
- SciPy: filtrado, interpolación, detección de máximos y cálculo de Pearson.
- Matplotlib: representación gráfica de los resultados.

Para YOLO11 se utiliza el modelo:

yolo11n-pose.pt


3. ESTRUCTURA DEL CÓDIGO
------------------------

El código está dividido en diferentes bloques.


3.1. Cálculo de ángulos

Las funciones angle() y angle3D() permiten calcular el ángulo formado por tres
puntos.

- angle() se utiliza con las coordenadas 2D obtenidas mediante MediaPipe y
  YOLO11.
- angle3D() se utiliza con las coordenadas tridimensionales de VICON.

Posteriormente se realiza una transformación del ángulo mediante:

180 - ángulo

para mantener la misma convención angular utilizada durante el trabajo.


3.2. Lectura del vídeo

El vídeo que se desea analizar se define mediante la variable:

video_path

OpenCV abre el archivo y obtiene, entre otros parámetros, la frecuencia de
fotogramas del vídeo.

Para analizar otra prueba es necesario modificar la ruta del archivo de vídeo.


3.3. Procesamiento mediante YOLO11

YOLO11 Pose analiza cada fotograma del vídeo y obtiene los keypoints del
corredor.

Del modelo se utilizan principalmente los puntos correspondientes a:

- hombros
- caderas
- rodillas
- tobillos

A partir de ellos se calcula el ángulo de cadera o rodilla seleccionado.

YOLO11 no se utiliza para calcular el ángulo de tobillo en este trabajo, ya que
el modelo empleado no proporciona un punto del pie equivalente al utilizado
para definir dicho ángulo.


3.4. Procesamiento mediante MediaPipe

MediaPipe Pose analiza los mismos fotogramas y obtiene los landmarks
correspondientes al cuerpo del corredor.

Se utilizan principalmente:

- hombros
- caderas
- rodillas
- tobillos
- FOOT_INDEX

Estos puntos permiten calcular los ángulos de cadera, rodilla y tobillo.


3.5. Selección de la articulación y del lado

Dentro de los bloques de MediaPipe, YOLO11 y VICON aparecen las diferentes
opciones para calcular los ángulos de:

- cadera
- rodilla
- tobillo

y para seleccionar el lado derecho o izquierdo.

Para cambiar la articulación analizada es necesario comentar la línea que se
está utilizando y descomentar la correspondiente al nuevo ángulo.

La misma articulación y el mismo lado deben seleccionarse en los bloques de
MediaPipe, YOLO11 y VICON para que la comparación sea coherente.


3.6. Filtrado Butterworth

Las señales obtenidas mediante MediaPipe y YOLO11 se filtran utilizando un
filtro Butterworth paso bajo.

En el código se utiliza:

- frecuencia de corte: 6 Hz
- orden: 4
- función filtfilt()

El filtrado se aplica tanto a la señal angular como a la distancia entre los
tobillos.

Los datos de VICON utilizados en el código no se filtran. :contentReference[oaicite:1]{index=1}


3.7. Lectura de los datos VICON

Los datos de VICON se importan desde un archivo Excel.

La ruta del archivo debe modificarse dependiendo de la prueba que se quiera
analizar.

A partir de las coordenadas tridimensionales de los puntos anatómicos se
calcula el mismo ángulo seleccionado para MediaPipe y YOLO11.


3.8. Distancia entre tobillos y detección de máximos

Para MediaPipe, YOLO11 y VICON se calcula la distancia horizontal entre ambos
tobillos.

Esta señal se utiliza para localizar los máximos correspondientes a la
separación de las piernas durante la carrera.

Los máximos se detectan mediante find_peaks().

Después se seleccionan máximos alternos para trabajar siempre con ciclos
correspondientes a la misma pierna. La selección [::2] o [1::2] debe adaptarse
al lado y a la prueba que se esté analizando. :contentReference[oaicite:2]{index=2}


3.9. Segmentación de los ciclos

Los máximos seleccionados se utilizan como puntos de inicio y final de cada
ciclo.

La parte de la señal comprendida entre dos máximos consecutivos se considera
un ciclo de carrera.

Los ciclos demasiado cortos se descartan.


3.10. Normalización temporal

Cada ciclo puede contener un número diferente de muestras.

La función normalize() interpola cada ciclo y lo transforma en una señal de
101 puntos.

De esta forma, todos los ciclos quedan representados entre el 0 % y el 100 %
del ciclo de carrera y pueden compararse directamente. :contentReference[oaicite:3]{index=3}


3.11. Curva media

Una vez normalizados los ciclos, el código calcula:

- la curva media
- la desviación estándar

para MediaPipe, YOLO11 y VICON.

Estas curvas son las utilizadas posteriormente para realizar las
comparaciones.


3.12. Métricas de comparación

El código calcula las siguientes métricas:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- coeficiente de correlación de Pearson

Estas métricas permiten evaluar las diferencias entre la curva estimada y la
curva obtenida mediante VICON.

En la versión actual del archivo, este bloque calcula directamente las métricas
entre MediaPipe y VICON. 


3.13. Representación de los resultados

Finalmente se generan dos tipos principales de gráficos:

1. Comparación de las curvas en grados.
2. Comparación de las curvas normalizadas mediante z-score.

En ambos casos se representan conjuntamente las curvas medias de:

- MediaPipe
- YOLO11
- VICON

junto con sus correspondientes desviaciones estándar.

La normalización z-score se utiliza para facilitar la comparación visual de la
forma de las curvas independientemente de sus diferencias de magnitud.


4. CÓMO UTILIZAR EL CÓDIGO
--------------------------

Para analizar una prueba diferente deben revisarse principalmente los
siguientes elementos:

1. Ruta del vídeo en video_path.
2. Ruta del archivo Excel correspondiente a VICON.
3. Articulación que se desea analizar.
4. Lado derecho o izquierdo que se desea analizar.
5. Selección de los máximos [::2] o [1::2].
6. Frecuencia de muestreo utilizada por el filtro cuando corresponda.

Después de realizar estas modificaciones, se puede ejecutar nuevamente el
script para obtener las curvas y las métricas correspondientes a la prueba.


5. OBSERVACIONES
----------------

El código fue desarrollado específicamente para los datos utilizados en este
Trabajo Fin de Máster.

Por este motivo, las rutas de los archivos, los nombres de las columnas de
VICON y algunos parámetros de procesamiento deben adaptarse cuando se cambia
de prueba o se utilizan otros datos.

El procedimiento de segmentación utilizado está planteado para las
grabaciones laterales analizadas en el trabajo.