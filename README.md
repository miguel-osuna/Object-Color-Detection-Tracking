# Object-Color-Detection-Tracking

## **Marco Teórico**

### *¿Qué es CV2?*
Es una biblioteca que habilita el uso de funciones básicas para Python y el
procesamiento de imágenes.

### *¿Qué es Numpy?*
Numpy es un módulo importante de Python, es el encargado de añadir toda la
capacidad matemática y vectorial a Python haciendo posible operar con cualquier
dato numérico o arreglo. Incorpora operaciones tan básicas como la suma o la
multiplicación u otras mucho más complejas como la transformada de Fourier o el
álgebra lineal.

El origen de Numpy se debe principalmente al diseñador de software Jim Hugunin
quien diseñó el módulo Numeric para dotar a Python de capacidades de cálculo
similares a las de otros softwares como MATLAB. Posteriormente, mejoró Numeric
incorporando nuevas funcionalidades naciendo lo que hoy conocemos como
Numpy.

En resumen es la biblioteca central para computación científica en Python la cual
proporciona un objeto de matriz multidimensional de alto rendimiento y herramientas
para trabajar con estas matrices, en diferentes lenguajes de programación.

### *Funciones utilizadas*
|   **Funciones**  |
|     :---:    |
| `cv2.namedWindow` |
| `cv2.videoCapture` |
| `camera.read()` |
| `cv2.GaussianBlur` |
| `cv2.cvtColor` |
| `cv2.inRange`  | 
| `cv2.morphologyExe` |
| `cv2.findContours`  |
| `cv2.minEnclosingCircle` |
| `cv2.moments`  |
| `cv2.circle`   | 
| `cv2.putText`  |
| `cv2.destroyAllWindows` |

### *¿Cómo se representa una imagen a color?*
Las imágenes se codifican como matrices. En particular, las imágenes de intensidad
o escala de grises se codifican como una matriz de dos dimensiones, donde cada
número representa la intensidad de un pixel.

Pero eso significa que cualquiera de estas matrices que generamos se puede
visualizar como una matriz. Para visualizar imágenes, usamos el módulo pyplot de
la librería matplotlib.

La función que nos permite visualizar matrices es imshow del módulo pyplot, que
invocamos como plt.imshow, y recibe la imagen como parámetro.
También podemos generar imágenes blancas, o grises. Si queremos mostrar más
de una imagen en una celda, vamos a tener que ejecutar plt.figure()para crear la
figura que contenga la imagen.

### *¿Qué es la representación de colores en HSV?*
El espacio de color HSV se compone de tres propiedades del color:

- H - Tono ("Hue": Longitud de onda dominante)
- S - Saturacion ("Saturation": Pureza / tonos del color)
- V - Valor ("Value": Intensidad

Usualmente se utiliza la figura geométrica de un cono para representar este espacio
de color, como se muestra en la figura.

<p align="center"> 
<img src="https://user-images.githubusercontent.com/48000150/57259121-5983a500-7013-11e9-99c1-c6dc4b65a55e.PNG">
</p>

Del contorno de la base o circunferencia emergen los colores primarios, la escala en
grados sexagesimales va de 0° = Rojo, a 120° donde aparece la tonalidad Verde,
luego la tonalidad Azul en los 240° y hasta los 360°, donde el ciclo es principio y fin,
porque aparece el Rojo nuevamente. [] Siendo una forma más parecida a la que el
ojo humano percibe los colores, pues las tonalidades se agrupan, es decir el modelo
hsv es una transformación no lineal del modelo RGB en coordenadas cilíndricas.

### *¿Qué es el ruido en una imagen?*
Se presenta como píxeles aislados que toman un nivel de gris diferente al de sus
vecinos, es decir el ruido aparece como una variación aleatoria en los valores de
cada pixel de la imagen.

### *¿Qué tipos de ruido existe?*

- **Gaussiano:** produce pequeñas variaciones en la imagen; generalmente se
debe a diferentes ganancias en la cámara, ruido en los digitalizadores,
perturbaciones en la transmisión, etc. Se considera que el valor final del píxel
sería el ideal más una cantidad correspondiente al error que puede
describirse como una variable aleatoria gaussiana.

- **Impulsional (Sal y Pimienta):** el valor que toma el píxel no tiene relación con
el valor ideal, sino con el valor del ruido que toma valores muy altos o bajos
(puntos blancos y/o negros) causados por una saturación del sensor o por un
valor mínimo captado, si se ha perdido la señal en ese punto. Se encuentran
también al trabajar con objetos a altas temperaturas, ya que las cámaras
tienen una ganancia en el infrarrojo que no es detectable por el ojo humano;
por ello las partes más calientes de un objeto pueden llegar a saturar un
píxel.

- **Multiplicativo:** la imagen obtenida es el resultado de la multiplicación de dos
señales.

<p align="center"> 
<img src="https://user-images.githubusercontent.com/48000150/57259124-5b4d6880-7013-11e9-8b8f-b01ef76f972e.PNG">
</p>

### *¿Qué es una máscara (kernel)?*
En el procesamiento de imágenes, un kernel, una matriz de convolución o una máscara es una matriz pequeña. Se usa para desenfoque, afilado, estampado, detección de bordes y más. Esto se logra haciendo una convolución entre un kernel y una imagen.

## **Desarrollo**

```python

# Importar las bibliotecas necesarias
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

# construye el argumento parse
ap = argparse.ArgumentParser()
args = vars(ap.parse_args())

# Se definen los límites para cada color en el espacio de color HSV
lower = { 'red' :( 166 , 84 , 141 ), 'green' :( 66 , 122 , 129 ), 'blue' :( 97 , 100 , 117 ),
'yellow' :( 23 , 59 , 119 ), 'orange' :( 0 , 50 , 80 )}
upper = { 'red' :( 186 , 255 , 255 ), 'green' :( 86 , 255 , 255 ), 'blue' :( 117 , 255 , 255 ),
'yellow' :( 54 , 255 , 255 ), 'orange' :( 20 , 255 , 255 )}

# Se definen los estándares de colores necesarios
colors = { 'red' :( 0 , 0 , 255 ), 'green' :( 0 , 255 , 0 ), 'blue' :( 255 , 0 , 0 ),
'yellow' :( 0 , 255 , 217 ), 'orange' :( 0 , 140 , 255 ), 'brown' :( 165 , 42 , 42 )}

# Se genera una captura de video y se asigna a una variable
camera = cv2.VideoCapture( 0 )

# Loop
while True :
    # Obtenemos cuadro por cuadro tomado, y lo convertimos a una formato
    # de color HSV, ya que se representa de una mejor manera con esto.
    (grabbed, frame) = camera.read()
    
    # Si estamos viendo un video y no tomamos un fotograma,
    # hemos llegado al final del video.
    if args.get( "video" ) and not grabbed:
        break
   
    # Se crea un tamano distinto
    frame = imutils.resize(frame, width= 600 )
    
    # Se suaviza la imagen utilizando un filtro gaussiano, eliminando
    # cualquier presencia de ruido en el frame
    blurred = cv2.GaussianBlur(frame, ( 11 , 11 ), 0 )
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Se utiliza una serie de dilataciones y erosiones a la imagen para
    # poder remover cualquier señal no deseada en la máscara utilizada
    for key, value in upper.items():
        # Utiliza un kernel de 9x9
        kernel = np.ones(( 9 , 9 ),np.uint8)
        mask = cv2.inRange(hsv, lower[key], upper[key])
        
        # Reduce y luego expande: opening
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Expandey luego reduce: closing
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Encuentra el contorno de una imagen binaria e iniciliza el centro
        de la bola (x,y)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[ -2 ]
        center = None
        
        # Prosigue si está presente al menos un contorno
        if len(cnts) > 0:
            # Encuentra el contorno mas largo de la mascara
            # Luego lo usa para calcular el circulo y centroide minimo

            # cv2.contourArea calcula el area de un contorno
            c = max(cnts, key=cv2.contourArea)

            # cv2.minEnclosingCircle()
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            # cv2.moments calcula todos los momentos hasta el tercer
            # orden de un poligono o una figura rasterizada
            M = cv2.moments(c)
            center = (int(M[ "m10" ] / M[ "m00" ]), int(M[ "m01" ] /
            M[ "m00" ]))
        
            # Se corrige el radio mínimo del círculo
            if radius > 0.5:
                cv2.circle(frame, (int(x), int(y)), int(radius),
                colors[key], 2 )
                cv2.putText(frame,key + " object" ,
                (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6 ,colors[key], 2 )
    
        # Muestra el cuadro a nuestra ventana
        cv2.imshow( "Frame" , frame)
        key = cv2.waitKey( 1 ) & 0xFF

        # Cuando 'q' es presionado, deten el ciclo for
        if key == ord( "q" ):
            break

            # Limpia la camara y cierra cualquier ventana abierta
            camera.release()
            cv2.destroyAllWindows()
```
En el código utilizamos la representación de colores HSV debido a que presenta una mayor
facilidad al momento de detectar los colores de la imagen.

Se definieron los límites inferior y superior indicados para 5 colores distintos:

- Rojo
- Verde
- Azul
- Amarillo
- Naranja

Al momento de obtener la imagen del video, se suaviza la imagen aplicando un filtro
Gaussiano para eliminar cualquier presencia de ruido. Después se utiliza una series de
dilataciones y erosiones para rellenar la imagen en caso de ser necesario.

Una vez que se tiene la imagen perfecta, se busca cualquier contorno que coincida con uno
de los colores predeterminados, y cuando lo hace, lo encierra en un círculo del mismo color,
indicando cuál es.

**Reconocimiento de Objeto Rojo**

<p align="center"> 
<img src="https://user-images.githubusercontent.com/48000150/57259135-630d0d00-7013-11e9-94ed-f2701a82b611.PNG">
</p>

**Reconocimiento de Objeto Verde**

<p align="center"> 
<img src="https://user-images.githubusercontent.com/48000150/57259137-63a5a380-7013-11e9-8f27-31dc40d2a9b0.PNG">
</p>

**Reconocimiento de Objeto Azul**

<p align="center"> 
<img src="https://user-images.githubusercontent.com/48000150/57259139-63a5a380-7013-11e9-92e9-b82f791430c5.PNG">
</p>

**Reconocimiento de Objeto Amarillo y Naranja**

<p align="center"> 
<img src="https://user-images.githubusercontent.com/48000150/57259154-6ef8cf00-7013-11e9-9d81-d145425332e2.PNG">
</p>



