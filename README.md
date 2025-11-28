# SESGO ALGORÍTMICO EN RECLUTAMIENTO CON IA

Este proyecto nace de una pregunta incómoda pero necesaria:
¿Un algoritmo de reclutamiento realmente evalúa talento, o solo reproduce patrones del mercado laboral?

Para explorarlo, entrené un modelo con microdatos reales de la ENOE (INEGI) y construí una aplicación que simula cómo un sistema automatizado podría aceptar o descartar perfiles laborales basándose únicamente en variables estructurales.

## OBJETIVO DEL PROYECTO

Construir y evaluar un modelo de Machine Learning capaz de estimar la probabilidad de que una persona esté empleada en México, y analizar si esa predicción refleja mérito individual o sesgos históricos relacionados con edad, escolaridad, zona de residencia y otras condiciones estructurales.

Este proyecto no busca reemplazar procesos humanos de reclutamiento, sino revelar los riesgos de automatizarlos sin supervisión ni criterios éticos.

## DATOS UTILIZADOS

- Fuente: Encuesta Nacional de Ocupación y Empleo (ENOE), INEGI

- Periodo analizado: 2° trimestre de 2025

- Registros utilizados: 20,000 observaciones

- Población: Personas de 15 a 80 años

### Descarga de datos

Los microdatos utilizados en este proyecto provienen de la Encuesta Nacional de Ocupación y Empleo (ENOE), disponibles para su descarga pública en el sitio oficial del INEGI:

https://www.inegi.org.mx/programas/enoe/15ymas/

### Variables consideradas en el modelo: 
| Variable      | Descripción                                 |
|---------------|---------------------------------------------|
| `sex`         | Sexo                                        |
| `eda_sdem`    | Edad                                        |
| `anios_esc`   | Años de escolaridad aprobados               |
| `ur_coei`     | Zona de residencia (urbano/rural)           |
| `n_hog`       | Tamaño del hogar                            |
| `n_pro_viv`   | Número de viviendas en el predio            |
| `h_mud`       | Movilidad reciente (si la persona se mudó)  |
| `empleo`      | Variable objetivo: 1 = ocupado, 0 = no ocupado |

Variable objetivo (empleo)
- 1 → persona ocupada
- 0 → persona no ocupada

Estas variables fueron seleccionadas porque son accesibles, numéricas y permiten analizar cómo un modelo puede aprender patrones socioeconómicos sin evaluar habilidades reales.

## ARQUITECTURA DE LA SOLUCIÓN

La solución completa está compuesta por tres elementos:

1️ **Notebook de análisis (Google Colab)**
- Limpieza, selección y transformación de microdatos de la ENOE
- Entrenamiento del modelo Random Forest
- Evaluación del desempeño y cálculo de importancia de variables
- Exportación del modelo entrenado (`modelo_empleo.pkl`) y las columnas predictoras (`columnas.pkl`)
- Generación de una muestra reducida del dataset (`df_model_muestra.csv`) para su uso en la aplicación

2️ **Modelo predictivo**
- Algoritmo: `RandomForestClassifier`
- Entrada: variables sociodemográficas
- Salida: probabilidad de estar empleado
- El modelo fue entrenado con 20,000 observaciones reales

3️ **Aplicación en Streamlit**
- Interfaz web donde el usuario ingresa sus datos
- El modelo predice si el perfil sería “aceptado” o “descartado”
- El umbral de decisión puede modificarse, mostrando cómo cambia la inclusión o exclusión de perfiles
- URL del deploy: https://sesgo-reclutamiento-ia-hwpzzwdxp2hdvqfpyqq5wy.streamlit.app/

## INSTRUCCIONES DE USO

### Requisitos previos

Asegúrese de tener instalado:

**- Python 3.8 o superior** 

**- pip** (gestor de paquetes) 
- Las dependencias del archivo `requirements.txt`

Para instalarlas, ejecute:
pip install -r requirements.txt

**1. Clone este repositorio**

git clone https://github.com/sambelectioner-maker/sesgo-reclutamiento-ia.git

**2. Ingrese en el directorio del proyecto**

cd sesgo-reclutamiento-ia

**3. Ejecute la aplicación**

streamlit run app.py

**La aplicación se abrirá en su navegador predeterminado**
http://localhost:8501/

### Uso en línea (sin instalación)

Puede probar el modelo directamente desde su navegador:

🔗 https://sesgo-reclutamiento-ia-hwpzzwdxp2hdvqfpyqq5wy.streamlit.app/

Solo ingrese los datos solicitados y el sistema indicará si el algoritmo lo aceptaría o descartaría para un empleo basado en patrones estadísticos.

### Archivos principales del repositorio
| Archivo                 | Descripción                                                 |
|------------------------|-------------------------------------------------------------|
| ProyectoFinal_CienciaDatos.ipynb | Cuaderno de análisis y entrenamiento del modelo        |
| aplicación.py           | Código principal de la aplicación en Streamlit              |
| columnas.pkl            | Columnas utilizadas para el modelo                          |
| modelo_empleo.pkl       | Modelo entrenado exportado en formato pickle                |
| df_modelo_muestra.csv   | Muestra de datos utilizada en la aplicación                 |
| requisitos.txt          | Dependencias necesarias para ejecutar el proyecto           |

## LIMITACIONES Y RIESGOS ÉTICOS

Este proyecto tiene fines académicos y de sensibilización. No debe utilizarse para tomar decisiones reales de contratación, ya que:

- El modelo aprende de datos históricos con posibles sesgos estructurales
- No evalúa habilidades, experiencia laboral ni competencias
- Puede discriminar perfiles basados en edad, escolaridad o zona de residencia
- Cualquier empleo real requiere supervisión humana y criterios transparentes

Este experimento demuestra que **automatizar procesos de reclutamiento sin control ético puede profundizar desigualdades preexistentes**.

## CONCLUSIÓN

Este proyecto evidencia que los modelos de Machine Learning pueden reproducir patrones históricos del mercado laboral mexicano sin evaluar talento real.  
Más que resolver el problema del reclutamiento, este trabajo **abre una conversación necesaria** sobre los peligros de delegar decisiones humanas a sistemas automatizados sin supervisión ética.

El código, modelo y aplicación son 100% reproducibles para fines educativos.


---
***
___

## 📦 Entregables del Proyecto

A continuación se listan todos los materiales correspondientes al Proyecto Final de Ciencia de Datos:

| Entregable | Link |
|-----------|------|
| 📄 Reporte Ejecutivo (PDF) | [Ver PDF](https://github.com/sambeliectioner-maker/sesgo-reclutamiento-ia/blob/main/reporte_ejecutivo.pdf) |
| 🎬 Elevator Pitch / Video | [Ver video en YouTube](https://youtube.com/shorts/5QV7Rr5Omjw?feature=share) |
| 📊 Presentación utilizada en clase | [Ver presentación en Canva](https://www.canva.com/design/DAG51Ewc0Ps/fpOjXjmewQxrR2yftWSLJQ/edit?utm_content=DAG51Ewc0Ps&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) |
| 🧠 Aplicación en Streamlit | [Abrir app](https://sesgo-reclutamiento-ia-hwpzzwdxp2hdvqfpyqq5wy.streamlit.app/) |
| 📓 Notebook de análisis | [Notebook principal](https://github.com/sambeliectioner-maker/sesgo-reclutamiento-ia/tree/main) |

> Este repositorio concentra todos los recursos necesarios para reproducir, evaluar y comprender el proyecto, incluyendo el código, el modelo, los resultados, el reporte ejecutivo, la presentación y el video pitch.
