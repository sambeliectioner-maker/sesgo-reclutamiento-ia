import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ---------------------------------------------------------
# Configuración general de la app
# ---------------------------------------------------------
st.set_page_config(
    page_title="Modelo de predicción de Empleo en México",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Modelo de predicción de Empleo en México")
st.write("Basado en microdatos de la **ENOE (INEGI)**.")

# ---------------------------------------------------------
# Carga de datos y modelo (con caché)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("df_model_muestra.csv")
    return df


@st.cache_resource
def load_model():
    model = joblib.load("modelo_empleo.pkl")
    columnas = joblib.load("columnas.pkl")
    return model, columnas


df = load_data()
model, columnas_modelo = load_model()

# Aseguramos tipos numéricos
for c in columnas_modelo + ["empleo"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=columnas_modelo + ["empleo"])
df[columnas_modelo] = df[columnas_modelo].fillna(0)

# ---------------------------------------------------------
# Probabilidades para todo el dataset (para dashboard/sesgos)
# ---------------------------------------------------------
X_all = df[columnas_modelo]
probas_all = model.predict_proba(X_all)[:, 1]
df["proba_empleo"] = probas_all

# Grupos de edad para gráficas
df["grupo_edad"] = pd.cut(
    df["eda_sdem"],
    bins=[15, 25, 35, 45, 60, 90],
    labels=["15–24", "25–34", "35–44", "45–59", "60+"],
    right=False,
)

# Etiquetas legibles
df["sexo_label"] = df["sex"].map({1: "Hombre", 2: "Mujer"}).fillna("Otro")
df["zona_label"] = df["ur_coei"].map({1: "Urbana", 2: "Rural"}).fillna("Otra")

# Mapeos para los inputs
SEXO_TO_CODE = {"Mujer": 2, "Hombre": 1}
ZONA_TO_CODE = {"Urbana": 1, "Rural": 2}


# ---------------------------------------------------------
# Función auxiliar: gráfico de descartes por edad
# ---------------------------------------------------------
def fig_descartes_por_edad(df_base, umbral=0.5):
    df_tmp = df_base.copy()
    df_tmp["descartado"] = df_tmp["proba_empleo"] < umbral

    desc_edad = (
        df_tmp.groupby("grupo_edad")["descartado"]
        .mean()
        .reset_index(name="prop_descartada")
    )

    fig = px.bar(
        desc_edad,
        x="grupo_edad",
        y="prop_descartada",
        labels={
            "grupo_edad": "Grupo de edad",
            "prop_descartada": "Proporción descartada",
        },
        title="Perfiles descartados por grupo de edad",
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))
    return fig


# ---------------------------------------------------------
# Barra lateral – inputs para la predicción individual
# ---------------------------------------------------------
st.sidebar.header("Ingresa los datos de la persona:")

sexo_input = st.sidebar.selectbox("Sexo", list(SEXO_TO_CODE.keys()))
edad_input = st.sidebar.slider("Edad", min_value=15, max_value=80, value=27)

anios_esc_input = st.sidebar.slider(
    "Años de escolaridad aprobados",
    min_value=0,
    max_value=20,
    value=12,
    help=(
        "📘 Años aprobados desde primaria en adelante (NO incluye kínder). "
        "Ejemplos: Secundaria completa = 9, Prepa = 12, Universidad = 16, "
        "Maestría = 18, Doctorado = 20."
    ),
)

zona_input = st.sidebar.selectbox("Zona de residencia", list(ZONA_TO_CODE.keys()))
n_hog_input = st.sidebar.slider(
    "¿Cuántas personas viven en tu casa?",
    min_value=1,
    max_value=15,
    value=3,
)
n_pro_viv_input = st.sidebar.slider(
    "¿Cuántas viviendas hay en el mismo terreno donde vives?",
    min_value=1,
    max_value=20,
    value=1,
)
h_mud_input = st.sidebar.selectbox(
    "¿Se mudó recientemente? (1 Sí / 0 No)",
    options=["No", "Sí"],
)

if st.sidebar.button("Calcular predicción"):
    st.session_state["do_predict"] = True

# ---------------------------------------------------------
# Tabs principales
# ---------------------------------------------------------
tab_pred, tab_dash, tab_descartes, tab_info = st.tabs(
    [
        "🔮 Predicción individual",
        "📊 Panel de control ENOE",
        "🚫 Perfiles descartados por el algoritmo",
        "📘 Acerca del modelo",
    ]
)

# =========================================================
# TAB 1 – PREDICCIÓN INDIVIDUAL
# =========================================================
with tab_pred:
    st.subheader("Resultado de la Predicción")

    if st.session_state.get("do_predict", False):
        # Construimos el registro con el mismo orden de columnas del modelo
        input_dict = {
            "sex": SEXO_TO_CODE[sexo_input],
            "eda_sdem": edad_input,
            "anios_esc": anios_esc_input,
            "ur_coei": ZONA_TO_CODE[zona_input],
            "n_hog": n_hog_input,
            "n_pro_viv": n_pro_viv_input,
            "h_mud": 1 if h_mud_input == "Sí" else 0,
        }

        df_input = pd.DataFrame([input_dict])[columnas_modelo]
        df_input = df_input.fillna(0)

        proba = model.predict_proba(df_input)[0][1]  # 0 a 1
        proba_pct = proba * 100
        umbral = 0.5

        # Mensaje principal (aceptado / rechazado)
        if proba >= umbral:
            st.success(
                "✅ El modelo predice que esta persona **sí estaría empleada** "
                "(o sería considerada 'aceptable' por un filtro automático)."
            )
        else:
            st.error(
                "❌ El modelo predice que esta persona **no sería contratada automáticamente** "
                "por un filtro basado solo en estos datos."
            )

            # Zona gris (sesgo / decisión dudosa)
            if 0.40 <= proba <= 0.60:
                st.warning(
                    "Esta predicción está en la **zona gris** (alrededor del 50%). "
                    "Aquí es donde un sistema automático tiende a **descartar por comodidad**, "
                    "aunque la persona podría tener talento, habilidades transferibles o "
                    "motivación para aprender el puesto."
                )
            else:
                st.info(
                    "Una probabilidad baja **no significa** que la persona no tenga talento; "
                    "solo refleja cómo se han comportado perfiles similares en los datos históricos."
                )

        st.write(
            f"**Probabilidad estimada de empleo / aceptación:** {proba_pct:.2f}%"
        )

        st.markdown("#### Cómo interpretar el resultado:")
        st.markdown(
            """
- Valores cercanos a 0% → muy baja probabilidad de ser contratada por este sistema.  
- Valores cercanos al 50% → escenario incierto, donde una revisión humana sería clave.  
- Valores cercanos al 100% → el sistema tiende a considerar a estos perfiles como 'seguros'.
            """
        )

        with st.expander("Ver datos que se envían al modelo (debug)"):
            st.json(input_dict)
            st.write("DataFrame que entra al modelo:")
            st.dataframe(df_input)

    else:
        st.info(
            "Utilice el botón **Calcular predicción** de la barra lateral para ver el resultado."
        )


# =========================================================
# TAB 2 – PANEL DE CONTROL ENOE
# =========================================================
with tab_dash:
    st.subheader("📊 Análisis descriptivo y Dashboard de la ENOE")
    st.write(
        f"El conjunto de datos de muestra contiene **{len(df):,} observaciones** "
        f"y **{len(columnas_modelo) + 1} variables** (incluyendo la variable objetivo)."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Distribución de la edad")
        fig_age = px.histogram(
            df,
            x="eda_sdem",
            nbins=30,
            labels={"eda_sdem": "Edad (años)", "count": "Número de personas"},
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        st.markdown("##### Distribución de años de escolaridad aprobados")
        fig_esc = px.histogram(
            df,
            x="anios_esc",
            nbins=25,
            labels={"anios_esc": "Años de escolaridad", "count": "Número de personas"},
        )
        st.plotly_chart(fig_esc, use_container_width=True)

    st.markdown("### Empleo por sexo y zona")
    col3, col4 = st.columns(2)

    with col3:
        tasa_sexo = (
            df.groupby("sexo_label")["empleo"]
            .mean()
            .reset_index(name="tasa_empleo")
        )
        fig_sexo = px.bar(
            tasa_sexo,
            x="sexo_label",
            y="tasa_empleo",
            labels={"sexo_label": "Sexo", "tasa_empleo": "Tasa de empleo"},
        )
        fig_sexo.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_sexo, use_container_width=True)

    with col4:
        tasa_zona = (
            df.groupby("zona_label")["empleo"]
            .mean()
            .reset_index(name="tasa_empleo")
        )
        fig_zona = px.bar(
            tasa_zona,
            x="zona_label",
            y="tasa_empleo",
            labels={"zona_label": "Zona", "tasa_empleo": "Tasa de empleo"},
        )
        fig_zona.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_zona, use_container_width=True)

    # ===== Matriz de correlación mejor presentada =====
    st.markdown("### Matriz de evaluación (variables numéricas)")

    nombres_corr = {
        "eda_sdem": "Edad",
        "anios_esc": "Escolaridad",
        "ur_coei": "Zona",
        "n_hog": "Tamaño del hogar",
        "n_pro_viv": "Viviendas en el predio",
        "h_mud": "Movilidad",
        "empleo": "Empleo",
    }

    cols_corr = list(nombres_corr.keys())
    corr = df[cols_corr].corr()
    corr = corr.rename(index=nombres_corr, columns=nombres_corr)

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="Blues",
        labels=dict(color="Correlación"),
    )

    fig_corr.update_layout(
        height=650,
        width=800,
        margin=dict(l=80, r=80, t=80, b=80),
        xaxis_title="Variables",
        yaxis_title="Variables",
    )
    fig_corr.update_xaxes(tickangle=-35)

    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("### Importancia de variables en el modelo (Random Forest)")

    importances = model.feature_importances_
    df_importances = pd.DataFrame(
        {"variable": columnas_modelo, "importancia": importances}
    ).sort_values("importancia", ascending=False)

    # Nombres más legibles
    nombres_bonitos = {
        "sex": "Sexo",
        "eda_sdem": "Edad",
        "anios_esc": "Años de escolaridad",
        "ur_coei": "Zona (urbana/rural)",
        "n_hog": "Tamaño del hogar",
        "n_pro_viv": "Número de viviendas en el predio",
        "h_mud": "Movilidad reciente",
    }
    df_importances["variable_legible"] = df_importances["variable"].map(
        nombres_bonitos
    )

    fig_imp = px.bar(
        df_importances,
        x="variable_legible",
        y="importancia",
        labels={"variable_legible": "Variable", "importancia": "Importancia"},
    )
    fig_imp.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown(
        """
**Cómo interpretar esto:**

- Entre más alta la barra → mayor peso tiene esa variable en la decisión del modelo.  
- Esto **no** determina causalidad, sino influencia estadística.  
- Aquí se ve cómo un sistema puede dar más peso a variables estructurales (edad, escolaridad, tipo de zona) que a lo que realmente define el talento.
        """
    )


# =========================================================
# TAB 3 – PERFILES DESCARTADOS POR EL ALGORITMO
# =========================================================
with tab_descartes:
    st.subheader("🚫 ¿Qué perfiles tendería a descartar un algoritmo como este?")

    st.markdown(
        """
Aquí usamos el **mismo modelo** para simular cómo se comportaría un filtro automatizado de reclutamiento:

- Calcula una probabilidad para cada persona en el dataset.  
- Compara esa probabilidad contra un umbral (un corte).  
- A quienes quedan por debajo del umbral los marcamos como **“Descartado”**, aunque en la vida real podrían tener talento, habilidades transferibles o motivación suficientes para aprender el trabajo.

La idea **NO** es decir quién merece o no un empleo, sino mostrar cómo un sistema automático puede excluir perfiles solo por cómo se ven en los datos.
        """
    )

    umbral = st.slider(
        "Umbral del sistema para considerar 'aceptado' a un perfil",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )

    df_umbral = df.copy()
    df_umbral["descartado"] = df_umbral["proba_empleo"] < umbral
    prop_descartados = df_umbral["descartado"].mean()

    st.markdown(
        f"Con el umbral actual, aproximadamente **{prop_descartados:.1%}** "
        f"de los perfiles serían **descartados automáticamente** sin que nadie revise su potencial."
    )

    st.markdown("### ¿A quién está descartando más el sistema?")

    col1, col2 = st.columns(2)

    # Por nivel educativo (años de escolaridad)
    with col1:
        desc_esc = (
            df_umbral.groupby("anios_esc")["descartado"]
            .mean()
            .reset_index(name="prop_descartada")
        )
        fig_esc_desc = px.bar(
            desc_esc,
            x="anios_esc",
            y="prop_descartada",
            labels={
                "anios_esc": "Años de escolaridad aprobados",
                "prop_descartada": "Proporción descartada",
            },
        )
        fig_esc_desc.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_esc_desc, use_container_width=True)

    # Por zona (urbana/rural)
    with col2:
        desc_zona = (
            df_umbral.groupby("zona_label")["descartado"]
            .mean()
            .reset_index(name="prop_descartada")
        )
        fig_zona_desc = px.bar(
            desc_zona,
            x="zona_label",
            y="prop_descartada",
            labels={
                "zona_label": "Zona",
                "prop_descartada": "Proporción descartada",
            },
        )
        fig_zona_desc.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_zona_desc, use_container_width=True)


# =========================================================
# TAB 4 – ACERCA DEL MODELO
# =========================================================
with tab_info:
    st.subheader("📘 Detalles del modelo y los datos")

    st.markdown(
        """
### Datos utilizados

- **Fuente:** Encuesta Nacional de Ocupación y Empleo (ENOE), INEGI.  
- **Año / trimestre:** 2025, segundo trimestre (T2).  
- **Tablas combinadas:** COE1 (condición de ocupación) y SDEM (sociodemográficos).  
- **Población analizada:** personas de 15 a 80 años de edad.

### Variable objetivo

- **empleo**: 1 = persona ocupada (empleada), 0 = persona no ocupada.  
- En este proyecto, la usamos como **proxy** de “perfil aceptado” por un filtro automático que aprende de los datos históricos.

### Modelo

- Tipo de modelo: **Random Forest Classifier**.  
- Objetivo: Estimar la probabilidad de que una persona sea clasificada como empleada/aceptada con base en variables estructurales:  
  sexo, edad, años de escolaridad, zona urbana/rural, tamaño del hogar, número de viviendas en el predio y movilidad reciente.

### Limitaciones importantes

- El modelo **no evalúa talento**, motivación, habilidades transferibles ni potencial de aprendizaje.  
- Solo ve lo que está en la base de datos: variables duras y simplificadas.  
- Justamente por eso sirve como ejemplo de cómo un sistema automatizado puede tomar decisiones laborales injustas **si se usa sin supervisión humana**.

### Interpretación crítica

Este modelo **NO debe usarse** para decidir sobre personas reales.

Su propósito es mostrar que:

1. Un algoritmo puede aprender patrones de desigualdad del propio mercado laboral.  
2. Si estos modelos se integran en procesos de reclutamiento, pueden profundizar la exclusión de perfiles que no encajan perfecto en el “molde”, aunque sí tengan talento para el puesto.  
3. Es necesario discutir regulación, transparencia y derecho a explicación cuando se usan sistemas de IA en decisiones que afectan el acceso al empleo.
        """
    )

