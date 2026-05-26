"""
🧹 App de Limpieza de Datos
Bootcamp IA
"""

import re
import io
import pandas as pd
import streamlit as st

# ─── Configuración de la página ───────────────────────────────────────────────
st.set_page_config(page_title="Limpieza de Datos", page_icon="🧹", layout="wide")

st.title("🧹 Limpieza de Datos")
st.caption("Carga un CSV o Excel, aplica las técnicas de limpieza y descarga el resultado.")

# ─── Funciones de limpieza ────────────────────────────────────────────────────

def reporte_calidad(df: pd.DataFrame) -> dict:
    """Calcula métricas básicas de calidad del dataset."""
    total_celdas = df.shape[0] * df.shape[1]
    nulos = int(df.isnull().sum().sum())
    duplicados = int(df.duplicated().sum())
    return {
        "Filas": df.shape[0],
        "Columnas": df.shape[1],
        "Valores nulos": nulos,
        "% Nulos": round(nulos / total_celdas * 100, 2) if total_celdas else 0,
        "Duplicados": duplicados,
        "Completitud (%)": round((1 - nulos / total_celdas) * 100, 1) if total_celdas else 100,
    }


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte los nombres de columnas a snake_case sin acentos."""
    mapa = {}
    for col in df.columns:
        nuevo = col.lower()
        for old, new in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ñ","n")]:
            nuevo = nuevo.replace(old, new)
        nuevo = re.sub(r"[^\w]", "_", nuevo)
        nuevo = re.sub(r"__+", "_", nuevo).strip("_")
        mapa[col] = nuevo
    return df.rename(columns=mapa)


def imputar_nulos(df: pd.DataFrame, estrategia: str) -> tuple[pd.DataFrame, list]:
    """Imputa valores nulos según la estrategia elegida."""
    log = []
    if estrategia == "Eliminar filas con nulos":
        antes = len(df)
        df = df.dropna().reset_index(drop=True)
        log.append(f"Se eliminaron {antes - len(df)} filas con nulos.")
        return df, log

    for col in df.columns:
        n_nulos = int(df[col].isna().sum())
        if n_nulos == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if estrategia == "Media / Moda":
                valor = df[col].mean()
                metodo = "media"
            else:
                valor = df[col].median()
                metodo = "mediana"
            df[col] = df[col].fillna(valor)
            log.append(f"'{col}': {n_nulos} nulos → {metodo} = {valor:.2f}")
        else:
            if df[col].notna().any():
                valor = df[col].mode()[0]
                df[col] = df[col].fillna(valor)
                log.append(f"'{col}': {n_nulos} nulos → moda = '{valor}'")
    return df, log


def tratar_outliers(df: pd.DataFrame, metodo: str) -> tuple[pd.DataFrame, list]:
    """Detecta y trata outliers en columnas numéricas usando IQR."""
    log = []
    for col in df.select_dtypes(include="number").columns:
        serie = df[col].dropna()
        Q1, Q3 = serie.quantile(0.25), serie.quantile(0.75)
        IQR = Q3 - Q1
        lim_inf, lim_sup = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_out = int(((serie < lim_inf) | (serie > lim_sup)).sum())
        if n_out == 0:
            continue
        if metodo == "Recortar (clip)":
            df[col] = df[col].clip(lower=lim_inf, upper=lim_sup)
            log.append(f"'{col}': {n_out} outliers recortados → [{lim_inf:.1f}, {lim_sup:.1f}]")
        else:
            antes = len(df)
            df = df[(df[col].isna()) | ((df[col] >= lim_inf) & (df[col] <= lim_sup))]
            log.append(f"'{col}': {antes - len(df)} filas eliminadas por outliers")
    return df.reset_index(drop=True), log


def limpiar_texto(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Normaliza columnas de texto: elimina espacios extra y aplica Title Case."""
    log = []
    for col in df.select_dtypes(include="object").columns:
        antes = df[col].copy()
        df[col] = df[col].apply(
            lambda s: re.sub(r"\s+", " ", str(s)).strip().title() if pd.notna(s) else s
        )
        cambios = int((df[col] != antes).sum())
        if cambios:
            log.append(f"'{col}': {cambios} celdas normalizadas")
    return df, log


def convertir_tipos(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Intenta convertir columnas object a numérico cuando el 80%+ de valores son válidos."""
    log = []
    for col in df.select_dtypes(include="object").columns:
        convertida = pd.to_numeric(df[col], errors="coerce")
        pct_ok = convertida.notna().sum() / len(df)
        if pct_ok >= 0.8:
            df[col] = convertida
            log.append(f"'{col}': convertida a numérico ({pct_ok*100:.0f}% válidos)")
    return df, log


def parsear_fechas(df: pd.DataFrame, cols_fecha: list) -> tuple[pd.DataFrame, list]:
    """Convierte columnas seleccionadas a datetime."""
    log = []
    for col in cols_fecha:
        antes_nulos = int(df[col].isna().sum())
        df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
        nuevos_nulos = int(df[col].isna().sum()) - antes_nulos
        log.append(f"'{col}': parseada como fecha ({nuevos_nulos} valores no reconocidos → NaT)")
    return df, log


# ─── Carga del archivo ────────────────────────────────────────────────────────

st.header("1. Cargar archivo")
archivo = st.file_uploader("Sube tu dataset (CSV o Excel)", type=["csv", "xlsx", "xls"])

if archivo is None:
    st.info("👆 Carga un archivo para comenzar.")
    st.stop()

nombre = archivo.name
try:
    if nombre.endswith(".csv"):
        muestra = archivo.read(4096).decode("utf-8", errors="ignore")
        archivo.seek(0)
        sep = ";" if muestra.count(";") > muestra.count(",") else ","
        df_original = pd.read_csv(archivo, sep=sep)
    else:
        df_original = pd.read_excel(archivo)
except Exception as e:
    st.error(f"No se pudo leer el archivo: {e}")
    st.stop()

st.success(f"✅ **{nombre}** — {df_original.shape[0]} filas × {df_original.shape[1]} columnas")

with st.expander("Vista previa del dataset original"):
    st.dataframe(df_original.head(20), use_container_width=True)

# ─── Diagnóstico inicial ──────────────────────────────────────────────────────

st.header("2. Diagnóstico")

rep = reporte_calidad(df_original)
cols_met = st.columns(len(rep))
for col, (k, v) in zip(cols_met, rep.items()):
    col.metric(k, v)

# Tabla de nulos por columna
nulos_col = df_original.isnull().sum()
nulos_col = nulos_col[nulos_col > 0].rename("Nulos")
if nulos_col.empty:
    st.success("✅ No hay valores nulos en el dataset.")
else:
    pct = (nulos_col / len(df_original) * 100).round(1).rename("% Nulos")
    st.dataframe(pd.concat([nulos_col, pct], axis=1), use_container_width=True)

with st.expander("Tipos de datos y estadísticas descriptivas"):
    st.dataframe(df_original.dtypes.rename("Tipo").to_frame(), use_container_width=True)
    st.dataframe(df_original.describe(include="all"), use_container_width=True)

# ─── Opciones de limpieza ─────────────────────────────────────────────────────

st.header("3. Opciones de limpieza")

with st.form("opciones"):
    c1, c2 = st.columns(2)
    with c1:
        opt_nombres  = st.checkbox("Normalizar nombres de columnas (snake_case)", value=True)
        opt_tipos    = st.checkbox("Convertir columnas a numérico cuando sea posible", value=True)
        opt_texto    = st.checkbox("Limpiar texto (espacios + Title Case)", value=True)
        opt_dup      = st.checkbox("Eliminar filas duplicadas", value=True)
    with c2:
        opt_nulos    = st.selectbox("Tratamiento de valores nulos",
                                    ["Mediana / Moda", "Media / Moda", "Eliminar filas con nulos", "No imputar"])
        opt_outliers = st.selectbox("Tratamiento de outliers (numéricos)",
                                    ["No tratar", "Recortar (clip)", "Eliminar filas"])
        cols_obj     = df_original.select_dtypes(include="object").columns.tolist()
        cols_fecha   = st.multiselect("Columnas de fecha a convertir (opcional)", options=cols_obj)

    ejecutar = st.form_submit_button("🚀 Ejecutar limpieza", use_container_width=True)

if not ejecutar:
    st.stop()

# ─── Ejecución del pipeline ───────────────────────────────────────────────────

df = df_original.copy()
log_total = []

if opt_nombres:
    df = normalizar_columnas(df)
    log_total.append("✔ Nombres de columnas normalizados a snake_case.")

if opt_tipos:
    df, log = convertir_tipos(df)
    log_total += [f"✔ {l}" for l in log]

if opt_nulos != "No imputar":
    df, log = imputar_nulos(df, opt_nulos)
    log_total += [f"✔ {l}" for l in log]

if opt_dup:
    antes = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log_total.append(f"✔ {antes - len(df)} duplicados eliminados.")

if opt_texto:
    df, log = limpiar_texto(df)
    log_total += [f"✔ {l}" for l in log]

if opt_outliers != "No tratar":
    df, log = tratar_outliers(df, opt_outliers)
    log_total += [f"✔ {l}" for l in log]

if cols_fecha:
    # Si se normalizaron columnas, mapear los nombres originales a los nuevos
    if opt_nombres:
        mapa_norm = {}
        for col in df_original.columns:
            nuevo = col.lower()
            for old, new in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ñ","n")]:
                nuevo = nuevo.replace(old, new)
            nuevo = re.sub(r"[^\w]", "_", nuevo)
            nuevo = re.sub(r"__+", "_", nuevo).strip("_")
            mapa_norm[col] = nuevo
        cols_fecha = [mapa_norm.get(c, c) for c in cols_fecha]
    cols_fecha = [c for c in cols_fecha if c in df.columns]
    df, log = parsear_fechas(df, cols_fecha)
    log_total += [f"✔ {l}" for l in log]

# ─── Resultados ───────────────────────────────────────────────────────────────

st.header("4. Resultado")

# Comparativa de métricas antes / después
rep_limpio = reporte_calidad(df)
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Antes")
    for k, v in rep.items():
        st.metric(k, v)
with col_b:
    st.subheader("Después")
    for k, v in rep_limpio.items():
        delta = v - rep[k]
        st.metric(k, v, delta=f"{delta:+}" if delta != 0 else None)

# Log de acciones
with st.expander("📋 Acciones realizadas", expanded=True):
    for linea in log_total:
        st.write(linea)

st.subheader("Dataset limpio")
st.dataframe(df, use_container_width=True)

# ─── Descarga ─────────────────────────────────────────────────────────────────

st.header("5. Descargar")
col_d1, col_d2 = st.columns(2)

with col_d1:
    st.download_button(
        "⬇️ Descargar CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="dataset_limpio.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_d2:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Datos_limpios")
    st.download_button(
        "⬇️ Descargar Excel",
        data=buffer.getvalue(),
        file_name="dataset_limpio.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

st.caption("Bootcamp IA")
