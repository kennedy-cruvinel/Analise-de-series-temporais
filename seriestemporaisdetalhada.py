import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from io import StringIO
from datetime import date

# Configuração da página
st.set_page_config(page_title="Sistema de Análise e Previsão de Séries Temporais", layout="wide")

# Título principal
st.markdown("<h1 style='text-align: center;'>Sistema de análise e previsão de séries temporais</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>Configurações</h2>", unsafe_allow_html=True)
    upload_file = st.file_uploader("📂 Carregar arquivo CSV", type=["csv"])
    if upload_file is not None:
        stringio = StringIO(upload_file.getvalue().decode("utf-8"))
        data = pd.read_csv(stringio, header=None)
        data_inicio = date(2000, 1, 1)
        periodo = st.date_input("📅 Data de início", data_inicio)
        periodo_previsao = st.number_input("🔮 Número de meses para previsão", min_value=1, max_value=48, value=1)
        processar = st.button("🚀 Processar")

if upload_file is not None and processar:
    try:
        ts_data = pd.Series(data.iloc[:, 0].values, index=pd.date_range(start=periodo, periods=len(data), freq='M'))

        # Decomposição
        decomposicao = seasonal_decompose(ts_data, model='additive')
        fig_decomposicao = decomposicao.plot()
        fig_decomposicao.set_size_inches(10, 6)
        for ax in fig_decomposicao.axes:
            ax.set_facecolor('#f0f0f0')  # fundo cinza claro para destacar
            ax.grid(True, linestyle='--', alpha=0.5)

        # Previsão SARIMAX
        modelo = SARIMAX(ts_data, order=(2, 0, 0), seasonal_order=(1, 1, 1, 12))
        modelo_fit = modelo.fit()
        previsao = modelo_fit.forecast(steps=periodo_previsao)

        # Gráfico de previsão
        fig_previsao, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor('#f0f0f0')  # fundo com sombra
        ts_data.plot(ax=ax, label='Dados Originais', color='blue')
        previsao.plot(ax=ax, label='Previsão', color='deepskyblue', linestyle='--')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        # TABS
        aba1, aba2 = st.tabs(["📊 Análise", "📈 Dados brutos"])

        with aba1:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📉 Decomposição da Série Temporal")
                st.pyplot(fig_decomposicao)

            with col2:
                st.subheader("📈 Previsão da Série Temporal")
                st.pyplot(fig_previsao)

            st.subheader("📋 Valores previstos")
            st.dataframe(previsao)

        with aba2:
            st.subheader("📅 Série Temporal Original")
            st.dataframe(ts_data)

    except Exception as e:
        st.error(f"❌ Erro ao processar o arquivo: {e}")
