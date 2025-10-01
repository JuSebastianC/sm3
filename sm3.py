# simulador_streamlit_v2.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# ==========================
# Presets por tecnología
# ==========================
tech_presets = {
    "5G": {"Pt_dBm": 23, "freq_GHz": 3.5, "distance_m": 100, "BW_MHz": 100},
    "5G-Advanced": {"Pt_dBm": 30, "freq_GHz": 28, "distance_m": 50, "BW_MHz": 200},
    "6G": {"Pt_dBm": 40, "freq_GHz": 100, "distance_m": 10, "BW_MHz": 1000}
}

# ==========================
# Funciones auxiliares
# ==========================
def awgn_channel(x, snr_db):
    snr_linear = 10 ** (snr_db / 10)
    power_signal = np.mean(np.abs(x) ** 2)
    noise_power = power_signal / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*x.shape))
    return x + noise

def rayleigh_channel(x, snr_db):
    h = np.random.rayleigh(scale=1.0, size=x.shape)
    x_faded = x * h
    return awgn_channel(x_faded, snr_db)

def bpsk_mod(bits):
    return 2 * bits - 1

def bpsk_demod(symbols):
    return (symbols >= 0).astype(int)

# Repetición 3x
def repetition_encode(bits):
    return np.repeat(bits, 3)

def repetition_decode(bits_rx):
    reshaped = bits_rx.reshape(-1, 3)
    return np.round(np.mean(reshaped, axis=1)).astype(int)

# Hamming(7,4)
G_hamming = np.array([
    [1,0,0,0,1,1,0],
    [0,1,0,0,1,0,1],
    [0,0,1,0,1,0,0],
    [0,0,0,1,0,1,1]
])
H_hamming = np.array([
    [1,1,1,0,1,0,0],
    [1,0,0,1,0,1,0],
    [0,1,0,1,0,0,1]
])

def hamming74_encode(bits):
    bits = bits.reshape(-1, 4)
    return (bits @ G_hamming % 2).astype(int).reshape(-1)

def hamming74_decode(bits_rx):
    bits_rx = bits_rx.reshape(-1, 7)
    syndromes = (bits_rx @ H_hamming.T % 2)
    decoded = []
    for i, s in enumerate(syndromes):
        syndrome_dec = s.dot(1 << np.arange(s.size)[::-1])
        if syndrome_dec != 0 and syndrome_dec <= 7:
            bits_rx[i, syndrome_dec-1] ^= 1
        decoded.append(bits_rx[i][:4])
    return np.array(decoded).reshape(-1)

# Convolucional (K=3, polinomios [7,5] en octal)
def conv_encode(bits):
    g1, g2 = 0b111, 0b101
    state = 0
    encoded = []
    for b in bits:
        state = ((state << 1) | b) & 0b111
        out1 = bin(state & g1).count("1") % 2
        out2 = bin(state & g2).count("1") % 2
        encoded.extend([out1, out2])
    return np.array(encoded)

def viterbi_decode(bits_rx):
    n = len(bits_rx) // 2
    trellis = {0: (0, [])}
    g1, g2 = 0b111, 0b101
    for i in range(n):
        new_trellis = {}
        for state, (metric, path) in trellis.items():
            for bit in [0, 1]:
                next_state = ((state << 1) | bit) & 0b111
                out1 = bin(next_state & g1).count("1") % 2
                out2 = bin(next_state & g2).count("1") % 2
                expected = np.array([out1, out2])
                received = bits_rx[2*i:2*i+2]
                branch_metric = np.sum(expected != received)
                new_metric = metric + branch_metric
                if next_state not in new_trellis or new_metric < new_trellis[next_state][0]:
                    new_trellis[next_state] = (new_metric, path + [bit])
        trellis = new_trellis
    best_state = min(trellis, key=lambda s: trellis[s][0])
    return np.array(trellis[best_state][1])

# Simulación con diferentes esquemas
def simulate_scheme(N_bits, EbN0_dB_range, channel_type, scheme, seed):
    np.random.seed(seed)
    ber_list = []
    start_time = time.time()

    for EbN0_dB in EbN0_dB_range:
        bits_tx = np.random.randint(0, 2, N_bits)

        if scheme == "Sin codificar":
            symbols = bpsk_mod(bits_tx)
        elif scheme == "Repetición 3x":
            coded = repetition_encode(bits_tx)
            symbols = bpsk_mod(coded)
        elif scheme == "Hamming(7,4)":
            pad_len = (-len(bits_tx)) % 4
            if pad_len: bits_tx = np.concatenate([bits_tx, np.zeros(pad_len, int)])
            coded = hamming74_encode(bits_tx)
            symbols = bpsk_mod(coded)
        elif scheme == "Convolucional (K=3)":
            coded = conv_encode(bits_tx)
            symbols = bpsk_mod(coded)

        # Canal
        if channel_type == "AWGN":
            rx = awgn_channel(symbols, EbN0_dB)
        else:
            rx = rayleigh_channel(symbols, EbN0_dB)

        # Decodificación
        if scheme == "Sin codificar":
            bits_rx = bpsk_demod(rx)
        elif scheme == "Repetición 3x":
            hard = bpsk_demod(rx)
            bits_rx = repetition_decode(hard)
        elif scheme == "Hamming(7,4)":
            hard = bpsk_demod(rx)
            bits_rx = hamming74_decode(hard)[:N_bits]
        elif scheme == "Convolucional (K=3)":
            hard = bpsk_demod(rx)
            bits_rx = viterbi_decode(hard)[:N_bits]

        errors = np.sum(bits_tx != bits_rx)
        ber = errors / len(bits_tx)
        ber_list.append(ber)

    sim_time = time.time() - start_time
    return ber_list, sim_time

# ==========================
# Interfaz Streamlit
# ==========================
st.title("📡 Simulador BER con técnicas de codificación")
st.sidebar.header("Parámetros de simulación")

# Tecnología
tech_choice = st.sidebar.selectbox("Tecnología", list(tech_presets.keys()))
preset = tech_presets[tech_choice]
st.sidebar.markdown("**Presets [Inferencia]:**")
st.sidebar.write(f"Pt = {preset['Pt_dBm']} dBm")
st.sidebar.write(f"Frecuencia = {preset['freq_GHz']} GHz")
st.sidebar.write(f"Distancia = {preset['distance_m']} m")
st.sidebar.write(f"Ancho de banda = {preset['BW_MHz']} MHz")

# Parámetros ajustables
EbN0_min = st.sidebar.number_input("Eb/N0 mínimo (dB)", value=0)
EbN0_max = st.sidebar.number_input("Eb/N0 máximo (dB)", value=8)
EbN0_step = st.sidebar.number_input("Paso (dB)", value=1)
N_bits = st.sidebar.number_input("Número de bits", value=10000, min_value=1000, max_value=200000)
channel_type = st.sidebar.radio("Tipo de canal", ["AWGN", "Rayleigh"])
seed = st.sidebar.number_input("Semilla aleatoria", value=42)

# Selección de esquemas
schemes = st.sidebar.multiselect("Esquemas de codificación", 
                                 ["Sin codificar", "Repetición 3x", "Hamming(7,4)", "Convolucional (K=3)"], 
                                 default=["Sin codificar"])

run_sim = st.sidebar.button("Ejecutar simulación")

# ==========================
# Simulación
# ==========================
if run_sim:
    EbN0_range = np.arange(EbN0_min, EbN0_max + 1, EbN0_step)
    results = {}

    for scheme in schemes:
        ber, sim_time = simulate_scheme(N_bits, EbN0_range, channel_type, scheme, seed)
        results[scheme] = ber

    df = pd.DataFrame(results, index=EbN0_range)
    df.index.name = "Eb/N0 (dB)"

    st.subheader("📊 Resultados")
    st.write(f"Tiempo total de simulación: {sim_time:.2f} segundos")
    st.dataframe(df)

    # Gráfica
    fig, ax = plt.subplots()
    for scheme in schemes:
        ax.semilogy(df.index, df[scheme], marker='o', label=scheme)
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("BER (escala log)")
    ax.set_title(f"Curva BER - Canal {channel_type}")
    ax.grid(True, which="both")
    ax.legend()
    st.pyplot(fig)

    # Descargar resultados
    csv = df.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Descargar resultados en CSV",
        data=csv,
        file_name='resultados_ber.csv',
        mime='text/csv',
    )
