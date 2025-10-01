# simulador_streamlit_v3_errors.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

st.set_page_config(layout="wide", page_title="Simulador BER — con teoría y errores")

# ==========================
# Presets por tecnología (informativo)
# ==========================
tech_presets = {
    "5G": {"Pt_dBm": 23, "freq_GHz": 3.5, "distance_m": 100, "BW_MHz": 100},
    "5G-Advanced": {"Pt_dBm": 30, "freq_GHz": 28, "distance_m": 50, "BW_MHz": 200},
    "6G": {"Pt_dBm": 40, "freq_GHz": 100, "distance_m": 10, "BW_MHz": 1000}
}

# ==========================
# Funciones auxiliares
# ==========================
def awgn_channel(x, EbN0_dB, rate=1.0):
    EbN0_lin = 10 ** (EbN0_dB / 10.0)
    sigma2 = 1.0 / (2.0 * rate * EbN0_lin)
    sigma = np.sqrt(sigma2)
    noise = sigma * np.random.randn(*x.shape)
    return x + noise

def rayleigh_channel(x, EbN0_dB, rate=1.0):
    fade = np.random.rayleigh(scale=1.0, size=x.shape)
    faded = x * fade
    return awgn_channel(faded, EbN0_dB, rate=rate)

def bpsk_mod(bits):
    return 2 * bits - 1

def bpsk_demod(symbols):
    return (symbols >= 0).astype(int)

# Codificación/decodificación
def repetition_encode(bits):
    return np.repeat(bits, 3)

def repetition_decode(hard_bits):
    resh = hard_bits.reshape(-1, 3)
    return (np.sum(resh, axis=1) >= 2).astype(int)

def hamming74_encode(bits):
    bits = bits.reshape(-1, 4)
    out = []
    for d in bits:
        d1,d2,d3,d4 = d
        p1 = (d1 ^ d2 ^ d4) & 1
        p2 = (d1 ^ d3 ^ d4) & 1
        p3 = (d2 ^ d3 ^ d4) & 1
        code = np.array([p1,p2,d1,p3,d2,d3,d4], dtype=int)
        out.extend(code.tolist())
    return np.array(out, dtype=int)

def hamming74_decode(hard_bits):
    blocks = hard_bits.reshape(-1,7)
    decoded = []
    for block in blocks:
        p1,p2,d1,p3,d2,d3,d4 = block
        s1 = p1 ^ d1 ^ d2 ^ d4
        s2 = p2 ^ d1 ^ d3 ^ d4
        s3 = p3 ^ d2 ^ d3 ^ d4
        syndrome = (s1 << 2) | (s2 << 1) | s3
        if syndrome != 0 and 1 <= syndrome <= 7:
            block[syndrome-1] ^= 1
        decoded.extend([block[2], block[4], block[5], block[6]])
    return np.array(decoded, dtype=int)

def conv_encode(bits):
    g1 = 0b111
    g2 = 0b101
    K = 3
    state = 0
    out = []
    for b in bits:
        state = ((state << 1) | int(b)) & ((1 << K) - 1)
        o1 = bin(state & g1).count("1") % 2
        o2 = bin(state & g2).count("1") % 2
        out.extend([o1, o2])
    for _ in range(K-1):
        state = ((state << 1) | 0) & ((1 << K) - 1)
        o1 = bin(state & g1).count("1") % 2
        o2 = bin(state & g2).count("1") % 2
        out.extend([o1, o2])
    return np.array(out, dtype=int)

def viterbi_decode(hard_bits):
    n = len(hard_bits)//2
    K = 3
    n_states = 1 << (K-1)
    g1 = 0b111
    g2 = 0b101
    trellis = {}
    for s in range(n_states):
        trellis[s] = []
        for b in [0,1]:
            next_state = ((s << 1) | b) & ((1 << (K-1)) - 1)
            full_state = ((s << 1) | b) & ((1 << K) - 1)
            o1 = bin(full_state & g1).count("1") % 2
            o2 = bin(full_state & g2).count("1") % 2
            trellis[s].append((next_state, np.array([o1,o2]), b))
    INF = 1e9
    path_metric = np.full((n+1, n_states), INF)
    path_bits = [[None]*n_states for _ in range(n+1)]
    path_metric[0,0] = 0
    path_bits[0][0] = []
    for t in range(n):
        r = hard_bits[2*t:2*t+2]
        for s in range(n_states):
            if path_bits[t][s] is None:
                continue
            for (ns, expected, b) in trellis[s]:
                metric = np.sum(expected != r)
                nm = path_metric[t,s] + metric
                if nm < path_metric[t+1, ns]:
                    path_metric[t+1, ns] = nm
                    path_bits[t+1][ns] = (path_bits[t][s] or []) + [b]
    end_state = int(np.argmin(path_metric[n]))
    decoded = np.array(path_bits[n][end_state], dtype=int)
    decoded = decoded[:-(K-1)] if K-1 > 0 else decoded
    return decoded

# Teórico BPSK
def ber_bpsk_theoretical(EbN0_dB_array):
    out = []
    for d in EbN0_dB_array:
        Eb_lin = 10**(d/10.0)
        val = 0.5 * math.erfc(math.sqrt(Eb_lin))
        out.append(val)
    return np.array(out)

def simulate_scheme(N_bits, EbN0_list, channel, scheme, seed):
    np.random.seed(seed)
    bers = []
    times = []
    if scheme == "Sin codificar":
        rate = 1.0
    elif scheme == "Repetición 3x":
        rate = 1.0/3.0
    elif scheme == "Hamming(7,4)":
        rate = 4.0/7.0
    elif scheme == "Convolucional (K=3)":
        rate = 1.0/2.0
    else:
        rate = 1.0
    for Eb in EbN0_list:
        start = time.time()
        bits_tx = np.random.randint(0,2, N_bits)
        if scheme == "Sin codificar":
            symbols = bpsk_mod(bits_tx)
        elif scheme == "Repetición 3x":
            enc = repetition_encode(bits_tx)
            symbols = bpsk_mod(enc)
        elif scheme == "Hamming(7,4)":
            pad = (-len(bits_tx)) % 4
            if pad: bits_p = np.concatenate([bits_tx, np.zeros(pad, dtype=int)])
            else: bits_p = bits_tx
            enc = hamming74_encode(bits_p)
            symbols = bpsk_mod(enc)
        elif scheme == "Convolucional (K=3)":
            enc = conv_encode(bits_tx)
            symbols = bpsk_mod(enc)
        if channel == "AWGN":
            rx = awgn_channel(symbols, Eb, rate=rate)
        else:
            rx = rayleigh_channel(symbols, Eb, rate=rate)
        hard = bpsk_demod(rx)
        if scheme == "Sin codificar":
            bits_rx = hard
            bits_comp = bits_tx
        elif scheme == "Repetición 3x":
            bits_rx = repetition_decode(hard)
            bits_comp = bits_tx
        elif scheme == "Hamming(7,4)":
            dec = hamming74_decode(hard)
            bits_rx = dec[:len(bits_tx)]
            bits_comp = bits_tx
        elif scheme == "Convolucional (K=3)":
            dec = viterbi_decode(hard)
            bits_rx = dec[:len(bits_tx)]
            bits_comp = bits_tx
        else:
            bits_rx = hard
            bits_comp = bits_tx
        errors = np.sum(bits_comp != bits_rx)
        ber = errors / len(bits_comp)
        bers.append(ber)
        times.append(time.time() - start)
    return np.array(bers), np.array(times), rate

# UI
st.title("📡 Simulador BER — teoría y errores visualizados")
st.sidebar.header("Parámetros")

tech = st.sidebar.selectbox("Tecnología (preset informativo)", list(tech_presets.keys()))
preset = tech_presets[tech]
st.sidebar.markdown("**Presets (informativo):**")
st.sidebar.write(f"Pt = {preset['Pt_dBm']} dBm | freq = {preset['freq_GHz']} GHz | dist = {preset['distance_m']} m | BW = {preset['BW_MHz']} MHz")

Eb_min = st.sidebar.number_input("Eb/N0 inicio (dB)", value=0.0, format="%.1f")
Eb_max = st.sidebar.number_input("Eb/N0 fin (dB)", value=8.0, format="%.1f")
Eb_step = st.sidebar.number_input("Paso (dB)", value=1.0, format="%.1f")
N_bits = st.sidebar.number_input("Número de bits (por punto)", value=10000, min_value=1000, step=1000)
channel = st.sidebar.radio("Tipo de canal", ["AWGN", "Rayleigh"])
seed = st.sidebar.number_input("Semilla (seed)", value=42)

schemes = st.sidebar.multiselect("Esquemas a comparar",
                                 ["Sin codificar", "Repetición 3x", "Hamming(7,4)", "Convolucional (K=3)"],
                                 default=["Sin codificar"])

run = st.sidebar.button("Ejecutar simulación")

col1, col2 = st.columns([2,1])

if run:
    Eb_list = np.arange(Eb_min, Eb_max + 1e-9, Eb_step)
    results = {}
    times_info = {}
    rates = {}
    for scheme in schemes:
        bers, times_arr, rate = simulate_scheme(N_bits, Eb_list, channel, scheme, seed)
        results[scheme] = bers
        times_info[scheme] = times_arr
        rates[scheme] = rate
    ber_theo = ber_bpsk_theoretical(Eb_list)
    df = pd.DataFrame(results, index=Eb_list)
    df.index.name = "Eb/N0 (dB)"
    with col1:
        fig, ax = plt.subplots(figsize=(8,4))
        for scheme in schemes:
            ax.semilogy(Eb_list, results[scheme], marker='o', label=f"{scheme} (sim.)")
        ax.semilogy(Eb_list, ber_theo, '--', label="Teórica BPSK (AWGN)")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("BER")
        ax.set_title(f"BER vs Eb/N0 — Canal: {channel}")
        ax.grid(True, which='both')
        ax.legend()
        st.pyplot(fig)
        st.markdown("**Interpretación rápida:**")
        for scheme in schemes:
            st.write(f"- **{scheme}**: tasa (R) = {rates[scheme]:.3f}. Tiempo medio = {np.mean(times_info[scheme]):.3f} s")
    with col2:
        st.subheader("Tabla de resultados")
        display_df = df.reset_index().rename(columns={"index":"Eb/N0 (dB)"})
        st.dataframe(display_df)

    # ==========================
    # Visualización de errores
    # ==========================
    st.markdown("---")
    st.subheader("Visualización de errores en un segmento de bits")

    scheme_show = schemes[0]  # primer esquema
    Eb_test = Eb_list[0]      # primer Eb/N0

    # Simulamos un segmento corto
    N_vis = 50
    bits_tx = np.random.randint(0,2, N_vis)

    if scheme_show == "Sin codificar":
        symbols = bpsk_mod(bits_tx)
        rate = 1.0
    elif scheme_show == "Repetición 3x":
        enc = repetition_encode(bits_tx)
        symbols = bpsk_mod(enc)
        rate = 1.0/3.0
    elif scheme_show == "Hamming(7,4)":
        pad = (-len(bits_tx)) % 4
        if pad: bits_p = np.concatenate([bits_tx, np.zeros(pad, dtype=int)])
        else: bits_p = bits_tx
        enc = hamming74_encode(bits_p)
        symbols = bpsk_mod(enc)
        rate = 4.0/7.0
    elif scheme_show == "Convolucional (K=3)":
        enc = conv_encode(bits_tx)
        symbols = bpsk_mod(enc)
        rate = 1.0/2.0

    if channel == "AWGN":
        rx = awgn_channel(symbols, Eb_test, rate=rate)
    else:
        rx = rayleigh_channel(symbols, Eb_test, rate=rate)

    hard = bpsk_demod(rx)

    if scheme_show == "Sin codificar":
        bits_rx = hard
        bits_comp = bits_tx
    elif scheme_show == "Repetición 3x":
        bits_rx = repetition_decode(hard)
        bits_comp = bits_tx
    elif scheme_show == "Hamming(7,4)":
        dec = hamming74_decode(hard)
        bits_rx = dec[:len(bits_tx)]
        bits_comp = bits_tx
    elif scheme_show == "Convolucional (K=3)":
        dec = viterbi_decode(hard)
        bits_rx = dec[:len(bits_tx)]
        bits_comp = bits_tx

    fig2, ax2 = plt.subplots(figsize=(10,2))
    ax2.plot(bits_comp[:N_vis], 'bo-', label="Bits transmitidos")
    ax2.plot(bits_rx[:N_vis], 'rx--', label="Bits recibidos")
    for i in range(N_vis):
        if bits_comp[i] != bits_rx[i]:
            ax2.axvspan(i-0.2, i+0.2, color="red", alpha=0.3)  # marcar error
    ax2.set_ylim(-0.5,1.5)
    ax2.set_xlabel("Índice de bit")
    ax2.set_ylabel("Valor")
    ax2.set_title(f"Errores en un segmento de {N_vis} bits — Esquema: {scheme_show}")
    ax2.legend()
    st.pyplot(fig2)

