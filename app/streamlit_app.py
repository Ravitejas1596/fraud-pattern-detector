from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st


API_URL = os.environ.get("FRAUD_API_URL", "http://127.0.0.1:8000")


def _score(payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{API_URL}/score", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def _sample_network(seed: int = 7, n_nodes: int = 60) -> Tuple[nx.Graph, Dict[int, float]]:
    rng = random.Random(seed)
    g = nx.barabasi_albert_graph(n=n_nodes, m=2, seed=seed)
    scores = {n: min(1.0, max(0.0, rng.random() ** 0.4)) for n in g.nodes()}  # skew to create clusters
    return g, scores


def _plot_network(g: nx.Graph, scores: Dict[int, float]) -> go.Figure:
    pos = nx.spring_layout(g, seed=7)
    edge_x, edge_y = [], []
    for u, v in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = [pos[n][0] for n in g.nodes()]
    node_y = [pos[n][1] for n in g.nodes()]
    node_color = [scores.get(n, 0.0) for n in g.nodes()]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=0.5, color="rgba(150,150,150,0.4)"),
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(
                size=10,
                color=node_color,
                colorscale="Reds",
                cmin=0,
                cmax=1,
                line=dict(width=1, color="white"),
                colorbar=dict(title="Fraud score"),
            ),
            text=[f"Tx {n} — score {scores.get(n, 0.0):.2f}" for n in g.nodes()],
            hoverinfo="text",
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        showlegend=False,
    )
    return fig


st.set_page_config(page_title="Fraud Pattern Detector", layout="wide")
st.title("Fraud Pattern Detector")
st.caption("XGBoost scorer + transaction-network patterns")

with st.sidebar:
    st.subheader("API")
    st.write(f"Using: `{API_URL}`")

tab1, tab2 = st.tabs(["Single transaction scorer", "Network patterns"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        amt = st.number_input("Transaction amount", min_value=0.0, value=117.0, step=1.0)
        product = st.selectbox("ProductCD", options=[None, "W", "C", "H", "S", "R"], index=1)
        device = st.selectbox("DeviceType", options=[None, "desktop", "mobile"], index=0)
    with c2:
        card1 = st.text_input("card1", value="10409")
        card6 = st.selectbox("card6", options=[None, "debit", "credit"], index=2)
        addr1 = st.text_input("addr1", value="299")
    with c3:
        p_email = st.text_input("P_emaildomain", value="gmail.com")
        r_email = st.text_input("R_emaildomain", value="gmail.com")
        dist1 = st.number_input("dist1", min_value=0.0, value=0.0, step=1.0)

    payload = {
        "TransactionAmt": float(amt),
        "ProductCD": product,
        "DeviceType": device,
        "card1": card1 or None,
        "card6": card6,
        "addr1": addr1 or None,
        "P_emaildomain": p_email or None,
        "R_emaildomain": r_email or None,
        "dist1": float(dist1),
    }

    if st.button("Score transaction", type="primary"):
        with st.spinner("Scoring..."):
            out = _score(payload)
        st.subheader(f"Decision: {out['decision']}")
        st.metric("Fraud probability", f"{out['probability_fraud']:.3f}")
        st.subheader("Top drivers")
        st.dataframe(out["top_factors"], use_container_width=True)

with tab2:
    st.subheader("Transaction network")
    st.caption(
        "A lightweight network visualization showing how suspicious transactions can form clusters around shared identifiers."
    )
    g, scores = _sample_network()
    fig = _plot_network(g, scores)
    st.plotly_chart(fig, use_container_width=True)

