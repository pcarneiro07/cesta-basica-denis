import os
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

# ================================================================
# 1. CARREGAR ARTEFATOS
# ================================================================
ART_DIR = "artifacts"

df_results = pd.read_csv(os.path.join(ART_DIR, "df_results.csv"))
df_analise = pd.read_csv(os.path.join(ART_DIR, "df_analise_produtos.csv"))

df_results["Data"] = pd.to_datetime(df_results["Data"])
produtos = sorted(df_results["Produto"].unique())

# ================================================================
# 2. DESIGN SYSTEM CORPORATIVO
# ================================================================
COLORS = {
    "bg": "#F4F6FA",
    "card": "rgba(255,255,255,0.75)",
    "card_border": "rgba(255,255,255,0.25)",
    "text": "#1A1A1A",
    "subtext": "#5A5A5A",
    "primary": "#0047FF",
    "primary_dark": "#0037CC",
    "accent": "#00B3FF",
    "shadow": "rgba(0,0,0,0.08)"
}

def glass_card(children, padding="25px"):
    return html.Div(
        children,
        style={
            "backdropFilter": "blur(10px)",
            "background": COLORS["card"],
            "padding": padding,
            "borderRadius": "18px",
            "border": f"1px solid {COLORS['card_border']}",
            "boxShadow": f"0 8px 20px {COLORS['shadow']}",
            "marginBottom": "25px",
        }
    )

def metric_card(title, value, color=COLORS["primary"]):
    return html.Div(
        style={
            "flex": 1,
            "background": COLORS["card"],
            "padding": "25px",
            "borderRadius": "14px",
            "border": f"1px solid {COLORS['card_border']}",
            "boxShadow": f"0 4px 15px {COLORS['shadow']}",
            "textAlign": "center"
        },
        children=[
            html.P(title, style={"color": COLORS["subtext"], "fontSize": "16px"}),
            html.H2(value, style={"color": color, "marginTop": "5px"})
        ]
    )

# ================================================================
# 3. DASHBOARD LAYOUT
# ================================================================
app = Dash(__name__)
app.title = "Cesta Básica — Análise Inteligente"

app.layout = html.Div(
    style={"background": COLORS["bg"], "minHeight": "100vh", "padding": "40px"},
    children=[

        # ===================== HERO HEADER =====================
        html.Div([
            html.H1(
                "📊 Monitor Inteligente de Preços da Cesta Básica",
                style={"color": COLORS["text"], "marginBottom": "0px"}
            ),
            html.P(
                "Análise preditiva, clusterização e acompanhamento de volatilidade dos produtos.",
                style={"color": COLORS['subtext'], "fontSize": "18px"}
            ),
        ], style={"marginBottom": "35px"}),

        # ===================== SELECTOR =====================
        glass_card([
            html.Label("Selecione o Produto:", style={"fontWeight": "600", "fontSize": "18px"}),
            dcc.Dropdown(
                id="produto_dropdown",
                options=[{"label": p, "value": p} for p in produtos],
                value=produtos[0],
                style={"marginTop": "10px", "fontSize": "16px"}
            )
        ]),

        # ===================== TABS =====================
        dcc.Tabs(
            id="abas",
            value="aba1",
            style={"marginTop": "20px"},
            children=[
                dcc.Tab(label="📈 Evolução Real vs Previsto", value="aba1", className="custom-tab"),
                dcc.Tab(label="🔎 Cluster & Estabilidade", value="aba2", className="custom-tab"),
                dcc.Tab(label="📉 Métricas de Erro", value="aba3", className="custom-tab"),
            ],
        ),

        html.Div(id="conteudo", style={"marginTop": "25px"})
    ]
)

# ================================================================
# 4. CALLBACK
# ================================================================
@app.callback(
    Output("conteudo", "children"),
    Input("produto_dropdown", "value"),
    Input("abas", "value")
)
def atualizar(produto, aba):

    df_p = df_results[df_results["Produto"] == produto]

    # ============================================================
    # ABA 1 — CHART PREMIUM
    # ============================================================
    if aba == "aba1":

        df_monthly = df_p.groupby(df_p["Data"].dt.to_period("M")).mean(numeric_only=True)
        df_monthly.index = df_monthly.index.to_timestamp()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_monthly.index,
            y=df_monthly["Real"],
            mode="lines+markers",
            name="Real",
            line=dict(color=COLORS["primary"], width=4),
            marker=dict(size=8)
        ))

        fig.add_trace(go.Scatter(
            x=df_monthly.index,
            y=df_monthly["Previsto"],
            mode="lines+markers",
            name="Previsto",
            line=dict(color=COLORS["accent"], width=4, dash="dot"),
            marker=dict(size=8)
        ))

        fig.update_layout(
            template="plotly_white",
            title=f"Evolução Mensal — {produto}",
            title_x=0.02,
            font=dict(color=COLORS["text"]),
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor="white",
        )

        return glass_card([dcc.Graph(figure=fig)])

    # ============================================================
    # ABA 2 — CLUSTER ANALYSIS
    # ============================================================
    if aba == "aba2":

        info = df_analise[df_analise["Produto"] == produto].iloc[0]

        return glass_card([
            html.H2("🔎 Cluster & Estabilidade do Produto", style={"color": COLORS["text"]}),
            html.H3(info["Cluster_Label"], style={"color": COLORS["primary"], "marginTop": "5px"}),

            html.Div([
                metric_card("MAPE (%)", f"{info['MAPE']:.2f}%"),
                metric_card("Desvio do Preço", f"{info['Desvio_Preco']:.2f}"),
                metric_card("Range do Preço", f"{info['Range_Preco']:.2f}")
            ], style={"display": "flex", "gap": "20px", "marginTop": "20px"}),

            html.Br(),
            html.P(
                "A clusterização usa PCA + KMeans para identificar padrões de estabilidade "
                "e volatilidade dos produtos ao longo do tempo.",
                style={"color": COLORS["subtext"], "fontSize": "16px"}
            )
        ])

    # ============================================================
    # ABA 3 — METRICS
    # ============================================================
    if aba == "aba3":

        mae = np.mean(np.abs(df_p["Real"] - df_p["Previsto"]))
        mse = np.mean((df_p["Real"] - df_p["Previsto"]) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((df_p["Real"] - df_p["Previsto"]) / df_p["Real"])) * 100

        return glass_card([
            html.H2("📉 Métricas de Erro do Modelo", style={"color": COLORS["text"], "marginBottom": "25px"}),

            html.Div([
                metric_card("MAE", f"{mae:.3f}"),
                metric_card("MSE", f"{mse:.3f}"),
                metric_card("RMSE", f"{rmse:.3f}"),
                metric_card("MAPE (%)", f"{mape:.2f}%", COLORS["primary_dark"])
            ], style={"display": "flex", "gap": "25px"})
        ])

# ================================================================
if __name__ == "__main__":
    app.run(debug=True)
