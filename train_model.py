
import os
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ================================================================
# 1. CONFIGURAÇÕES BÁSICAS
# ================================================================
DATA_PATH = os.path.join("data", "ICB_2s-2025.xlsx")
ARTIFACTS_DIR = "artifacts"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ================================================================
# 2. CARREGAR DADOS
# ================================================================
print(f"Lendo base em: {DATA_PATH}")
df = pd.read_excel(DATA_PATH)

# Garante tipos
df["Data_Coleta"] = pd.to_datetime(df["Data_Coleta"])
df["Data_num"] = df["Data_Coleta"].astype("int64") // 10**9
df["Produto_id"] = df["Produto"].astype("category").cat.codes

# ================================================================
# 3. SEPARAR FEATURES E TARGET
# ================================================================
X = df[["Data_num", "Produto_id"]]
y = df["Preco"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================================================
# 4. TREINAR MODELO XGBOOST
# ================================================================
print("Treinando modelo XGBoost...")

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    tree_method="hist"
)

model.fit(X_train, y_train)

# ================================================================
# 5. PREVISÕES NO TESTE
# ================================================================
y_pred = model.predict(X_test)

df_results = pd.DataFrame({
    "Data": df.loc[y_test.index, "Data_Coleta"],
    "Produto": df.loc[y_test.index, "Produto"],
    "Real": y_test.values,
    "Previsto": y_pred
})

# Salva resultados de teste para o dashboard
results_path = os.path.join(ARTIFACTS_DIR, "df_results.csv")
df_results.to_csv(results_path, index=False, encoding="utf-8")
print(f"Resultados de teste salvos em: {results_path}")

# ================================================================
# 6. MÉTRICAS POR PRODUTO (MAPE, MAE, MSE, RMSE)
# ================================================================
print("Calculando métricas por produto...")

def calc_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

metricas_prod = (
    df_results.groupby("Produto")
    .apply(lambda g: pd.Series({
        "MAPE": calc_mape(g["Real"], g["Previsto"]),
        "MAE": mean_absolute_error(g["Real"], g["Previsto"]),
        "MSE": mean_squared_error(g["Real"], g["Previsto"]),
        "RMSE": np.sqrt(mean_squared_error(g["Real"], g["Previsto"]))
    }))
    .reset_index()
)

# ================================================================
# 7. VARIAÇÃO DE PREÇO (DESVIO E RANGE)
# ================================================================
var_preco = (
    df.groupby("Produto")["Preco"]
    .agg(Desvio_Preco="std", Range_Preco=lambda x: x.max() - x.min())
    .reset_index()
)

df_analise = metricas_prod.merge(var_preco, on="Produto", how="left")
df_analise = df_analise.dropna()

# ================================================================
# 8. PCA + KMEANS (3 CLUSTERS)
# ================================================================
print("Rodando PCA + KMeans (3 clusters)...")

X_feats = df_analise[["MAPE", "Desvio_Preco", "Range_Preco"]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_feats)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df_analise["PC1"] = X_pca[:, 0]
df_analise["PC2"] = X_pca[:, 1]

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_pca)
df_analise["cluster_id"] = clusters

# Ordena clusters pelo MAPE médio (baixo → alto)
cluster_order = (
    df_analise.groupby("cluster_id")["MAPE"]
    .mean()
    .sort_values()
    .index
    .tolist()
)

cluster_label_map = {
    cluster_order[0]: "Baixa variação de preço (MAPE baixo)",
    cluster_order[1]: "Média variação de preço",
    cluster_order[2]: "Alta variação de preço (MAPE alto)"
}

df_analise["Cluster_Label"] = df_analise["cluster_id"].map(cluster_label_map)

# ================================================================
# 9. SALVAR ARTEFATOS DE ANÁLISE
# ================================================================
analise_path = os.path.join(ARTIFACTS_DIR, "df_analise_produtos.csv")
df_analise.to_csv(analise_path, index=False, encoding="utf-8")
print(f"Análise por produto (com clusters) salva em: {analise_path}")

# Modelo treinado (para uso futuro em API / Azure Function)
model_path = os.path.join(ARTIFACTS_DIR, "xgb_model.json")
model.save_model(model_path)
print(f"Modelo XGBoost salvo em: {model_path}")

print("\nTreino e geração de artefatos concluídos com sucesso.")
print("→ df_results.csv, df_analise_produtos.csv e xgb_model.json gerados em 'artifacts/'.")
