import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import os

# --- Configurações ---
# Em um projeto real, os dados viriam de um banco de dados ou arquivo maior.
TRANSACTIONS_DATA = [
 ['leite', 'pao', 'manteiga'], ['cerveja', 'fraldas',
'batata_chips'],
 ['leite', 'fraldas', 'pao', 'cerveja'], ['pao', 'manteiga',
'geleia'],
 ['cafe', 'pao', 'acucar'], ['leite', 'pao', 'cerveja'],
 ['cerveja', 'fraldas'], ['leite', 'cafe', 'pao', 'manteiga']
]
OUTPUT_RULES_PATH = '../data/regras_de_associacao.csv'
MIN_SUPPORT = 0.25
MIN_LIFT = 1.2

# --- Lógica Principal ---
print("Iniciando o script de geração de regras de associação...")
# 1. Pré-processamento
print("Pré-processando dados transacionais...")
te = TransactionEncoder()
te_ary = te.fit(TRANSACTIONS_DATA).transform(TRANSACTIONS_DATA)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
# 2. Aplicar Apriori
print(f"Executando o algoritmo Apriori com suporte mínimo de {MIN_SUPPORT}...")
frequent_itemsets = apriori(df_encoded, min_support=MIN_SUPPORT,
use_colnames=True)
# 3. Gerar e Filtrar Regras
print(f"Gerando regras com lift mínimo de {MIN_LIFT}...")
rules = association_rules(frequent_itemsets, metric="lift",
min_threshold=MIN_LIFT)

# Adicionar uma formatação mais legível para as regras
rules['rule'] = rules['antecedents'].apply(lambda x: ', '.join(list(x))) + ' -> ' + rules['consequents'].apply(lambda x: ', '.join(list(x)))
# Selecionar e reordenar colunas para o relatório final
report_df = rules[['rule', 'support', 'confidence',
'lift']].sort_values(by='lift', ascending=False)
# 4. Salvar o Relatório de Regras
print(f"Salvando as regras geradas em {OUTPUT_RULES_PATH}...")
os.makedirs('../data', exist_ok=True)
report_df.to_csv(OUTPUT_RULES_PATH, index=False)

print("Processo de geração de regras concluído com sucesso!")