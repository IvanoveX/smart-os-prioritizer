# 🛠️ Sistema de Priorização Inteligente de OS — Telecontrol

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![SentenceTransformers](https://img.shields.io/badge/Sentence--Transformers-Multilingual-orange.svg)](https://sbert.net)
[![F1-Score](https://img.shields.io/badge/F1--Macro%20(Teste)-0.8919-brightgreen.svg)]()

**Fábrica de Projetos Ágeis III | UNIMAR — 1º Semestre de 2026**

Pipeline de Machine Learning que classifica automaticamente Ordens de Serviço em níveis de prioridade (**Baixa · Média · Alta · Urgente**) com base no conteúdo semântico do chamado e nos metadados estruturados.

---

## Integrantes

| Nome | RA |
|---|---|
| Ivan Luís Geronimo Del Roio | 2031330 |
| Erick Augusto Silva De Freitas | 2089849 |
| Gabriel Saes Cominale Tonette | 2068363 |
| Hugo Alves da Silva | 2045165 |
| João Daniel Caçador Araújo | 2031562 |
| Samuel Alves Vieira | 2041169 |

---

## Arquitetura

```
Texto do chamado (Ticket_Subject + Ticket_Description)
        ↓
SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2)
        → 384 dimensões semânticas
        +
7 features estruturadas (produto, canal, idade, tipo, severidade textual...)
        ↓
SMOTE → balanceia classes no treino
        ↓
XGBoost (early stopping no conjunto de validação)
        ↓
{"priority": "Urgente", "confidence": 0.91, "distribution": {...}}
```

**Resultado:** F1-Macro de **0,8919** no teste holdout com gap validação→teste de apenas **0,0046**.

---

## Estrutura do Repositório

```
.
├── os_classifier.py                 # Pipeline principal (treino + predição)
├── requirements.txt                 # Dependências
├── README.md
├── data/
│   └── tickets_sintetico_6000.csv   # Dataset sintético com contra-exemplos
└── model/                           # Gerado automaticamente ao treinar
    ├── modelo.pkl
    ├── label_encoder.pkl
    └── backend.pkl
```

---

## Instalação

```bash
# 1. Clone o repositório
git clone https://https://github.com/IvanoveX/smart-os-prioritizer.git
cd smart-os-prioritizer

# 2. (Opcional) Crie um ambiente virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install -r requirements.txt
```

> **Nota:** Na primeira execução, o SentenceTransformer baixa o modelo automaticamente (~110 MB). Requer conexão com a internet.

---

## Execução

### Treinar o modelo e avaliar
```bash
python os_classifier.py
```
Executa as fases de treino (split 60/20/20 estratificado + SMOTE), exibe o F1-Macro de validação e teste, e salva os artefatos em `model/`.

### Usar modelo já treinado
```bash
python os_classifier.py --producao
```

### Classificar uma OS via terminal (modo interativo)
```bash
python os_classifier.py --producao --nova-os
```

### Uso como módulo Python
```python
from os_classifier import carregar_artefatos, predict_new_ticket

model, encoder, le = carregar_artefatos()

ticket = {
    "Product_Purchased":  "Geladeira",
    "Ticket_Type":        "Problema Técnico",
    "Ticket_Subject":     "Fumaça saindo da geladeira",
    "Ticket_Description": "Minha geladeira emite fumaça e cheiro de queimado. "
                          "Tenho 72 anos e guardo insulina dentro dela.",
    "Ticket_Channel":     "Telefone",
    "Customer_Age":       72,
}

resultado = predict_new_ticket(ticket, model, encoder, le)
print(resultado)
# {'priority': 'Urgente', 'confidence': 0.9121, 'distribution': {...}}
```

---

## Dependências

```
sentence-transformers>=2.7
xgboost>=2.0
imbalanced-learn>=0.12
scikit-learn>=1.3
pandas>=2.0
numpy>=1.26
joblib>=1.3
shap>=0.43
```

---

## Resultados

| Métrica | Valor |
|---|---|
| F1-Macro (Validação) | 0,8965 |
| F1-Macro (Teste Final) | 0,8919 |
| Gap Validação → Teste | 0,0046 |
| Erros críticos (Urgente → Baixa) | 0 |
