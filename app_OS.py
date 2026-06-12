import streamlit as st
import pandas as pd
from datetime import datetime
import uuid

st.set_page_config(
    page_title="Telecontrol — Priorização de OS",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilo global ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .ticket-card { border-radius:8px; padding:14px 18px; margin-bottom:10px; border-left-width:5px; border-left-style:solid; }
    .badge { display:inline-block; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:700; margin-right:6px; }
    .metric-box { text-align:center; padding:12px; border-radius:8px; }
    section[data-testid="stSidebar"] { background:#f8f9fa; }
    div[data-testid="stExpander"] { border:1px solid #dee2e6; border-radius:8px; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ── Configurações de prioridade ────────────────────────────────────────────────
PRIORITY_CFG = {
    "Urgente": {"color":"#DC3545","bg":"#FFF5F5","icon":"🔴","order":0},
    "Alta":    {"color":"#FD7E14","bg":"#FFF8F0","icon":"🟠","order":1},
    "Média":   {"color":"#D4A017","bg":"#FFFCF0","icon":"🟡","order":2},
    "Baixa":   {"color":"#28A745","bg":"#F0FFF4","icon":"🟢","order":3},
}

PRODUCTS = sorted([
    "Airfryer Philips","Amazon Kindle","Aquecedor a Gás","Aspirador Dyson",
    "Câmera Canon EOS","Câmera GoPro Hero","Caixa de Som JBL",
    "Computador Desktop","Controle Xbox","Fone Bluetooth JBL",
    "Fogão","Fogão 5 Bocas","Freezer","Geladeira","Geladeira Side by Side",
    "HP Pavilion","iPad Pro","iPhone 13","iPhone 14","Impressora HP",
    "Lavadora a Pressão","Lenovo ThinkPad","LG Smart TV","LG Washing Machine",
    "MacBook Air","MacBook Pro","Máquina de Lavar","Micro-ondas",
    "Microsoft Surface","Monitor LG 27","Nintendo Switch","Notebook Dell XPS",
    "PlayStation 5","Purificador de Água","Roteador Wi-Fi",
    "Samsung Galaxy S23","Smart TV LG 50","Smart TV Samsung 55",
    "Smartwatch Garmin","Soundbar Sony","Tablet Samsung",
    "Xbox Series X","Outro",
])

CHANNELS      = ["Telefone", "Chat", "Email"]
TICKET_TYPES  = ["Problema Técnico", "Reclamação", "Solicitação de reembolso", "Cancelamento", "Dúvida"]

# ── Carregar modelo (cache) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando modelo de IA...")
def load_model():
    try:
        from os_classifier import carregar_artefatos
        return carregar_artefatos()
    except Exception as e:
        return None, None, None

model, encoder, le = load_model()

# ── Estado da sessão ───────────────────────────────────────────────────────────
if "tickets" not in st.session_state:
    st.session_state.tickets = []

# ── SIDEBAR — Formulário de nova OS ───────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Nova Ordem de Serviço")
    if model is None:
        st.error("Modelo não encontrado. Execute `python os_classifier.py` para treinar antes de usar o app.")
    
    with st.form("nova_os", clear_on_submit=True):
        product  = st.selectbox("Produto / Equipamento", PRODUCTS)
        subject  = st.text_input("Assunto", placeholder="Ex: Geladeira com faísca elétrica")
        desc     = st.text_area("Descrição do problema", placeholder="Descreva o que está acontecendo...", max_chars=200, height=100)
        channel  = st.selectbox("Canal de atendimento", CHANNELS)
        ttype    = st.selectbox("Tipo do chamado", TICKET_TYPES)
        age      = st.number_input("Idade do cliente", min_value=18, max_value=85, value=35, step=1)
        submitted = st.form_submit_button("🚀 Classificar e Abrir OS", use_container_width=True, type="primary")

    if submitted:
        if not subject.strip() or not desc.strip():
            st.error("Preencha o assunto e a descrição antes de abrir o chamado.")
        elif model is None:
            st.error("Modelo indisponível. Treine o modelo primeiro.")
        else:
            from os_classifier import predict_new_ticket
            ticket_data = {
                "Product_Purchased": product,
                "Ticket_Type":       ttype,
                "Ticket_Subject":    subject,
                "Ticket_Description":desc,
                "Ticket_Channel":    channel,
                "Customer_Age":      int(age),
            }
            with st.spinner("Classificando..."):
                result = predict_new_ticket(ticket_data, model, encoder, le)

            ticket = {
                "id":          str(uuid.uuid4())[:8].upper(),
                "subject":     subject,
                "description": desc,
                "product":     product,
                "channel":     channel,
                "ticket_type": ttype,
                "age":         int(age),
                "priority":    result["priority"],
                "confidence":  result["confidence"],
                "distribution":result["distribution"],
                "opened_at":   datetime.now(),
            }
            st.session_state.tickets.append(ticket)
            cfg = PRIORITY_CFG[result["priority"]]
            st.success(f"{cfg['icon']} OS **#{ticket['id']}** aberta como **{result['priority']}** ({result['confidence']:.0%} de confiança)")
            st.rerun()

    if st.session_state.tickets:
        st.divider()
        if st.button("🗑️ Limpar todas as OS", use_container_width=True):
            st.session_state.tickets = []
            st.rerun()

# ── MAIN — Dashboard ──────────────────────────────────────────────────────────
st.title("🎯 Dashboard de Priorização de OS")
st.caption("Chamados urgentes aparecem no topo. Dentro de cada nível: mais antigos primeiro, depois por confiança decrescente.")

if not st.session_state.tickets:
    st.info("Nenhuma OS aberta ainda. Use o formulário na barra lateral para criar um chamado.")
    st.stop()

df = pd.DataFrame(st.session_state.tickets)
df["opened_at"] = pd.to_datetime(df["opened_at"])

# ── Contadores por prioridade ──────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, priority in zip([c1, c2, c3, c4], ["Urgente", "Alta", "Média", "Baixa"]):
    cfg   = PRIORITY_CFG[priority]
    count = len(df[df["priority"] == priority])
    col.markdown(
        f"""<div class="metric-box" style="background:{cfg['bg']};border:1px solid {cfg['color']}20;">
        <div style="font-size:28px;">{cfg['icon']}</div>
        <div style="font-size:22px;font-weight:700;color:{cfg['color']};">{count}</div>
        <div style="font-size:13px;color:#666;">{priority}</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.divider()

# ── Seções por prioridade ──────────────────────────────────────────────────────
for priority in ["Urgente", "Alta", "Média", "Baixa"]:
    cfg    = PRIORITY_CFG[priority]
    subset = df[df["priority"] == priority].copy()

    if subset.empty:
        continue

    # Ordenação: mais antigo primeiro → maior confiança primeiro
    subset = subset.sort_values(
        ["opened_at", "confidence"],
        ascending=[True, False],
    )

    expanded = priority in ["Urgente", "Alta"]
    with st.expander(f"{cfg['icon']} **{priority}** — {len(subset)} chamado(s)", expanded=expanded):
        for _, t in subset.iterrows():
            elapsed  = datetime.now() - t["opened_at"].to_pydatetime()
            mins     = int(elapsed.total_seconds() // 60)
            secs     = int(elapsed.total_seconds() % 60)
            time_str = f"{mins}min {secs}s" if mins < 60 else f"{mins // 60}h {mins % 60}min"
            desc_preview = t["description"][:130] + ("..." if len(t["description"]) > 130 else "")

            # Badges do canal e tipo
            badges = (
                f'<span class="badge" style="background:#e9ecef;color:#495057;">{t["channel"]}</span>'
                f'<span class="badge" style="background:#e9ecef;color:#495057;">{t["ticket_type"]}</span>'
                f'<span class="badge" style="background:#e9ecef;color:#495057;">{t["age"]} anos</span>'
            )
            if t["age"] >= 60:
                badges += '<span class="badge" style="background:#FFF3CD;color:#856404;">⚖️ Lei 10.741</span>'

            st.markdown(
                f"""<div class="ticket-card" style="background:{cfg['bg']};border-left-color:{cfg['color']};">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                        <span style="font-size:13px;color:#6c757d;">#{t['id']}</span>
                        <strong style="font-size:16px;margin-left:8px;">{t['subject']}</strong>
                    </div>
                    <div style="text-align:right;min-width:140px;">
                        <div style="color:{cfg['color']};font-weight:700;font-size:15px;">{t['confidence']:.0%} confiança</div>
                        <div style="color:#999;font-size:12px;">Aberto há {time_str}</div>
                    </div>
                </div>
                <div style="margin:6px 0;">{badges}</div>
                <div style="font-size:13px;color:#666;margin-top:4px;">
                    <strong>{t['product']}</strong> · {desc_preview}
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Detalhe da distribuição de probabilidade (expansível)
            with st.expander(f"Ver distribuição completa — #{t['id']}", expanded=False):
                dist = t["distribution"]
                for p, prob in sorted(dist.items(), key=lambda x: x[1], reverse=True):
                    pcfg = PRIORITY_CFG[p]
                    st.markdown(
                        f"{pcfg['icon']} **{p}**",
                    )
                    st.progress(prob, text=f"{prob:.1%}")
