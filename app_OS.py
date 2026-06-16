import streamlit as st
import pandas as pd
from datetime import datetime
import uuid

# ── Configuração de Página Executiva ──────────────────────────────────────────
st.set_page_config(
    page_title="Telecontrol — AI Priority Core",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS High-End (Estética SaaS Enterprise) ──────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Reset de Fonte Global */
    html, body, [data-testid="stAppViewContainer"], .main {
        font-family: 'Inter', sans-serif !important;
        background-color: #F8FAFC;
    }
    
    /* Sidebar Profissional */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Cards de Métricas Analíticas */
    .metric-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border-top: 4px solid #CBD5E1;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1E293B;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    
    /* Card de Ticket Estilo Central de Atendimento */
    .ticket-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        transition: transform 0.15s ease;
    }
    .ticket-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    
    /* Sistema de Badges Técnicos */
    .custom-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #E2E8F0;
        margin-right: 6px;
    }
    .badge-law {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    
    /* Customização de Input/Botões do Streamlit */
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Configurações de Identidade Visual Corporativa ───────────────────────────
PRIORITY_CFG = {
    "Urgente": {"color": "#9B1C1C", "border": "#E02424", "bg": "#FDF2F2", "label": "CRITICAL / URGENT"},
    "Alta":    {"color": "#B45309", "border": "#D97706", "bg": "#FEF3C7", "label": "HIGH PRIORITY"},
    "Média":   {"color": "#1E40AF", "border": "#2563EB", "bg": "#EFF6FF", "label": "MEDIUM"},
    "Baixa":   {"color": "#065F46", "border": "#059669", "bg": "#ECFDF5", "label": "LOW"},
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

CHANNELS     = ["Telefone", "Chat", "Email"]
TICKET_TYPES = ["Problema Técnico", "Reclamação", "Solicitação de reembolso", "Cancelamento", "Dúvida"]

# ── Carregamento do Pipeline de IA ───────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando modelo de IA...")
def load_model():
    try:
        from os_classifier import load_artifacts 
        return load_artifacts()              
    except Exception as e:
        return None, None, None

model, encoder, le = load_model()

if "tickets" not in st.session_state:
    st.session_state.tickets = []

# ── SIDEBAR — Entrada de Ordens de Serviço ───────────────────────────────────
with st.sidebar:
    st.markdown("<h3 style='color:#0F172A;font-weight:700;margin-bottom:4px;'>Input de Chamados</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B;font-size:13px;margin-bottom:24px;'>Simulação de entrada via Posto Autorizado ou API</p>", unsafe_allow_html=True)
    
    if model is None:
        st.error("Engine indisponível. Execute o treinamento do modelo (`os_classifier.py`) antes de iniciar.")
    
    with st.form("nova_os", clear_on_submit=True):
        product = st.selectbox("Equipamento / Produto afetado", PRODUCTS)
        subject = st.text_input("Sumário do Incidente", placeholder="Ex: Vazamento de gás no duto traseiro")
        desc    = st.text_area("Descrição Técnica do Defeito", placeholder="Relato completo do cliente final...", max_chars=200, height=100)
        
        c_left, c_right = st.columns(2)
        channel = c_left.selectbox("Origem / Canal", CHANNELS)
        ttype   = c_right.selectbox("Categoria", TICKET_TYPES)
        age     = st.number_input("Idade do Titular", min_value=18, max_value=85, value=35, step=1)
        
        st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Analisar e Enfileirar OS", use_container_width=True, type="primary")

    if submitted:
        if not subject.strip() or not desc.strip():
            st.error("Campos compulsórios (Sumário e Descrição) ausentes.")
        elif model is None:
            st.error("Pipeline offline.")
        else:
            from os_classifier import predict_new_ticket
            ticket_data = {
                "Product_Purchased": product,
                "Ticket_Type":       ttype,
                "Ticket_Subject":    subject,
                "Ticket_Description": desc,
                "Ticket_Channel":    channel,
                "Customer_Age":      int(age),
            }
            
            with st.spinner("Iniciando inferência..."):
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
                "distribution": result["distribution"],
                "opened_at":   datetime.now(),
            }
            st.session_state.tickets.append(ticket)
            st.rerun()

    if st.session_state.tickets:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Limpar Fila de Operações", use_container_width=True):
            st.session_state.tickets = []
            st.rerun()

# ── CORPO PRINCIPAL — Dashboard de Governança e Operações ─────────────────────
st.markdown("<h1 style='color:#0F172A; font-weight:700; padding-bottom:0px; margin-bottom:0px;'>Smart OS Prioritizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#64748B; font-size:14px; margin-top:0px;'>Painel Operacional de Monitoramento de Demanda Core — Telecontrol Business Intelligence</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if not st.session_state.tickets:
    st.info("Fila limpa. Aguardando novas requisições na barra lateral para triagem em tempo real.")
    st.stop()

df = pd.DataFrame(st.session_state.tickets)
df["opened_at"] = pd.to_datetime(df["opened_at"])

# ── Painel de Volumetria Analítica (Grid Executivo) ───────────────────────────
cols = st.columns(4)
for col, priority in zip(cols, ["Urgente", "Alta", "Média", "Baixa"]):
    cfg = PRIORITY_CFG[priority]
    count = len(df[df["priority"] == priority])
    col.markdown(
        f"""<div class="metric-container" style="border-top-color:{cfg['color']};">
            <div class="metric-value">{count}</div>
            <div class="metric-label">{cfg['label']}</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br><h4 style='color:#1E293B; font-weight:600;'>Fila de Despacho Dinâmico (SLA)</h4>", unsafe_allow_html=True)

# ── Renderização Estruturada da Fila por Criticidade ─────────────────────────
for priority in ["Urgente", "Alta", "Média", "Baixa"]:
    cfg = PRIORITY_CFG[priority]
    subset = df[df["priority"] == priority].copy()

    if subset.empty:
        continue

    # Regra de Negócio: Ordenação cronológica estrita com desempate por confiança algorítmica
    subset = subset.sort_values(["opened_at", "confidence"], ascending=[True, False])

    with st.expander(f"📌 {cfg['label']} ({len(subset)} pendentes)", expanded=(priority in ["Urgente", "Alta"])):
        for _, t in subset.iterrows():
            elapsed = datetime.now() - t["opened_at"].to_pydatetime()
            mins = int(elapsed.total_seconds() // 60)
            secs = int(elapsed.total_seconds() % 60)
            time_str = f"{mins}m {secs}s" if mins < 60 else f"{mins // 60}h {mins % 60}m"
            
            # Formatação estruturada de metadados
            badges_html = (
                f'<span class="custom-badge">{t["channel"]}</span>'
                f'<span class="custom-badge">{t["ticket_type"]}</span>'
                f'<span class="custom-badge">{t["age"]} Anos</span>'
            )
            if t["age"] >= 60:
                badges_html += '<span class="custom-badge badge-law">⚖️ Estatuto do Idoso</span>'

            # Card Renderizado via CSS Avançado
            st.markdown(
                f"""<div class="ticket-card" style="border-left: 5px solid {cfg['color']};">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <span style="font-family:monospace; font-size:12px; color:#94A3B8; font-weight:600;">ID: #{t['id']}</span>
                        <span style="font-size:15px; font-weight:600; color:#1E293B; margin-left:12px;">{t['subject']}</span>
                    </div>
                    <div style="text-align:right;">
                        <span style="background:{cfg['bg']}; color:{cfg['color']}; font-size:12px; font-weight:700; padding:3px 8px; border-radius:4px; border:1px solid {cfg['border']}40;">
                            {t['confidence']:.1%} CONFIDENCE
                        </span>
                        <div style="color:#64748B; font-size:11px; margin-top:6px; font-weight:500;">Aberto há {time_str}</div>
                    </div>
                </div>
                <div style="margin:8px 0 10px 0;">{badges_html}</div>
                <div style="font-size:13px; color:#475569; line-height:1.5; border-top:1px solid #F1F5F9; padding-top:8px;">
                    <strong style="color:#0F172A;">{t['product']}</strong> — {t['description']}
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Distribuição de Probabilidades Multiclasse (Mecanismo de Explicabilidade)
            with st.expander(f"Ver probabilidade vetorial — #{t['id']}", expanded=False):
                dist = t["distribution"]
                for p, prob in sorted(dist.items(), key=lambda x: x[1], reverse=True):
                    pcfg = PRIORITY_CFG[p]
                    col_lbl, col_bar = st.columns([1, 4])
                    col_lbl.markdown(f"<span style='font-size:12px; font-weight:600; color:#475569;'>{p}</span>", unsafe_allow_html=True)
                    col_bar.progress(prob)
