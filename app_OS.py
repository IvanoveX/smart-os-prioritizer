import streamlit as st
import pandas as pd
from datetime import datetime
import uuid

st.set_page_config(
    page_title="Telecontrol · Central de OS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System (Enterprise NOC) ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    background: #F8FAFC;
    color: #0F172A;
}
section[data-testid="stSidebar"] {
    background: #F1F5F9;
    border-right: 1px solid #E2E8F0;
}
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 16px 20px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
.kpi-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 30px;
    font-weight: 700;
    line-height: 1;
}
.ticket-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left-width: 4px;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 8px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.priority-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.07em;
    border-width: 1px;
    border-style: solid;
}
.meta-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
    background: #F1F5F9;
    color: #475569;
    margin-right: 4px;
    border: 1px solid #E2E8F0;
}
.ticket-id      { font-size:11px; font-weight:600; color:#94A3B8;
                  letter-spacing:0.06em; font-family:'Courier New',monospace; }
.ticket-subject { font-size:15px; font-weight:600; color:#0F172A; }
.ticket-time    { font-size:11px; color:#94A3B8; font-weight:500; }
.ticket-conf    { font-size:13px; font-weight:700; }
.ticket-product { font-size:13px; font-weight:600; color:#1E293B; }
.ticket-desc    { font-size:13px; color:#475569; line-height:1.5; margin-top:6px; }
div[data-testid="stExpander"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 6px !important;
    margin-bottom: 8px !important;
    background: #FFFFFF !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Priority Configuration (corporate palette, no emoji) ──────────────────────
PRIORITY_CFG = {
    "Urgente": {"color":"#9B1C1C","bg":"#FEF2F2","border":"#FCA5A5","label":"URGENTE","order":0},
    "Alta":    {"color":"#B45309","bg":"#FFFBEB","border":"#FCD34D","label":"ALTA",   "order":1},
    "Média":   {"color":"#1E40AF","bg":"#EFF6FF","border":"#93C5FD","label":"MÉDIA",  "order":2},
    "Baixa":   {"color":"#065F46","bg":"#ECFDF5","border":"#6EE7B7","label":"BAIXA",  "order":3},
}

CHANNELS     = ["Telefone", "Chat", "Email"]
TICKET_TYPES = ["Problema Técnico", "Reclamação", "Solicitação de reembolso", "Cancelamento", "Dúvida"]

# ── Backend ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Inicializando motor de classificação...")
def load_model():
    try:
        from os_classifier import load_artifacts
        return load_artifacts()
    except Exception:
        return None, None, None

model, encoder, le = load_model()

# ── Session State ──────────────────────────────────────────────────────────────
if "tickets" not in st.session_state:
    st.session_state.tickets = []

# ── Sidebar — Formulário de nova OS ───────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:10px 0 12px;">
        <div style="font-size:11px;font-weight:700;letter-spacing:0.1em;
                    text-transform:uppercase;color:#64748B;">Telecontrol</div>
        <div style="font-size:18px;font-weight:700;color:#0F172A;margin-top:2px;">
            Nova Ordem de Serviço
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    if model is None:
        st.error(
            "Motor indisponível. Execute `python os_classifier.py` para treinar o modelo.",
            icon="⚠️",
        )

    with st.form("nova_os", clear_on_submit=True):
        # Produto como texto livre — o SentenceTransformer interpreta qualquer string
        product = st.text_input(
            "Produto / Equipamento",
            placeholder="Ex: Geladeira Frost Free Brastemp, Notebook Dell Inspiron...",
            help="Digite livremente. O modelo interpreta semanticamente qualquer equipamento.",
        )
        subject = st.text_input(
            "Assunto",
            placeholder="Ex: Geladeira com faísca elétrica",
        )
        desc = st.text_area(
            "Descrição do problema",
            placeholder="Descreva o que está acontecendo com o maior nível de detalhe...",
            max_chars=200,
            height=110,
        )
        col_l, col_r = st.columns(2)
        with col_l:
            channel = st.selectbox("Canal", CHANNELS)
        with col_r:
            age = st.number_input("Idade", min_value=18, max_value=85, value=35, step=1)
        ttype     = st.selectbox("Tipo do chamado", TICKET_TYPES)
        submitted = st.form_submit_button(
            "Classificar e Abrir OS",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        if not product.strip():
            st.error("Informe o produto ou equipamento.")
        elif not subject.strip() or not desc.strip():
            st.error("Preencha o assunto e a descrição.")
        elif model is None:
            st.error("Motor de IA indisponível.")
        else:
            from os_classifier import predict_new_ticket
            ticket_data = {
                "Product_Purchased":  product.strip(),
                "Ticket_Type":        ttype,
                "Ticket_Subject":     subject.strip(),
                "Ticket_Description": desc.strip(),
                "Ticket_Channel":     channel,
                "Customer_Age":       int(age),
            }
            with st.spinner("Analisando chamado..."):
                result = predict_new_ticket(ticket_data, model, encoder, le)

            ticket = {
                "id":           str(uuid.uuid4())[:8].upper(),
                "subject":      subject.strip(),
                "description":  desc.strip(),
                "product":      product.strip(),
                "channel":      channel,
                "ticket_type":  ttype,
                "age":          int(age),
                "priority":     result["priority"],
                "confidence":   result["confidence"],
                "distribution": result["distribution"],
                "opened_at":    datetime.now(),
            }
            st.session_state.tickets.append(ticket)
            st.success(
                f"OS #{ticket['id']} registrada · {result['priority']} · {result['confidence']:.0%} confiança"
            )
            st.rerun()

    if st.session_state.tickets:
        st.divider()
        if st.button("Limpar fila de OS", use_container_width=True):
            st.session_state.tickets = []
            st.rerun()

# ── Main Dashboard ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:24px;">
    <div style="font-size:11px;font-weight:700;letter-spacing:0.1em;
                text-transform:uppercase;color:#64748B;">
        Telecontrol · Central de Operações
    </div>
    <div style="font-size:24px;font-weight:700;color:#0F172A;margin-top:2px;">
        Fila de Triagem Inteligente
    </div>
    <div style="font-size:13px;color:#64748B;margin-top:4px;">
        Chamados urgentes têm prioridade. Dentro de cada nível: mais antigos primeiro,
        depois por confiança decrescente.
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.tickets:
    st.markdown("""
    <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:6px;
                padding:48px;text-align:center;margin-top:16px;">
        <div style="font-size:14px;font-weight:600;color:#1E293B;">Fila de OS vazia</div>
        <div style="font-size:13px;color:#64748B;margin-top:4px;">
            Use o formulário na barra lateral para registrar um novo chamado.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = pd.DataFrame(st.session_state.tickets)
df["opened_at"] = pd.to_datetime(df["opened_at"])

# ── KPI Summary ────────────────────────────────────────────────────────────────
kpi_cols = st.columns(4)
for col, priority in zip(kpi_cols, ["Urgente", "Alta", "Média", "Baixa"]):
    cfg   = PRIORITY_CFG[priority]
    count = len(df[df["priority"] == priority])
    with col:
        st.markdown(
            f"""<div class="kpi-card" style="border-top:3px solid {cfg['color']};">
                <div class="kpi-label">{cfg['label']}</div>
                <div class="kpi-value" style="color:{cfg['color']};">{count}</div>
            </div>""",
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── Ticket Sections by Priority ────────────────────────────────────────────────
for priority in ["Urgente", "Alta", "Média", "Baixa"]:
    cfg    = PRIORITY_CFG[priority]
    subset = df[df["priority"] == priority].copy()

    if subset.empty:
        continue

    # Oldest first → highest confidence first (tie-break)
    subset = subset.sort_values(["opened_at", "confidence"], ascending=[True, False])

    n     = len(subset)
    label = f"{cfg['label']} — {n} chamado{'s' if n > 1 else ''}"

    with st.expander(label, expanded=(priority in ["Urgente", "Alta"])):
        for _, t in subset.iterrows():
            elapsed  = datetime.now() - t["opened_at"].to_pydatetime()
            mins     = int(elapsed.total_seconds() // 60)
            secs     = int(elapsed.total_seconds() % 60)
            time_str = f"{mins}min {secs}s" if mins < 60 else f"{mins // 60}h {mins % 60}min"
            desc_preview = (
                t["description"][:140] + "..."
                if len(t["description"]) > 140 else t["description"]
            )

            meta_badges = (
                f'<span class="meta-badge">{t["channel"]}</span>'
                f'<span class="meta-badge">{t["ticket_type"]}</span>'
                f'<span class="meta-badge">{t["age"]} anos</span>'
            )
            if t["age"] >= 60:
                meta_badges += (
                    '<span class="meta-badge" '
                    'style="background:#FEF3C7;color:#92400E;border-color:#FCD34D;">'
                    'Lei 10.741</span>'
                )

            st.markdown(
                f"""<div class="ticket-card" style="border-left-color:{cfg['color']};">
                    <div style="display:flex;justify-content:space-between;
                                align-items:flex-start;gap:12px;">
                        <div style="flex:1;min-width:0;">
                            <div style="display:flex;align-items:center;
                                        gap:8px;margin-bottom:4px;">
                                <span class="ticket-id">#{t['id']}</span>
                                <span class="priority-badge"
                                      style="background:{cfg['bg']};color:{cfg['color']};
                                             border-color:{cfg['border']};">
                                    {cfg['label']}
                                </span>
                            </div>
                            <div class="ticket-subject">{t['subject']}</div>
                        </div>
                        <div style="text-align:right;flex-shrink:0;">
                            <div class="ticket-conf" style="color:{cfg['color']};">
                                {t['confidence']:.0%}
                            </div>
                            <div class="ticket-time">{time_str}</div>
                        </div>
                    </div>
                    <div style="margin:8px 0 6px;">{meta_badges}</div>
                    <div class="ticket-desc">
                        <span class="ticket-product">{t['product']}</span>
                        <span style="color:#CBD5E1;margin:0 6px;">·</span>
                        {desc_preview}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            with st.expander(f"Distribuição de probabilidade — #{t['id']}", expanded=False):
                dist = t["distribution"]
                for p, prob in sorted(dist.items(), key=lambda x: x[1], reverse=True):
                    pcfg = PRIORITY_CFG[p]
                    st.markdown(
                        f'<span style="font-size:12px;font-weight:700;color:{pcfg["color"]};'
                        f'text-transform:uppercase;letter-spacing:0.06em;">'
                        f'{pcfg["label"]}</span>',
                        unsafe_allow_html=True,
                    )
                    st.progress(prob, text=f"{prob:.1%}")
