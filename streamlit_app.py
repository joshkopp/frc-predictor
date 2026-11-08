"""FRC 2026 (REBUILT) toolkit, themed to match joshkoppanalil.com.

Two pages behind one custom nav bar:
  - Match Predictor       (app_pages/predictor.py)
  - Alliance Recommender  (app_pages/alliance.py)

Run locally:  streamlit run streamlit_app.py
"""
import pathlib

import streamlit as st

import ui

ROOT = pathlib.Path(__file__).parent

st.set_page_config(page_title="FRC 2026 Toolkit · Josh Koppanalil", page_icon="🤖", layout="wide",
                   initial_sidebar_state="collapsed")
# CSS goes through st.markdown: st.html silently drops a stylesheet this large.
st.markdown(f"<style>{(ROOT / 'assets' / 'style.css').read_text()}</style>", unsafe_allow_html=True)

predictor = st.Page("app_pages/predictor.py", title="Match Predictor", icon=":material/insights:", default=True)
alliance = st.Page("app_pages/alliance.py", title="Alliance Recommender", icon=":material/groups:", url_path="alliance")
page = st.navigation([predictor, alliance], position="hidden")

# ---- one themed nav bar for every page (replaces Streamlit's sidebar menu)
with st.container(key="jknav"):
    brand, links, ext = st.columns([1, 4, 1.4], vertical_alignment="center", gap="small")
    brand.html(ui.brand())
    with links:
        with st.container(horizontal=True, horizontal_alignment="center", gap="small", key="jklinks"):
            for pg_, label, icon in ((predictor, "Match Predictor", ":material/insights:"),
                                     (alliance, "Alliance Recommender", ":material/groups:")):
                with st.container(key="plon" if pg_.url_path == page.url_path else f"pl_{label[:5]}", width="content"):
                    st.page_link(pg_, label=label, icon=icon)
    ext.html(ui.ext_link())

page.run()
st.html(ui.footer())
