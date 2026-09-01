import base64
import importlib
import io
import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v2 as components_v2
from PIL import Image

# Streamlit keeps imported modules cached between script reruns.  Reload the
# local pipeline module so newly added helpers are available without having to
# kill the development server first.
import detection as _detection
import classification as _classification
import roboflow_metrics as _roboflow_metrics

importlib.reload(_detection)
importlib.reload(_classification)
importlib.reload(_roboflow_metrics)

from classification import classify_ship
from roboflow_metrics import get_detection_metrics, get_display_metrics


clickable_detection_image = components_v2.component(
    "clickable_detection_image",
    html='<div class="clickable-image-root"></div>',
    css="""
    .clickable-image-root { position: relative; width: 100%; line-height: 0; }
    .clickable-image-root img { width: 100%; height: auto; border-radius: 8px; display: block; }
    .clickable-image-root button { position: absolute; z-index: 5; padding: 0; margin: 0;
        border: 0; background: transparent; cursor: pointer; box-sizing: border-box; }
    .clickable-image-root button:hover { background: rgba(0,201,167,.16); outline: 3px solid #00c9a7; }
    """,
    js="""
    export default function(component) {
      const { data, parentElement, setStateValue } = component;
      const root = parentElement.querySelector('.clickable-image-root');
      root.replaceChildren();
      const image = document.createElement('img');
      image.src = data.image;
      root.appendChild(image);
      for (const box of data.boxes) {
        const button = document.createElement('button');
        button.type = 'button';
        button.title = `${box.object_class} 선택`;
        button.style.left = `${box.left}%`;
        button.style.top = `${box.top}%`;
        button.style.width = `${box.width}%`;
        button.style.height = `${box.height}%`;
        button.addEventListener('click', () => setStateValue('selected_box', box.index));
        root.appendChild(button);
      }
    }
    """,
)

from detection import (
    draw_predictions,
    format_json_output,
    get_model_status,
    pipeline_to_json,
    read_image_from_bytes,
    run_pipeline,
    summarize_predictions,
)

# ============================================================
# 페이지 설정
# ============================================================

st.set_page_config(
    page_title="해양 침투 객체 감시 시스템",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');

:root {
    --ocean-deep: #061525;
    --ocean-mid: #0c2847;
    --ocean-light: #134074;
    --accent-teal: #00c9a7;
    --accent-cyan: #38bdf8;
    --accent-glow: rgba(0, 201, 167, 0.35);
    --card-bg: rgba(12, 40, 71, 0.55);
    --card-border: rgba(56, 189, 248, 0.18);
    --text-primary: #e8f4fc;
    --text-muted: #7da8c4;
    --success: #34d399;
    --warning: #fbbf24;
}

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0, 201, 167, 0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 100% 100%, rgba(56, 189, 248, 0.08) 0%, transparent 50%),
        linear-gradient(180deg, #061525 0%, #0a1e35 50%, #061525 100%);
    background-attachment: fixed;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 1400px;
}

[data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
}

[data-testid="column"] {
    gap: 1rem;
}

#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
}

/* ── Hero Header ── */
.hero-header {
    background: linear-gradient(135deg, rgba(12, 40, 71, 0.9) 0%, rgba(6, 21, 37, 0.95) 100%);
    border: 1px solid var(--card-border);
    border-radius: 18px;
    padding: 20px 28px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255,255,255,0.05);
}

.hero-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 20px;
    flex-wrap: wrap;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-teal), var(--accent-cyan), var(--accent-teal));
    background-size: 200% 100%;
    animation: shimmer 4s ease infinite;
}

@keyframes shimmer {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 201, 167, 0.12);
    border: 1px solid rgba(0, 201, 167, 0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 11px;
    font-weight: 500;
    color: var(--accent-teal);
    letter-spacing: 0.5px;
    margin-bottom: 10px;
}

.hero-title-block {
    flex: 1;
    min-width: 240px;
}

.main-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 6px 0;
    letter-spacing: -0.4px;
}

.sub-title {
    color: var(--text-muted);
    font-size: 0.88rem;
    font-weight: 400;
    margin: 0;
    line-height: 1.5;
}

.hero-stats {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    align-items: center;
    padding-top: 4px;
}

.hero-stat-item {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--text-muted);
    font-size: 0.82rem;
}

.hero-stat-item span {
    color: var(--accent-cyan);
    font-weight: 600;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(6, 21, 37, 0.6);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 6px;
    margin-bottom: 16px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 9px 22px;
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text-muted);
    background: transparent;
    border: none;
    transition: all 0.25s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0, 201, 167, 0.2), rgba(56, 189, 248, 0.15)) !important;
    color: var(--text-primary) !important;
    border: 1px solid rgba(0, 201, 167, 0.3) !important;
    box-shadow: 0 2px 12px rgba(0, 201, 167, 0.15);
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 8px;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}

.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* ── Section Cards ── */
.section-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    margin-bottom: 16px;
}

.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 4px 0;
}

.section-desc {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin: 0 0 14px 0;
}

.section-header-block {
    margin-bottom: 4px;
}

/* ── Video / Upload Area ── */
.video-placeholder {
    height: 380px;
    border: 2px dashed rgba(56, 189, 248, 0.25);
    border-radius: 14px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    background:
        radial-gradient(circle at 50% 50%, rgba(56, 189, 248, 0.06) 0%, transparent 70%),
        rgba(6, 21, 37, 0.8);
    position: relative;
    overflow: hidden;
}

.video-placeholder::before {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(56, 189, 248, 0.02) 2px,
        rgba(56, 189, 248, 0.02) 4px
    );
    pointer-events: none;
}

.video-placeholder-icon {
    font-size: 2.5rem;
    opacity: 0.6;
}

.video-placeholder-text {
    font-size: 0.95rem;
    color: var(--text-muted);
    font-weight: 400;
}

.video-placeholder-sub {
    font-size: 0.75rem;
    color: rgba(125, 168, 196, 0.6);
}

/* ── Workflow (Roboflow-style) ── */
.workflow-panel {
    background: rgba(6, 21, 37, 0.75);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 14px 12px;
    min-height: 420px;
}

.workflow-panel-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 12px;
}

.workflow-block {
    background: rgba(12, 40, 71, 0.65);
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    font-size: 0.78rem;
    color: var(--text-primary);
    font-weight: 500;
}

.workflow-block.purple { border-left-color: #a855f7; }
.workflow-block.green { border-left-color: #22c55e; }
.workflow-block.orange { border-left-color: #f97316; }
.workflow-block.blue { border-left-color: #3b82f6; }
.workflow-block.cyan { border-left-color: #06b6d4; }

.workflow-connector {
    width: 2px;
    height: 10px;
    background: rgba(56, 189, 248, 0.2);
    margin: 0 auto 8px;
}

.io-panel {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 16px;
    min-height: 420px;
    display: flex;
    flex-direction: column;
}

.io-panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(56, 189, 248, 0.12);
}

.io-panel-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
}

.io-panel-subtitle {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.media-card {
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(6, 21, 37, 0.65);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 10px;
    padding: 10px 12px;
    margin-top: 10px;
}

.media-thumb {
    width: 52px;
    height: 52px;
    border-radius: 8px;
    object-fit: cover;
    border: 1px solid rgba(56, 189, 248, 0.15);
    flex-shrink: 0;
}

.media-name {
    font-size: 0.82rem;
    color: var(--text-primary);
    word-break: break-all;
}

.media-meta {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 2px;
}

.output-visual-wrap {
    background: rgba(6, 21, 37, 0.55);
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 10px;
    padding: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 0;
    overflow: hidden;
}

.output-visual-wrap.empty {
    height: 220px;
    min-height: 220px;
    flex-direction: column;
    box-sizing: border-box;
}

.output-json-wrap {
    background: rgba(6, 21, 37, 0.75);
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 10px;
    padding: 0;
    min-height: 0;
    max-height: 70vh;
    overflow: auto;
}

.detection-stats-bar {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-top: 12px;
}

.stat-chip {
    background: rgba(6, 21, 37, 0.6);
    border: 1px solid rgba(56, 189, 248, 0.1);
    border-radius: 8px;
    padding: 8px 10px;
    text-align: center;
}

.stat-chip-label {
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

.stat-chip-value {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-top: 2px;
}

.view-toggle label {
    min-width: 72px;
    text-align: center;
}

/* ── Dashboard Panel ── */
.dashboard-panel {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 18px;
    backdrop-filter: blur(12px);
    min-height: 380px;
    display: flex;
    flex-direction: column;
}

.dashboard-panel-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(56, 189, 248, 0.12);
    flex-shrink: 0;
}

.metric-grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 10px;
}

/* ── Metric Cards ── */
.metric-card {
    background: rgba(6, 21, 37, 0.6);
    border: 1px solid rgba(56, 189, 248, 0.1);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 0;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    border-color: rgba(0, 201, 167, 0.25);
    box-shadow: 0 0 16px rgba(0, 201, 167, 0.08);
}

.metric-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
}

.metric-value.accent { color: var(--accent-teal); }
.metric-value.cyan { color: var(--accent-cyan); }

.metric-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 14px;
    margin-bottom: 16px;
}

.metric-card.compact {
    padding: 18px;
    text-align: center;
}

.metric-card.compact .metric-value {
    font-size: 1.6rem;
}

.metric-card.compact .metric-label {
    font-size: 0.78rem;
    text-transform: none;
    letter-spacing: 0;
}

.compact-divider {
    border: none;
    border-top: 1px solid rgba(56, 189, 248, 0.1);
    margin: 12px 0;
}

/* ── Status Badge ── */
.active-status {
    background: linear-gradient(135deg, rgba(52, 211, 153, 0.12), rgba(0, 201, 167, 0.08));
    border: 1px solid rgba(52, 211, 153, 0.35);
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    margin-top: auto;
    font-weight: 600;
    color: var(--success);
    font-size: 0.85rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
}

.status-dot {
    width: 8px;
    height: 8px;
    background: var(--success);
    border-radius: 50%;
    animation: pulse 2s ease infinite;
    box-shadow: 0 0 8px var(--success);
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
}

/* ── Environment Badge ── */
.env-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 10px;
    padding: 8px 16px;
    color: var(--accent-cyan);
    font-size: 0.82rem;
    font-weight: 500;
    margin-bottom: 14px;
}

.tab2-controls {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 6px;
    flex-wrap: wrap;
}

/* ── Object Tags ── */
.object-tags {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 4px;
}

.object-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(6, 21, 37, 0.8);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 0.78rem;
    color: var(--text-muted);
}

.object-label {
    font-size: 0.72rem;
    color: #7da8c4;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 6px;
}

.object-tag strong {
    color: var(--accent-teal);
    font-weight: 600;
}

/* ── Model Status Grid ── */
.tab3-layout {
    display: flex;
    flex-direction: column;
    gap: 20px;
    padding: 8px 0 16px;
}

.model-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
}

.model-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.model-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 201, 167, 0.1);
}

.model-card-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}

.model-card-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
}

.model-card-value.active {
    color: var(--success);
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 16px 0 8px;
    color: rgba(125, 168, 196, 0.5);
    font-size: 0.75rem;
    border-top: 1px solid rgba(56, 189, 248, 0.08);
    margin-top: 20px;
}

/* ── Streamlit Overrides ── */
[data-testid="stMetric"] {
    background: rgba(6, 21, 37, 0.6);
    border: 1px solid rgba(56, 189, 248, 0.1);
    border-radius: 12px;
    padding: 12px 16px;
}

[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
}

[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-size: 1.4rem !important;
}

[data-testid="stFileUploader"] {
    background: rgba(6, 21, 37, 0.55);
    border: 2px dashed rgba(56, 189, 248, 0.28);
    border-radius: 12px;
    padding: 0;
    margin-top: 48px !important;
    margin-bottom: 10px !important;
    height: 220px;
    min-height: 220px;
    width: 100%;
    box-sizing: border-box;
    overflow: hidden;
    position: relative;
}

[data-testid="stFileUploader"] section {
    padding: 0 !important;
    height: 216px !important;
    min-height: 216px !important;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 0 !important;
    border-radius: 10px !important;
    background: transparent !important;
    box-sizing: border-box;
    position: relative;
    flex-direction: column;
    gap: 10px;
}

[data-testid="stFileUploader"] section::before {
    content: '📁';
    font-size: 3.2rem;
    line-height: 1;
    filter: saturate(0.7);
    opacity: 0.55;
    pointer-events: none;
    position: absolute;
    left: 50%;
    top: calc(50% - 14px);
    transform: translate(-50%, -50%);
}

[data-testid="stFileUploader"] section::after {
    content: 'JPG · PNG · MP4 · AVI · MOV';
    color: var(--text-muted);
    font-size: 0.72rem;
    font-weight: 500;
    line-height: 1;
    opacity: 0.65;
    pointer-events: none;
    position: absolute;
    left: 50%;
    top: calc(50% + 30px);
    transform: translate(-50%, -50%);
    white-space: nowrap;
}

[data-testid="stFileUploader"] section > div {
    display: none !important;
}

[data-testid="stFileUploader"] section button {
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    padding: 0 !important;
    border: 0 !important;
    opacity: 0 !important;
    cursor: pointer !important;
    box-shadow: none !important;
    background: transparent !important;
}

[data-testid="stFileUploaderFile"] {
    position: absolute !important;
    top: 12px;
    right: 12px;
    z-index: 5;
    width: auto !important;
    min-width: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    border: 0 !important;
}

[data-testid="stFileUploaderFile"] > div:not(:last-child) {
    display: none !important;
}

[data-testid="stFileUploaderDeleteBtn"],
[data-testid="stFileUploaderFile"] button {
    position: static !important;
    width: 30px !important;
    height: 30px !important;
    min-height: 30px !important;
    padding: 0 !important;
    opacity: 1 !important;
    color: #e8f4fc !important;
    background: rgba(239, 68, 68, 0.18) !important;
    border: 1px solid rgba(248, 113, 113, 0.45) !important;
    border-radius: 50% !important;
    box-shadow: none !important;
}

[data-testid="stFileUploaderDeleteBtn"] svg,
[data-testid="stFileUploaderFile"] button svg {
    display: none !important;
}

[data-testid="stFileUploaderDeleteBtn"]::before,
[data-testid="stFileUploaderFile"] button::before {
    content: '×';
    font-size: 1.25rem;
    line-height: 1;
}

.st-key-live_upload_card,
.st-key-intrusion_upload_card {
    position: relative;
}

.st-key-live_upload_card .media-card,
.st-key-intrusion_upload_card .media-card {
    padding-right: 48px;
}

.st-key-live_upload_card [data-testid="stButton"],
.st-key-intrusion_upload_card [data-testid="stButton"] {
    position: absolute;
    top: 50%;
    right: 12px;
    transform: translateY(-50%);
    z-index: 10;
}

.st-key-live_upload_card [data-testid="stButton"] button,
.st-key-intrusion_upload_card [data-testid="stButton"] button {
    width: 30px;
    height: 30px;
    min-height: 30px;
    padding: 0;
    color: #fca5a5;
    font-size: 1.2rem;
    line-height: 1;
    background: rgba(239, 68, 68, 0.16);
    border: 1px solid rgba(248, 113, 113, 0.45);
    border-radius: 50%;
}

.st-key-ship_result_box,
.st-key-intrusion_result_box {
    min-height: 420px;
    overflow: auto;
    box-sizing: border-box;
}

.st-key-ship_result_box [data-testid="stVideo"] {
    width: 100%;
    max-height: 420px;
}

.st-key-ship_result_box [data-testid="stVideo"] video {
    width: 100%;
    max-height: 398px;
    object-fit: contain;
}

.st-key-ship_result_box [data-testid="stImage"] img,
.st-key-intrusion_result_box [data-testid="stImage"] img {
    height: 398px !important;
    max-height: 398px !important;
    object-fit: contain !important;
}

.st-key-intrusion_result_box [data-testid="stVideo"],
.st-key-intrusion_result_box [data-testid="stVideo"] video {
    width: 100%;
    max-height: 398px;
    object-fit: contain;
}

.result-file-bar { color: var(--text-muted); font-size: 0.78rem; margin: 2px 0 8px; }
.evaluation-grid { display: flex; flex-direction: column; gap: 12px; margin-top: 14px; }
.evaluation-card { display: grid; grid-template-columns: 170px 150px 1fr; align-items: center; gap: 18px; background: rgba(6, 21, 37, 0.72); border: 1px solid rgba(56, 189, 248, 0.16); border-radius: 12px; padding: 18px 22px; min-height: 92px; }
.evaluation-name { color: var(--accent-cyan); font-size: 0.9rem; font-weight: 700; }
.evaluation-value { color: var(--text-primary); font-size: 1.25rem; font-weight: 700; }
.evaluation-desc { color: var(--text-muted); font-size: 0.78rem; line-height: 1.55; }
@media (max-width: 900px) { .evaluation-card { grid-template-columns: 1fr; gap: 8px; } }

.st-key-replace_live_upload button,
.st-key-replace_intrusion_upload button {
    min-height: 34px !important;
    padding: 5px 14px !important;
    color: var(--accent-cyan) !important;
    background: rgba(56, 189, 248, 0.08) !important;
    border: 1px solid rgba(56, 189, 248, 0.28) !important;
    border-radius: 9px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}

.st-key-replace_live_upload button:hover,
.st-key-replace_intrusion_upload button:hover {
    color: var(--text-primary) !important;
    background: rgba(0, 201, 167, 0.12) !important;
    border-color: rgba(0, 201, 167, 0.42) !important;
}

.dropzone-hint {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 6px;
    text-align: center;
}

.mode-badge {
    display: inline-block;
    background: rgba(0, 201, 167, 0.12);
    border: 1px solid rgba(0, 201, 167, 0.25);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.68rem;
    color: var(--accent-teal);
    margin-left: 8px;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(0, 201, 167, 0.4);
}

.stAlert {
    background: rgba(56, 189, 248, 0.08) !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    border-radius: 12px !important;
    color: var(--text-muted) !important;
}

hr {
    border-color: rgba(56, 189, 248, 0.1) !important;
    margin: 12px 0 !important;
}

[data-testid="stRadio"] > div {
    gap: 8px;
}

[data-testid="stRadio"] label {
    background: rgba(6, 21, 37, 0.6) !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
    border-radius: 10px !important;
    padding: 7px 14px !important;
    font-size: 0.82rem !important;
    transition: all 0.2s ease !important;
}

[data-testid="stRadio"] label:hover {
    border-color: rgba(0, 201, 167, 0.3) !important;
}

.st-key-intrusion_environment [data-testid="stRadio"] > div {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.st-key-intrusion_environment [data-testid="stRadio"] label {
    color: #e8f4fc !important;
    background: rgba(56, 189, 248, 0.08) !important;
    border: 1px solid rgba(56, 189, 248, 0.22) !important;
    border-radius: 10px !important;
    padding: 8px 16px !important;
}

.st-key-intrusion_environment [data-testid="stRadio"] label p,
.st-key-intrusion_environment [data-testid="stRadio"] label span {
    color: #e8f4fc !important;
    font-weight: 600 !important;
}

.st-key-intrusion_environment [data-testid="stRadio"] label:has(input:checked) {
    color: #67e8f9 !important;
    background: rgba(56, 189, 248, 0.14) !important;
    border-color: rgba(56, 189, 248, 0.4) !important;
    box-shadow: 0 0 14px rgba(56, 189, 248, 0.1);
}

.st-key-intrusion_environment [data-testid="stRadio"] label:has(input:checked) p,
.st-key-intrusion_environment [data-testid="stRadio"] label:has(input:checked) span {
    color: #67e8f9 !important;
}

[data-testid="stImage"] img {
    border-radius: 8px;
    border: none;
    max-height: none !important;
    object-fit: contain;
    width: 100%;
}

[data-testid="stVideo"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--card-border);
    max-height: 380px;
}

[data-testid="stVideo"] video {
    max-height: 380px;
}

h3 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

.stSubheader, [data-testid="stSubheader"] {
    color: var(--text-primary) !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Helper Functions
# ============================================================

def render_workflow_sidebar():
    st.markdown("""
    <div class="workflow-panel">
        <div class="workflow-panel-title">Pipeline</div>
        <div class="workflow-block purple">Input · Media Upload</div>
        <div class="workflow-connector"></div>
        <div class="workflow-block purple">YOLOv26 Object Detection</div>
        <div class="workflow-block" style="margin:-4px 0 8px 12px;font-size:0.68rem;color:#7da8c4;border:none;background:transparent;padding:0;">
            함정 · 드론 · 사람
        </div>
        <div class="workflow-connector"></div>
        <div class="workflow-block green">Dynamic Crop</div>
        <div class="workflow-connector"></div>
        <div class="workflow-block orange">ViT Classification</div>
        <div class="workflow-block" style="margin:-4px 0 8px 12px;font-size:0.68rem;color:#7da8c4;border:none;background:transparent;padding:0;">
            함정 톤급 세부 분류
        </div>
        <div class="workflow-connector"></div>
        <div class="workflow-block blue">Bounding Box Visualization</div>
        <div class="workflow-connector"></div>
        <div class="workflow-block">Output · Visual / JSON</div>
    </div>
    """, unsafe_allow_html=True)


def clear_uploaded_file(widget_key: str):
    st.session_state.pop(widget_key, None)
    st.session_state.pop(f"{widget_key}_saved", None)


def render_clickable_detections(image: Image.Image, predictions: list) -> int | None:
    """이미지 위 bbox를 클릭 가능한 컴포넌트로 렌더링한다."""
    annotated = draw_predictions(image, predictions)
    buffer = io.BytesIO()
    annotated.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    width, height = image.size
    boxes = []
    for index, pred in enumerate(predictions):
        boxes.append({
            "index": index,
            "object_class": pred.object_class,
            "left": pred.x / width * 100,
            "top": pred.y / height * 100,
            "width": pred.width / width * 100,
            "height": pred.height / height * 100,
        })
    selection = clickable_detection_image(
        key="detection_bbox_selector",
        data={"image": f"data:image/jpeg;base64,{encoded}", "boxes": boxes},
        default={"selected_box": None},
        on_selected_box_change=lambda: None,
    )
    return selection.selected_box


@st.cache_data(show_spinner="선택한 함정의 톤급을 분류하고 있습니다...")
def classify_crop_bytes(crop_bytes: bytes) -> tuple[str, float]:
    crop = read_image_from_bytes(crop_bytes)
    return classify_ship(crop)


def render_selected_crop(image: Image.Image, predictions: list, selected_index: int | None) -> None:
    if selected_index is None:
        st.info("이미지의 바운딩 박스를 클릭하면 해당 영역의 crop과 분류 결과가 표시됩니다.")
        return
    try:
        selected_index = int(selected_index)
        prediction = predictions[selected_index]
    except (TypeError, ValueError, IndexError):
        st.warning("선택한 바운딩 박스를 찾을 수 없습니다. 다시 선택해 주세요.")
        return

    crop = image.crop((prediction.x, prediction.y, prediction.x2, prediction.y2))
    st.markdown('<div class="section-title">선택 영역 Classification</div>', unsafe_allow_html=True)
    crop_col, result_col = st.columns([1.2, 2], gap="medium")
    with crop_col:
        st.image(crop, caption=f"Crop · {prediction.object_class}", use_container_width=True)
    with result_col:
        st.metric("탐지 객체", prediction.object_class)
        st.metric("Detection Confidence", f"{prediction.detection_confidence:.2f}")
        if prediction.object_class != "ship":
            st.warning("함정 톤급 Classification 모델은 함정 바운딩 박스에만 적용됩니다.")
            return
        crop_buffer = io.BytesIO()
        crop.save(crop_buffer, format="PNG")
        try:
            class_name, confidence = classify_crop_bytes(crop_buffer.getvalue())
            st.metric("함정 톤급", class_name)
            st.metric("Classification Confidence", f"{confidence:.4f}")
        except Exception as exc:
            st.error(f"Classification 처리 중 오류: {exc}")


def render_media_card(filename: str, meta: str, thumbnail: Image.Image, uploader_key: str):
    buffer = io.BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    with st.container(key=f"{uploader_key}_card"):
        st.markdown(f"""
        <div class="media-card">
            <img class="media-thumb" src="data:image/jpeg;base64,{encoded}" alt="thumbnail">
            <div>
                <div class="media-name">{filename}</div>
                <div class="media-meta">{meta}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.button(
            "×",
            key=f"clear_{uploader_key}",
            help="업로드 파일 지우기",
            on_click=clear_uploaded_file,
            args=(uploader_key,),
        )


def render_detection_stats(summary: dict, mode: str):
    counts = summary["size_counts"]
    st.markdown(f"""
    <div class="detection-stats-bar">
        <div class="stat-chip">
            <div class="stat-chip-label">Total</div>
            <div class="stat-chip-value">{summary["total"]}</div>
        </div>
        <div class="stat-chip">
            <div class="stat-chip-label">대형</div>
            <div class="stat-chip-value">{counts.get("대형", 0)}</div>
        </div>
        <div class="stat-chip">
            <div class="stat-chip-label">중형</div>
            <div class="stat-chip-value">{counts.get("중형", 0)}</div>
        </div>
        <div class="stat-chip">
            <div class="stat-chip-label">소형</div>
            <div class="stat-chip-value">{counts.get("소형", 0)}</div>
        </div>
    </div>
    <div class="dropzone-hint">처리 모드: {mode} · 탐지 {summary["avg_detection_confidence"]} · 분류 {summary["avg_classification_confidence"]}</div>
    """, unsafe_allow_html=True)


def render_object_detection_stats(summary: dict, mode: str):
    counts = summary.get("object_counts", {})
    st.markdown(f"""
    <div class="detection-stats-bar">
        <div class="stat-chip">
            <div class="stat-chip-label">Total</div>
            <div class="stat-chip-value">{summary["total"]}</div>
        </div>
        <div class="stat-chip">
            <div class="stat-chip-label">Ship</div>
            <div class="stat-chip-value">{counts.get("ship", 0)}</div>
        </div>
        <div class="stat-chip">
            <div class="stat-chip-label">Drone</div>
            <div class="stat-chip-value">{counts.get("drone", 0)}</div>
        </div>
        <div class="stat-chip">
            <div class="stat-chip-label">Person</div>
            <div class="stat-chip-value">{counts.get("person", 0)}</div>
        </div>
    </div>
    <div class="dropzone-hint">처리 모드: {mode} · 평균 탐지 Confidence {summary["avg_detection_confidence"]}</div>
    """, unsafe_allow_html=True)


@st.cache_data(show_spinner="영상 전체 프레임을 탐지하고 있습니다...")
def process_video_bytes(data: bytes, suffix: str, task: str):
    input_path = output_path = None
    capture = writer = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".mp4") as input_file:
            input_file.write(data)
            input_path = input_file.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as output_file:
            output_path = output_file.name

        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            raise ValueError("영상 파일을 열 수 없습니다.")

        fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise ValueError("결과 영상을 생성할 수 없습니다.")

        first_image = first_result = None
        all_predictions = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pipeline_result = run_pipeline(image, task=task)
            annotated = draw_predictions(image, pipeline_result.predictions)
            annotated_array = np.asarray(annotated, dtype=np.uint8)
            annotated_bgr = cv2.cvtColor(annotated_array, cv2.COLOR_RGB2BGR)
            writer.write(annotated_bgr)
            all_predictions.extend(pipeline_result.predictions)
            if first_image is None:
                first_image = image
                first_result = pipeline_result

        capture.release()
        capture = None
        writer.release()
        writer = None
        if first_image is None or first_result is None:
            raise ValueError("영상에 처리할 프레임이 없습니다.")

        with open(output_path, "rb") as result_file:
            video_bytes = result_file.read()
        return video_bytes, first_image, first_result, all_predictions, frame_count
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        for path in (input_path, output_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except PermissionError:
                    # Windows/FFmpeg can release a video handle a moment late.
                    pass


def process_upload(uploaded_file, task: str = "detection", environment: str | None = None):
    data = uploaded_file.getvalue()
    name = uploaded_file.name
    suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""

    image_suffixes = {".jpg", ".jpeg", ".png"}
    video_suffixes = {".mp4", ".avi", ".mov"}
    mime_type = uploaded_file.type or ""

    if mime_type.startswith("image") or suffix in image_suffixes:
        image = read_image_from_bytes(data)
        frame_index = None
        media_meta = f"Image · {image.size[0]} x {image.size[1]}"
        pipeline_result = run_pipeline(image, task=task)
        predictions = pipeline_result.predictions
        annotated = draw_predictions(image, predictions)
        video_bytes = None
    elif mime_type.startswith("video") or suffix in video_suffixes:
        video_bytes, image, pipeline_result, predictions, frame_count = process_video_bytes(
            data, suffix or ".mp4", task
        )
        frame_index = 0
        media_meta = f"Video · {frame_count} frames"
        annotated = None
    else:
        raise ValueError("지원하지 않는 파일 형식입니다.")

    summary = summarize_predictions(predictions)
    json_payload = pipeline_to_json(
        pipeline_result,
        image.size,
        name,
        frame_index=frame_index,
    )
    if environment is not None:
        json_payload["environment"] = environment

    return {
        "image": image,
        "annotated": annotated,
        "video_bytes": video_bytes,
        "summary": summary,
        "predictions": predictions,
        "pipeline_result": pipeline_result,
        "json_text": format_json_output(json_payload),
        "media_meta": media_meta,
        "is_video": mime_type.startswith("video") or suffix in video_suffixes,
        "mode": pipeline_result.mode,
    }


def render_dashboard_tab1(summary: dict | None = None, mode: str = "demo"):
    counts = (summary or {}).get("object_counts", {})
    total = str((summary or {}).get("total", 0))
    avg_det = str((summary or {}).get("avg_detection_confidence", 0.0))
    ship = str(counts.get("ship", 0))
    drone = str(counts.get("drone", 0))
    person = str(counts.get("person", 0))

    st.markdown(f"""
    <div class="dashboard-panel">
        <div class="dashboard-panel-title">통합 객체 탐지 현황</div>
        <div class="metric-grid-2">
            <div class="metric-card">
                <div class="metric-label">전체 탐지 수</div>
                <div class="metric-value cyan">{total}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">처리 모드</div>
                <div class="metric-value accent">{mode}</div>
            </div>
        </div>
        <hr class="compact-divider">
        <div class="object-label">객체별 탐지 수 (Object Detection)</div>
        <div class="object-tags">
            <div class="object-tag">함정 <strong>{ship}</strong></div>
            <div class="object-tag">드론 <strong>{drone}</strong></div>
            <div class="object-tag">사람 <strong>{person}</strong></div>
        </div>
        <hr class="compact-divider">
        <div class="metric-card">
            <div class="metric-label">평균 탐지 Confidence</div>
            <div class="metric-value accent">{avg_det}</div>
        </div>
        <div class="active-status">
            <span class="status-dot"></span>
            Detection Status : ACTIVE
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_dashboard_tab2(summary: dict | None = None, mode: str = "demo"):
    counts = (summary or {}).get("size_counts", {})
    fine_counts = (summary or {}).get("fine_counts", {})
    total = (summary or {}).get("total", 0)
    avg_confidence = (summary or {}).get("avg_classification_confidence", 0.0)
    tonnage_tags = "".join(
        f'<div class="object-tag">{label} <strong>{count}</strong></div>'
        for label, count in fine_counts.items()
    ) or '<div class="dropzone-hint">분류 결과 없음</div>'
    st.markdown(f"""
    <div class="dashboard-panel">
        <div class="dashboard-panel-title">함정 톤급 분류 현황</div>
        <div class="metric-card">
            <div class="metric-label">분류 함정 수</div>
            <div class="metric-value accent">{total}</div>
        </div>
        <hr class="compact-divider">
        <div class="metric-grid-2">
            <div class="metric-card">
                <div class="metric-label">대형</div>
                <div class="metric-value">{counts.get("대형", 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">중형</div>
                <div class="metric-value">{counts.get("중형", 0)}</div>
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">소형</div>
            <div class="metric-value">{counts.get("소형", 0)}</div>
        </div>
        <hr class="compact-divider">
        <div class="object-label">톤급별 분류 결과</div>
        <div class="object-tags">{tonnage_tags}</div>
        <hr class="compact-divider">
        <div class="metric-card">
            <div class="metric-label">평균 분류 Confidence</div>
            <div class="metric-value cyan">{avg_confidence:.2f}</div>
        </div>
        <div class="dropzone-hint">처리 모드: {mode}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Hero Header
# ============================================================

st.markdown("""
<div class="hero-header">
    <div class="hero-badge">● LIVE MONITORING</div>
    <div class="hero-top">
        <div class="hero-title-block">
            <div class="main-title">해양 침투 객체 감시 시스템</div>
        </div>
        <div class="hero-stats">
            <div class="hero-stat-item">탐지 <span>YOLOv26</span></div>
            <div class="hero-stat-item">분류 <span>ViT</span></div>
            <div class="hero-stat-item">데이터 <span>Roboflow</span></div>
            <div class="hero-stat-item">상태 <span style="color:#34d399">ACTIVE</span></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Tabs
# ============================================================

tab1, tab3 = st.tabs([
    "객체 탐지",
    "모델 성능평가",
])


# ============================================================
# TAB 1 — 실시간 감시
# ============================================================

with tab1:
    result = None
    uploaded_file = st.session_state.get("live_upload_saved")
    if uploaded_file is None:
        uploaded_file = st.file_uploader(
            "파일을 드래그 앤 드롭하거나 클릭하여 선택",
            type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
            label_visibility="collapsed",
            key="live_upload",
        )
        if uploaded_file is not None:
            st.session_state["live_upload_saved"] = uploaded_file
            st.rerun()
    if uploaded_file is not None:
        try:
            result = process_upload(uploaded_file, task="detection")
        except Exception as exc:
            st.error(f"파일 처리 중 오류: {exc}")

    if result is not None:
        info_col, replace_col = st.columns([7, 1.4])
        with info_col:
            st.markdown(f'<div class="result-file-bar">{uploaded_file.name} · {result["media_meta"]}</div>', unsafe_allow_html=True)
        with replace_col:
            st.button("↻ 파일 변경", key="replace_live_upload", on_click=clear_uploaded_file, args=("live_upload",), use_container_width=True)

        output_col, status_col = st.columns([3.2, 1.25], gap="medium")
        with output_col:
            view_mode = st.radio("출력 형식", ["Visual", "JSON"], horizontal=True, label_visibility="collapsed", key="output_view")
            selected_box = None
            if view_mode == "Visual":
                with st.container(border=True, key="ship_result_box"):
                    if result["is_video"]:
                        st.video(result["video_bytes"], format="video/mp4")
                        st.caption("아래 첫 프레임의 바운딩 박스를 클릭해 분류할 영역을 선택하세요.")
                        selected_box = render_clickable_detections(result["image"], result["predictions"])
                    else:
                        selected_box = render_clickable_detections(result["image"], result["predictions"])
            else:
                with st.container(border=True, key="ship_result_box"):
                    st.code(result["json_text"], language="json")
        with status_col:
            render_dashboard_tab1(result["summary"], result["mode"])
        if view_mode == "Visual":
            render_selected_crop(result["image"], result["predictions"], selected_box)
    elif uploaded_file is not None:
        st.button("파일 다시 선택", key="retry_live_upload", on_click=clear_uploaded_file, args=("live_upload",))
    else:
        st.markdown("""
        <div class="output-visual-wrap empty">
            <div class="video-placeholder-text">파일을 업로드하면 함정 · 드론 · 사람 통합 탐지 결과를 표시합니다</div>
            <div class="video-placeholder-sub">하나의 Object Detection 모델이 세 객체를 중복 탐지합니다</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# TAB 3 — 탐지 결과
# ============================================================

with tab3:
    st.markdown('<div class="object-label">평가 대상</div>', unsafe_allow_html=True)

    evaluation_options = ["전체", "대형 함정", "중형 함정", "소형 함정", "드론", "사람"]
    if st.session_state.get("evaluation_target") not in evaluation_options:
        st.session_state["evaluation_target"] = "전체"

    evaluation_target = st.radio(
        "평가 대상",
        evaluation_options,
        horizontal=True,
        label_visibility="collapsed",
        key="evaluation_target",
    )

    try:
        detection_metrics = get_detection_metrics()
        display_metrics = get_display_metrics(evaluation_target)
        target_metrics = display_metrics["metrics"]

        precision = target_metrics.get("precision", 0.0)
        recall = target_metrics.get("recall", 0.0)
        ap50 = target_metrics.get("map50", 0.0)
        ap50_95 = target_metrics.get("map50_95", 0.0)
        images = target_metrics.get("images", 0)
        targets = target_metrics.get("targets", 0)
        is_overall = display_metrics["type"] == "overall"
        metric_prefix = "mAP" if is_overall else "AP"
        source = detection_metrics.get("overall_source", "-") if is_overall else display_metrics["class_name"]
        scope_text = "전체 모델" if is_overall else "선택 클래스"

        st.markdown(f"""
        <div class="tab3-layout">
            <div class="object-label">선택된 평가 대상: {evaluation_target}</div>
            <div class="evaluation-grid">
                <div class="evaluation-card">
                    <div class="evaluation-name">Precision</div>
                    <div class="evaluation-value">{precision:.4f}</div>
                    <div class="evaluation-desc">{scope_text}의 정밀도입니다.</div>
                </div>
                <div class="evaluation-card">
                    <div class="evaluation-name">Recall</div>
                    <div class="evaluation-value">{recall:.4f}</div>
                    <div class="evaluation-desc">{scope_text}의 재현율입니다.</div>
                </div>
                <div class="evaluation-card">
                    <div class="evaluation-name">{metric_prefix}@0.5</div>
                    <div class="evaluation-value">{ap50:.4f}</div>
                    <div class="evaluation-desc">{scope_text}의 IoU 0.5 기준 Average Precision입니다.</div>
                </div>
                <div class="evaluation-card">
                    <div class="evaluation-name">{metric_prefix}@0.5:0.95</div>
                    <div class="evaluation-value">{ap50_95:.4f}</div>
                    <div class="evaluation-desc">{scope_text}에서 IoU 0.5~0.95를 반영한 Average Precision입니다.</div>
                </div>
                <div class="evaluation-card">
                    <div class="evaluation-name">평가 데이터</div>
                    <div class="evaluation-value">{images if not is_overall else detection_metrics["split"]}</div>
                    <div class="evaluation-desc">{f'Images {images} · Targets {targets}' if not is_overall else f'전체 모델 성능 · Source {source}'}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as exc:
        st.error(f"Roboflow 성능평가 연동 오류: {exc}")


# ============================================================
# Footer
# ============================================================

st.markdown("""
<div class="app-footer">
    Marine Ship Detection &amp; Classification System &nbsp;·&nbsp; YOLOv26 + ViT
</div>
""", unsafe_allow_html=True)
