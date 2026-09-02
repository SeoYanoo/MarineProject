import base64
import io
import math
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

if os.getenv("MARINE_DEV_RELOAD", "").strip() == "1":
    import importlib

    importlib.reload(_detection)
    importlib.reload(_classification)
    importlib.reload(_roboflow_metrics)

from classification import classify_ship
from config import (
    VIDEO_INFERENCE_FPS,
    VIDEO_MAX_DIMENSION,
    VIDEO_MAX_INFERENCE_CALLS,
    VIDEO_OUTPUT_FPS,
)
from roboflow_metrics import get_detection_metrics, get_display_metrics


clickable_detection_image = components_v2.component(
    "clickable_detection_image",
    html='<div class="clickable-image-root"></div>',
    css="""
    .clickable-image-root { position: relative; width: 100%; line-height: 0; }
    .clickable-image-root img { width: 100%; height: auto; border-radius: 8px; display: block; }
    .clickable-image-root button { position: absolute; z-index: 5; padding: 0; margin: 0;
        border: 0; background: transparent; cursor: pointer; box-sizing: border-box; }
    .clickable-image-root button:hover { background: rgba(126, 169, 205, .12); outline: 2px solid #8fb9dc; }
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

/* ── Refined maritime console theme ────────────────────────── */
:root {
    --ocean-deep: #07111d;
    --ocean-mid: #0d1826;
    --ocean-light: #142235;
    --accent-teal: #9bbcd6;
    --accent-cyan: #78a6cd;
    --accent-glow: transparent;
    --card-bg: #0d1826;
    --card-border: #233247;
    --text-primary: #f2f5f8;
    --text-muted: #8997a9;
    --success: #7fc6a4;
    --warning: #d9b879;
}

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    letter-spacing: -0.015em;
}

.stApp {
    background: #07111d;
    color: var(--text-primary);
}

.block-container {
    max-width: 1320px;
    padding: 28px 32px 20px;
}

[data-testid="stVerticalBlock"] { gap: .65rem !important; }

.hero-header {
    background: transparent;
    border: 0;
    border-bottom: 1px solid var(--card-border);
    border-radius: 0;
    padding: 12px 2px 24px;
    margin-bottom: 22px;
    box-shadow: none;
    overflow: visible;
}

.hero-header::before { display: none; }

.hero-badge {
    gap: 8px;
    margin: 0 0 13px;
    padding: 0;
    background: transparent;
    border: 0;
    border-radius: 0;
    color: #7990a8;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .16em;
}

.hero-badge::before {
    content: '';
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--success);
}

.hero-top { align-items: flex-end; gap: 28px; }

.main-title {
    color: #f5f7fa;
    font-size: clamp(1.55rem, 2.3vw, 2.15rem);
    font-weight: 600;
    letter-spacing: -.045em;
    line-height: 1.25;
}

.sub-title {
    margin-top: 8px;
    color: var(--text-muted);
    font-size: .8rem;
}

.hero-stats {
    gap: 0;
    padding: 0 0 3px;
    flex-wrap: nowrap;
}

.hero-stat-item {
    gap: 5px;
    padding: 0 13px;
    border-right: 1px solid var(--card-border);
    color: #708095;
    font-size: .73rem;
    white-space: nowrap;
}

.hero-stat-item:first-child { padding-left: 0; }
.hero-stat-item:last-child { padding-right: 0; border-right: 0; }
.hero-stat-item span { color: #c5d0db; font-weight: 500; }

.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
    min-height: 42px;
    margin-bottom: 20px;
    padding: 0;
    background: transparent;
    border: 0;
    border-bottom: 1px solid var(--card-border);
    border-radius: 0;
}

.stTabs [data-baseweb="tab"] {
    min-height: 42px;
    padding: 0 2px 13px;
    border: 0 !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0;
    color: #718096;
    font-size: .84rem;
    font-weight: 500;
    transition: color .16s ease, border-color .16s ease;
}

.stTabs [data-baseweb="tab"]:hover { color: #c8d1da; }

.stTabs [aria-selected="true"] {
    background: transparent !important;
    border-color: #89abc8 !important;
    box-shadow: none;
    color: #eef3f7 !important;
}

.stTabs [data-baseweb="tab-panel"] { padding-top: 4px; }

[data-testid="stFileUploader"] {
    padding: 0 !important;
    background: transparent !important;
    border: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}

[data-testid="stFileUploader"] section {
    min-height: 230px;
    padding: 52px 28px 38px !important;
    background: #0a1522 !important;
    border: 1px dashed #34465b !important;
    border-radius: 10px !important;
    transition: background .16s ease, border-color .16s ease;
}

[data-testid="stFileUploader"] section:hover {
    background: #0d1927 !important;
    border-color: #6684a0 !important;
}

[data-testid="stFileUploader"] section::before {
    content: 'MEDIA INPUT';
    top: 38px;
    color: #67788c;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .16em;
}

[data-testid="stFileUploader"] section::after {
    content: 'JPG, PNG, MP4, AVI, MOV';
    bottom: 30px;
    color: #536275;
    font-size: 10px;
    letter-spacing: .07em;
}

[data-testid="stFileUploader"] section button,
.stButton button {
    min-height: 36px !important;
    padding: 7px 15px !important;
    background: #152336 !important;
    border: 1px solid #304259 !important;
    border-radius: 7px !important;
    box-shadow: none !important;
    color: #d9e2ea !important;
    font-size: .78rem !important;
    font-weight: 500 !important;
}

[data-testid="stFileUploader"] section button:hover,
.stButton button:hover {
    background: #1a2b40 !important;
    border-color: #58738e !important;
    color: #ffffff !important;
}

.output-visual-wrap {
    background: #0a1522;
    border: 1px solid var(--card-border);
    border-radius: 10px;
    box-shadow: none;
}

.output-visual-wrap.empty {
    min-height: 360px;
    background: #0a1522;
    border: 1px solid var(--card-border);
}

.video-placeholder-text {
    color: #b7c3cf;
    font-size: .9rem;
    font-weight: 500;
}

.video-placeholder-sub { color: #637388; font-size: .74rem; }

.dashboard-panel {
    padding: 20px;
    background: #0d1826;
    border: 1px solid var(--card-border);
    border-radius: 10px;
    box-shadow: none;
}

.dashboard-panel-title {
    margin-bottom: 18px;
    padding-bottom: 13px;
    border-bottom: 1px solid var(--card-border);
    color: #c8d2dc;
    font-size: .76rem;
    font-weight: 600;
    letter-spacing: .04em;
}

.metric-grid-2 { gap: 8px; }

.metric-card {
    padding: 13px 14px;
    background: #101d2c;
    border: 1px solid #223249;
    border-radius: 8px;
    box-shadow: none;
}

.metric-card:hover {
    background: #101d2c;
    border-color: #344861;
    transform: none;
    box-shadow: none;
}

.metric-label {
    color: #748399;
    font-size: .69rem;
    font-weight: 500;
    letter-spacing: .01em;
}

.metric-value {
    margin-top: 7px;
    color: #eef2f6;
    font-size: 1.25rem;
    font-weight: 600;
    letter-spacing: -.03em;
}

.metric-value.accent,
.metric-value.cyan { color: #a8c2d8; }

.compact-divider,
hr { border-color: #223047 !important; margin: 13px 0 !important; }

.object-label {
    margin-bottom: 9px;
    color: #78889d;
    font-size: .68rem;
    font-weight: 600;
    letter-spacing: .04em;
}

.object-tags { gap: 6px; }

.object-tag {
    padding: 6px 9px;
    background: #111f30;
    border: 1px solid #263950;
    border-radius: 6px;
    color: #8e9db0;
    font-size: .7rem;
}

.object-tag strong { color: #dce4eb; font-weight: 600; }

.active-status {
    margin-top: 13px;
    padding: 9px 11px;
    background: rgba(127, 198, 164, .06);
    border: 1px solid rgba(127, 198, 164, .2);
    border-radius: 7px;
    color: #88bca3;
    font-size: .68rem;
    letter-spacing: .06em;
}

.status-dot {
    width: 5px;
    height: 5px;
    background: var(--success);
    box-shadow: none;
    animation: none;
}

[data-testid="stRadio"] label,
.st-key-intrusion_environment [data-testid="stRadio"] label {
    padding: 7px 12px !important;
    background: transparent !important;
    border: 1px solid #2a3a4f !important;
    border-radius: 7px !important;
    color: #9eacbb !important;
    font-size: .76rem !important;
}

[data-testid="stRadio"] label:hover,
.st-key-intrusion_environment [data-testid="stRadio"] label:hover {
    background: #0f1d2d !important;
    border-color: #516a84 !important;
}

[data-testid="stRadio"] label:has(input:checked),
.st-key-intrusion_environment [data-testid="stRadio"] label:has(input:checked) {
    background: #17273a !important;
    border-color: #6f91af !important;
    box-shadow: none !important;
}

[data-testid="stRadio"] label:has(input:checked) p,
[data-testid="stRadio"] label:has(input:checked) span,
.st-key-intrusion_environment [data-testid="stRadio"] label:has(input:checked) p,
.st-key-intrusion_environment [data-testid="stRadio"] label:has(input:checked) span {
    color: #e4ebf1 !important;
    font-weight: 500 !important;
}

.result-file-bar {
    margin: 4px 0 10px;
    color: #8291a4;
    font-size: .75rem;
}

.st-key-ship_result_box,
.st-key-intrusion_result_box,
[data-testid="stVideo"] {
    background: #08121e;
    border-color: var(--card-border) !important;
    border-radius: 9px !important;
    box-shadow: none;
}

[data-testid="stImage"] img { border-radius: 7px; }

[data-testid="stMetric"] {
    padding: 12px 14px;
    background: #0f1c2b;
    border: 1px solid #223249;
    border-radius: 8px;
}

[data-testid="stMetricLabel"] { color: #7d8da1 !important; }
[data-testid="stMetricValue"] { color: #eef2f6 !important; font-weight: 600 !important; }

.stAlert {
    background: #0f1c2b !important;
    border: 1px solid #293b51 !important;
    border-radius: 8px !important;
    color: #aeb9c5 !important;
}

/* ── Processing state ── */
.processing-state {
    position: relative;
    display: grid;
    grid-template-columns: 34px minmax(0, 1fr) auto;
    align-items: center;
    gap: 15px;
    min-height: 86px;
    margin: 2px 0 12px;
    padding: 17px 20px;
    overflow: hidden;
    background: #0b1724;
    border: 1px solid #2a3a4f;
    border-radius: 2px;
}

.processing-state::before {
    content: '';
    position: absolute;
    top: -1px;
    left: -1px;
    width: 42px;
    height: 1px;
    background: #8aaac4;
}

.processing-indicator {
    position: relative;
    width: 28px;
    height: 28px;
    border: 1px solid #30445b;
    border-radius: 50%;
}

.processing-indicator::before {
    content: '';
    position: absolute;
    inset: 4px;
    border: 2px solid transparent;
    border-top-color: #9bbbd3;
    border-right-color: #607f9b;
    border-radius: 50%;
    animation: processing-rotate .9s linear infinite;
}

.processing-copy { min-width: 0; }

.processing-kicker {
    margin-bottom: 4px;
    color: #6f8499;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: .16em;
}

.processing-title {
    color: #edf2f6;
    font-size: .86rem;
    font-weight: 600;
    letter-spacing: -.015em;
}

.processing-desc {
    margin-top: 4px;
    color: #7e8da0;
    font-size: .69rem;
}

.processing-meta {
    color: #64778b;
    font-size: 9px;
    font-weight: 600;
    letter-spacing: .11em;
    white-space: nowrap;
}

.processing-track {
    position: absolute;
    right: 0;
    bottom: 0;
    left: 0;
    height: 2px;
    overflow: hidden;
    background: #132236;
}

.processing-track::after {
    content: '';
    position: absolute;
    inset: 0;
    width: 28%;
    background: linear-gradient(90deg, transparent, #7399b8, transparent);
    animation: processing-scan 1.8s ease-in-out infinite;
}

@keyframes processing-rotate { to { transform: rotate(360deg); } }
@keyframes processing-scan {
    from { transform: translateX(-110%); }
    to { transform: translateX(390%); }
}

[data-testid="stSpinner"] {
    padding: 12px 14px !important;
    background: #0b1724 !important;
    border: 1px solid #2a3a4f !important;
    border-radius: 2px !important;
}

[data-testid="stSpinner"] p,
[data-testid="stSpinner"] span { color: #dfe7ee !important; }

.evaluation-grid { gap: 8px; margin-top: 12px; }

.evaluation-card {
    grid-template-columns: minmax(140px, 170px) minmax(120px, 145px) 1fr;
    gap: 20px;
    min-height: 78px;
    padding: 16px 20px;
    background: #0d1826;
    border: 1px solid var(--card-border);
    border-radius: 8px;
}

.evaluation-name {
    color: #91abc2;
    font-size: .78rem;
    font-weight: 600;
}

.evaluation-value {
    color: #f0f4f7;
    font-size: 1.1rem;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}

.evaluation-desc { color: #78879a; font-size: .74rem; }

.dropzone-hint { color: #68788c; font-size: .7rem; }

.app-footer {
    margin-top: 28px;
    padding: 18px 0 6px;
    border-top: 1px solid #1d2b3d;
    color: #526176;
    font-size: .65rem;
    letter-spacing: .08em;
}

@media (max-width: 900px) {
    .block-container { padding: 18px 18px 14px; }
    .hero-top { align-items: flex-start; }
    .hero-stats { width: 100%; overflow-x: auto; }
    .evaluation-card { grid-template-columns: 1fr 1fr; gap: 7px 14px; }
    .evaluation-desc { grid-column: 1 / -1; }
}

@media (max-width: 640px) {
    .hero-header { padding-top: 4px; }
    .hero-stat-item { padding: 0 9px; }
    .stTabs [data-baseweb="tab-list"] { gap: 18px; }
    .output-visual-wrap.empty { min-height: 280px; padding: 24px; }
    .evaluation-card { grid-template-columns: 1fr; }
    .evaluation-desc { grid-column: auto; }
    .processing-state { grid-template-columns: 30px 1fr; padding: 15px 16px; }
    .processing-meta { display: none; }
}

@media (prefers-reduced-motion: reduce) {
    .processing-indicator::before,
    .processing-track::after { animation: none; }
}

/* ── Technical edge & ambient navigation field ────────────── */
html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #07111d !important;
}

[data-testid="stMain"] {
    background-image:
        linear-gradient(rgba(126, 159, 188, .038) 1px, transparent 1px),
        linear-gradient(90deg, rgba(126, 159, 188, .038) 1px, transparent 1px),
        linear-gradient(rgba(126, 159, 188, .016) 1px, transparent 1px),
        linear-gradient(90deg, rgba(126, 159, 188, .016) 1px, transparent 1px),
        radial-gradient(circle at 82% 8%, rgba(71, 112, 148, .10), transparent 34%),
        radial-gradient(circle at 18% 88%, rgba(35, 72, 101, .08), transparent 30%);
    background-size: 48px 48px, 48px 48px, 12px 12px, 12px 12px, auto, auto;
}

.section-card,
.workflow-panel,
.io-panel,
.media-card,
.output-visual-wrap,
.output-visual-wrap.empty,
.dashboard-panel,
.metric-card,
.object-tag,
.active-status,
.evaluation-card,
[data-testid="stMetric"],
.stAlert,
[data-testid="stVideo"],
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section button,
.stButton button,
[data-testid="stRadio"] label,
.st-key-intrusion_environment [data-testid="stRadio"] label,
.st-key-ship_result_box,
.st-key-intrusion_result_box {
    border-radius: 2px !important;
}

[data-testid="stImage"] img,
.clickable-image-root img,
.media-thumb { border-radius: 1px !important; }

.dashboard-panel,
.evaluation-card,
.output-visual-wrap,
.st-key-ship_result_box,
.st-key-intrusion_result_box {
    position: relative;
}

.dashboard-panel::before,
.evaluation-card::before {
    content: '';
    position: absolute;
    top: -1px;
    left: -1px;
    width: 34px;
    height: 1px;
    background: #789ab8;
    pointer-events: none;
}

.hero-header {
    border-bottom-color: #304158;
}

.hero-header::after {
    content: 'SURVEILLANCE GRID  ·  SECTOR 01';
    position: absolute;
    right: 2px;
    bottom: -18px;
    color: #435369;
    font-size: 9px;
    font-weight: 500;
    letter-spacing: .12em;
}

.stTabs [aria-selected="true"] {
    border-bottom-color: #9bb8cf !important;
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


def _new_video_summary() -> dict:
    return {
        "total": 0,
        "size_counts": {"대형": 0, "중형": 0, "소형": 0},
        "object_counts": {"ship": 0, "drone": 0, "person": 0},
        "confidence_sum": 0.0,
    }


def _accumulate_video_summary(summary: dict, predictions: list) -> None:
    for prediction in predictions:
        summary["total"] += 1
        summary["confidence_sum"] += prediction.detection_confidence
        if prediction.object_class == "ship":
            size_counts = summary["size_counts"]
            size_counts[prediction.size_class] = size_counts.get(prediction.size_class, 0) + 1
        object_counts = summary["object_counts"]
        object_counts[prediction.object_class] = object_counts.get(prediction.object_class, 0) + 1


def _finalize_video_summary(summary: dict) -> dict:
    total = summary["total"]
    return {
        "total": total,
        "size_counts": summary["size_counts"],
        "object_counts": summary["object_counts"],
        "avg_detection_confidence": round(summary["confidence_sum"] / total, 2) if total else 0.0,
        "avg_classification_confidence": 0.0,
    }


@st.cache_data(
    show_spinner=False,
    ttl=3600,
    max_entries=1,
)
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

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
        if not math.isfinite(source_fps) or source_fps <= 0:
            source_fps = 24.0
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        output_scale = min(
            1.0,
            VIDEO_MAX_DIMENSION / max(source_width, source_height, 1),
        )
        width = max(2, int(round(source_width * output_scale / 2)) * 2)
        height = max(2, int(round(source_height * output_scale / 2)) * 2)
        output_stride = max(1, int(round(source_fps / min(source_fps, VIDEO_OUTPUT_FPS))))
        output_fps = source_fps / output_stride
        inference_stride = max(1, int(round(source_fps / VIDEO_INFERENCE_FPS)))
        if frame_count > 0:
            inference_stride = max(
                inference_stride,
                math.ceil(frame_count / VIDEO_MAX_INFERENCE_CALLS),
            )

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (width, height),
        )
        if not writer.isOpened():
            raise ValueError("결과 영상을 생성할 수 없습니다.")

        first_image = first_result = None
        latest_predictions = []
        video_summary = _new_video_summary()
        source_frame_index = 0
        next_inference_frame = 0
        analyzed_frames = 0
        written_frames = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if source_frame_index % output_stride == 0:
                if (frame.shape[1], frame.shape[0]) != (width, height):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if first_result is None or source_frame_index >= next_inference_frame:
                    pipeline_result = run_pipeline(image, task=task)
                    latest_predictions = pipeline_result.predictions
                    _accumulate_video_summary(video_summary, latest_predictions)
                    analyzed_frames += 1
                    next_inference_frame = source_frame_index + inference_stride
                    if first_result is None:
                        first_image = image
                        first_result = pipeline_result

                annotated = draw_predictions(image, latest_predictions)
                annotated_array = np.asarray(annotated, dtype=np.uint8)
                annotated_bgr = cv2.cvtColor(annotated_array, cv2.COLOR_RGB2BGR)
                writer.write(annotated_bgr)
                written_frames += 1

            source_frame_index += 1

        capture.release()
        capture = None
        writer.release()
        writer = None
        if first_image is None or first_result is None or written_frames == 0:
            raise ValueError("영상에 처리할 프레임이 없습니다.")

        with open(output_path, "rb") as result_file:
            video_bytes = result_file.read()
        return (
            video_bytes,
            first_image,
            first_result,
            first_result.predictions,
            frame_count,
            analyzed_frames,
            _finalize_video_summary(video_summary),
            output_fps,
        )
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
        summary = summarize_predictions(predictions)
        annotated = draw_predictions(image, predictions)
        video_bytes = None
        analyzed_frames = None
    elif mime_type.startswith("video") or suffix in video_suffixes:
        (
            video_bytes,
            image,
            pipeline_result,
            predictions,
            frame_count,
            analyzed_frames,
            summary,
            output_fps,
        ) = process_video_bytes(data, suffix or ".mp4", task)
        frame_index = 0
        media_meta = (
            f"Video · {frame_count} frames · "
            f"{analyzed_frames} analyzed · {output_fps:.1f} fps output"
        )
        annotated = None
    else:
        raise ValueError("지원하지 않는 파일 형식입니다.")

    json_payload = pipeline_to_json(
        pipeline_result,
        image.size,
        name,
        frame_index=frame_index,
    )
    if environment is not None:
        json_payload["environment"] = environment
    if analyzed_frames is not None:
        json_payload["video_analysis"] = {
            "source_frames": frame_count,
            "analyzed_frames": analyzed_frames,
            "output_fps": round(output_fps, 2),
        }

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
    <div class="hero-badge">MARITIME MONITORING</div>
    <div class="hero-top">
        <div class="hero-title-block">
            <div class="main-title">해양 침투 객체 감시 시스템</div>
            <div class="sub-title">영상 기반 객체 탐지 및 함정 분류 통합 관제</div>
        </div>
        <div class="hero-stats">
            <div class="hero-stat-item">Detection <span>YOLOv26</span></div>
            <div class="hero-stat-item">Classification <span>ViT</span></div>
            <div class="hero-stat-item">Status <span style="color:#7fc6a4">Active</span></div>
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
        upload_name = uploaded_file.name.lower()
        is_processing_video = (
            (uploaded_file.type or "").startswith("video")
            or upload_name.endswith((".mp4", ".avi", ".mov"))
        )
        processing_title = (
            "영상을 분석하고 있습니다"
            if is_processing_video
            else "이미지를 분석하고 있습니다"
        )
        processing_desc = (
            "프레임 선별 · 객체 탐지 · 결과 영상 구성 순서로 처리됩니다"
            if is_processing_video
            else "객체 탐지와 결과 시각화를 준비하고 있습니다"
        )
        processing_meta = "FRAME PIPELINE" if is_processing_video else "IMAGE PIPELINE"
        processing_placeholder = st.empty()
        processing_placeholder.markdown(
            f"""
            <div class="processing-state" role="status" aria-live="polite">
                <div class="processing-indicator" aria-hidden="true"></div>
                <div class="processing-copy">
                    <div class="processing-kicker">ANALYSIS IN PROGRESS</div>
                    <div class="processing-title">{processing_title}</div>
                    <div class="processing-desc">{processing_desc}</div>
                </div>
                <div class="processing-meta">{processing_meta}</div>
                <div class="processing-track" aria-hidden="true"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        try:
            result = process_upload(uploaded_file, task="detection")
        except Exception as exc:
            st.error(f"파일 처리 중 오류: {exc}")
        finally:
            processing_placeholder.empty()

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
