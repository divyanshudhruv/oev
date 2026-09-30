# Gradio theme + CSS for the OEV Space demo. One place for the look.
import gradio as gr

theme = gr.themes.Soft(
    primary_hue="orange",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#171412",
    body_background_fill_dark="#171412",
    body_text_color="#f5eee8",
    body_text_color_dark="#f5eee8",
    body_text_color_subdued="#b9aa9f",
    body_text_color_subdued_dark="#b9aa9f",
    block_background_fill="#211d1a",
    block_background_fill_dark="#211d1a",
    block_border_color="#3a2f28",
    block_border_color_dark="#3a2f28",
    border_color_primary="#3a2f28",
    border_color_primary_dark="#3a2f28",
    color_accent="#e8b48c",
    color_accent_soft="#5a4032",
    color_accent_soft_dark="#5a4032",
    border_color_accent="#e8b48c",
    border_color_accent_dark="#e8b48c",
    input_background_fill="#1b1816",
    input_background_fill_dark="#1b1816",
    input_border_color="#4a3b32",
    input_border_color_dark="#4a3b32",
    input_border_color_focus="#e8b48c",
    input_border_color_focus_dark="#e8b48c",
    button_primary_background_fill="#e8b48c",
    button_primary_background_fill_dark="#e8b48c",
    button_primary_background_fill_hover="#dca47a",
    button_primary_background_fill_hover_dark="#dca47a",
    button_primary_border_color="#e8b48c",
    button_primary_border_color_dark="#e8b48c",
    button_primary_border_color_hover="#dca47a",
    button_primary_border_color_hover_dark="#dca47a",
    button_primary_text_color="#2b1d16",
    button_primary_text_color_dark="#2b1d16",
    button_primary_text_color_hover="#2b1d16",
    button_primary_text_color_hover_dark="#2b1d16",
    button_secondary_background_fill="#2a2420",
    button_secondary_background_fill_dark="#2a2420",
    button_secondary_background_fill_hover="#352c26",
    button_secondary_background_fill_hover_dark="#352c26",
    button_secondary_border_color="#4a3b32",
    button_secondary_border_color_dark="#4a3b32",
    button_secondary_border_color_hover="#5a4639",
    button_secondary_border_color_hover_dark="#5a4639",
    button_secondary_text_color="#f5eee8",
    button_secondary_text_color_dark="#f5eee8",
    button_secondary_text_color_hover="#ffffff",
    button_secondary_text_color_hover_dark="#ffffff",
    block_border_width="1px",
    block_radius="0px",
    button_large_radius="0px",
    button_small_radius="0px",
    input_radius="0px",
)

CSS = """
# page frame: centered column
.gradio-container { width: 1020px !important; max-width: calc(100vw - 32px) !important; min-width: 0 !important; box-sizing: border-box !important; margin: 0 auto; }
footer { visibility: hidden; }
:root, .gradio-container { --primary-pastel: #e8b48c; --color-accent: var(--primary-pastel); --border-color-accent: var(--primary-pastel); }
.gradio-container > main,
.gradio-container .main,
.tab-container,
.tab-container > div,
.tab-container .tab-panel,
.tab-container .row { width: 100% !important; min-width: 0 !important; box-sizing: border-box !important; }
.tab-container .column { min-width: 0 !important; box-sizing: border-box !important; }

.gradio-container { font-size: 14px; line-height: 1.5; }
.gradio-container p, .gradio-container textarea, .gradio-container input { font-size: 14px; line-height: 1.5; }
.gradio-container h1 { font-size: 24px; line-height: 1.25; font-weight: 600; }
.gradio-container h2 { font-size: 18px; line-height: 1.25; font-weight: 600; }
.gradio-container h3 { font-size: 16px; line-height: 1.25; font-weight: 600; }
.gradio-container label, .gradio-container .form-label, .gradio-container .block-label { font-size: 12px; line-height: 1.4; font-weight: 500; }
.gradio-container button, .tab-nav button { font-size: 13px; line-height: 1.25; }

# everything square: override any gradio rounding
.gradio-container * { border-radius: 0 !important; }

#header { padding: 12px 0 16px; border-bottom: 1px solid var(--border-color-primary);
          margin-bottom: 16px; }
#header h1 { margin: 0 0 8px; letter-spacing: -0.01em; }
#header .prose p { color: var(--body-text-color-subdued); margin: 0 0 4px; }
#header .metrics { font-family: var(--font-mono); font-size: 12px;
                   color: var(--body-text-color-subdued);
                   letter-spacing: 0.02em; }
#header .metrics b { color: var(--body-text-color); font-weight: 600; }

# preset buttons: equal width, one row
#presets { gap: 8px; margin-bottom: 16px; }
#presets button { flex: 1 1 0; min-width: 0; width: auto; height: 32px !important;
                  min-height: 32px !important; padding: 0 12px !important; }

#qs-help { margin: 12px 0; }
#qs-help button { min-height: 32px !important; padding: 4px 12px !important; }

# decide: full width, own row
#decide-btn { width: 100%; min-height: 40px; margin-top: 16px; }
.tab-container .row { gap: 16px; }

.cm-editor, .cm-content, .cm-line, textarea.code,
#state-box textarea, #qs-box textarea {
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
}

.bars .qname { display: block; font-size: 12px; text-transform: uppercase;
               letter-spacing: 0.1em; color: var(--body-text-color-subdued);
               margin: 16px 0 8px; }
.bars .row { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
.bars .row .name { flex: 0 0 38%; overflow: hidden; text-overflow: ellipsis;
                   white-space: nowrap; font-size: 12px; }
.bars .row .track { flex: 1 1 auto; height: 8px;
                    background: var(--block-background-fill);
                    border: 1px solid var(--border-color-primary); position: relative; }
.bars .row .fill { position: absolute; inset: 0 auto 0 0; height: 100%;
                   background: var(--color-accent); opacity: 0.75; }
.bars .row.win .fill { background: var(--color-accent); opacity: 1; }
.bars .row .val { flex: 0 0 52px; text-align: right;
                  font-family: var(--font-mono); font-size: 12px;
                  color: var(--body-text-color-subdued); }
.bars .row.win .val { color: var(--body-text-color); font-weight: 600; }

#statusline { font-family: var(--font-mono); font-size: 12px;
              color: var(--body-text-color-subdued); text-align: right;
              min-height: 1.4em; margin-top: 8px; }

# error: red left rule
#errorbox { margin: 12px 0; padding: 12px;
            border-left: 3px solid var(--color-danger) !important; }

#state-box textarea:focus, #qs-box textarea:focus, #qs-box input:focus,
.cm-editor.cm-focused, .cm-content:focus {
  outline: 2px solid var(--border-color-accent) !important;
  outline-offset: -1px;
}

.tab-nav { display: flex; width: 100%; gap: 8px; margin-bottom: 16px; }
.tab-nav button { flex: 1 1 0; min-width: 0; min-height: 32px; padding: 6px 12px; font-size: 13px; letter-spacing: 0.04em; }
.tab-nav button:hover,
.tab-nav button:focus-visible,
.tab-nav button[aria-selected="true"] {
  color: var(--color-accent) !important;
  -webkit-text-fill-color: var(--color-accent) !important;
  background: var(--color-accent-soft) !important;
  border-color: var(--color-accent) !important;
  box-shadow: inset 0 -2px 0 var(--color-accent) !important;
}
.tab-nav button:hover *,
.tab-nav button:focus-visible *,
.tab-nav button[aria-selected="true"] * {
  color: var(--color-accent) !important;
  -webkit-text-fill-color: var(--color-accent) !important;
}

# responsive: presets wrap on narrow screens
@media (max-width: 640px) {
  #presets { flex-wrap: wrap; }
  #presets button { flex: 1 1 45%; width: auto; }
}
"""
