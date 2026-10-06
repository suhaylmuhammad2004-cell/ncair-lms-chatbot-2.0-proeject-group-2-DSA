"""Gradio chat UI for the NCAIR LMS assistant.

Run:  python src/app.py
Local-only by default; set NCAIR_SHARE=1 to get a public link, NCAIR_DEBUG=1 to show the routing trace.
"""
import config
from orchestrator import Assistant

assistant = Assistant()


def _text(content):
    """Gradio 6 may give content as a list of blocks like [{"type": "text", "text": "..."}]."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", ""))
    if isinstance(content, (list, tuple)):
        return " ".join(_text(c) for c in content)
    return str(content)


def chat(message, history):
    message = _text(message)
    hist = []
    for m in history or []:
        if isinstance(m, dict):  # messages format (Gradio 5/6)
            hist.append({"role": m["role"], "content": _text(m["content"])})
        elif isinstance(m, (list, tuple)) and len(m) == 2:  # old tuple format (Gradio 4)
            hist += [{"role": "user", "content": _text(m[0])}, {"role": "assistant", "content": _text(m[1])}]
    r = assistant.reply(message, hist)
    return r.text + (f"\n\n`{r.debug()}`" if config.SHOW_DEBUG else "")


if __name__ == "__main__":
    import gradio as gr

    GREEN, GREEN_DARK, GREEN_SOFT = "#16a34a", "#15803d", "#dcfce7"

    CSS = """
    .gradio-container h1 { color: #16a34a !important; }
    .prose a, .message a, .gradio-container a { color: #16a34a !important; }
    """
    FORCE_LIGHT = "() => { document.body.classList.remove('dark'); }"

    def build_theme(gr):
        theme = gr.themes.Soft(primary_hue="green", secondary_hue="green").set(
            body_background_fill="white", background_fill_primary="white",
            background_fill_secondary="white", block_background_fill="white",
            input_background_fill="white", input_background_fill_focus="white",
            input_background_fill_hover="white",
            body_text_color="black", body_text_color_subdued="black",
            block_title_text_color="black", block_label_text_color="black",
            input_placeholder_color="#4b5563",
            button_primary_background_fill=GREEN, button_primary_background_fill_hover=GREEN_DARK,
            button_primary_text_color="white", button_primary_border_color=GREEN,
            button_secondary_background_fill="white", button_secondary_background_fill_hover=GREEN_SOFT,
            button_secondary_text_color="black", button_secondary_border_color=GREEN,
            button_cancel_background_fill=GREEN, button_cancel_text_color="white",
            block_border_width="1px", block_border_color=GREEN, border_color_primary=GREEN,
            border_color_accent=GREEN, border_color_accent_subdued=GREEN,
            input_border_color=GREEN, input_border_color_focus=GREEN, input_border_color_hover=GREEN,
            block_label_background_fill=GREEN_SOFT, block_label_border_color=GREEN,
            color_accent=GREEN, color_accent_soft=GREEN_SOFT,
            link_text_color=GREEN, link_text_color_hover=GREEN_DARK,
            link_text_color_active=GREEN_DARK, link_text_color_visited=GREEN_DARK,
        )
        # make dark mode identical to light mode
        for name in dir(theme):
            if name.endswith("_dark") and hasattr(theme, name[:-5]):
                setattr(theme, name, getattr(theme, name[:-5]))
        return theme

    demo = gr.ChatInterface(
        chat,
        title="NCAIR LMS Assistant",
        description=(
            "Ask about registration steps, course rules or portal links, and so on... in English, Hausa, Yorùbá, or Igbo.\n\nPowered by Llama3.2 and N-ATLaS: "
            f"_{config.ATTRIBUTION}_"
        ),
        examples=["Where do I sign in?", "A ina zan shiga ciki?", "Nibo ni mo ti wọlé?", "Ebee ka m ga-abanye?",
                  ],
    )
    demo.launch(share=config.GRADIO_SHARE, theme=build_theme(gr), css=CSS, js=FORCE_LIGHT)