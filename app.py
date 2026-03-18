import gradio as gr
from pipeline import MerzifyPipeline
from PIL import Image
import torch
import os

# Pipeline initialisieren
merzify = MerzifyPipeline()
MERZ_REF = "static/merz_reference.jpg"

def load_pipeline():
    if not os.path.exists(MERZ_REF):
        return "❌ Bitte merz_reference.jpg in static/ ablegen!"
    merzify.load_models(MERZ_REF)
    return "✅ KI geladen und bereit!"

def generate(prompt, negative_prompt, steps, guidance, ip_scale, cn_scale, seed):
    try:
        if merzify.pipe is None:
            return None, "❌ Erst auf 'KI Laden' klicken!"
        
        full_prompt = f"professional photo, {prompt}, highly detailed face, sharp"
        img = merzify.merzify(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            num_steps=int(steps),
            guidance_scale=guidance,
            ip_adapter_scale=ip_scale,
            controlnet_scale=cn_scale,
            seed=int(seed),
        )
        return img, "✅ Erfolgreich merzifiziert!"
    except Exception as e:
        return None, f"❌ Fehler: {str(e)}"

# Gradio UI
with gr.Blocks(title="Merzify AI 🇩🇪", theme=gr.themes.Dark()) as demo:
    gr.Markdown("# 🇩🇪 Merzify AI\n### Powered by InstantID + Stable Diffusion XL")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Einstellungen")
            load_btn = gr.Button("🚀 KI Laden", variant="primary", size="lg")
            load_status = gr.Textbox(label="Status", value="Noch nicht geladen...")
            
            prompt = gr.Textbox(
                label="Beschreibung",
                value="Friedrich Merz as an astronaut in space",
                lines=2
            )
            neg_prompt = gr.Textbox(
                label="Negativ-Prompt",
                value="blurry, low quality, distorted face, ugly, bad anatomy",
                lines=2
            )
            
            with gr.Accordion("🔧 Erweiterte Einstellungen", open=False):
                steps = gr.Slider(10, 50, value=30, label="Schritte")
                guidance = gr.Slider(1, 15, value=5.0, step=0.5, label="Guidance Scale")
                ip_scale = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Gesichts-Stärke")
                cn_scale = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="ControlNet Stärke")
                seed = gr.Number(value=42, label="Seed (für Reproduzierbarkeit)")
            
            gen_btn = gr.Button("✨ MERZIFY!", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("### 🖼️ Ergebnis")
            output_img = gr.Image(label="Merzifiziertes Bild", height=600)
            status = gr.Textbox(label="Status")
    
    # Beispiel-Prompts
    gr.Markdown("### 💡 Beispiel-Prompts")
    gr.Examples(
        examples=[
            ["Friedrich Merz as a superhero", "blurry, low quality"],
            ["Friedrich Merz on the beach in Hawaii", "ugly, distorted"],
            ["Friedrich Merz as a medieval knight", "bad anatomy, blurry"],
            ["Friedrich Merz cooking in a kitchen", "low quality, ugly"],
        ],
        inputs=[prompt, neg_prompt]
    )
    
    load_btn.click(fn=load_pipeline, outputs=load_status)
    gen_btn.click(
        fn=generate,
        inputs=[prompt, neg_prompt, steps, guidance, ip_scale, cn_scale, seed],
        outputs=[output_img, status]
    )

if __name__ == "__main__":
    demo.launch(share=True)  # share=True = öffentliche URL!
