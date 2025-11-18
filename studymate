# =========================================================
# 📘 StudyMate AI — SINGLE CELL WORKING VERSION
# Chatbot + Translation + Image Analysis
# IBM Granite 3.3 2B + Granite Vision 3.2 2B + Gradio
# =========================================================

!pip install transformers accelerate gradio pillow sentencepiece -q

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gradio as gr


# =========================================================
# 1. CHATBOT — IBM Granite 3.3 2B Instruct
# KEY FIX --> use_fast=False
# =========================================================

chat_model_name = "ibm-granite/granite-3.3-2b-instruct"

chat_tokenizer = AutoTokenizer.from_pretrained(
    chat_model_name,
    use_fast=False  # <<< FIXES YOUR TOKENIZER ERROR
)

chat_model = AutoModelForCausalLM.from_pretrained(
    chat_model_name,
    device_map="auto",
    torch_dtype="auto"
)

chatbot = pipeline(
    "text-generation",
    model=chat_model,
    tokenizer=chat_tokenizer,
    temperature=0.3,
    max_new_tokens=300
)

def chatbot_response(user_input):
    prompt = f"User: {user_input}\nAssistant:"
    result = chatbot(prompt)[0]["generated_text"]
    return result[len(prompt):].strip()


# =========================================================
# 2. IMAGE ANALYSIS — IBM Granite Vision 3.2 2B
# =========================================================

vision_pipeline = pipeline(
    "image-to-text",
    model="ibm-granite/granite-vision-3.2-2b"
)

def analyze_image(img):
    return vision_pipeline(img)[0]["generated_text"]


# =========================================================
# 3. TRANSLATION — NLLB 200 (Indian Languages)
# =========================================================

translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    device_map="auto"
)

language_codes = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Malayalam": "mal_Mlym",
    "Kannada": "kan_Knda",
    "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr",
    "Punjabi": "pan_Guru",
    "Bengali": "ben_Beng"
}

def translate_text(text, target_lang):
    tgt = language_codes[target_lang]
    out = translator(text, tgt_lang=tgt)
    return out[0]["translation_text"]


# =========================================================
# 4. GRADIO USER INTERFACE
# =========================================================

with gr.Blocks(title="StudyMate AI") as app:

    gr.Markdown("""
    # 📘 StudyMate AI  
    ### Chatbot • Translation • Image Analysis  
    """)

    # ---------------- CHATBOT TAB ----------------
    with gr.Tab("💬 Chatbot"):
        q = gr.Textbox(label="Ask StudyMate")
        a = gr.Textbox(label="Response")
        btn = gr.Button("Send")
        btn.click(fn=chatbot_response, inputs=q, outputs=a)

    # ---------------- TRANSLATION TAB ----------------
    with gr.Tab("🌐 Translation"):
        t_in = gr.Textbox(label="Enter text to translate")
        lang_select = gr.Dropdown(list(language_codes.keys()), label="Translate To")
        t_out = gr.Textbox(label="Translated Output")
        btn_t = gr.Button("Translate")
        btn_t.click(fn=translate_text, inputs=[t_in, lang_select], outputs=t_out)

    # ---------------- IMAGE ANALYSIS TAB ----------------
    with gr.Tab("🖼️ Image Analysis"):
        img = gr.Image(label="Upload Image")
        caption = gr.Textbox(label="Image Caption")
        img_btn = gr.Button("Analyze Image")
        img_btn.click(fn=analyze_image, inputs=img, outputs=caption)

app.launch()
