import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="Chatbot IA", page_icon=":)")

st.title(" Chatbot IA con Python y Streamlit")
st.write("Este chatbot está implementado con un modelo de lenguaje open-source.")

# cagar modelo
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return tokenizer, model
    
tokenizer, model = load_model()

#Historial de conversacion
if "history" not in st.session_state:
    st.session_state.history = []

def responder(mensaje):
    inputs = tokenizer.encode(mensaje + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return respuesta

# Caja de texto
user_input = st.text_input("Escribe tu mensaje:")

if st.button("Enviar"):
    if user_input:
        respuesta = responder(user_input)
        st.session_state.history.append(("Tú", user_input))
        st.session_state.history.append(("IA", respuesta))

# Mostrar historial
for sender, msg in st.session_state.history:
    if sender == "Tú":
        st.markdown(f"** Tú:** {msg}")
    else:
        st.markdown(f"** IA:** {msg}")
