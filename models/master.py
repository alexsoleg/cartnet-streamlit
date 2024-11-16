import torch
import streamlit as st
from models.cartnet import CartNet

# We cache the loading function to make is very fast on reload.
@st.cache_resource
def create_model():
    model = CartNet(dim_in=256, dim_rbf=64, num_layers=4, radius=5.0, invariant=False, temperature=True, use_envelope=True, cholesky=True)
    ckpt_path = "cpkt/cartnet_adp.ckpt"
    load = torch.load(ckpt_path, map_location=torch.device('cpu'))["model_state"]
    
    model.load_state_dict(load)
    model.eval()
    return model