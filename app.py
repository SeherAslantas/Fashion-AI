import os
import streamlit as st
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.neighbors import NearestNeighbors

model = load_model("fashion_model.h5", compile=False)

features = np.array(pickle.load(open("Images_features.pkl", "rb")))
filenames = pickle.load(open("filenames.pkl", "rb"))
filenames=[os.path.join("images",os.path.basename(f.replace("\\","/"))) for f in filenames]
neighbors = NearestNeighbors(
    n_neighbors=6,
    algorithm="brute",
    metric="euclidean"
)

neighbors.fit(features)

st.title("Moda Yapay Zekası - Benzer Ürün Önerisi")

st.write("Benzer ürünleri bulmak için bir ürün seçin.")

product_index = st.selectbox(
    "Bir ürün seçin:",
    range(len(filenames)),
    format_func=lambda x: f"Ürün {x + 1}"
)

img = Image.open(filenames[product_index])

st.image(img, caption="Seçilen Ürün", width=250)

if st.button("Benzer Ürünleri Bul"):

    distances, indices = neighbors.kneighbors(
        features[product_index].reshape(1, -1)
    )

    st.subheader("Önerilen Benzer Ürünler")

    cols = st.columns(5)

    for i, index in enumerate(indices[0][1:6]):

        product_image = Image.open(filenames[index])

        with cols[i]:
            st.image(product_image, use_container_width=True)