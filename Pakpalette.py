import streamlit as st
from collections import Counter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import ImageColor
import io

def load_image_cv(uploaded_file):
    # Read the uploaded file with OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert the image to a different color space
    converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return converted_image
def rgb_to_hex(rgb_color):
    hex_color = '#'
    for i in rgb_color:
        i = int(i)
        hex_color += ("{:02x}".format(i))
    return hex_color
def prep_image(converted_image):
    modified_img = cv2.resize(converted_image, (900, 600), interpolation=cv2.INTER_AREA)
    modified_img = modified_img.reshape(modified_img.shape[0] * modified_img.shape[1], 3)
    return modified_img
#สร้างmodel
def save_model(img):#color_analysis
    clf = KMeans(n_clusters=5)
    color_labels = clf.fit_predict(img)
    center_colors = clf.cluster_centers_
    counts = Counter(color_labels)
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
    return hex_colors
#สร้าง palette จาก hexcolor
def load_model(hex_colors): #show_palette
    palette = np.array([ImageColor.getcolor(color, "RGB") for color in hex_colors])
    fig, ax = plt.subplots(dpi=100)
    ax.imshow(palette[np.newaxis, :, :])
    ax.axis('off')
    return fig

def main():
    st.image('logo.png')
    st.write('Upload your image and generate your color palette.This color palette help your artwork or graphic design more beautiful and cool.Let try!!!!!!!')
    #อัปโหลดรูปและแสดงผล
    uploaded_file = st.file_uploader("Upload an image...:camera:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with st.expander("**_IMAGE_** :camera_with_flash:", expanded=True):
            st.image(uploaded_file, use_column_width=True)
        #เรียกใช้def()
        #เตรียมรูปจากการอัปโหลด
        myimage = load_image_cv(uploaded_file)
        modified_image = prep_image(myimage)
        #นำรูปไป cluters แล้วแสดงเป็น hexcolor
        codecolor = save_model(modified_image)
        #นำ hexcolor ไปสร้าง palette
        palette = load_model(codecolor)

        with st.expander("**_COLOR_ PALETTE** :art: ", expanded=True):
            st.write(palette)
            st.markdown("**_COLOR_ CODE** :")
            st.markdown(codecolor)

            #dowload_palette as png
            file_obj = io.BytesIO()
            palette.savefig(file_obj, format='png')
            file_obj.seek(0)
            st.download_button(
                label="Download palette",
                data= file_obj,
                file_name="palette.png",
                mime="image/png",
             )

    else:
        uploaded_file = 'flower.jpeg'
        with st.expander("**_IMAGE_** :camera_with_flash:", expanded=True):
            st.image(uploaded_file, use_column_width=True)

        raw_img = cv2.imread(uploaded_file)
        myimage = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        modified_image = prep_image(myimage)
        codecolor = save_model(modified_image)
        palette = load_model(codecolor)
        with st.expander("**_COLOR_ PALETTE** :art: ", expanded=True):
            st.write(palette)
            st.markdown("**_COLOR_ CODE** :")
            st.markdown(codecolor)
            # dowload_palette as png
            file_obj = io.BytesIO()
            palette.savefig(file_obj, format='png')
            file_obj.seek(0)
            st.download_button(
                label="Download palette",
                data=file_obj,
                file_name="palette.png",
                mime="image/png",
            )
if __name__ == '__main__':
    main()
