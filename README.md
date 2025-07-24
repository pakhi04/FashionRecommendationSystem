# Fashion Recommender System

This project is an **image-based fashion recommendation system** built using deep learning and Streamlit. Users can upload an image of a clothing item, and the system will recommend visually similar items from a predefined dataset.

---

##  Features

- Upload an image of clothing and receive top 5–6 similar item recommendations
- Utilizes **ResNet50** (ImageNet pre-trained) to extract feature embeddings
- Embeddings are precomputed and stored for efficient similarity matching
- Easy-to-use web interface built with **Streamlit**

---

## Tech Stack

- **Python 3**
- **TensorFlow / Keras** (ResNet50 for feature extraction)
- **Scikit-learn** (NearestNeighbors for similarity search)
- **NumPy, PIL, Pickle**
- **Streamlit** (for the web interface)


