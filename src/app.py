import streamlit as st
from pathlib import Path
import os
import tempfile
from PIL import Image
import pandas as pd 
import numpy as np 
import torch 
import requests
import faiss 
import warnings 
import logging
from typing import Optional, List, Tuple, Union 
from transformers import CLIPProcessor, CLIPModel 
from sentence_transformers import SentenceTransformer 


warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ImageMusicRecommender:
    """Optimized image-to-music recommendation system using CLIP embeddings and FAISS."""

    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize the recommender with CLIP model, dataset, and FAISS index."""
        self.clip_model_name = clip_model_name
        self.clip_model = None
        self.processor = None
        self.emid_df = None
        self.song_description_embeddings_clip = None # CLIP text embeddings of song descriptions
        self.index = None  # FAISS index attribute
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self._load_models()
        self._load_dataset()
        self._prepare_song_description_embeddings() # CLIP text embeddings for song descriptions
        self._build_faiss_index() #  FAISS index on song description embeddings


    def _load_models(self) -> None:
        """Load CLIP model and processor."""
        try:
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self.clip_model.eval()  # Set CLIP to evaluation mode
            logging.info(f"✓ CLIP model loaded successfully on {self.device}")
        except Exception as e:
            logging.error(f"✗ Error loading CLIP model: {e}")

    def _load_dataset(self, url: str = "https://raw.githubusercontent.com/ecnu-aigc/EMID/main/EMID_data.csv") -> None:
        """Load the EMID dataset."""
        try:
            self.emid_df = pd.read_csv(url)
            logging.info(f"✓ Dataset loaded: {len(self.emid_df)} songs")
        except Exception as e:
            logging.error(f"✗ Error loading dataset: {e}")

    @torch.no_grad()
    def _get_image_embedding(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Extract normalized image embeddings using CLIP."""
        if self.clip_model is None or self.processor is None:
            logging.error("✗ CLIP Model or processor not loaded. Cannot extract features.")
            return None

        try:
            image = None
            if str(image_path).startswith('http'):
                try:
                    # url se leke yeh analyse karega pehle download karega image then process karega and open  
                    # karke image object banayega
                    # then uske baad feature extract karega and return karega uske output
                    response = requests.get(image_path, stream=True, timeout=10, verify=False) # Added verify=False
                    response.raise_for_status()
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        for chunk in response.iter_content(chunk_size=8192):
                            tmp.write(chunk)
                        tmp_path = tmp.name
                    image = Image.open(tmp_path).convert("RGB")
                    os.unlink(tmp_path)
                    logging.info("Successfully downloaded and opened image from URL.")
                except requests.exceptions.RequestException as e:
                    logging.error(f"✗ Error downloading image from URL {image_path}: {e}")
                    return None
                except Exception as e:
                    logging.error(f"✗ Error processing downloaded image from URL {image_path}: {e}")
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    return None
            else:
                try:
                    image = Image.open(image_path).convert("RGB")
                    logging.info("Successfully opened image from local path.")
                except FileNotFoundError:
                    logging.error(f"✗ Error: Image file not found at local path {image_path}")
                    return None
                except Exception as e:
                    logging.error(f"✗ Error processing image from local path {image_path}: {e}")
                    return None

            if image is None:
                return None

            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()} 

            features = self.clip_model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            # features ko normalize kar raha hai taaki unit length pe aa jaye
            # isse similarity comparisons mein madad milegi 

            return features.cpu().numpy()

        except Exception as e:
            #  unexpected errors during feature extraction
            # mostly network issues or file corruption ke karan se hi hota hai 
            logging.error(f"✗ Unexpected error during image feature extraction for {image_path}: {e}")
            return None


    @torch.no_grad()
    #The torch.no_grad() context manager in PyTorch disables
    #  gradient calculation for any computations that occur within its scope.
    #  This has two key benefits: it saves memory and speeds up computations.
    def _get_text_embeddings(self, texts: List[str], batch_size: int = 32) -> Optional[np.ndarray]:
        """Extract normalized text embeddings in batches using CLIP text encoder."""
        if self.clip_model is None or self.processor is None:
            logging.error("✗ CLIP Model or processor not loaded. Cannot generate text embeddings.")
            return None

        all_embeddings = []

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                # basically yeh ensure karega ki agar koi null value ya non-string value aajaye to wo error na de
                batch = [str(t) if t is not None else '' for t in batch]
                inputs = self.processor(text=batch, return_tensors="pt",
                                       padding=True, truncation=True, max_length=77)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                embeddings = self.clip_model.get_text_features(**inputs)
                embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
                # Normalize the embeddings and usko unit length pe le aana
                all_embeddings.append(embeddings.cpu())

            return torch.cat(all_embeddings, dim=0).numpy()
        except Exception as e:
            logging.error(f"✗ Error generating CLIP text embeddings: {e}")
            return None


    def _prepare_song_description_embeddings(self):
        """Generate CLIP text embeddings for song descriptions."""
        if self.clip_model is None or self.processor is None or self.emid_df is None:
            logging.error("✗ CLIP model, processor, or dataset not loaded. Cannot prepare song description embeddings.")
            return

        logging.info("Generating CLIP text embeddings for song descriptions...")

        # Combine image texts to create a proxy for song text
        # Ensure non-string values are handled
        song_texts_proxy = (
            self.emid_df['Image1_text'].fillna('').astype(str) + ' ' +
            self.emid_df['Image2_text'].fillna('').astype(str) + ' ' +
            self.emid_df['Image3_text'].fillna('').astype(str)
        ).tolist()

        # Generate CLIP text embeddings
        try:
             self.song_description_embeddings_clip = self._get_text_embeddings(song_texts_proxy)
             if self.song_description_embeddings_clip is not None:
                logging.info("✓ CLIP text embeddings for song descriptions generated.")
                logging.info(f"Shape of CLIP text embeddings: {self.song_description_embeddings_clip.shape}")
             else:
                 logging.error("✗ Failed to generate CLIP text embeddings for song descriptions.")

        except Exception as e:
             logging.error(f"✗ Error generating CLIP text embeddings during preparation: {e}")
             self.song_description_embeddings_clip = None


    def _build_faiss_index(self):
        """Build a FAISS index for fast similarity search on song description embeddings."""
        if self.song_description_embeddings_clip is None:
            logging.error("✗ Song description embeddings not available. Cannot build FAISS index.")
            return

        logging.info("Building FAISS index on song description embeddings...")
        try:
            dimension = self.song_description_embeddings_clip.shape[1]
            # Use IndexFlatIP for cosine similarity search (embeddings are normalized)
            self.index = faiss.IndexFlatIP(dimension)

            # Add the embeddings to the index
            self.index.add(self.song_description_embeddings_clip.astype('float32'))

            logging.info(f"✓ FAISS index built with {self.index.ntotal} vectors.")
        except Exception as e:
            logging.error(f"✗ Error building FAISS index: {e}")
            self.index = None


    def recommend(self, image_path: Union[str, Path],
                 top_k: int = 10,
                 show_scores: bool = True) -> pd.DataFrame:
        """
        Generate music recommendations based on an input image using FAISS.

        Args:
            image_path: Path or URL to the input image
            top_k: Number of songs to recommend
            show_scores: Whether to include similarity scores

        Returns:
            DataFrame with recommended songs
        """
        logging.info(f"Processing image for recommendation: {image_path}")

        if self.emid_df is None or self.index is None:
             logging.error("✗ Dataset or FAISS index not available. Cannot recommend.")
             return pd.DataFrame()

        # Extract image embedding using CLIP
        img_emb = self._get_image_embedding(image_path)
        if img_emb is None:
            logging.error("✗ Failed to get image embedding. Cannot proceed with recommendation.")
            return pd.DataFrame()

        # --- Recommendation Logic using FAISS ---
        # Use the CLIP image embedding as the query vector for the FAISS index
        # The index contains CLIP text embeddings of song descriptions, so dimensions match (512).
        try:
            # Perform the search using FAISS
            # D is the distance matrix, I is the index matrix
            distances, indices = self.index.search(img_emb.astype('float32'), top_k)

            # FAISS IndexFlatIP returns inner products. For normalized vectors, inner product = cosine similarity.
            # Higher inner product means higher similarity.
            similarities = distances[0] #  similarities for the top_k results

            # Retrieve the corresponding songs from the full emid_df using the indices
            results = self.emid_df.iloc[indices[0]].copy()

            if show_scores:
                results['similarity_score'] = similarities

            logging.info(f"✓ Found {len(results)} recommendations.")

            return results

        except Exception as e:
            logging.error(f"✗ Error during FAISS search: {e}")
            return pd.DataFrame()


    def display_recommendations(self, recommendations: pd.DataFrame) -> None:
        """Display recommendations in a formatted way using Streamlit."""
        if recommendations.empty:
            st.write("No recommendations to display")
            return

        cols = ['Audio_Filename', 'genre', 'feeling'] # Simplified display
        if 'similarity_score' in recommendations.columns:
            cols.append('similarity_score')

        display_df = recommendations[cols].reset_index(drop=True)

        for idx, row in display_df.iterrows():
            st.write(f"**{idx + 1}. {row['Audio_Filename']}**")
            st.write(f"   Genre: {row['genre']}")
            st.write(f"   Feeling: {row['feeling']}")
            if 'similarity_score' in row:
                st.write(f"   Score: {row['similarity_score']:.4f}")
            st.write("---") # Separator


st.title("Image to Music Recommender")


# Initialize the recommender as a cached resource 
# helps in loading the model only once and reusing it across interactions
@st.cache_resource(show_spinner="Loading the music recommender...")
def get_recommender():
    return ImageMusicRecommender()

recommender = get_recommender()

# Add input methods (file upload and URL)
image_source = st.radio("Choose image source:", ("Upload Image", "Enter Image URL"))

image_path = None
if image_source == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location to get a path
        # Streamlit handles file uploads in memory, save to temp file for path
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            image_path = tmp_file.name
        # Only display the image if it was successfully uploaded
        st.image(uploaded_file, caption="Uploaded Image.", use_container_width=True)

elif image_source == "Enter Image URL":
    image_url = st.text_input("Enter Image URL:")
    if image_url:
        image_path = image_url
        try:
            st.image(image_url, caption="Image from URL.", use_container_width=True)
        except Exception as e:
            st.error(f"Could not display image from URL. Please check the URL. Error: {e}")
            image_path = None # Reset image_path if URL is invalid or causes error


# Get recommendations when a button is clicked or image is available
if image_path and st.button("Get Music Recommendations"):
    with st.spinner("Getting recommendations..."):
        recommendations = recommender.recommend(image_path, top_k=10)

    if not recommendations.empty:
        st.subheader("Recommended Songs:")
        recommender.display_recommendations(recommendations) # Call the updated display method
    else:
        st.info("No recommendations found.")

    # Clean up the temporary file if it was uploaded
    if image_source == "Upload Image" and image_path and os.path.exists(image_path):
         os.unlink(image_path)