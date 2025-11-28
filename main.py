import logging
import os
import warnings
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import umap
from adjustText import adjust_text

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress INFO level logs from sentence_transformers and transformers libraries
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.", category=UserWarning, module='umap') # UMAP warning
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib') # Generic Matplotlib UserWarnings


class Config:
    """Holds configuration parameters for the application."""
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FILE_NAME = "001data"
    EXCEL_FILE_PATH: str = os.path.join(_BASE_DIR, FILE_NAME+".xlsx")
    CHROMA_DB_PATH: str = os.path.join(_BASE_DIR, "chroma_db")
    COLLECTION_NAME: str = "word_vectors"
    MODEL_NAME: str = 'all-MiniLM-L6-v2'

class SemanticComparer:
    """
    A class to handle semantic comparison of word sets, including
    data loading, vector database management, analysis, and visualization.
    """
    def __init__(self, config: Config):
        """
        Initializes the SemanticComparer with a configuration.
        """
        self.config = config
        self.model = SentenceTransformer(self.config.MODEL_NAME)
        self.client = chromadb.PersistentClient(path=self.config.CHROMA_DB_PATH)
        self.collection = self._get_or_create_collection()
        logging.info(f"ChromaDB persistent storage path: {os.path.abspath(self.config.CHROMA_DB_PATH)}")


    def _get_or_create_collection(self):
        """
        Clears any existing collection and creates a new one.
        """
        try:
            self.client.delete_collection(name=self.config.COLLECTION_NAME)
            logging.info(f"Collection '{self.config.COLLECTION_NAME}' cleared.")
        except Exception as e:
            logging.warning(f"Collection '{self.config.COLLECTION_NAME}' could not be cleared (it might not exist): {e}")
        
        logging.info(f"Creating collection: '{self.config.COLLECTION_NAME}'")
        return self.client.get_or_create_collection(name=self.config.COLLECTION_NAME)

    def add_word(self, word: str, metadata: Dict):
        """
        Encodes a word to a vector and adds it to the ChromaDB collection.
        """
        try:
            embedding = self.model.encode(word).tolist()
            # ID must be unique. Using core_word and context for uniqueness.
            clean_id = f"{metadata.get('core_word', '')}_{metadata.get('context', '')}_{word}".lower().replace(" ", "_").replace(",", "")
            self.collection.add(
                ids=[clean_id],
                embeddings=[embedding],
                metadatas=[{"word": word, **metadata}]
            )
        except Exception as e:
            logging.error(f"Failed to add word '{word}' to ChromaDB: {e}")

    def load_data_from_excel(self):
        """
        Reads word sets from the configured Excel file and populates ChromaDB.
        Expected columns: 'word', 'responses', 'familiarity'.
        """

        try:
            df = pd.read_excel(self.config.EXCEL_FILE_PATH)
        except FileNotFoundError:
            logging.error(f"Excel file not found at: {self.config.EXCEL_FILE_PATH}")
            return

        for _, row in df.iterrows():
            core_word = str(row['word']).strip().lower()
            responses = str(row['responses']).strip()
            
            raw_context_type = str(row['familiarity']).strip().lower()
            context_type = None
            if raw_context_type == 'f':
                context_type = 'familiar'
            elif raw_context_type == 'u':
                context_type = 'unfamiliar'
            else:
                logging.warning(f"Skipping row due to unexpected familiarity value: '{raw_context_type}'. Expected 'f' or 'u'.")
                continue # Skip this row if context is not understood

            # Add the response words as related words, skipping the first 3
            response_words = [word.strip() for word in responses.split(',')][3:]
            for ans_word in response_words:
                if ans_word:
                    self.add_word(ans_word, {"type": "related_word", "core_word": core_word, "context": context_type})
        
        logging.info(f"Successfully loaded data from '{self.config.EXCEL_FILE_PATH}'.")
        logging.info(f"Total items in collection: {self.collection.count()}")

    def get_mean_vector(self, core_word: str, context_type: str) -> np.ndarray:
        """
        Calculates the mean vector for a set of related words based on context.
        """
        results = self.collection.get(
            where={"$and": [
                {"type": "related_word"},
                {"core_word": core_word},
                {"context": context_type}
            ]},
            include=['embeddings']
        )
        
        if results['embeddings'] is not None and len(results['embeddings']) > 0:
            return np.mean(np.array(results['embeddings']), axis=0)
        return np.array([])

    @staticmethod
    def calculate_euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculates the Euclidean distance between two vectors.
        """
        if vec1.size == 0 or vec2.size == 0:
            return float('inf')
        return float(np.linalg.norm(vec1 - vec2))

    def analyze_and_compare(self, core_word: str):
        """
        Performs the core analysis: calculates mean vectors for 'familiar' and 'unfamiliar' sets
        and computes the distance between them.
        """
        logging.info(f"--- Analyzing and comparing sets for core word: '{core_word}' ---")
        
        mean_vec_familiar = self.get_mean_vector(core_word, "familiar")
        mean_vec_unfamiliar = self.get_mean_vector(core_word, "unfamiliar")

        if mean_vec_familiar.size > 0 and mean_vec_unfamiliar.size > 0:
            distance = self.calculate_euclidean_distance(mean_vec_familiar, mean_vec_unfamiliar)
            logging.info(f"Euclidean distance between mean vectors of 'familiar' and 'unfamiliar' for '{core_word}': {distance:.4f}")
            return distance
        else:
            logging.warning(f"Could not calculate distance for '{core_word}'; one or both context sets are empty.")
            return None

    def visualize_multiple(self, core_words: List[str], output_filename: str):
        """
        Generates a combined UMAP visualization for multiple core words and their contexts.
        """
        logging.info(f"--- Generating combined visualization for core words: {core_words} ---")

        plot_embeddings = []
        plot_metadatas = []

        # Color palette for core words (main colors)
        core_word_base_colors = plt.get_cmap('tab10')
        if len(core_words) > 10:
            logging.warning("More than 10 core words; colors will start to repeat.")
            core_word_base_colors = plt.get_cmap('tab20')
        
        # Dictionary to map core_word -> colors
        core_word_color_map = {}
        for i, word in enumerate(core_words):
            base_color = np.array(core_word_base_colors(i % core_word_base_colors.N))
            pale_color = base_color.copy()
            pale_color[:3] = pale_color[:3] + (1 - pale_color[:3]) * 0.6  # Mix with white
            
            core_word_color_map[word] = {'familiar': base_color, 'unfamiliar': pale_color}

        # 1. Gather all data points
        for core_word in core_words:
            core_embedding = self.model.encode(core_word).tolist()
            plot_embeddings.append(core_embedding)
            plot_metadatas.append({'word': core_word, 'type': 'core_word', 'core_word': core_word})

            related_results = self.collection.get(where={"core_word": core_word}, include=['embeddings', 'metadatas'])
            if related_results['ids']:
                plot_embeddings.extend(related_results['embeddings'])
                plot_metadatas.extend(related_results['metadatas'])
            
        if len(plot_embeddings) < 2:
            logging.warning("Not enough data points to create a visualization.")
            return

        embeddings_np = np.array(plot_embeddings)
        
        # 2. UMAP reduction
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings_np)-1))
        reduced_embeddings = reducer.fit_transform(embeddings_np)
        
        # 3. Plot the data
        fig, ax = plt.subplots(figsize=(16, 12))
        
        texts = []
        for i, (x, y) in enumerate(reduced_embeddings):
            meta = plot_metadatas[i]
            word_type = meta.get('type')
            context_type = meta.get('context')
            current_core_word = meta.get('core_word')
            word_text = meta.get('word')

            color = 'gray'
            edge_color = 'none' # No border by default
            linewidth = 0 # No linewidth by default
            
            if current_core_word:
                if word_type == 'core_word':
                    color = core_word_color_map[current_core_word]['familiar']
                    edge_color = 'black'
                    linewidth = 2.5 # Thicker border for core words
                elif context_type == 'familiar':
                    color = core_word_color_map[current_core_word]['familiar']
                    edge_color = 'black'
                    linewidth = 1.5 # Thicker border for familiar related words
                elif context_type == 'unfamiliar':
                    color = core_word_color_map[current_core_word]['unfamiliar']


            if word_type == 'core_word':
                ax.scatter(x, y, color=color, marker='*', s=500, alpha=1.0, 
                           edgecolor=edge_color, zorder=10, linewidths=linewidth)
                texts.append(ax.text(x, y, word_text, fontsize=10, weight='bold'))
            elif word_type == 'related_word':
                marker = 'o'
                ax.scatter(x, y, color=color, marker=marker, s=100, alpha=0.8,
                           edgecolor=edge_color, linewidths=linewidth)
                texts.append(ax.text(x, y, word_text, fontsize=9))

        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

        ax.set_title(f"Semantic Clusters for: {', '.join([w.title() for w in core_words])}", fontsize=18)
        ax.set_xlabel("UMAP Dimension 1", fontsize=12)
        ax.set_ylabel("UMAP Dimension 2", fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='*', color='gray', label='Core Word', markersize=15, linestyle='None', markeredgecolor='black', markeredgewidth=2.5),
            Line2D([0], [0], marker='o', color='gray', label='Related (Familiar)', markersize=10, linestyle='None', markeredgecolor='black', markeredgewidth=1.5),
            Line2D([0], [0], marker='o', color='lightgray', label='Related (Unfamiliar)', markersize=10, linestyle='None', markeredgecolor='none')
        ]

        # Add color entries to legend
        for word, colors in core_word_color_map.items():
             legend_elements.append(Line2D([0], [0], marker='s', color=colors['familiar'], 
                                          label=f'{word.title()}', markersize=10, linestyle='None'))

        ax.legend(handles=legend_elements, title="Point Types and Core Words", loc='upper right', fontsize=10, title_fontsize='12')
        
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()
        logging.info(f"Combined visualization saved to '{output_filename}'")


def main():
    """
    Main function to run the semantic comparison and visualization.
    """
    config = Config()
    comparer = SemanticComparer(config)
    
    # Load data from Excel
    comparer.load_data_from_excel()
    
    core_words_to_analyze = ["pencil", "violin", "ladder", "napkin", "butter", "helmet", "zipper", "toilet", "candle", "hanger"]
    
    # Analyze each core word
    for word in core_words_to_analyze:
        comparer.analyze_and_compare(word)
    
    # Generate a combined visualization for all specified core words
    comparer.visualize_multiple(core_words_to_analyze, config.FILE_NAME+"_semantic_clusters.png")

if __name__ == "__main__":
    main()