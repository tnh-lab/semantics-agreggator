import logging
import os
import warnings
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import chromadb
import fasttext
import fasttext.util
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
        self.distance_logs = []
        
        # fastText model handling
        model_path = 'cc.en.300.bin'
        if not os.path.exists(model_path):
            logging.info(f"fastText model '{model_path}' not found. Downloading...")
            fasttext.util.download_model('en', if_exists='ignore')
            logging.info("Download complete.")
        
        logging.info("Loading fastText model...")
        self.model = fasttext.load_model(model_path)
        logging.info("fastText model loaded.")

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
            embedding = self.model.get_word_vector(word).tolist()
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
            
            self.distance_logs.append({'core_word': core_word, 'distance': distance})
            
            return distance
        else:
            logging.warning(f"Could not calculate distance for '{core_word}'; one or both context sets are empty.")
            return None

    def update_distance_log_with_new_run(self):
        log_file = 'distance_log.xlsx'
        
        if not self.distance_logs:
            logging.warning("No new distances to log for this run.")
            return

        current_run_df = pd.DataFrame(self.distance_logs)
        current_run_df = current_run_df.set_index('core_word')

        timestamp_col = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        current_run_df = current_run_df.rename(columns={'distance': timestamp_col})
        
        try:
            if os.path.exists(log_file):
                log_df = pd.read_excel(log_file, index_col=0)
                updated_df = log_df.merge(current_run_df, on='core_word', how='outer')
            else:
                updated_df = current_run_df

            updated_df.to_excel(log_file)
            logging.info(f"Updated '{log_file}' with new run data in column '{timestamp_col}'.")

        except Exception as e:
            logging.error(f"Failed to update distance log: {e}")

    def visualize_distance_scatterplot(self, output_filename: str):
        log_file = 'distance_log.xlsx'
        if not os.path.exists(log_file):
            logging.warning(f"Distance log file '{log_file}' not found. Cannot generate scatter plot.")
            return

        try:
            df = pd.read_excel(log_file, index_col=0)
            
            df['mean_distance'] = df.mean(axis=1)
            df = df.reset_index()

            fig, ax = plt.subplots(figsize=(16, 10))
            
            ax.scatter(df['core_word'], df['mean_distance'], color='purple', marker='x')
            
            ax.set_title("Mean of Euclidean Distances per Core Word (All Runs)")
            ax.set_xlabel("Core Word")
            ax.set_ylabel("Mean Euclidean Distance")
            plt.xticks(rotation=45)
            ax.grid(True)
            plt.tight_layout()

            plt.savefig(output_filename)
            plt.close()
            logging.info(f"Mean distance scatter plot saved to '{output_filename}'")
        except Exception as e:
            logging.error(f"Failed to generate distance scatter plot: {e}")

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
            core_word_color_map[word] = core_word_base_colors(i % core_word_base_colors.N)

        # 1. Gather all data points
        for core_word in core_words:
            core_embedding = self.model.get_word_vector(core_word).tolist()
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
            edge_color = 'black'
            linewidth = 1.5
            marker = 'o'

            if current_core_word:
                color = core_word_color_map.get(current_core_word, 'gray')
                if word_type == 'core_word':
                    marker = '*'
                    linewidth = 2.5
                elif context_type == 'unfamiliar':
                    marker = 's'
            
            size = 100
            if word_type == 'core_word':
                size = 500

            ax.scatter(x, y, color=color, marker=marker, s=size, alpha=0.8,
                       edgecolor=edge_color, linewidths=linewidth, zorder=10 if word_type == 'core_word' else 5)
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
            Line2D([0], [0], marker='s', color='gray', label='Related (Unfamiliar)', markersize=10, linestyle='None', markeredgecolor='black', markeredgewidth=1.5)
        ]

        # Add color entries to legend
        for word, color in core_word_color_map.items():
             legend_elements.append(Line2D([0], [0], marker='s', color=color, 
                                          label=f'{word.title()}', markersize=10, linestyle='None'))

        ax.legend(handles=legend_elements, title="Point Types and Core Words", loc='upper right', fontsize=10, title_fontsize='12')
        
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()
        logging.info(f"Combined visualization saved to '{output_filename}'")


    def visualize_graph(self, core_word: str, output_filename: str):
        """
        Generates a graph visualization for a single core word and its contexts.
        """
        logging.info(f"--- Generating graph visualization for core word: '{core_word}' ---")

        plot_embeddings = []
        plot_metadatas = []

        # 1. Gather data
        core_embedding = self.model.get_word_vector(core_word).tolist()
        plot_embeddings.append(core_embedding)
        plot_metadatas.append({'word': core_word, 'type': 'core_word', 'core_word': core_word})

        related_results = self.collection.get(where={"core_word": core_word}, include=['embeddings', 'metadatas'])
        if related_results and related_results['ids']:
            plot_embeddings.extend(related_results['embeddings'])
            plot_metadatas.extend(related_results['metadatas'])
        
        if len(plot_embeddings) < 2:
            logging.warning(f"Not enough data points for '{core_word}' to create a graph visualization.")
            return

        embeddings_np = np.array(plot_embeddings)
        
        # 2. UMAP reduction
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings_np)-1))
        reduced_embeddings = reducer.fit_transform(embeddings_np)
        
        # 3. Plot the data
        fig, ax = plt.subplots(figsize=(16, 12))
        
        core_word_coords = reduced_embeddings[0]
        
        color_map = {'familiar': 'blue', 'unfamiliar': 'green'}
        core_word_color = 'red'

        texts = []
        for i, (x, y) in enumerate(reduced_embeddings):
            meta = plot_metadatas[i]
            word_type = meta.get('type')
            context_type = meta.get('context')
            word_text = meta.get('word')

            color = 'gray'
            if word_type == 'core_word':
                color = core_word_color
                ax.scatter(x, y, color=color, marker='*', s=600, alpha=1.0, zorder=10, edgecolor='black', linewidths=2.5)
                texts.append(ax.text(x, y, word_text, fontsize=12, weight='bold'))
            else: # related_word
                color = color_map.get(context_type, 'gray')
                ax.scatter(x, y, color=color, marker='o', s=150, alpha=0.8, edgecolor='black', linewidths=1.0)
                texts.append(ax.text(x, y, word_text, fontsize=9))
                # Draw line to core word
                ax.plot([core_word_coords[0], x], [core_word_coords[1], y], color='gray', linestyle='--', linewidth=0.7, zorder=1)

    
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

        ax.set_title(f"Example of Semantic Representation of the Word {core_word.title()}", fontsize=18)
        ax.set_xlabel("UMAP Dimension 1", fontsize=12)
        ax.set_ylabel("UMAP Dimension 2", fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='*', color=core_word_color, label='Core Word', markersize=15, linestyle='None', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color=color_map['familiar'], label='Related (Familiar)', markersize=10, linestyle='None', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color=color_map['unfamiliar'], label='Related (Unfamiliar)', markersize=10, linestyle='None', markeredgecolor='black')
        ]
        ax.legend(handles=legend_elements, title="Node Types", loc='upper right')

        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()
        logging.info(f"Graph visualization for '{core_word}' saved to '{output_filename}'")


    def visualize_multiple_graph(self, core_words: List[str], output_filename: str):
        """
        Generates a combined UMAP graph visualization for multiple core words and their contexts.
        """
        logging.info(f"--- Generating combined graph visualization for core words: {core_words} ---")

        plot_embeddings = []
        plot_metadatas = []

        # Color palette for core words
        core_word_base_colors = plt.get_cmap('tab10')
        if len(core_words) > 10:
            logging.warning("More than 10 core words; colors will start to repeat.")
            core_word_base_colors = plt.get_cmap('tab20')
        
        core_word_color_map = {word: core_word_base_colors(i % core_word_base_colors.N) for i, word in enumerate(core_words)}

        # 1. Gather all data points
        for core_word in core_words:
            core_embedding = self.model.get_word_vector(core_word).tolist()
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

        # Store core word coordinates
        core_word_coords = {meta.get('word'): reduced_embeddings[i] for i, meta in enumerate(plot_metadatas) if meta.get('type') == 'core_word'}
        
        texts = []
        for i, (x, y) in enumerate(reduced_embeddings):
            meta = plot_metadatas[i]
            word_type = meta.get('type')
            context_type = meta.get('context')
            current_core_word = meta.get('core_word')
            word_text = meta.get('word')

            color = 'gray'
            edge_color = 'black'
            linewidth = 1.5
            marker = 'o'

            if current_core_word:
                color = core_word_color_map.get(current_core_word, 'gray')
                if word_type == 'core_word':
                    marker = '*'
                    linewidth = 2.5
                elif context_type == 'unfamiliar':
                    marker = 's'
            
            size = 100
            if word_type == 'core_word':
                size = 500

            ax.scatter(x, y, color=color, marker=marker, s=size, alpha=0.8,
                       edgecolor=edge_color, linewidths=linewidth, zorder=10 if word_type == 'core_word' else 5)
            texts.append(ax.text(x, y, word_text, fontsize=9))

            # Draw line to core word
            if word_type == 'related_word' and current_core_word in core_word_coords:
                core_coords = core_word_coords[current_core_word]
                ax.plot([core_coords[0], x], [core_coords[1], y], color=color, linestyle='--', linewidth=0.7, zorder=1)

        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

        ax.set_title(f"Semantic Network Representation of Displayed Words", fontsize=18)
        ax.set_xlabel("UMAP Dimension 1", fontsize=12)
        ax.set_ylabel("UMAP Dimension 2", fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='*', color='gray', label='Core Word', markersize=15, linestyle='None', markeredgecolor='black', markeredgewidth=2.5),
            Line2D([0], [0], marker='o', color='gray', label='Related (Familiar)', markersize=10, linestyle='None', markeredgecolor='black', markeredgewidth=1.5),
            Line2D([0], [0], marker='s', color='gray', label='Related (Unfamiliar)', markersize=10, linestyle='None', markeredgecolor='black', markeredgewidth=1.5)
        ]

        for word, color in core_word_color_map.items():
             legend_elements.append(Line2D([0], [0], marker='s', color=color, label=f'{word.title()}', markersize=10, linestyle='None'))

        ax.legend(handles=legend_elements, title="Point Types and Core Words", loc='upper right', fontsize=10, title_fontsize='12')
        
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()
        logging.info(f"Combined graph visualization saved to '{output_filename}'")


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
        
    # Update and visualize distances
    comparer.update_distance_log_with_new_run()
    comparer.visualize_distance_scatterplot(config.FILE_NAME+"_distance_scatterplot.png")
    
    # Other visualizations
    comparer.visualize_multiple(core_words_to_analyze, config.FILE_NAME+"_semantic_clusters.png")
    comparer.visualize_graph("hanger", "hanger_graph.png")
    comparer.visualize_multiple_graph(core_words_to_analyze, config.FILE_NAME+"_semantic_graph_clusters.png")


if __name__ == "__main__":
    main()