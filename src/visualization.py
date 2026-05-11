"""
Visualization module for text summarization project.

Generates professional charts and diagrams for:
- Model performance comparison
- ROUGE score analysis
- Compression ratio visualization
- Word frequency analysis
- Sentence scoring visualization
- Training curves (if available)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# Color palette
COLORS = {
    'extractive': '#3b82f6',
    'abstractive': '#f59e0b',
    'finetuned': '#10b981',
    'reference': '#6b7280',
    'bg': '#f8fafc',
    'grid': '#e5e7eb',
}


class SummarizationVisualizer:
    """Generates visualizations for summarization models."""
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = Path(__file__).parent / "visualizations"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_rouge_comparison(self, results: Dict[str, Dict[str, float]], 
                             save_path: Optional[str] = None) -> Path:
        """Plot ROUGE scores comparison bar chart.
        
        Args:
            results: Dict of model_name -> {'rouge1': float, 'rouge2': float, 'rougeL': float}
            save_path: Optional custom save path
        """
        models = list(results.keys())
        rouge1 = [results[m]['rouge1'] for m in models]
        rouge2 = [results[m]['rouge2'] for m in models]
        rougeL = [results[m]['rougeL'] for m in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width, rouge1, width, label='ROUGE-1', color=COLORS['extractive'], alpha=0.8)
        bars2 = ax.bar(x, rouge2, width, label='ROUGE-2', color=COLORS['abstractive'], alpha=0.8)
        bars3 = ax.bar(x + width, rougeL, width, label='ROUGE-L', color=COLORS['finetuned'], alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('ROUGE Score')
        ax.set_title('ROUGE Score Comparison by Model')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.legend()
        ax.set_ylim(0, max(max(rouge1), max(rouge2), max(rougeL)) * 1.2)
        
        path = save_path or self.output_dir / 'rouge_comparison.png'
        plt.savefig(path)
        plt.close()
        return path
    
    def plot_compression_comparison(self, results: Dict[str, Dict[str, float]],
                                   save_path: Optional[str] = None) -> Path:
        """Plot compression ratio comparison."""
        models = list(results.keys())
        ratios = [results[m]['compression'] for m in models]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        colors = [COLORS['extractive'], COLORS['abstractive'], COLORS['finetuned']][:len(models)]
        bars = ax.barh(models, ratios, color=colors, alpha=0.8, height=0.5)
        
        for bar, ratio in zip(bars, ratios):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{ratio:.1%}', va='center', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Compression Ratio (Summary / Original)')
        ax.set_title('Compression Ratio by Model')
        ax.set_xlim(0, max(ratios) * 1.3)
        
        path = save_path or self.output_dir / 'compression_comparison.png'
        plt.savefig(path)
        plt.close()
        return path
    
    def plot_performance_radar(self, results: Dict[str, Dict[str, float]],
                              save_path: Optional[str] = None) -> Path:
        """Plot radar chart comparing models on multiple dimensions."""
        categories = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Speed', 'Compression']
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        for model, color in zip(results.keys(), [COLORS['extractive'], COLORS['abstractive'], COLORS['finetuned']]):
            r = results[model]
            # Normalize speed (inverse - faster is better)
            speed_norm = 1.0 / (r['time'] / 100 + 1)  # Normalize to 0-1
            # Normalize compression (lower is better for compression)
            comp_norm = 1.0 - r['compression']
            
            values = [r['rouge1'], r['rouge2'], r['rougeL'], speed_norm, comp_norm]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        path = save_path or self.output_dir / 'performance_radar.png'
        plt.savefig(path)
        plt.close()
        return path
    
    def plot_word_frequency(self, text: str, top_n: int = 20,
                           save_path: Optional[str] = None) -> Path:
        """Plot word frequency distribution."""
        from collections import Counter
        import re
        
        # Simple word frequency (lowercase, remove punctuation)
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out'}
        words = [w for w in words if w not in stop_words]
        
        word_counts = Counter(words).most_common(top_n)
        words_list = [w[0] for w in word_counts]
        counts = [w[1] for w in word_counts]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(words_list)))
        bars = ax.barh(range(len(words_list)), counts, color=colors[::-1])
        
        ax.set_yticks(range(len(words_list)))
        ax.set_yticklabels(words_list[::-1])
        ax.set_xlabel('Frequency')
        ax.set_title(f'Top {top_n} Most Frequent Words')
        ax.invert_yaxis()
        
        path = save_path or self.output_dir / 'word_frequency.png'
        plt.savefig(path)
        plt.close()
        return path
    
    def plot_sentence_scores(self, sentences: List[str], scores: np.ndarray,
                            save_path: Optional[str] = None) -> Path:
        """Plot sentence importance scores."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.RdYlGn(scores)
        bars = ax.bar(range(len(sentences)), scores, color=colors, alpha=0.8)
        
        # Highlight top sentences
        top_indices = np.argsort(scores)[-3:]
        for idx in top_indices:
            bars[idx].set_color('#3b82f6')
            bars[idx].set_alpha(1.0)
        
        ax.set_xlabel('Sentence Index')
        ax.set_ylabel('Importance Score')
        ax.set_title('Sentence Importance Scores')
        ax.set_xticks(range(len(sentences)))
        ax.set_xticklabels([f'{i+1}' for i in range(len(sentences))])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3b82f6', label='Top 3 Sentences'),
            Patch(facecolor=plt.cm.RdYlGn(0.5), label='Other Sentences')
        ]
        ax.legend(handles=legend_elements)
        
        path = save_path or self.output_dir / 'sentence_scores.png'
        plt.savefig(path)
        plt.close()
        return path
    
    def plot_training_curves(self, train_losses: List[float], val_losses: List[float] = None,
                            save_path: Optional[str] = None) -> Path:
        """Plot training loss curves."""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss', linewidth=2)
        if val_losses:
            ax.plot(range(1, len(val_losses) + 1), val_losses, 'r--', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        path = save_path or self.output_dir / 'training_curves.png'
        plt.savefig(path)
        plt.close()
        return path
    
    def generate_all_charts(self, results: Dict[str, Dict[str, float]] = None):
        """Generate all standard charts for the project."""
        if results is None:
            # Default results from our evaluation
            results = {
                'Extractive': {
                    'rouge1': 0.290, 'rouge2': 0.099, 'rougeL': 0.188,
                    'compression': 0.189, 'time': 25.2
                },
                'Abstractive (BART)': {
                    'rouge1': 0.391, 'rouge2': 0.169, 'rougeL': 0.284,
                    'compression': 0.086, 'time': 504.8
                },
            }
        
        print("Generating visualizations...")
        
        # Generate all charts
        self.plot_rouge_comparison(results)
        print("  [OK] ROUGE comparison")
        
        self.plot_compression_comparison(results)
        print("  [OK] Compression comparison")
        
        self.plot_performance_radar(results)
        print("  [OK] Performance radar")
        
        print(f"\nAll charts saved to: {self.output_dir}")
        return self.output_dir


def main():
    """Generate all visualizations."""
    viz = SummarizationVisualizer()
    
    # Example results from our evaluation
    results = {
        'Extractive': {
            'rouge1': 0.290, 'rouge2': 0.099, 'rougeL': 0.188,
            'compression': 0.189, 'time': 25.2
        },
        'Abstractive (BART)': {
            'rouge1': 0.391, 'rouge2': 0.169, 'rougeL': 0.284,
            'compression': 0.086, 'time': 504.8
        },
    }
    
    viz.generate_all_charts(results)
    
    # Example word frequency
    sample_text = """Artificial intelligence has rapidly evolved from a theoretical concept to a transformative technology impacting nearly every industry. Machine learning enables computers to learn patterns from data without explicit programming. Deep learning uses neural networks with many layers to achieve remarkable success in image recognition and natural language processing."""
    viz.plot_word_frequency(sample_text)
    print("  [OK] Word frequency")
    
    print("\nDone! Check the 'visualizations/' folder for all charts.")


if __name__ == "__main__":
    main()
