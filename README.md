# Molecular Search Pipeline

A comprehensive pipeline for molecular similarity search using state-of-the-art molecular generation, featurization, and vector similarity search techniques.

## Features

- **Molecular Generation**: Generate novel molecules similar to reference compounds using COATI models
- **Molecular Featurization**: Convert molecules to high-dimensional vectors using RDKit descriptors or pretrained models
- **Vector Similarity Search**: Fast similarity search using Pinecone vector database
- **Performance Analysis**: Comprehensive evaluation using Tanimoto similarity metrics
- **Modular Design**: Clean, well-documented, and easily extensible codebase

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Reference     │    │   Generated     │    │   ZINC Dataset  │
│   Molecule      │    │   Molecules     │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │   Molecular Featurizer    │
                    │  (RDKit/Graphormer/ChemGPT)│
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │   Vector Database         │
                    │   (Pinecone)              │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │   Similarity Search       │
                    │   & Analysis              │
                    └───────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- RDKit (for molecular operations)
- CUDA-compatible GPU (optional, for faster processing)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd MolSearch-GPT
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   # Create .env file
   echo "PINECONE_API_KEY=your_pinecone_api_key_here" > .env
   ```

5. **Download required models**:
   ```bash
   # The COATI model will be downloaded automatically on first use
   # or you can manually download it to the models/ directory
   ```

## Configuration

The pipeline is configured via `config.yaml`. Key configuration options:

```yaml
# Model configuration
model:
  featurizer_type: "rdkit_2d"  # Options: "rdkit_2d", "chemgpt", "graphormer"

# Generation parameters
generation:
  num_variations: 100
  noise_scale: 0.35

# Pinecone configuration
pinecone:
  index_name: "molsearch"
```

## Usage

### Basic Usage

```python
from molsearch_pipeline import MolecularSearchPipeline, Config

# Load configuration
config = Config()

# Initialize pipeline
pipeline = MolecularSearchPipeline(config)

# Run pipeline with fentanyl as reference
reference_smiles = "CCC(=O)N(C1CCN(CC1)CCC2=CC=CC=C2)C3=CC=CC=C3"
analysis, results = pipeline.run_full_pipeline(reference_smiles)
```

### Command Line Interface

```bash
# Run the pipeline
python molsearch_pipeline.py

# Run with custom configuration
python molsearch_pipeline.py --config custom_config.yaml
```

### Jupyter Notebook

For interactive exploration, use the provided Jupyter notebook:

```bash
jupyter notebook main.ipynb
```

## Pipeline Components

### 1. Molecular Generator (`MolecularGenerator`)

Generates novel molecules similar to a reference compound using COATI models.

```python
generator = MolecularGenerator(config)
generated_smiles = generator.generate_molecules(
    reference_smiles="CCC(=O)N(C1CCN(CC1)CCC2=CC=CC=C2)C3=CC=CC=C3",
    num_variations=100,
    noise_scale=0.35
)
```

### 2. Molecular Featurizer (`MolecularFeaturizer`)

Converts molecules to high-dimensional feature vectors.

```python
featurizer = MolecularFeaturizer(config)
features = featurizer.featurize_molecules(molecules)
```

**Supported featurization methods**:
- RDKit 2D descriptors
- ChemGPT embeddings
- Graphormer embeddings

### 3. Vector Search Manager (`VectorSearchManager`)

Handles vector similarity search using Pinecone.

```python
vector_manager = VectorSearchManager(config)
vector_manager.upsert_vectors(ids, vectors)
search_results = vector_manager.search_similar(query_vector, top_k=100)
```

### 4. Similarity Analyzer (`SimilarityAnalyzer`)

Analyzes search results and computes performance metrics.

```python
analyzer = SimilarityAnalyzer()
analysis, sorted_results = analyzer.analyze_search_results(results_df, query_mol)
```

## Output

The pipeline generates several output files:

```
output/
├── analysis_<reference_smiles>.json    # Analysis metrics
├── results_<reference_smiles>.csv      # Search results
├── visualization_<reference_smiles>.png # Similarity plots
└── pipeline.log                        # Execution log
```

### Analysis Metrics

- **Number of results**: Total molecules returned
- **Mean similarity**: Average Tanimoto similarity
- **Max/Min similarity**: Best and worst similarities
- **Similarity distribution**: Count of high/medium/low similarity molecules

## Performance Optimization

### Parallel Processing

The pipeline supports parallel processing for:
- Molecular featurization
- Descriptor computation
- Database operations

### Memory Management

- Batch processing for large datasets
- Efficient vector storage
- Automatic garbage collection

### GPU Acceleration

- TensorFlow/GPU support for model inference
- CUDA-optimized molecular operations

## Error Handling

The pipeline includes comprehensive error handling:

- **Model loading errors**: Graceful fallback to alternative models
- **Network errors**: Retry mechanisms for API calls
- **Invalid molecules**: Automatic filtering of invalid SMILES
- **Memory errors**: Batch processing to handle large datasets

## Testing

Run the test suite:

```bash
pytest tests/
```

## Roadmap

- [ ] Support for more molecular generation models
- [ ] Integration with additional vector databases
- [ ] Web interface for interactive search
- [ ] Real-time molecular property prediction
- [ ] Multi-objective optimization
