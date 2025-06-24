# MolSearch

A comprehensive pipeline for molecular similarity search using RDKit and Streamlit for web interface.

## Features

- **Molecular Similarity Search**: Find similar molecules using Tanimoto coefficients
- **Molecular Featurization**: Convert molecules to feature vectors using RDKit descriptors
- **SQLite Database Storage**: Local database for molecule storage and retrieval
- **Streamlit Web Interface**: Interactive web application for molecular search
- **Performance Analysis**: Comprehensive evaluation using similarity metrics
- **Modular Design**: Clean, well-documented, and easily extensible codebase

## Quick Start

### Live Demo
Try the live application: [MolSearch on Streamlit Cloud](https://molsearch.streamlit.app)

### Local Development
```bash
git clone https://github.com/rehanbchinoy/MolSearch.git
cd MolSearch
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Query         │    │   Molecular     │    │   Similarity    │
│   Molecule      │    │   Featurizer    │    │   Search        │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │   SQLite Database         │
                    │   (Local Storage)         │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │   Streamlit Interface     │
                    │   (Web UI)                │
                    └───────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+ (3.11 recommended for best compatibility)
- RDKit (for molecular operations)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rehanbchinoy/MolSearch.git
   cd MolSearch
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

## Deployment

### Streamlit Cloud (Live Demo)

The application is currently deployed on Streamlit Cloud and available at:
**https://molsearch.streamlit.app**

### Local Deployment

To run the Streamlit web interface locally:

```bash
streamlit run app.py
```

The web interface will be available at `http://localhost:8501`

### Other Deployment Options

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions on:
- Heroku deployment
- Docker deployment
- Railway deployment
- Vercel deployment

## Configuration

The pipeline is configured via `config.yaml`. Key configuration options:

```yaml
# Model configuration
model:
  featurizer_type: "rdkit_2d"  # Options: "rdkit_2d"

# Database configuration
database:
  db_path: "molsearch.db"

# Processing parameters
processing:
  batch_size: 10
  max_molecules: 1000
```

## Usage

### Web Interface

1. **Visit the live demo**: [https://molsearch.streamlit.app](https://molsearch.streamlit.app)
2. **Choose input method**: Single SMILES, CSV upload, or example molecules
3. **Enter a molecule**: Use SMILES notation (e.g., "CCO" for ethanol)
4. **Configure search**: Set number of results to return
5. **Search**: Click "Search Similar Molecules" to find similar compounds
6. **View results**: See molecular structures, properties, and similarity scores
7. **Download**: Export results as CSV

### Programmatic Usage

```python
from molsearch_pipeline import MolecularSearchPipeline, Config

# Load configuration
config = Config()

# Initialize pipeline
pipeline = MolecularSearchPipeline(config)

# Process molecules
smiles_list = ["CCO", "CCCO", "c1ccccc1"]
df = pipeline.process_molecules(smiles_list)

# Find similar molecules
results = pipeline.find_similar_molecules("CCO", df, top_k=5)
```

### Command Line Interface

```bash
# Run the pipeline
python molsearch_pipeline.py

# Run with custom configuration
python molsearch_pipeline.py --config custom_config.yaml
```

## Pipeline Components

### 1. Similarity Calculator (`SimilarityCalculator`)

Calculates molecular similarities using Tanimoto coefficients.

```python
calculator = SimilarityCalculator()
similarity = calculator.compute_tanimoto_similarity(mol1, mol2)
```

### 2. Molecular Featurizer (`MolecularFeaturizer`)

Converts molecules to feature vectors using RDKit descriptors.

```python
featurizer = MolecularFeaturizer(config)
features = featurizer.featurize_molecules(molecules)
```

**Supported descriptors**:
- Molecular Weight
- LogP
- Number of H-donors/acceptors
- TPSA
- Number of rotatable bonds
- Number of aromatic rings
- Heavy atom count
- Ring count

### 3. Database Manager (`DatabaseManager`)

Handles SQLite database operations for molecule storage.

```python
with DatabaseManager(config) as db:
    db.create_tables()
    db.insert_molecules(df)
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

## Error Handling

The pipeline includes comprehensive error handling:

- **Invalid molecules**: Automatic filtering of invalid SMILES
- **Database errors**: Graceful handling of database operations
- **Memory errors**: Batch processing to handle large datasets

## Testing

Run the test suite:

```bash
pytest tests/
```

## Development

### Local Development

1. **Set up development environment**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Run tests**:
   ```bash
   pytest tests/
   ```

3. **Run Streamlit app locally**:
   ```bash
   streamlit run app.py
   ```

### File Structure

```
MolSearch/
├── molsearch_pipeline.py    # Main pipeline implementation
├── app.py                   # Streamlit web interface
├── cli.py                   # Command line interface
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── runtime.txt              # Python version specification
├── packages.txt             # System dependencies
├── style.css                # Custom CSS styling
├── tests/                   # Test suite
├── data/                    # Data directory
├── models/                  # Model directory
└── output/                  # Output directory
```



## Roadmap

- [x] Basic molecular similarity search
- [x] Streamlit web interface
- [x] RDKit integration
- [x] Deployment to Streamlit Cloud
- [ ] Support for more molecular descriptors
- [ ] Integration with additional databases

