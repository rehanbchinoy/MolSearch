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
Try the live application: [MolSearch on Streamlit Cloud](https://your-app-name.streamlit.app)

### Local Development
```bash
git clone <repository-url>
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

- Python 3.8+
- RDKit (for molecular operations)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
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

### Streamlit Cloud (Recommended)

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Add MolSearch interface"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set main file to `app.py`
   - Click "Deploy"

3. **Get your app URL**: `https://your-app-name.streamlit.app`

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

### Basic Usage

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

### Streamlit Web Interface

To run the Streamlit web interface:

```bash
streamlit run app.py
```

The web interface will be available at `http://localhost:8501`

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
├── tests/                   # Test suite
├── data/                    # Data directory
├── models/                  # Model directory
└── output/                  # Output directory
```

## Blog Integration

To integrate MolSearch into your blog posts, use the HTML snippet from `blog-integration-example.html`. This provides:

- Professional call-to-action button
- App preview image
- Feature highlights
- Example usage instructions

## Roadmap

- [ ] Enhanced Streamlit interface with molecule visualization
- [ ] Support for more molecular descriptors
- [ ] Integration with additional databases
- [ ] Real-time molecular property prediction
- [ ] Multi-objective optimization
- [ ] Deployment to web hosting platform
