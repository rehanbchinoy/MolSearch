# MolSearch - Molecular Similarity Search Pipeline

A robust molecular similarity search pipeline using RDKit and Streamlit, designed for chemical informatics and drug discovery applications.

## Features

- **Molecular Similarity Search**: Find similar molecules using Tanimoto coefficients
- **RDKit Integration**: Full molecular processing and featurization
- **Interactive Web Interface**: Beautiful Streamlit app with molecular visualization
- **Database Support**: SQLite storage for molecular data
- **Robust Drawing**: Multiple fallback methods for molecular visualization
- **Export Capabilities**: Download results as CSV files

## Quick Start

### Prerequisites

- Python 3.10 (required for RDKit compatibility)
- pip or conda package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd MolSearch-GPT
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## Usage

### Web Interface

1. **Enter a SMILES string** or select from example molecules
2. **Configure search parameters** (number of results, similarity threshold)
3. **Click "Search Similar Molecules"** to find similar compounds
4. **View results** with molecular properties and similarity scores
5. **Download results** as CSV for further analysis

### Example SMILES

- **Ethanol**: `CCO`
- **Benzene**: `c1ccccc1`
- **Aspirin**: `CC(=O)OC1=CC=CC=C1C(=O)O`
- **Fentanyl**: `CCC(=O)N(C1CCN(CC1)CCC2=CC=CC=C2)C3=CC=CC=C3`

## Deployment

### Streamlit Cloud

1. **Push your code** to GitHub
2. **Connect your repository** to Streamlit Cloud
3. **Set Python version** to 3.10 in Streamlit Cloud settings
4. **Deploy** - the app will automatically install dependencies

### Configuration Files

- `requirements.txt`: Python dependencies with NumPy <2.0.0 for RDKit compatibility
- `runtime.txt`: Specifies Python 3.10
- `packages.txt`: System dependencies for molecular drawing

## Technical Details

### Dependencies

- **Core**: numpy<2.0.0, pandas, scipy
- **Chemistry**: rdkit-pypi==2023.3.1b1
- **Web**: streamlit>=1.28.0
- **Visualization**: matplotlib, seaborn, Pillow

### Architecture

- **Pipeline**: Modular design with separate components for similarity calculation, featurization, and analysis
- **Database**: SQLite for molecular storage and retrieval
- **Drawing**: Robust fallback system for molecular visualization
- **API**: Clean interfaces for integration with other tools

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- RDKit community for the excellent cheminformatics toolkit
- Streamlit team for the amazing web framework
- Open source contributors who made this possible

## Features

- **Molecular Similarity Search**: Find similar molecules using Tanimoto coefficients
- **Molecular Featurization**: Convert molecules to feature vectors using RDKit descriptors
- **SQLite Database Storage**: Local database for molecule storage and retrieval
- **Streamlit Web Interface**: Interactive web application for molecular search

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

- **Python 3.11** (required for RDKit compatibility)
  - RDKit does not support Python 3.12+ yet
  - Python 3.10 and below may have compatibility issues
- RDKit (for molecular operations)

### Setup

#### Option 1: Using pip (Recommended for local development)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rehanbchinoy/MolSearch.git
   cd MolSearch
   ```

2. **Create a virtual environment with Python 3.11**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

#### Option 2: Using conda (Recommended for deployment)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rehanbchinoy/MolSearch.git
   cd MolSearch
   ```

2. **Create conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate molsearch
   ```

3. **Verify installation**:
   ```bash
   python -c "import rdkit; print('RDKit version:', rdkit.__version__)"
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
├── requirements.txt         # Python dependencies (pip)
├── environment.yml          # Conda environment file
├── conda-requirements.txt   # Alternative conda requirements
├── runtime.txt              # Python version specification
├── packages.txt             # System dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
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

## Troubleshooting

### Deployment Issues

#### RDKit Installation Problems

If you encounter RDKit installation issues:

1. **Python Version**: Ensure you're using Python 3.11
   ```bash
   python --version  # Should show Python 3.11.x
   ```

2. **Streamlit Cloud**: The app is configured to use Python 3.11 via `runtime.txt`
   - If deployment fails, check that Streamlit Cloud is using Python 3.11
   - RDKit doesn't support Python 3.12+ yet

3. **Conda Installation (Recommended)**: Use conda for easier RDKit installation
   ```bash
   conda create -n molsearch python=3.11
   conda activate molsearch
   conda install -c conda-forge rdkit=2023.3.1
   pip install -r requirements.txt
   ```

4. **Alternative: Use environment.yml**:
   ```bash
   conda env create -f environment.yml
   conda activate molsearch
   ```

#### Common Errors

- **"No matching distribution found for rdkit-pypi"**: Use conda installation instead
- **"Unable to locate package #"**: This is a packages.txt parsing error - fixed in latest version
- **"Module not found"**: Ensure all dependencies are installed
- **"Port already in use"**: Change port with `streamlit run app.py --server.port 8502`

### Performance Issues

- **Slow search**: Reduce the number of molecules in your dataset
- **Memory errors**: Use smaller batch sizes in the configuration
- **Timeout errors**: Increase timeout settings in Streamlit configuration

### Getting Help

1. Check the [live demo](https://molsearch.streamlit.app) to see if the issue is local
2. Review the test files for working examples
3. Open an issue on GitHub with:
   - Python version
   - Error message
   - Steps to reproduce

