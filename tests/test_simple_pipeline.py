"""
Tests for the simplified molecular search pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Import the simplified pipeline
from molsearch_pipeline import (
    Config,
    SimilarityCalculator,
    MolecularFeaturizer,
    DatabaseManager,
    SimilarityAnalyzer,
    MolecularSearchPipeline,
)
from rdkit import Chem


class TestConfig:
    """Test configuration management."""

    def test_config_initialization(self):
        """Test Config class initialization."""
        config = Config()
        assert config.data_dir.exists()
        assert config.models_dir.exists()
        assert config.output_dir.exists()
        assert config.db_path == "molsearch.db"
        assert config.batch_size == 10
        assert config.max_molecules == 1000


class TestSimilarityCalculator:
    """Test similarity calculations."""

    def test_tanimoto_similarity_same_molecule(self):
        """Test Tanimoto similarity for identical molecules."""
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCO")
        similarity = SimilarityCalculator.compute_tanimoto_similarity(mol1, mol2)
        assert similarity == 1.0

    def test_tanimoto_similarity_different_molecules(self):
        """Test Tanimoto similarity for different molecules."""
        mol1 = Chem.MolFromSmiles("CCO")  # ethanol
        mol2 = Chem.MolFromSmiles("CCCO")  # propanol
        similarity = SimilarityCalculator.compute_tanimoto_similarity(mol1, mol2)
        assert 0.0 < similarity < 1.0

    def test_tanimoto_similarity_invalid_smiles(self):
        """Test Tanimoto similarity with invalid SMILES."""
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("invalid_smiles")
        similarity = SimilarityCalculator.compute_tanimoto_similarity(mol1, mol2)
        assert similarity == 0.0

    def test_tanimoto_similarity_string_input(self):
        """Test Tanimoto similarity with SMILES strings."""
        similarity = SimilarityCalculator.compute_tanimoto_similarity("CCO", "CCCO")
        assert 0.0 < similarity < 1.0

    def test_similarity_matrix(self):
        """Test similarity matrix computation."""
        mols = [
            Chem.MolFromSmiles("CCO"),
            Chem.MolFromSmiles("CCCO"),
            Chem.MolFromSmiles("CCCC"),
        ]
        matrix = SimilarityCalculator.compute_fingerprint_similarity_matrix(mols)
        assert matrix.shape == (3, 3)
        assert np.allclose(matrix, matrix.T)  # Symmetric
        assert np.allclose(np.diag(matrix), 1.0)  # Diagonal is 1


class TestMolecularFeaturizer:
    """Test molecular featurization."""

    def test_featurizer_initialization(self):
        """Test featurizer initialization."""
        config = Config()
        featurizer = MolecularFeaturizer(config)
        assert featurizer.descriptor_calculators is not None
        assert len(featurizer.descriptor_calculators) > 0

    def test_featurization(self):
        """Test molecule featurization."""
        config = Config()
        featurizer = MolecularFeaturizer(config)

        mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCCO")]

        features = featurizer.featurize_molecules(mols)
        assert features.shape[0] == 2
        assert features.shape[1] > 0
        assert not np.isnan(features).any()

    def test_featurization_with_invalid_mol(self):
        """Test featurization with invalid molecules."""
        config = Config()
        featurizer = MolecularFeaturizer(config)

        mols = [
            Chem.MolFromSmiles("CCO"),
            None,  # Invalid molecule
            Chem.MolFromSmiles("CCCO"),
        ]

        features = featurizer.featurize_molecules(mols)
        assert features.shape[0] == 3
        assert features.shape[1] > 0
        # Invalid molecule should have zero features
        assert np.all(features[1] == 0.0)

    def test_get_feature_names(self):
        """Test getting feature names."""
        config = Config()
        featurizer = MolecularFeaturizer(config)
        feature_names = featurizer.get_feature_names()
        assert len(feature_names) > 0
        assert "MolWt" in feature_names
        assert "LogP" in feature_names


class TestDatabaseManager:
    """Test database operations."""

    def test_database_creation(self):
        """Test database table creation."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            config = Config()
            config.db_path = db_path

            with DatabaseManager(config) as db:
                db.create_tables()

                # Check if table exists
                cursor = db.connection.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='molecules'"
                )
                result = cursor.fetchone()
                assert result is not None

        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_molecule_insertion(self):
        """Test molecule insertion."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            config = Config()
            config.db_path = db_path

            with DatabaseManager(config) as db:
                db.create_tables()

                # Create test data
                df = pd.DataFrame(
                    {"mol_id": ["test1", "test2"], "smiles": ["CCO", "CCCO"]}
                )

                db.insert_molecules(df)

                # Verify insertion
                cursor = db.connection.cursor()
                cursor.execute("SELECT COUNT(*) FROM molecules")
                count = cursor.fetchone()[0]
                assert count == 2

        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_database_context_manager(self):
        """Test database context manager functionality."""
        config = Config()
        config.db_path = ":memory:"  # Use in-memory database

        db_manager = DatabaseManager(config)
        with db_manager as db:
            assert db.connection is not None
            db.create_tables()

        # Connection should be closed after context exit
        assert db_manager.connection is None


class TestSimilarityAnalyzer:
    """Test similarity analysis."""

    def test_analyze_search_results(self):
        """Test search results analysis."""
        analyzer = SimilarityAnalyzer()

        # Create mock results
        mols = [
            Chem.MolFromSmiles("CCO"),
            Chem.MolFromSmiles("CCCO"),
            Chem.MolFromSmiles("CCCC"),
        ]

        df_results = pd.DataFrame({"mols": mols, "score": [0.9, 0.7, 0.5]})

        query_mol = Chem.MolFromSmiles("CCO")

        analysis, sorted_results = analyzer.analyze_search_results(
            df_results, query_mol
        )

        assert "num_results" in analysis
        assert "mean_similarity" in analysis
        assert "max_similarity" in analysis
        assert analysis["num_results"] == 3
        assert "tanimoto" in sorted_results.columns
        assert len(sorted_results) == 3
        assert (
            sorted_results.iloc[0]["tanimoto"] == 1.0
        )  # Should be most similar to itself


class TestMolecularSearchPipeline:
    """Test the main pipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        config = Config()
        pipeline = MolecularSearchPipeline(config)
        assert pipeline.config == config
        assert pipeline.similarity_calc is not None
        assert pipeline.featurizer is not None
        assert pipeline.analyzer is not None

    def test_process_molecules(self):
        """Test molecule processing."""
        config = Config()
        pipeline = MolecularSearchPipeline(config)

        test_smiles = ["CCO", "CCCO", "CCCC"]
        df = pipeline.process_molecules(test_smiles)

        assert len(df) == 3
        assert "mol_id" in df.columns
        assert "smiles" in df.columns
        assert "mols" in df.columns
        assert "MolWt" in df.columns
        assert "LogP" in df.columns

    def test_process_molecules_with_invalid_smiles(self):
        """Test processing with invalid SMILES."""
        config = Config()
        pipeline = MolecularSearchPipeline(config)

        test_smiles = ["CCO", "invalid_smiles", "CCCO"]
        df = pipeline.process_molecules(test_smiles)

        # Should only process valid SMILES
        assert len(df) == 2
        assert "CCO" in df["smiles"].values
        assert "CCCO" in df["smiles"].values

    def test_find_similar_molecules(self):
        """Test similarity search."""
        config = Config()
        pipeline = MolecularSearchPipeline(config)

        # Create test data
        test_smiles = ["CCO", "CCCO", "CCCC", "c1ccccc1"]
        df = pipeline.process_molecules(test_smiles)

        # Search for molecules similar to ethanol
        results = pipeline.find_similar_molecules("CCO", df, top_k=2)

        assert len(results) == 2
        assert results.iloc[0]["smiles"] == "CCO"  # Should be most similar to itself
        assert results.iloc[0]["similarity"] == 1.0

    def test_find_similar_molecules_invalid_query(self):
        """Test similarity search with invalid query."""
        config = Config()
        pipeline = MolecularSearchPipeline(config)

        test_smiles = ["CCO", "CCCO"]
        df = pipeline.process_molecules(test_smiles)

        with pytest.raises(ValueError):
            pipeline.find_similar_molecules("invalid_smiles", df)


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_similarity(self):
        """Test end-to-end similarity calculation."""
        # Test molecules
        mol1 = Chem.MolFromSmiles("CCO")  # ethanol
        mol2 = Chem.MolFromSmiles("CCCO")  # propanol
        mol3 = Chem.MolFromSmiles("CCCC")  # butane

        # Calculate similarities
        sim_12 = SimilarityCalculator.compute_tanimoto_similarity(mol1, mol2)
        sim_13 = SimilarityCalculator.compute_tanimoto_similarity(mol1, mol3)

        # Ethanol should be more similar to propanol than butane
        assert sim_12 > sim_13

    def test_featurization_and_similarity(self):
        """Test that featurization works with similarity calculation."""
        config = Config()
        featurizer = MolecularFeaturizer(config)

        mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCCO")]

        # Featurize molecules
        features = featurizer.featurize_molecules(mols)

        # Calculate similarity
        similarity = SimilarityCalculator.compute_tanimoto_similarity(mols[0], mols[1])

        # Both should work without errors
        assert features.shape[0] == 2
        assert 0.0 <= similarity <= 1.0

    def test_full_pipeline_workflow(self):
        """Test the complete pipeline workflow."""
        config = Config()
        config.db_path = ":memory:"  # Use in-memory database for testing

        pipeline = MolecularSearchPipeline(config)

        # Test molecules
        test_smiles = ["CCO", "CCCO", "CCCC"]

        # Process molecules
        df = pipeline.process_molecules(test_smiles)
        assert len(df) == 3

        # Find similar molecules
        results = pipeline.find_similar_molecules("CCO", df, top_k=2)
        assert len(results) == 2

        # Store in database
        with DatabaseManager(config) as db:
            db.create_tables()
            db.insert_molecules(df)

            # Verify storage
            cursor = db.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM molecules")
            count = cursor.fetchone()[0]
            assert count == 3


if __name__ == "__main__":
    pytest.main([__file__])
