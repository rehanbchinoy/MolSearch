#!/usr/bin/env python3
"""
Molecular Search Pipeline

A robust pipeline for molecular similarity search using RDKit.
This pipeline provides:
- Molecular similarity calculation using Tanimoto coefficients
- Molecular featurization using RDKit descriptors
- SQLite database storage and retrieval
- Similarity search and analysis
"""

import os
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import json
import sqlite3
import itertools

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit.Chem.Descriptors import MolWt, NumHDonors, NumHAcceptors

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


@dataclass
class Config:
    """Configuration class for the minimal molecular search pipeline."""

    # Data paths
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    output_dir: Path = Path("output")

    # Database configuration
    db_path: str = "molsearch.db"

    # Processing parameters
    batch_size: int = 10
    max_molecules: int = 1000

    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [self.data_dir, self.models_dir, self.output_dir]:
            path.mkdir(exist_ok=True)


class SimilarityCalculator:
    """Calculate molecular similarities using RDKit only."""

    @staticmethod
    def compute_tanimoto_similarity(
        mol1: Union[str, Chem.Mol], mol2: Union[str, Chem.Mol]
    ) -> float:
        """
        Compute Tanimoto similarity between two molecules using RDKit.

        Args:
            mol1: First molecule (SMILES string or RDKit Mol)
            mol2: Second molecule (SMILES string or RDKit Mol)

        Returns:
            Tanimoto similarity score
        """
        try:
            # Convert to RDKit Mol objects if needed
            if isinstance(mol1, str):
                mol1 = Chem.MolFromSmiles(mol1)
            if isinstance(mol2, str):
                mol2 = Chem.MolFromSmiles(mol2)

            if mol1 is None or mol2 is None:
                return 0.0

            # Generate Morgan fingerprints
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)

            # Calculate Tanimoto similarity
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
            return float(similarity)

        except Exception as e:
            logger.error(f"Error computing Tanimoto similarity: {e}")
            return 0.0

    @staticmethod
    def compute_fingerprint_similarity_matrix(mols: List[Chem.Mol]) -> np.ndarray:
        """
        Compute similarity matrix for a list of molecules.

        Args:
            mols: List of RDKit molecules

        Returns:
            Similarity matrix as numpy array
        """
        n_mols = len(mols)
        similarity_matrix = np.zeros((n_mols, n_mols))

        for i in range(n_mols):
            for j in range(i, n_mols):
                similarity = SimilarityCalculator.compute_tanimoto_similarity(
                    mols[i], mols[j]
                )
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        return similarity_matrix


class MolecularFeaturizer:
    """Handles molecular featurization using RDKit descriptors only."""

    def __init__(self, config: Config):
        self.config = config
        self.descriptor_calculators = self._setup_descriptors()

    def _setup_descriptors(self):
        """Setup RDKit descriptor calculators."""
        return {
            "MolWt": MolWt,
            "LogP": Descriptors.MolLogP,
            "NumHDonors": NumHDonors,
            "NumHAcceptors": NumHAcceptors,
            "TPSA": Descriptors.TPSA,
            "NumRotatableBonds": Descriptors.NumRotatableBonds,
            "NumAromaticRings": Descriptors.NumAromaticRings,
            "HeavyAtomCount": Descriptors.HeavyAtomCount,
            "RingCount": Descriptors.RingCount,
        }

    def featurize_molecules(self, molecules: List[Chem.Mol]) -> np.ndarray:
        """
        Featurize a list of molecules using RDKit descriptors.

        Args:
            molecules: List of RDKit molecules

        Returns:
            Feature matrix as numpy array
        """
        try:
            features = []
            for mol in molecules:
                if mol is None:
                    # Use zeros for invalid molecules
                    mol_features = [0.0] * len(self.descriptor_calculators)
                else:
                    mol_features = []
                    for name, calculator in self.descriptor_calculators.items():
                        try:
                            value = calculator(mol)
                            mol_features.append(
                                float(value) if value is not None else 0.0
                            )
                        except:
                            mol_features.append(0.0)

                features.append(mol_features)

            features_array = np.array(features)
            logger.info(
                f"Featurized {len(molecules)} molecules to {features_array.shape[1]} features"
            )
            return features_array

        except Exception as e:
            logger.error(f"Error featurizing molecules: {e}")
            raise

    def get_feature_names(self) -> List[str]:
        """Get the names of the computed features."""
        return list(self.descriptor_calculators.keys())


class DatabaseManager:
    """Handles SQLite database operations."""

    def __init__(self, config: Config):
        self.config = config
        self.connection = None

    def __enter__(self):
        self.connection = sqlite3.connect(self.config.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()
            self.connection = None

    def create_tables(self):
        """Create necessary database tables."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS molecules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mol_id TEXT UNIQUE NOT NULL,
                smiles TEXT NOT NULL,
                mol_weight REAL,
                logp REAL,
                num_h_donors INTEGER,
                num_h_acceptors INTEGER,
                tpsa REAL,
                num_rotatable_bonds INTEGER,
                num_aromatic_rings INTEGER,
                heavy_atom_count INTEGER,
                ring_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.connection.commit()

    def insert_molecules(self, df: pd.DataFrame):
        """Insert molecules into the database."""
        try:
            # Prepare data for insertion
            data_to_insert = []
            for _, row in df.iterrows():
                mol = Chem.MolFromSmiles(row["smiles"])
                if mol is not None:
                    data_to_insert.append(
                        {
                            "mol_id": row["mol_id"],
                            "smiles": row["smiles"],
                            "mol_weight": MolWt(mol),
                            "logp": Descriptors.MolLogP(mol),
                            "num_h_donors": NumHDonors(mol),
                            "num_h_acceptors": NumHAcceptors(mol),
                            "tpsa": Descriptors.TPSA(mol),
                            "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                            "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
                            "heavy_atom_count": Descriptors.HeavyAtomCount(mol),
                            "ring_count": Descriptors.RingCount(mol),
                        }
                    )

            # Insert into database
            if data_to_insert:
                df_to_insert = pd.DataFrame(data_to_insert)
                df_to_insert.to_sql(
                    "molecules", self.connection, if_exists="replace", index=False
                )
                self.connection.commit()
                logger.info(f"Inserted {len(data_to_insert)} molecules into database")

        except Exception as e:
            logger.error(f"Error inserting molecules: {e}")
            raise


class SimilarityAnalyzer:
    """Handles molecular similarity analysis."""

    def __init__(self):
        self.similarity_calc = SimilarityCalculator()

    def analyze_search_results(
        self, df_results: pd.DataFrame, query_mol: Chem.Mol
    ) -> Dict:
        """
        Analyze search results and compute performance metrics.

        Args:
            df_results: DataFrame with search results
            query_mol: Query molecule

        Returns:
            Dictionary with analysis results
        """
        try:
            # Compute Tanimoto similarities
            df_results["tanimoto"] = df_results["mols"].apply(
                lambda mol: self.similarity_calc.compute_tanimoto_similarity(
                    mol, query_mol
                )
            )

            # Sort by similarity
            df_results = df_results.sort_values("tanimoto", ascending=False)

            # Compute metrics
            analysis = {
                "num_results": len(df_results),
                "mean_similarity": df_results["tanimoto"].mean(),
                "max_similarity": df_results["tanimoto"].max(),
                "min_similarity": df_results["tanimoto"].min(),
                "std_similarity": df_results["tanimoto"].std(),
                "high_similarity_count": (df_results["tanimoto"] > 0.7).sum(),
                "medium_similarity_count": (
                    (df_results["tanimoto"] > 0.5) & (df_results["tanimoto"] <= 0.7)
                ).sum(),
                "low_similarity_count": (df_results["tanimoto"] <= 0.5).sum(),
            }

            return analysis, df_results

        except Exception as e:
            logger.error(f"Error analyzing search results: {e}")
            raise


class MolecularSearchPipeline:
    """Main pipeline class for molecular search."""

    def __init__(self, config: Config):
        self.config = config
        self.similarity_calc = SimilarityCalculator()
        self.featurizer = MolecularFeaturizer(config)
        self.analyzer = SimilarityAnalyzer()

    def process_molecules(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Process a list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            DataFrame with processed molecules
        """
        try:
            # Convert SMILES to molecules
            molecules = []
            valid_smiles = []

            for i, smiles in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    molecules.append(mol)
                    valid_smiles.append(smiles)
                else:
                    logger.warning(f"Invalid SMILES: {smiles}")

            logger.info(
                f"Processed {len(molecules)} valid molecules from {len(smiles_list)} total"
            )

            # Create DataFrame
            df = pd.DataFrame(
                {
                    "mol_id": [f"mol_{i}" for i in range(len(molecules))],
                    "smiles": valid_smiles,
                    "mols": molecules,
                }
            )

            # Add features
            features = self.featurizer.featurize_molecules(molecules)
            feature_names = self.featurizer.get_feature_names()

            for i, name in enumerate(feature_names):
                df[name] = features[:, i]

            return df

        except Exception as e:
            logger.error(f"Error processing molecules: {e}")
            raise

    def find_similar_molecules(
        self, query_smiles: str, molecule_df: pd.DataFrame, top_k: int = 10
    ) -> pd.DataFrame:
        """
        Find molecules similar to the query.

        Args:
            query_smiles: Query molecule SMILES
            molecule_df: DataFrame with molecules to search
            top_k: Number of top results to return

        Returns:
            DataFrame with similar molecules
        """
        try:
            query_mol = Chem.MolFromSmiles(query_smiles)
            if query_mol is None:
                raise ValueError(f"Invalid query SMILES: {query_smiles}")

            # Calculate similarities
            similarities = []
            for mol in molecule_df["mols"]:
                sim = self.similarity_calc.compute_tanimoto_similarity(query_mol, mol)
                similarities.append(sim)

            # Add similarity scores
            result_df = molecule_df.copy()
            result_df["similarity"] = similarities

            # Sort and return top k
            result_df = result_df.sort_values("similarity", ascending=False)
            return result_df.head(top_k)

        except Exception as e:
            logger.error(f"Error finding similar molecules: {e}")
            raise


def main():
    """Main function to run the molecular search pipeline."""
    # Load configuration
    config = Config()

    # Example usage
    reference_smiles = "CCC(=O)N(C1CCN(CC1)CCC2=CC=CC=C2)C3=CC=CC=C3"  # Fentanyl

    print("MolSearch Pipeline")
    print("=" * 50)
    print(f"Reference molecule: {reference_smiles}")
    print(f"Output directory: {config.output_dir}")
    print()

    # Test molecules
    test_smiles = [
        "CCO",  # ethanol
        "CCCO",  # propanol
        "CCCC",  # butane
        "c1ccccc1",  # benzene
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
        reference_smiles,  # fentanyl
    ]

    # Test basic functionality
    try:
        # Initialize pipeline
        pipeline = MolecularSearchPipeline(config)

        # Test similarity calculation
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCCO")
        similarity = SimilarityCalculator.compute_tanimoto_similarity(mol1, mol2)
        print(f"SUCCESS: Similarity calculation works: {similarity:.3f}")

        # Test featurization
        featurizer = MolecularFeaturizer(config)
        features = featurizer.featurize_molecules([mol1, mol2])
        print(f"SUCCESS: Featurization works: {features.shape}")
        print(f"   Feature names: {featurizer.get_feature_names()}")

        # Test molecule processing
        df = pipeline.process_molecules(test_smiles)
        print(f"SUCCESS: Molecule processing works: {len(df)} molecules processed")

        # Test similarity search
        results = pipeline.find_similar_molecules("CCO", df, top_k=3)
        print(f"SUCCESS: Similarity search works: {len(results)} results found")
        print(
            f"   Top result: {results.iloc[0]['smiles']} (similarity: {results.iloc[0]['similarity']:.3f})"
        )

        # Test database operations
        with DatabaseManager(config) as db:
            db.create_tables()
            db.insert_molecules(df)
            print("SUCCESS: Database operations work")

        print("\nALL TESTS PASSED! Pipeline is ready to use.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
