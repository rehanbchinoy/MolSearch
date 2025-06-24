#!/usr/bin/env python3
"""
MolSearch Streamlit App

A molecular similarity search application using RDKit and Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import Draw, but make it optional
try:
    from rdkit.Chem import Draw
    DRAW_AVAILABLE = True
except ImportError:
    DRAW_AVAILABLE = False
    st.warning("⚠️ Molecular drawing is not available in this environment. Molecules will be displayed as SMILES strings only.")

# Page configuration
st.set_page_config(
    page_title="MolSearch - Molecular Similarity Search",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .molecule-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .similarity-high { color: #2e8b57; font-weight: bold; }
    .similarity-medium { color: #ff8c00; font-weight: bold; }
    .similarity-low { color: #dc143c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def compute_tanimoto_similarity(mol1, mol2):
    """Compute Tanimoto similarity between two molecules."""
    try:
        if mol1 is None or mol2 is None:
            return 0.0
        
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        return float(similarity)
    except:
        return 0.0

def get_molecular_properties(mol):
    """Get molecular properties for a given molecule."""
    if mol is None:
        return {}
    
    try:
        return {
            'Molecular Weight': round(Descriptors.MolWt(mol), 2),
            'LogP': round(Descriptors.MolLogP(mol), 2),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'TPSA': round(Descriptors.TPSA(mol), 2),
            'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
            'Aromatic Rings': Descriptors.NumAromaticRings(mol),
            'Heavy Atoms': Descriptors.HeavyAtomCount(mol),
            'Ring Count': Descriptors.RingCount(mol)
        }
    except:
        return {}

def draw_molecule(mol, size=(300, 300)):
    """Draw a molecule and return as PIL Image."""
    try:
        if mol is None:
            return None
        
        if not DRAW_AVAILABLE:
            return None
        
        # Create a figure with white background
        fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100))
        ax.set_facecolor('white')
        
        # Draw the molecule
        img = Chem.Draw.MolToImage(mol, size=size)
        
        plt.close(fig)
        return img
    except:
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🧪 MolSearch</h1>', unsafe_allow_html=True)
    st.markdown("### Molecular Similarity Search using RDKit")
    
    # Sidebar for input
    with st.sidebar:
        st.header("🔬 Search Configuration")
        
        # Input method selection
        input_method = st.selectbox(
            "Choose input method:",
            ["Single SMILES", "CSV Upload", "Example Molecules"]
        )
        
        # Number of results
        num_results = st.slider("Number of results to return:", 1, 20, 10)
        
        # Similarity threshold
        similarity_threshold = st.slider("Minimum similarity threshold:", 0.0, 1.0, 0.3, 0.05)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        if input_method == "Single SMILES":
            smiles_input = st.text_input(
                "Enter SMILES string:",
                placeholder="e.g., CCO for ethanol"
            )
            
            if smiles_input:
                mol = Chem.MolFromSmiles(smiles_input)
                if mol is not None:
                    st.success("✅ Valid molecule!")
                    img = draw_molecule(mol)
                    if img:
                        st.image(img, caption="Molecular Structure")
                    else:
                        st.write(f"**SMILES:** `{smiles_input}`")
                    
                    # Show properties
                    props = get_molecular_properties(mol)
                    if props:
                        st.subheader("Molecular Properties")
                        for prop, value in props.items():
                            st.write(f"**{prop}:** {value}")
                else:
                    st.error("❌ Invalid SMILES string")
        
        elif input_method == "CSV Upload":
            uploaded_file = st.file_uploader(
                "Upload CSV file with SMILES column:",
                type=['csv']
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Uploaded {len(df)} molecules")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")
        
        elif input_method == "Example Molecules":
            example_molecules = {
                "Ethanol": "CCO",
                "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "Glucose": "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O",
                "Benzene": "c1ccccc1"
            }
            
            selected_example = st.selectbox("Choose example molecule:", list(example_molecules.keys()))
            smiles_input = example_molecules[selected_example]
            st.text_input("SMILES:", value=smiles_input, key="example_smiles")
            
            mol = Chem.MolFromSmiles(smiles_input)
            if mol is not None:
                img = draw_molecule(mol)
                if img:
                    st.image(img, caption=f"{selected_example} Structure")
                else:
                    st.write(f"**{selected_example} SMILES:** `{smiles_input}`")
    
    with col2:
        st.header("🔍 Search Results")
        
        if 'smiles_input' in locals() and smiles_input:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol is not None:
                # Generate example results (in a real app, this would search a database)
                example_results = [
                    ("CCO", "Ethanol", 1.0),
                    ("CCCO", "Propanol", 0.8),
                    ("c1ccccc1", "Benzene", 0.2),
                    ("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin", 0.1),
                    ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine", 0.05)
                ]
                
                results = []
                for smiles, name, sim in example_results[:num_results]:
                    result_mol = Chem.MolFromSmiles(smiles)
                    if result_mol is not None:
                        similarity = compute_tanimoto_similarity(mol, result_mol)
                        if similarity >= similarity_threshold:
                            results.append({
                                'SMILES': smiles,
                                'Name': name,
                                'Similarity': similarity,
                                'Mol': result_mol
                            })
                
                if results:
                    st.success(f"Found {len(results)} similar molecules")
                    
                    for i, result in enumerate(results):
                        with st.container():
                            st.markdown(f"### Result {i+1}: {result['Name']}")
                            
                            col_a, col_b = st.columns([1, 2])
                            
                            with col_a:
                                img = draw_molecule(result['Mol'], size=(200, 200))
                                if img:
                                    st.image(img, caption=result['Name'])
                                else:
                                    st.write(f"**SMILES:** `{result['SMILES']}`")
                            
                            with col_b:
                                # Similarity score with color coding
                                sim = result['Similarity']
                                if sim >= 0.7:
                                    sim_class = "similarity-high"
                                elif sim >= 0.4:
                                    sim_class = "similarity-medium"
                                else:
                                    sim_class = "similarity-low"
                                
                                st.markdown(f"<span class='{sim_class}'>Similarity: {sim:.3f}</span>", unsafe_allow_html=True)
                                
                                # Properties
                                props = get_molecular_properties(result['Mol'])
                                if props:
                                    st.write("**Properties:**")
                                    for prop, value in list(props.items())[:4]:  # Show first 4 properties
                                        st.write(f"• {prop}: {value}")
                                
                                st.write(f"**SMILES:** `{result['SMILES']}`")
                            
                            st.divider()
                else:
                    st.warning("No molecules found above the similarity threshold")
            else:
                st.error("Please enter a valid SMILES string")
        else:
            st.info("👈 Enter a molecule in the input section to start searching")

if __name__ == "__main__":
    main() 