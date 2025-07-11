import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Descriptors import MolWt
import io
from molsearch_pipeline import MolecularSearchPipeline, Config

# Try to import Draw, but make it optional
try:
    from rdkit.Chem import Draw
    DRAW_AVAILABLE = True
except ImportError:
    DRAW_AVAILABLE = False
    st.warning("⚠️ Molecular drawing is not available in this environment. Molecules will be displayed as SMILES strings only.")

def draw_molecule_robust(mol, size=(300, 300)):
    """Try multiple methods to draw a molecule."""
    if mol is None:
        return None
    
    # Method 1: Try RDKit Draw
    if DRAW_AVAILABLE:
        try:
            img = Draw.MolToImage(mol, size=size)
            return img
        except:
            pass
    
    # Method 2: Try RDKit with different backend
    if DRAW_AVAILABLE:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100))
            img = Draw.MolToImage(mol, size=size)
            plt.close(fig)
            return img
        except:
            pass
    
    # Method 3: Generate a simple text representation
    try:
        from PIL import Image, ImageDraw, ImageFont
        # Create a simple image with SMILES text
        img = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font, fallback to basic if not available
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        smiles = Chem.MolToSmiles(mol)
        # Center the text
        bbox = draw.textbbox((0, 0), smiles, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2
        
        draw.text((x, y), smiles, fill='black', font=font)
        return img
    except:
        return None

# Load custom CSS
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

st.set_page_config(page_title="MolSearch", page_icon="🧪", layout="wide")

st.title("MolSearch")
st.markdown("Molecular similarity search using RDKit and local database")


# Initialize pipeline
@st.cache_resource
def get_pipeline():
    config = Config()
    return MolecularSearchPipeline(config)


pipeline = get_pipeline()

# Main content - Input section at the top
st.header("Input")
st.markdown("Enter a query molecule to find similar compounds in our database.")

# Configuration in input section
col1, col2 = st.columns([3, 1])
with col1:
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Single SMILES", "Upload CSV", "Example molecules"],
        horizontal=True,
    )
with col2:
    # Configuration
    st.markdown("**Configuration**")
    top_k = st.slider("Number of results", 1, 50, 10)

query_smiles = None

if input_method == "Single SMILES":
    query_smiles = st.text_input(
        "Enter SMILES string:",
        value="CCO",
        help="Enter a valid SMILES string for the query molecule",
    )

    if query_smiles:
        mol = Chem.MolFromSmiles(query_smiles)
        if mol:
            st.success("Valid SMILES entered")
            # Display molecule
            col1, col2 = st.columns([1, 2])
            with col1:
                img = draw_molecule_robust(mol, size=(300, 300))
                if img:
                    st.image(img, caption=f"Query molecule: {query_smiles}")
                else:
                    st.write(f"Query molecule: {query_smiles}")
            with col2:
                st.markdown("**Molecular Properties:**")
                st.write(f"**Molecular Weight:** {MolWt(mol):.2f}")
                st.write(f"**LogP:** {Descriptors.MolLogP(mol):.2f}")
                st.write(f"**TPSA:** {Descriptors.TPSA(mol):.1f}")
                st.write(f"**H-bond Donors:** {Descriptors.NumHDonors(mol)}")
                st.write(f"**H-bond Acceptors:** {Descriptors.NumHAcceptors(mol)}")
        else:
            st.error("Invalid SMILES string")
            query_smiles = None

elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader(
        "Upload CSV file with SMILES column",
        type=["csv"],
        help="CSV file should contain a 'smiles' column",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "smiles" in df.columns:
                st.success(f"Loaded {len(df)} molecules from CSV")
                st.dataframe(df.head())
                query_smiles = df["smiles"].iloc[0]  # Use first molecule as query

                # Display the first molecule
                mol = Chem.MolFromSmiles(query_smiles)
                if mol:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        img = draw_molecule_robust(mol, size=(300, 300))
                        if img:
                            st.image(img, caption=f"Query molecule: {query_smiles}")
                        else:
                            st.write(f"Query molecule: {query_smiles}")
                    with col2:
                        st.markdown("**Molecular Properties:**")
                        st.write(f"**Molecular Weight:** {MolWt(mol):.2f}")
                        st.write(f"**LogP:** {Descriptors.MolLogP(mol):.2f}")
                        st.write(f"**TPSA:** {Descriptors.TPSA(mol):.1f}")
                        st.write(f"**H-bond Donors:** {Descriptors.NumHDonors(mol)}")
                        st.write(
                            f"**H-bond Acceptors:** {Descriptors.NumHAcceptors(mol)}"
                        )
            else:
                st.error("CSV file must contain a 'smiles' column")
                query_smiles = None
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            query_smiles = None

else:  # Example molecules
    example_molecules = {
        "Ethanol": "CCO",
        "Benzene": "c1ccccc1",
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Fentanyl": "CCC(=O)N(C1CCN(CC1)CCC2=CC=CC=C2)C3=CC=CC=C3",
    }

    selected_example = st.selectbox(
        "Choose example molecule:", list(example_molecules.keys())
    )

    query_smiles = example_molecules[selected_example]
    mol = Chem.MolFromSmiles(query_smiles)
    if mol:
        col1, col2 = st.columns([1, 2])
        with col1:
            img = draw_molecule_robust(mol, size=(300, 300))
            if img:
                st.image(img, caption=f"Query molecule: {query_smiles}")
            else:
                st.write(f"Query molecule: {query_smiles}")
        with col2:
            st.markdown("**Molecular Properties:**")
            st.write(f"**Molecular Weight:** {MolWt(mol):.2f}")
            st.write(f"**LogP:** {Descriptors.MolLogP(mol):.2f}")
            st.write(f"**TPSA:** {Descriptors.TPSA(mol):.1f}")
            st.write(f"**H-bond Donors:** {Descriptors.NumHDonors(mol)}")
            st.write(f"**H-bond Acceptors:** {Descriptors.NumHAcceptors(mol)}")

# Search section
st.markdown("---")
st.header("Search Results")

if query_smiles:
    if st.button("Search Similar Molecules", type="primary", use_container_width=True):
        with st.spinner("Processing molecules..."):
            try:
                # Create a simple test dataset
                test_smiles = [
                    "CCO",  # ethanol
                    "CCCO",  # propanol
                    "CCCC",  # butane
                    "c1ccccc1",  # benzene
                    "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
                    "CCC(=O)N(C1CCN(CC1)CCC2=CC=CC=C2)C3=CC=CC=C3",  # fentanyl
                    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # ibuprofen
                    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
                    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # testosterone
                    "CC(C)(C)OC(=O)N[C@@H](CC1=CC=CC=C1)C(=O)O",  # phenylalanine
                ]

                # Process molecules
                df = pipeline.process_molecules(test_smiles)

                # Find similar molecules
                results = pipeline.find_similar_molecules(query_smiles, df, top_k=top_k)

                if len(results) > 0:
                    st.success(f"Found {len(results)} similar molecules")

                    # Display summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Mean Similarity", f"{results['similarity'].mean():.3f}"
                        )
                    with col2:
                        st.metric(
                            "Max Similarity", f"{results['similarity'].max():.3f}"
                        )
                    with col3:
                        st.metric(
                            "Min Similarity", f"{results['similarity'].min():.3f}"
                        )
                    with col4:
                        st.metric("Results", len(results))

                    # Display results in a more organized way
                    st.subheader("Similar Molecules")

                    # Create a grid layout for results
                    cols = st.columns(3)
                    for i, (_, row) in enumerate(results.iterrows()):
                        col_idx = i % 3
                        with cols[col_idx]:
                            st.markdown(
                                f"**{i+1}. Similarity: {row['similarity']:.3f}**"
                            )
                            mol = row["mols"]
                            if mol:
                                img = draw_molecule_robust(mol, size=(200, 200))
                                if img:
                                    st.image(img)
                                else:
                                    st.write(f"SMILES: {row['smiles']}")

                            st.markdown(f"**SMILES:** `{row['smiles']}`")

                            # Display key properties
                            st.markdown("**Properties:**")
                            st.write(
                                f"MW: {row['MolWt']:.1f} | LogP: {row['LogP']:.2f} | TPSA: {row['TPSA']:.1f}"
                            )

                            st.markdown("---")

                    # Download results
                    csv = results[
                        ["smiles", "similarity", "MolWt", "LogP", "TPSA"]
                    ].to_csv(index=False)
                    st.download_button(
                        label="Download results as CSV",
                        data=csv,
                        file_name=f"similar_molecules_{query_smiles[:10]}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                else:
                    st.warning("No similar molecules found")

            except Exception as e:
                st.error(f"Error during search: {e}")
else:
    st.info("Enter a query molecule above to start searching")

# Footer
st.markdown("---")
st.markdown(
    "**MolSearch** - A molecular similarity search pipeline using RDKit and Streamlit"
)
