#!/usr/bin/env python3
"""
Simple MolSearch App - Test Version
"""

import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="MolSearch - Test",
    page_icon="🧪",
    layout="wide"
)

def main():
    st.title("🧪 MolSearch - Test Version")
    st.write("This is a test version to verify deployment works.")
    
    # Test basic functionality
    st.header("Test Components")
    
    # Test pandas
    df = pd.DataFrame({
        'Molecule': ['Ethanol', 'Benzene', 'Aspirin'],
        'SMILES': ['CCO', 'c1ccccc1', 'CC(=O)OC1=CC=CC=C1C(=O)O'],
        'MW': [46.07, 78.11, 180.16]
    })
    st.dataframe(df)
    
    # Test numpy
    arr = np.random.rand(5, 5)
    st.write("Random similarity matrix:")
    st.write(arr)
    
    # Test user input
    user_input = st.text_input("Enter a SMILES string:", "CCO")
    st.write(f"You entered: {user_input}")
    
    # Test button
    if st.button("Test Button"):
        st.success("✅ Button works!")
    
    st.info("If you can see this, the basic deployment is working!")

if __name__ == "__main__":
    main() 