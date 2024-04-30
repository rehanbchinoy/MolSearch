import streamlit as st
import pandas as pd
import numpy as np

st.title('MolSearch-GPT')

st.file_uploader('Query molecules')

## Read in file, parse it to ID and SMILES
## Get embedding using featurization pipeline
## Query db
## Return mols2grid with molecules and Tanimoto distribution