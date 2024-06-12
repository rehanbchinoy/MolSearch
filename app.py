import streamlit as st
import pandas as pd
import numpy as np
from pinecone import Pinecone
import datamol as dm
import molfeat
from molfeat.calc import  RDKitDescriptors2D
from molfeat.trans import MoleculeTransformer

st.title('MolSearch-GPT')

uploaded_file = st.file_uploader('Query molecules', type={"csv"})
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

    mols = dm.convert.from_df(df)
    calc = RDKitDescriptors2D(replace_nan=True)
    featurizer = MoleculeTransformer(calc, dtype=np.float32)

    with dm.without_rdkit_log():
        feats = np.stack(featurizer(mols))

    df['embeddings'] = feats

    ## Start PC index
    pc = Pinecone(api_key= "c2c9ba1d-9710-472b-a950-a3db5b40a67c") ## supplied by Pinecone
    index = pc.Index('molsearch') ## already created

    for feat in feats.tolist():
        query_results = index.query(vector = feat, top_k = 100)
        st.write(query_results[0][0])

