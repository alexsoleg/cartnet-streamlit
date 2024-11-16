import streamlit as st
import os
from ase.io import read
from CifFile import ReadCif
from torch_geometric.data import Data, Batch
import torch
from models.master import create_model
from process import process_data
from utils import radius_graph_pbc
import gc

MEAN_TEMP = torch.tensor(192.1785) #training temp mean
STD_TEMP = torch.tensor(81.2135) #training temp std


@torch.no_grad()
def main():
    model = create_model()
    st.title("CartNet ADP Prediction")
    st.image('fig/pipeline.png')

    st.markdown("""
                CartNet is a graph neural network specifically designed for predicting Anisotropic Displacement Parameters (ADPs) in crystal structures. The model has been trained on over 220,000 molecular crystal structures from the Cambridge Structural Database (CSD), making it highly accurate and robust for ADP prediction tasks. CartNet addresses the computational challenges of traditional methods by encoding the full 3D geometry of atomic structures into a Cartesian reference frame, bypassing the need for unit cell encoding. The model incorporates innovative features, including a neighbour equalization technique to enhance interaction detection and a Cholesky-based output layer to ensure valid ADP predictions. Additionally, it introduces a rotational SO(3) data augmentation technique to improve generalization across different crystal structure orientations, making the model highly efficient and accurate in predicting ADPs while significantly reducing computational costs.
    """)

    uploaded_file = st.file_uploader("Upload a CIF file", type=["cif"], accept_multiple_files=False)
    # uploaded_file = "ABABEM.cif"
    if uploaded_file is not None:
        try:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            filename = str(uploaded_file.name)
            # Read the CIF file using ASE
            atoms = read(filename, format="cif")
            cif = ReadCif(filename)
            cif_data = cif.first_block()
            if "_diffrn_ambient_temperature" in cif_data.keys():
                temperature = float(cif_data["_diffrn_ambient_temperature"])
            else:
                raise ValueError("Temperature not found in the CIF file. \
                                    Please provide a temperature in the field _diffrn_ambient_temperature from the CIF file.")
            st.success("CIF file successfully read.")

            data = Data()
            data.x = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32)
            
            data.pos = torch.tensor(atoms.positions, dtype=torch.float32)
            data.temperature_og = torch.tensor([temperature], dtype=torch.float32)
            data.temperature = (data.temperature_og - MEAN_TEMP) / STD_TEMP
            data.cell = torch.tensor(atoms.cell.array, dtype=torch.float32).unsqueeze(0)

            data.pbc = torch.tensor([True, True, True])
            data.natoms = len(atoms)

            del atoms
            gc.collect()
            batch = Batch.from_data_list([data])
            

            edge_index, _, _, edge_attr = radius_graph_pbc(batch, 5.0, 64)
            del batch
            gc.collect()
            data.cart_dist = torch.norm(edge_attr, dim=-1)
            data.cart_dir = torch.nn.functional.normalize(edge_attr, dim=-1)
            data.edge_index = edge_index
            data.non_H_mask = data.x != 1
            delattr(data, "pbc")
            delattr(data, "natoms")
            batch = Batch.from_data_list([data])
            del data, edge_index, edge_attr
            gc.collect()

            st.success("Graph successfully created.")

            process_data(batch, model)
            st.success("ADPs successfully predicted.")
            
            # Create a download button for the processed CIF file
            with open("output.cif", "r") as f:
                cif_contents = f.read()
            
            st.download_button(
                label="Download processed CIF file",
                data=cif_contents,
                file_name="output.cif",
                mime="text/plain"
            )

            os.remove("output.cif")
            os.remove(filename)
            gc.collect()
        except Exception as e:
            st.error(f"An error occurred while reading the CIF file: {e}")

    
    st.markdown("""
    📌 The official implementation of the paper with all experiments can be found at [CartNet GitHub Repository](https://github.com/imatge-upc/CartNet).
    """)

    st.markdown("""
    ### How to cite

    If you use CartNet in your research, please cite our paper:

    ```bibtex
    @article{your_paper_citation,
    title={Title of the Paper},
    author={Author1 and Author2 and Author3},
    journal={Journal Name},
    year={2023},
    volume={XX},
    number={YY},
    pages={ZZZ}
    }
    ```
    """)

if __name__ == "__main__":
    main()
