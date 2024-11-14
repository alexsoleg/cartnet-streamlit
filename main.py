import streamlit as st
from models.cartnet import CartNet
import os
from ase.io import read, write
from ase import Atoms
from CifFile import ReadCif
from torch_geometric.data import Data, Batch
import torch
from utils import radius_graph_pbc
from ase import Atoms
from ase.io import write
import gc

MEAN_TEMP = torch.tensor(192.1785) #training temp mean
STD_TEMP = torch.tensor(81.2135) #training temp std

# We cache the loading function to make is very fast on reload.
@st.cache_resource
def create_model():
    model = CartNet(dim_in=256, dim_rbf=64, num_layers=4, radius=5.0, invariant=False, temperature=True, use_envelope=True, cholesky=True)
    ckpt_path = "cpkt/cartnet_adp.ckpt"
    load = torch.load(ckpt_path, map_location=torch.device('cpu'))["model_state"]
    
    model.load_state_dict(load)
    model.eval()
    return model

@torch.no_grad()
def process_data(batch, model):
    atoms = batch.x.numpy().astype(int)  # Atomic numbers
    positions = batch.pos.numpy()  # Atomic positions
    cell = batch.cell.squeeze(0).numpy()  # Cell parameters
    temperature = batch.temperature_og.numpy()[0]


    adps = model(batch)
    M = batch.cell.squeeze(0)
    N = torch.diag(torch.linalg.norm(torch.linalg.inv(M.transpose(-1,-2)).squeeze(0), dim=-1))

    M = torch.linalg.inv(M)
    N = torch.linalg.inv(N)

    adps = M.transpose(-1,-2)@adps@M
    adps = N.transpose(-1,-2)@adps@N
    del M, N
    gc.collect()
    
    
    non_H_mask = batch.non_H_mask.numpy()
    indices = torch.arange(len(atoms))[non_H_mask].numpy()
    indices = {indices[i]: i for i in range(len(indices))}
    # Create ASE Atoms object
    ase_atoms = Atoms(numbers=atoms, positions=positions, cell=cell, pbc=True)

    # Convert positions to fractional coordinates
    fractional_positions = ase_atoms.get_scaled_positions()

    # Write to CIF file
    write('output.cif', ase_atoms)

    with open('output.cif', 'r') as file:
        lines = file.readlines()

    # Find the line where "loop_" appears and remove lines from there to the end
    for i, line in enumerate(lines):
        if line.strip().startswith('loop_'):
            lines = lines[:i]
            break

    # Write the modified lines to a new output file
    with open('output.cif', 'w') as file:
        file.writelines(lines)

    # Manually append positions and ADPs to the CIF file
    with open('output.cif', 'a') as cif_file:

        # Write temperature
        cif_file.write(f"\n_diffrn_ambient_temperature    {temperature}\n")
        # Write atomic positions
        cif_file.write("\nloop_\n")
        cif_file.write("_atom_site_label\n")
        cif_file.write("_atom_site_type_symbol\n")
        cif_file.write("_atom_site_fract_x\n")
        cif_file.write("_atom_site_fract_y\n")
        cif_file.write("_atom_site_fract_z\n")
        cif_file.write("_atom_site_U_iso_or_equiv\n")
        cif_file.write("_atom_site_thermal_displace_type\n")
        
        element_count = {}
        for i, (atom_number, frac_pos) in enumerate(zip(atoms, fractional_positions)):
            element = ase_atoms[i].symbol
            assert atom_number == ase_atoms[i].number
            if element not in element_count:
                element_count[element] = 0
            element_count[element] += 1
            label = f"{element}{element_count[element]}"
            u_iso = torch.trace(adps[indices[i]]).mean() if element != 'H' else 0.01
            type = "Uani" if element != 'H' else "Uiso"
            cif_file.write(f"{label} {element} {frac_pos[0]} {frac_pos[1]} {frac_pos[2]} {u_iso} {type}\n")

        # Write ADPs
        cif_file.write("\nloop_\n")
        cif_file.write("_atom_site_aniso_label\n")
        cif_file.write("_atom_site_aniso_U_11\n")
        cif_file.write("_atom_site_aniso_U_22\n")
        cif_file.write("_atom_site_aniso_U_33\n")
        cif_file.write("_atom_site_aniso_U_23\n")
        cif_file.write("_atom_site_aniso_U_13\n")
        cif_file.write("_atom_site_aniso_U_12\n")
        
        element_count = {}
        for i, atom_number in enumerate(atoms):
            if atom_number == 1:
                continue
            element = ase_atoms[i].symbol
            if element not in element_count:
                element_count[element] = 0
            element_count[element] += 1
            label = f"{element}{element_count[element]}"
            cif_file.write(f"{label} {adps[indices[i],0,0]} {adps[indices[i],1,1]} {adps[indices[i],2,2]} {adps[indices[i],1,2]} {adps[indices[i],0,2]} {adps[indices[i],0,1]}\n")


@torch.no_grad()
def main():
    model = create_model()
    st.title("CartNet ADP Prediction")
    st.image('fig/pipeline.png')
    
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

            if len(atoms.positions) > 300:
                raise ValueError("This online implementation is not optimized for large systems. For large systems, please use the local version.")
            
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
    ⚠️ **Warning**: This online web application is designed for structures with up to 300 atoms in the unit cell. For larger structures, please use the [local implementation of CartNet](https://github.com/alexsoleg/cartnet-streamlit/).
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
