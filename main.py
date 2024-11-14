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


def process_data(batch, model):
    atoms = batch.x.numpy().astype(int)  # Atomic numbers
    positions = batch.pos.numpy()  # Atomic positions
    cell = batch.cell.squeeze(0).numpy()  # Cell parameters

    with torch.no_grad():
        adps = model(batch)
    

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
        for i, (atom_number, frac_pos, adp) in enumerate(zip(atoms, fractional_positions, adps)):
            element = ase_atoms[i].symbol
            assert atom_number == ase_atoms[i].number
            if element not in element_count:
                element_count[element] = 0
            element_count[element] += 1
            label = f"{element}{element_count[element]}"
            u_iso = torch.trace(adp).mean() if element != 'H' else 0
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
            cif_file.write(f"{label} {adp[0,0]} {adp[1,1]} {adp[2,2]} {adp[1,2]} {adp[0,2]} {adp[0,1]}\n")

def main():
    model = create_model()
    st.success("Model successfully loaded.")
    st.title("CartNet ADP Prediction")
    
    uploaded_file = st.file_uploader("Upload a CIF file", type=["cif"], accept_multiple_files=False)
    # uploaded_file = "ABABEM.cif"
    if uploaded_file is not None:
        st.info("Uploaded file: " + uploaded_file.name)
        st.info(uploaded_file)
        try:
            filename = str(uploaded_file.name)
            # Read the CIF file using ASE
            atoms = read(uploaded_file, format="cif")
            st.info(atoms)
            st.success("CIF file successfully read using ASE.")
            st.info(uploaded_file)
            cif = ReadCif(uploaded_file)
            cif_data = cif.first_block()
            if "_diffrn_ambient_temperature" in cif_data.keys():
                temperature = float(cif_data["_diffrn_ambient_temperature"])
            else:
                raise ValueError("Temperature not found in the CIF file. \
                                    Please provide a temperature in the field _diffrn_ambient_temperature from the CIF file.")
            

            data = Data()
            data.x = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32)
            data.pos = torch.tensor(atoms.positions, dtype=torch.float32)
            data.temperature_og = torch.tensor([temperature], dtype=torch.float32)
            data.temperature = (data.temperature_og - MEAN_TEMP) / STD_TEMP
            data.cell = torch.tensor(atoms.cell.array, dtype=torch.float32).unsqueeze(0)

            data.pbc = torch.tensor([True, True, True])
            data.natoms = len(atoms)
            batch = Batch.from_data_list([data])

            edge_index, _, _, edge_attr = radius_graph_pbc(batch, 5.0, 64)
            data.cart_dist = torch.norm(edge_attr, dim=-1)
            data.cart_dir = torch.nn.functional.normalize(edge_attr, dim=-1)
            data.edge_index = edge_index
            data.non_H_mask = data.x != 1
            delattr(data, "pbc")
            delattr(data, "natoms")
            batch = Batch.from_data_list([data])

            st.success("Torch graph successfully created.")

            process_data(batch, model)
            
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
            os.remove(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred while reading the CIF file: {e}")

if __name__ == "__main__":
    main()
