import argparse
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
def process_cif(input_file, output_file):
    model = create_model()
    
    try:
        # Read the CIF file using ASE
        atoms = read(input_file, format="cif")
        cif = ReadCif(input_file)
        cif_data = cif.first_block()
        if "_diffrn_ambient_temperature" in cif_data.keys():
            temperature = float(cif_data["_diffrn_ambient_temperature"])
        else:
            raise ValueError("Temperature not found in the CIF file. \
                                Please provide a temperature in the field _diffrn_ambient_temperature from the CIF file.")
        
        data = Data()
        data.x = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32)

        if len(atoms.positions) > 300:
            raise ValueError("This implementation is not optimized for large systems. For large systems, please use the local version.")
        
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

        process_data(batch, model, output_file)

        gc.collect()
    except Exception as e:
        print(f"An error occurred while processing the CIF file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a CIF file and output the result.")
    parser.add_argument("input_file", type=str, help="Path to the input CIF file.")
    parser.add_argument("output_file", type=str, help="Path to the output CIF file.")
    args = parser.parse_args()

    process_cif(args.input_file, args.output_file) 