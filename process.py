import torch
from ase.io import write
from ase import Atoms
import gc

@torch.no_grad()
def process_data(batch, model, output_file="output.cif"):
    atoms = batch.x.numpy().astype(int)  # Atomic numbers
    positions = batch.pos.numpy()  # Atomic positions
    cell = batch.cell.squeeze(0).numpy()  # Cell parameters
    temperature = batch.temperature_og.numpy()[0]


    adps = model(batch)
    
    # Convert Ucart to Ucif
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
    write(output_file, ase_atoms)

    with open(output_file, 'r') as file:
        lines = file.readlines()

    # Find the line where "loop_" appears and remove lines from there to the end
    for i, line in enumerate(lines):
        if line.strip().startswith('loop_'):
            lines = lines[:i]
            break

    # Write the modified lines to a new output file
    with open(output_file, 'w') as file:
        file.writelines(lines)

    # Manually append positions and ADPs to the CIF file
    with open(output_file, 'a') as cif_file:

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
