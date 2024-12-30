import numpy as np
import os
def parse_dump_file(dump_file):
    atoms = []
    box_bounds = []
    nbonds = []

    try:
        with open(dump_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Dump file '{dump_file}' not found.")
        return None, None, None, None

    in_atoms_section = False
    atom_columns = None

    line_iterator = iter(lines)
    for line in line_iterator:
        if "ITEM: BOX BOUNDS" in line:
            # Parse the next three lines for box bounds
            for _ in range(3):
                bounds = next(line_iterator).split()
                if len(bounds) == 2:
                    box_bounds.append([float(bounds[0]), float(bounds[1])])
        elif "ITEM: ATOMS" in line:
            # Start parsing atoms
            in_atoms_section = True
            atom_columns = line.split()[2:]  # Extract column names
            continue
        elif "ITEM:" in line:
            # End atoms section on new ITEM line
            in_atoms_section = False
            continue

        if in_atoms_section:
            split_line = line.split()
            if len(split_line) == len(atom_columns):
                atoms.append([float(x) if i > 1 else int(x)  # id and type are integers
                              for i, x in enumerate(split_line)])
                nbonds.append(float(split_line[-2]))  # Extract the c_nbond column (second to last)

    # Convert to NumPy arrays
    if not atoms or not box_bounds:
        print(f"Error: Failed to parse atoms or box bounds from '{dump_file}'.")
        return None, None, None, None

    atoms = np.array(atoms)
    box_bounds = np.array(box_bounds)
    return atom_columns, atoms, box_bounds, nbonds

def parse_bond_file(bond_file):
    bonds = []

    try:
        with open(bond_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Bond file '{bond_file}' not found.")
        return None

    in_bonds_section = False
    for line in lines:
        if "ITEM: ENTRIES" in line:
            in_bonds_section = True
            continue
        elif "ITEM:" in line:
            in_bonds_section = False
            continue

        if in_bonds_section:
            bonds.append([int(x) for x in line.split()])

    return np.array(bonds)

def write_data_file(output_file, atom_columns, atoms, box_bounds, bonds, nbonds):
    if atoms is None or box_bounds is None or bonds is None or nbonds is None:
        print("Error: Missing data. Data file generation aborted.")
        return

    with open(output_file, 'w') as f:
        # Header with LAMMPS version, timestep, and units
        f.write("LAMMPS data file via write_data, version 2 Aug 2023, timestep = 0, units = cgs\n\n")

        # Meta information
        f.write(f"{len(atoms)} atoms\n")
        f.write(f"{len(set(atoms[:, 1]))} atom types\n")  # Unique atom types
        f.write(f"{len(bonds)} bonds\n")
        f.write(f"{len(set(bonds[:, 1]))} bond types\n\n")  # Unique bond types

        # Box dimensions
        f.write(f"{box_bounds[0][0]} {box_bounds[0][1]} xlo xhi\n")
        f.write(f"{box_bounds[1][0]} {box_bounds[1][1]} ylo yhi\n")
        f.write(f"{box_bounds[2][0]} {box_bounds[2][1]} zlo zhi\n\n")

        # Atoms section with a comment
        f.write("Atoms # bpm/sphere\n\n")

        # Write atoms with the additional columns (molecule-ID, lineflag, density, nx, ny, nz)
        for i, atom in enumerate(atoms):
            atom_id = int(atom[0])
            atom_type = int(atom[1])
            x, y, z = atom[3], atom[4], atom[5]

            # Set additional columns with default values
            molecule_id = 0
            lineflag = nbonds[i]  # Use the c_nbond value from the dump file
            density = 2.2
            nx, ny, nz = 0, 0, 0  # You can adjust nx, ny, nz if needed

            # Write each atom's data with the new format
            f.write(f"{atom_id} {molecule_id} {atom_type} {lineflag} {density} {x} {y} {z} {nx} {ny} {nz}\n")

        # Bonds section
        f.write("\nBonds\n\n")
        for bond in bonds:
            bond_id = int(bond[0])  # Ensure bond ID is an integer
            bond_type = int(bond[1])  # Ensure bond type is an integer
            atom1_id = int(bond[2])  # Atom 1 ID should be an integer
            atom2_id = int(bond[3])  # Atom 2 ID should be an integer

            # Write each bond's data with the correct formatting
            f.write(f"{bond_id} {bond_type} {atom1_id} {atom2_id}\n")

# Main script


PATH="/media/gmora/Data/Mora/impact/Purdue/3PB/"
#os.mkdir(f"{PATH}data")
for n in range(0,1000001,5000):
    dump_file = f"{PATH}dump/dump.lammps.{n}"
    bond_file = f"{PATH}bond/bond.lammps.{n}"
    output_file = f"{PATH}/data/data.lammps.{n}"

    atom_columns, atoms, box_bounds, nbonds = parse_dump_file(dump_file)
    bonds = parse_bond_file(bond_file)

    write_data_file(output_file, atom_columns, atoms, box_bounds, bonds, nbonds)
