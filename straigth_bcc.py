import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import os

# Parámetros
A = 0.125  # Amplitud de la onda senoidal
lamb = 1  # Longitud de onda
d = 0.05  # Distancia entre capas a lo largo de los ejes Y y Z
bd = 0.01  # Distancia entre puntos
gamma = 0  # Ángulo de rotación en grados por capa
ln = 5 # Número de capas
mass = 0.09091965489  # Valor de la masa
xlo, xhi = -1, 5  # Límites del eje X
ylo, yhi = 0, 2  # Límites del eje Y
zlo, zhi = 0, ln * d  # Límite superior a lo largo del eje Z
w = 2 * np.pi / lamb  # Frecuencia angular
mass_formatted = f"{mass:.10e}"  # Valor de la masa formateado
id_offset = 39
def create_equidistant_line(bd, xlo, xhi):
    x_points = np.arange(xlo, xhi + bd, bd)
    z_points = np.zeros_like(x_points)  # Línea recta a lo largo del eje X
    return np.array(x_points), np.array(z_points)

x, z = create_equidistant_line(bd, xlo, xhi)

def generate_layer(x, z, y_offset):
    y = np.full_like(x, y_offset)
    return np.vstack((x, y, z)).T

def rotate_and_reposition(layer, angle, xlo, ylo):
    rotation = R.from_euler('z', angle, degrees=True)
    center = np.array([xlo / 2, ylo / 2, 0])
    shifted_positions = layer - center
    rotated_positions = rotation.apply(shifted_positions)
    rotated_positions += center
    return rotated_positions

def plot_3d_sine_wave(positions, ids):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    unique_ids = np.unique(ids)
    
    for id in unique_ids:
        mask = ids == id
        ax.plot(x[mask], y[mask], z[mask], marker='o', markersize=1, linestyle='None')
        
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.show()

def calculate_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

positions_y = []
ids = []
current_id = 1  # Inicializar en 1 para empezar los IDs desde 1

# Generar las capas con desplazamientos en X e Y
for y_offset in np.arange(-yhi, yhi + d, d):
    layer_y = generate_layer(x, z.copy(), y_offset)
    positions_y.append(layer_y)
    ids.extend([current_id] * len(layer_y))
    current_id += 1

positions_y = np.vstack(positions_y)
ids = np.array(ids)

final_positions = []
final_ids = []
z_offset = zlo
current_id = 0  # Reiniciar ID para las capas en Z

# Loop para aumentar las posiciones en X y Y en bd/2 y d/2
while z_offset < zhi:
    # Calcular el desplazamiento para la capa actual
    x_offset = (current_id % 2)* bd / 2
    y_offset = (current_id % 2) * d / 2  # Desplazamiento alternado para Y
    
    rotated_layer = rotate_and_reposition(positions_y, gamma * (z_offset // d), xhi, yhi)
    rotated_layer[:, 2] += z_offset
    rotated_layer[:, 0] += x_offset  # Aumentar X
    rotated_layer[:, 1] += y_offset  # Aumentar Y

    final_positions.append(rotated_layer)
    final_ids.extend(ids + (current_id - 1))  # Asignar IDs consecutivos
    current_id += id_offset  # Incrementar el ID para la próxima capa
    z_offset += d

final_positions = np.vstack(final_positions)
final_ids = np.array(final_ids)

# Asegurar que los IDs sean consecutivos desde 1
final_ids = final_ids - id_offset

# Filtrar posiciones finales
filter = (
    (final_positions[:, 0] >= 0) & (final_positions[:, 0] <= xhi) & 
    (final_positions[:, 1] >= ylo) & (final_positions[:, 1] <= yhi) & 
    (final_positions[:, 2] >= zlo) & (final_positions[:, 2] <= zhi + 1)
)

final_positions = final_positions[filter]
final_ids = final_ids[filter]
unique_ids, counts = np.unique(final_ids, return_counts=True)
print(counts)

types = np.full(len(final_positions), 1)
aid = np.arange(1, len(final_positions) + 1)
atoms = np.hstack((aid[:, np.newaxis], final_ids[:, np.newaxis], final_ids[:, np.newaxis], final_positions))

print("Total number of points:", len(atoms))
print(atoms)
k = 1
k1 = 1
output = []

for count in counts:
    i = 1
    while i < count:
        if k >= len(final_positions):
            break
        output.append([k1, 1, k, k + 1])
        k += 1
        k1 += 1
        i += 1
    k += 1

Bonds = np.array(output)

n_atoms = len(final_positions)
filtered_bonds = []

for bond in Bonds:
    i, j = bond[2] - 1, bond[3] - 1
    if i < n_atoms and j < n_atoms:
        p1 = final_positions[i]
        p2 = final_positions[j]
        distance = calculate_distance(p1, p2)
        if distance <= bd + 0.000000001:
            filtered_bonds.append(bond)

Bonds = np.array(filtered_bonds)

angles = []
angle_id = 1

for i in range(len(Bonds) - 1):
    bond1 = Bonds[i]
    bond2 = Bonds[i + 1]
    if bond1[3] == bond2[2]:
        angles.append([angle_id, 1, bond1[2], bond1[3], bond2[3]])
        angle_id += 1

Angles = np.array(angles)

# Guardar los datos
path = 'G:/Simaf/Mantis/Indent Gera'
filename = os.path.join(path, f'Sin_bouligand_{gamma}_{ln}.dat')
os.makedirs(path, exist_ok=True)

formats_atoms = ['%10d', '%10d', '%10d', '%21.12e', '%20.12e', '%20.12e']
formats_bonds = ['%10d', '%10d', '%7d', '%7d']
formats_angles = ['%10d', '%10d', '%7d', '%7d', '%7d']
formats_data = ['%10d', '%10d']
max_id = np.max(final_ids)
masses = np.arange(1, max_id + 1)
mass_array = np.full((len(masses)), mass_formatted)

lon1 = len(atoms)
lon2 = len(Bonds)
lon3 = len(Angles)

data = np.array([[lon1, 'atoms'],
                 [lon2, 'bonds'],
                 [lon3, 'angles']
                 ])
types = np.array([[len(np.unique(final_ids)), 'atom types'],
                  [1, 'bond types'],
                  [1, 'angle types']
                  ])
limits = np.array([[xlo , xhi + 1, 'xlo', 'xhi'],
                   [ylo - 1, yhi + 1, 'ylo', 'yhi'],
                   [zlo - 1, zhi + 1, 'zlo', 'zhi']], dtype=object)

def format_sci(val):
    if isinstance(val, (int, float)):
        return f'{val:.10e}'
    return val

limits_transformed = np.array([[format_sci(val) for val in row] for row in limits], dtype=object)

with open(filename, 'w') as f:
    f.write("\n")
    for number, label in data:
        f.write(f"    {int(number):9d}  {label:10s}\n")
    f.write("\n")
    for number, label in types:
        f.write(f"    {int(number):9d}  {label:10s}\n")
    f.write("\n")
    for nlo, nhi, lo, hi in limits_transformed:
        f.write(f"   {nlo:10s} {nhi:10s} {lo:3} {hi:3}\n")
    f.write("\n")
    f.write("Masses\n\n")
    for number, label in zip(masses, mass_array):
        f.write(f"        {int(number):1d}           {label:10s}\n")
    f.write("\n")
    f.write("Atoms\n\n")
    np.savetxt(f, atoms, fmt=formats_atoms)
    f.write("\n")
    f.write("Bonds\n\n")
    np.savetxt(f, Bonds, fmt=formats_bonds)
    f.write("\n")
    f.write("Angles\n\n")
    np.savetxt(f, Angles, fmt=formats_angles)
    f.write("\n")

print(f"Arrays saved in {filename}")

plot_3d_sine_wave(final_positions, final_ids)
