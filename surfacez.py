import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import os

# Parámetros
A = 0.5
lamb = 0.3146
d = 0.4
b = 1.414215 * (d / 2)
bd = 0.4
gamma = 10
ln = 20
mass = 0.2
diameter = 0.4
xlo, xhi = 0, 12
ylo, yhi = 0, 12
zlo, zhi = (-ln * d) + d, 0.2
w = 2 * np.pi / lamb
diameter_formatted = f"{diameter:.10e}"
mass_formatted = f"{mass:.10e}"
id_offset =-5


# Crear una línea recta en el plano X-Z
def create_equidistant_line(bd, xlo, xhi):
    x_points = np.arange(xlo, xhi + bd, bd)
    z_points = np.zeros_like(x_points)
    return np.array(x_points), np.array(z_points)

x, z = create_equidistant_line(bd, xlo, xhi)

# Generar una capa a partir de la línea
def generate_layer(x, z, y_offset):
    y = np.full_like(x, y_offset)
    return np.vstack((x, y, z)).T

# Rotar y reposicionar una capa
def rotate_and_reposition(layer, angle, xlo, ylo):
    rotation = R.from_euler('z', angle, degrees=True)
    center = np.array([xlo / 2, ylo / 2, 0])
    shifted_positions = layer - center
    rotated_positions = rotation.apply(shifted_positions)
    rotated_positions += center
    return rotated_positions

# Plot opcional en 3D
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

# Cálculo de distancia
def calculate_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

# Generar posiciones en el eje Y
positions_y = []
ids = []
current_id = 1

for y_offset in np.arange(-yhi, yhi + d, d):
    layer_y = generate_layer(x, z.copy(), y_offset)
    positions_y.append(layer_y)
    ids.extend([current_id] * len(layer_y))
    current_id += 1

positions_y = np.vstack(positions_y)
ids = np.array(ids)

# Generar posiciones finales con desplazamientos en Z
final_positions = []
final_ids = []
z_offset = zlo
current_id = 0

while z_offset < zhi:
    x_offset = 0  # (current_id % 2) * bd / 2
    y_offset = 0  # (current_id % 2) * d / 2

    angle = gamma * (current_id)  # Usamos el índice de la capa para definir el ángulo

    rotated_layer = rotate_and_reposition(positions_y, angle, xhi, yhi)
    rotated_layer[:, 2] += z_offset
    rotated_layer[:, 0] += x_offset
    rotated_layer[:, 1] += y_offset

    final_positions.append(rotated_layer)
    final_ids.extend(ids + (current_id - 1))
    current_id += id_offset  # Incrementa el id de forma continua
    z_offset += d  # Avanza al siguiente valor de z

final_positions = np.vstack(final_positions)
final_ids = np.array(final_ids)
x = final_positions[:, 0]  
y = final_positions[:, 1]  
z = final_positions[:, 2]  
# Calcular el nuevo array z basado en la ecuación
z_new = z + 0.5 * np.sin(2 * x) * np.sin(2 * y)

# Sumar el nuevo array z_new a la columna z del array existente
final_positions[:, 2] += z_new  # Columna 2 corresponde a z
# Filtrar posiciones
filter = (
    (final_positions[:, 0] >= 3.3) & (final_positions[:, 0] <= 8.7) &
    (final_positions[:, 1] >= 2.27) & (final_positions[:, 1] <= 9.73) &
    (final_positions[:, 2] >= zlo + 0.4 ) & (final_positions[:, 2] <= zhi-0.37 )
)

final_positions = final_positions[filter]
final_ids = final_ids[filter]
unique_ids, counts = np.unique(final_ids, return_counts=True)
#print(counts)


def check_particle_intersections(positions, diameter):
    intersecting_pairs = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distance = calculate_distance(positions[i], positions[j])
            if distance < diameter:  # Si la distancia es menor al diámetro, hay intersección
                intersecting_pairs.append((i, j))
    return intersecting_pairs

intersections = check_particle_intersections(final_positions, diameter)

if intersections:
    print(f"Se encontraron {len(intersections)} pares de partículas que se intersectan:")
    #for pair in intersections:
        #print(f"Partículas {pair[0]} y {pair[1]} están demasiado cerca.")
    # Aquí puedes decidir eliminar partículas problemáticas o ajustar posiciones
    # Por ejemplo:
    # final_positions = resolver_intersecciones(final_positions, intersections, diameter)
else:
    print("No se encontraron partículas que se intersecten.")
#diam = np.linspace(diameter,diameter,len(final_positions))
#print(diam)

types = np.full(len(final_positions), 1)
#print(types)
aid = np.arange(1, len(final_positions) + 1)
atoms = np.hstack((aid[:, np.newaxis],
                    types[:, np.newaxis],
                   #  final_ids[:, np.newaxis],
                     final_positions))
type = np.hstack((aid[:,np.newaxis],final_ids[:, np.newaxis]))

print("Total number of points:", len(atoms))
print(atoms)
# Crear bonds de tipo 1 (consecutivos en moléculas)
output = []
k = 1
k1 = 1
unique_ids, counts = np.unique(final_ids, return_counts=True)

for count in counts:
    i = 1
    while i < count:
        if k >= len(final_positions):
            break
        distance = calculate_distance(final_positions[k-1], final_positions[k])  # Calcular distancia entre átomos consecutivos
        if distance <= bd+0.16:  # Solo crear bond si la distancia es menor o igual a b
            output.append([k1, 1, k, k + 1])
        k += 1
        k1 += 1
        i += 0
    k += 1

Bonds = np.array(output)

# Crear bonds de tipo 2 (entre moléculas diferentes en la misma capa)
bonds_type2 = []
bond_id = len(Bonds) + 1

for z_value in np.unique(final_positions[:, 2]):
    indices_in_layer = np.where(np.isclose(final_positions[:, 2], z_value, atol=1e-6))[0]
    positions_in_layer = final_positions[indices_in_layer]
    ids_in_layer = final_ids[indices_in_layer]

    for i in range(len(positions_in_layer)):
        for j in range(i + 1, len(positions_in_layer)):
            if ids_in_layer[i] != ids_in_layer[j]:  # Moleculas distintas
                distance = calculate_distance(positions_in_layer[i], positions_in_layer[j])
                if distance <= d+0.3:  # Condición de distancia
                    bond = [bond_id, 2, indices_in_layer[i] + 1, indices_in_layer[j] + 1]
                    bonds_type2.append(bond)
                    bond_id += 1

bonds_type2 = np.array(bonds_type2)
if len(bonds_type2) > 0:
    Bonds = np.vstack((Bonds, bonds_type2))
bonds_type3 = []  # Bonds entre diferentes capas
bond_id = len(Bonds) + 1  # Continuar desde el último bond

# Iterar sobre todos los valores únicos de z para crear bonds entre partículas en diferentes capas
for i, z_value_i in enumerate(np.unique(final_positions[:, 2])):
    # Identificar las partículas en la capa con valor z_value_i
    indices_in_layer_i = np.where(np.isclose(final_positions[:, 2], z_value_i, atol=1e-6))[0]
    positions_in_layer_i = final_positions[indices_in_layer_i]
    ids_in_layer_i = final_ids[indices_in_layer_i]

    # Comparar con las partículas de las capas superiores
    for j, z_value_j in enumerate(np.unique(final_positions[:, 2])):
        if i >= j:  # Para evitar hacer la misma comparación en reversa (i == j se ignora)
            continue

        # Identificar las partículas en la capa con valor z_value_j
        indices_in_layer_j = np.where(np.isclose(final_positions[:, 2], z_value_j, atol=1e-6))[0]
        positions_in_layer_j = final_positions[indices_in_layer_j]
        ids_in_layer_j = final_ids[indices_in_layer_j]

        # Comparar partículas de las dos capas
        for k in range(len(positions_in_layer_i)):
            for l in range(len(positions_in_layer_j)):
                # Calcular distancia entre las partículas de diferentes capas
                distance = calculate_distance(positions_in_layer_i[k], positions_in_layer_j[l])
                if distance <= d+0.3:  # Si la distancia es menor o igual a 0.5
                    bond = [bond_id, 2, indices_in_layer_i[k] + 1, indices_in_layer_j[l] + 1]
                    bonds_type3.append(bond)
                    bond_id += 1

# Convertir bonds a array y añadirlos a la lista de bonds generales
bonds_type3 = np.array(bonds_type3)
if len(bonds_type3) > 0:
    Bonds = np.vstack((Bonds, bonds_type3))


# Guardar los datos
path = 'D:/Simaf/Mantis/Indent Gera'
filename = os.path.join(path, f'Straigth_bouligand_{gamma}_{ln}.dat')
os.makedirs(path, exist_ok=True)

formats_atoms = ['%10d', 
                 '%10d',
                 #  '%10d',
                  '%21.12e', '%20.12e', '%20.12e']
formats_bonds = ['%10d', '%10d', '%7d', '%7d']
formats_angles = ['%10d', '%10d', '%7d', '%7d', '%7d']
formats_data = ['%10d', '%10d']
formats_type = ['%10d', '%10d']
max_id = np.max(final_ids)
masses = np.arange(1, max_id + 1)
mass_array = np.full(#(len(masses))
    1, mass_formatted)
diameters = np.arange(1, max_id + 1)
diameter_array = np.full(#(len(diameters))
                         1, diameter_formatted)
#print(diameter_array)

lon1 = len(atoms)
lon2 = len(Bonds)
#lon3 = len(Angles)

data = np.array([[lon1, 'atoms'],
                 [lon2, 'bonds'],
                # [lon3, 'angles']
                 ])
types = np.array([[len(np.unique(types)), 'atom types'],
                  [3, 'bond types'],
                 # [1, 'angle types']
                  ])
limits = np.array([[xlo-1 , xhi + 2, 'xlo', 'xhi'],
                   [ylo - 2, yhi + 2, 'ylo', 'yhi'],
                   [zlo - 3, zhi + 2, 'zlo', 'zhi']], dtype=object)

def format_sci(val):
    if isinstance(val, (int, float)):
        return f'{val:.10e}'
    return val

limits_transformed = np.array([[format_sci(val) for val in row] for row in limits], dtype=object)

with open(filename, 'w') as f:
    f.write("\n")
    for number, label in data:
        f.write(f"{int(number):9d}  {label:10s}\n")
    f.write("\n")
    for number, label in types:
        f.write(f"{int(number):9d}  {label:10s}\n")
    f.write("\n")
    for nlo, nhi, lo, hi in limits_transformed:
        f.write(f"{nlo:10s} {nhi:10s} {lo:3} {hi:3}\n")
    f.write("\n")
 
   
    f.write("Atoms\n\n")
    np.savetxt(f, atoms, fmt=formats_atoms)
    #f.write("Types\n\n")
    #np.savetxt(f, type, fmt=formats_type)
    f.write("\n")
    
    f.write("diameters\n\n")
    for number, label in zip(diameters, diameter_array):
        f.write(f"{int(number):1d}  {label:10s}\n")
    f.write("\n")
    f.write("Masses\n\n")
    for number, label in zip(masses, mass_array):
        f.write(f"{int(number):1d}  {label:10s}\n")
    f.write("\n")
    f.write("Bonds\n\n")
    np.savetxt(f, Bonds, fmt=formats_bonds)
    f.write("\n")
    #f.write("Angles\n\n")
    #np.savetxt(f, Angles, fmt=formats_angles)
    #f.write("\n")

print(f"Arrays saved in {filename}")

plot_3d_sine_wave(final_positions, final_ids)
