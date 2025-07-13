import pennylane as qml
from jax import numpy as np
import jax
import optax
jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_enable_x64', True)


ANG_TO_BOHR = 1.0 / 0.52917721092
import numpy as np

symbols = ["H", "O", "H"]
coordinates = np.array([[-0.0399, -0.0038, 0.0], [1.5780, 0.8540, 0.0], [2.7909, -0.5159, 0.0]])
charge = 0
multiplicity = 1
basis_set = "sto-3g"

electrons = 10
orbitals = 7
core, active = qml.qchem.active_space(electrons, orbitals, active_electrons=4, active_orbitals=4)

molecule = qml.qchem.Molecule(
    symbols,
    coordinates,
    charge=charge,
    mult=multiplicity,
    basis_name=basis_set
)

H, qubits = qml.qchem.molecular_hamiltonian(
    molecule,
    active_electrons=4,
    active_orbitals=4,
)
print(f"Number of qubits = {qubits}")
print(H)



#mol = qml.qchem.Molecule(symbols, geometry,charges=0,basis="sto-3g")
#args = [mol.coordinates]
#e_nuc = qml.qchem.nuclear_energy(mol.nuclear_charges, mol.coordinates)(*args)  # ← NEW
#print(f"Nuclear repulsion energy = {e_nuc[0]}")     # ← NEW

dev = qml.device("lightning.qubit", wires=qubits)
electrons = 10  # Oxygen (8) + 2×Hydrogen (1 each)
hf = qml.qchem.hf_state(4,8)
#hf = [int(x) for x in hf_arr]
print(f"Hartree–Fock bitstring = {hf}")

electrons = 4
orbitals = 4

singles, doubles = qml.qchem.excitations(electrons, 8)
s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
n_params = len(s_wires) + len(d_wires)
print(f"Number of excitation amplitudes = {n_params}")

@qml.qnode(dev, wires=range(qubits),interface="jax")
def circuit(params):
    # prepare HF reference
    qml.BasisState(hf, wires=range(qubits))
    # apply UCCSD ansatz
    qml.UCCSD(
        params,
        wires=range(qubits),
        s_wires=s_wires,
        d_wires=d_wires,
        init_state=hf
    )
    return qml.expval(H)

def cost_fn(param):
    return circuit(param)


max_iterations = 200
conv_tol = 1e-6
opt = optax.adam(learning_rate=0.1)


theta = np.zeros(n_params)
opt_state = opt.init(theta)
print("Size of theta = ", theta.shape)

opt_state = opt.init(theta)

# record the VQE energy
energy = []
angle  = []

# VQE loop
for n in range(max_iterations):
    e = cost_fn(theta)
    energy.append(e)
    angle.append(theta)

    grad = jax.grad(cost_fn)(theta)
    updates, opt_state = opt.update(grad, opt_state)
    theta = optax.apply_updates(theta, updates)

    if n % 10 == 0:
        print(f"Step {n:4d}  E = {e:.8f} Ha")

    if n > 0 and abs(energy[-1] - energy[-2]) < conv_tol:
        break

print(f"\nConverged in {n} steps")
print(f"Ground-state energy ≈ {energy[-1]:.8f} Ha")
#E_tot = energy[-1] + e_nuc[0]  # ← NEW: add the constant shift
print(f"Electronic energy     ≈ {energy[-1]:.8f} Ha")
#print(f"Total ground-state    ≈ {E_tot:.8f} Ha")  # ← NEW
