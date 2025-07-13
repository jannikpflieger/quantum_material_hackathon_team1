from tangelo import SecondQuantizedMolecule

H2 = [('Ni', (0, 0, 0)),('H', (0, 0, 0.74137727))]
mol_H2 = SecondQuantizedMolecule(H2, q=0, spin=1, basis="sto-3g")


print(f"{mol_H2.n_active_mos} active molecular orbitals")
print(f"{mol_H2.n_active_electrons} active electrons")

from tangelo.algorithms import VQESolver

vqe_options = {"molecule": mol_H2}
vqe_solver = VQESolver(vqe_options)
vars(vqe_solver)

vqe_solver.build()
vars(vqe_solver)

print(f"Variational parameters: {vqe_solver.ansatz.var_params}\n")
print(vqe_solver.ansatz.circuit)

energy_vqe = vqe_solver.simulate()
print(f"\nOptimal energy: \t {energy_vqe}")
print(f"Optimal parameters: \t {vqe_solver.optimal_var_params}")

from tangelo.algorithms import FCISolver, CCSDSolver
fci_solver = FCISolver(mol_H2)
energy_fci = fci_solver.simulate()
ccsd_solver = CCSDSolver(mol_H2)
energy_ccsd = ccsd_solver.simulate()
print(f"FCI energy: \t {energy_fci}")
print(f"CCSD energy: \t {energy_ccsd}")
print(f"VQE energy: \t {energy_vqe}")



# Compare energies associated to different variational parameters
energy = vqe_solver.energy_estimation("ones")
print(f"{energy:.7f} (params = {vqe_solver.ansatz.var_params})")

energy = vqe_solver.energy_estimation("MP2")
print(f"{energy:.7f} (params = {vqe_solver.ansatz.var_params})")

energy = vqe_solver.energy_estimation(vqe_solver.optimal_var_params)
print(f"{energy:.7f} (params = {vqe_solver.ansatz.var_params})")

# You can retrieve the circuit corresponding to the last parameters you have used
optimal_circuit = vqe_solver.ansatz.circuit


resources = vqe_solver.get_resources()
print(resources)