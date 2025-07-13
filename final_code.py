# ─────────────────────────────────────────────────────────────────────────────
# General VQE adsorption‐energy script for M–H⁺ on any element M
# Change just ELEMENT and SPIN to run for Cu, Pt, Fe, etc.
# ─────────────────────────────────────────────────────────────────────────────

from pyscf import gto, scf
from openfermion import MolecularData, jordan_wigner
import pennylane as qml
from pennylane import Identity, PauliX, PauliY, PauliZ
import numpy as np

# ───────────── USER INPUT ─────────────
ELEMENT = "Ag"        # e.g. "Cu", "Fe", "Ir", "Au", ...
SPIN    = 1           # 2S = spin multiplicity − 1; for Pt use 2, Cu use 1, Ni use 0, etc.
# ────────────────────────────────────────

# Common basis/ecp for all metals
BASIS = {ELEMENT: "lanl2dz", "H": "sto-3g"}
ECP   = {ELEMENT: "lanl2dz"}

def build_qubit_op(geom, basis, ecp=None, charge=0, spin=0):
    """Return (QubitOperator, nuclear_repulsion)."""
    mol = gto.M(atom=geom, basis=basis, ecp=ecp or {},
                charge=charge, spin=spin, verbose=0)
    mf = scf.RHF(mol).run()
    h1 = mf.get_hcore()
    g2 = mol.intor("int2e")
    e_nuc = mol.energy_nuc()

    # Freeze down to ≤8 spatial orbitals
    n_orb = h1.shape[0]
    na    = min(8, n_orb)
    idx   = list(range(n_orb - na, n_orb))
    h1 = h1[np.ix_(idx, idx)]
    g2 = g2[np.ix_(idx, idx, idx, idx)]

    md = MolecularData(geometry=[L.split() for L in geom],
                       basis="frozen", charge=charge, multiplicity=1)
    md.one_body_integrals = h1
    md.two_body_integrals = g2.transpose((0,2,1,3))
    md.nuclear_repulsion  = e_nuc
    md.n_orbitals         = na
    md.n_electrons        = mol.nelectron - 2*(n_orb - na)

    ferm_h   = md.get_molecular_hamiltonian()
    qubit_op = jordan_wigner(ferm_h)
    return qubit_op, e_nuc

def vqe_energy(qubit_op, e_nuc):
    """Run VQE and return total energy (electronic + nuclear)."""
    coeffs, ops, max_wire = [], [], -1
    for pstr, coeff in qubit_op.terms.items():
        c = float(np.real(coeff))
        if not pstr:
            ops.append(Identity(0)); coeffs.append(c); max_wire = max(max_wire, 0)
        else:
            term = None
            for w, p in pstr:
                max_wire = max(max_wire, w)
                gate = {"X": PauliX, "Y": PauliY, "Z": PauliZ}[p]
                term = gate(wires=w) if term is None else term @ gate(wires=w)
            ops.append(term); coeffs.append(c)

    H = qml.Hamiltonian(coeffs, ops)
    n_qubits = max_wire + 1
    print(f"{ELEMENT}: using {n_qubits} qubits")

    dev = qml.device("lightning.qubit", wires=n_qubits)
    @qml.qnode(dev)
    def circuit(params):
        qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
        return qml.expval(H)

    opt    = qml.GradientDescentOptimizer(stepsize=0.2)
    params = np.random.normal(0, 0.1, (1, n_qubits, 3))
    for _ in range(50):
        params, e_elec = opt.step_and_cost(circuit, params)

    return e_elec + e_nuc

# ─────────────────────────────────────────────────────────────────────────────
# 1) E(M) for bare atom
geom_M   = [f"{ELEMENT} 0 0 0"]
qop_M, e_nuc_M = build_qubit_op(geom_M, BASIS, ECP, charge=0, spin=SPIN)
E_M = vqe_energy(qop_M, e_nuc_M); print(f"E({ELEMENT}) = {E_M:.6f} Ha")

# 2) E(H2) → E(H⁺+e⁻)=½E(H2)
geom_H2  = ["H 0 0 0", "H 0 0 0.74"]
qop_H2, e_nuc_H2 = build_qubit_op(geom_H2, {"H":"sto-3g"}, None, charge=0, spin=0)
E_H2 = vqe_energy(qop_H2, e_nuc_H2); E_Hp = 0.5*E_H2
print(f"E(H2) = {E_H2:.6f} Ha, E(H⁺+e⁻)=½E(H2) = {E_Hp:.6f} Ha")

# 3) E(M–H⁺)
geom_MH  = [f"{ELEMENT} 0 0 0", "H 0 0 1.6"]
qop_MH, e_nuc_MH = build_qubit_op(geom_MH, BASIS, ECP, charge=1, spin=SPIN)
E_MH = vqe_energy(qop_MH, e_nuc_MH); print(f"E({ELEMENT}–H⁺) = {E_MH:.6f} Ha")

# 4) ΔE_ads
ΔE = E_MH - E_M - E_Hp
print(f"ΔE_ads = {ΔE:.6f} Ha = {ΔE*27.2114:.3f} eV")