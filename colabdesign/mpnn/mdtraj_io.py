import jax
import numpy as np

from colabdesign.af.alphafold.common import residue_constants
from colabdesign.shared.protein import _np_get_cb

order_aa = {b: a for a, b in residue_constants.restype_order.items()}
aa_order = residue_constants.restype_order


def prep_from_mdtraj(traj, chain=None, **kwargs):
    """
    Extracts features directly from an mdtraj.Trajectory object.
    """
    
    chains_to_process = []
    if chain is None:
        chains_to_process = list(traj.topology.chains)
    else:
        requested_chain_ids = list(chain)
        for c in traj.topology.chains:
            if c.chain_id in requested_chain_ids:
                chains_to_process.append(c)

    all_chains_data = []
    last_res_idx = 0
    full_lengths = []

    for chain_obj in chains_to_process:
        chain_id = chain_obj.chain_id
        atom_indices = [a.index for a in chain_obj.atoms]

        chain_top = traj.topology.subset(atom_indices)
        chain_xyz = traj.xyz[:, atom_indices, :] * 10  # Convert nm to Angstroms
        n_res = chain_top.n_residues

        all_atom_positions = np.zeros((traj.n_frames, n_res, residue_constants.atom_type_num, 3))
        all_atom_mask = np.zeros((n_res, residue_constants.atom_type_num))
        aatype = np.zeros(n_res, dtype=int)
        residue_index = np.zeros(n_res, dtype=int)

        for res_idx, residue in enumerate(chain_top.residues):
            res_name = residue.name
            aatype[res_idx] = residue_constants.resname_to_idx.get(
                res_name, residue_constants.resname_to_idx["UNK"]
            )
            residue_index[res_idx] = residue.resSeq

            for atom in residue.atoms:
                if atom.name in residue_constants.atom_order:
                    atom_type_idx = residue_constants.atom_order[atom.name]
                    chain_atom_index = next(
                        a.index for a in chain_top.atoms if a.serial == atom.serial
                    )
                    all_atom_positions[:,res_idx, atom_type_idx] = chain_xyz[:,chain_atom_index]
                    all_atom_mask[res_idx, atom_type_idx] = 1

        batch = {
            "aatype": aatype,
            "all_atom_positions": all_atom_positions,
            "all_atom_mask": all_atom_mask,
        }

        p, m = batch["all_atom_positions"], batch["all_atom_mask"]
        atom_idx = residue_constants.atom_order
        atoms = {k: p[..., atom_idx[k], :] for k in ["N", "CA", "C",]}

        cb_atoms = _np_get_cb(**atoms, use_jax=False)
        cb_mask = np.prod([m[..., atom_idx[k]] for k in ["N", "CA", "C"]], 0)
        cb_idx = atom_idx["CB"]
        batch["all_atom_positions"][..., cb_idx, :] = np.where(
            m[:, cb_idx, None], p[..., cb_idx, :], cb_atoms
        )
        batch["all_atom_mask"][..., cb_idx] = (m[:, cb_idx] + cb_mask) > 0
        #batch["all_atom_positions"] = np.moveaxis(batch["all_atom_positions"], 0, -1)

        chain_data = {
            "batch": batch,
            "residue_index": residue_index + last_res_idx,
            "chain_id": [chain_id] * n_res,
            "res_indices_original": residue_index,
        }
        all_chains_data.append(chain_data)

        last_res_idx += n_res + 50
        full_lengths.append(n_res)

    if not all_chains_data:
        raise ValueError("No valid chains found or processed from the mdtraj frame.")

    final_batch = jax.tree_util.tree_map(
        lambda *x: np.concatenate(x, 0), *[d.pop("batch") for d in all_chains_data]
    )
    final_residue_index = np.concatenate([d.pop("residue_index") for d in all_chains_data])
    final_idx = {
        "residue": np.concatenate([d.pop("res_indices_original") for d in all_chains_data]),
        "chain": np.concatenate([d.pop("chain_id") for d in all_chains_data]),
    }

    return {
        "batch": final_batch,
        "residue_index": final_residue_index,
        "idx": final_idx,
        "lengths": full_lengths,
    }
