import jax
import jax.numpy as jnp
import numpy as np
import re
import copy
import random
import os
import joblib
from functools import partial

from .modules import RunModel
from .mdtraj_io import prep_from_mdtraj

from scipy.special import softmax, log_softmax

from colabdesign.shared.prep import prep_pos
from colabdesign.shared.utils import Key, copy_dict

# borrow some stuff from AfDesign
from colabdesign.af.prep import prep_pdb
from colabdesign.af.alphafold.common import protein, residue_constants
aa_order = residue_constants.restype_order
order_aa = {b:a for a,b in aa_order.items()}

class mk_mpnn_ensemble_model():
  def __init__(self, model_name="v_48_020",
               backbone_noise=0.0, dropout=0.0,
               seed=None, verbose=False, weights="original", # weights can be set to either original or soluble
               ): 

    # load model
    if weights == "original":
      from .weights import __file__ as mpnn_path
    elif weights == "soluble":
      from .weights_soluble import __file__ as mpnn_path
    else:
      raise ValueError(f'Invalid value {weights} supplied for weights. Value must be either "original" or "soluble".')

    path = os.path.join(os.path.dirname(mpnn_path), f'{model_name}.pkl')
    checkpoint = joblib.load(path)
    config = {'num_letters': 21,
              'node_features': 128,
              'edge_features': 128,
              'hidden_dim': 128,
              'num_encoder_layers': 3,
              'num_decoder_layers': 3,
              'augment_eps': backbone_noise,
              'k_neighbors': checkpoint['num_edges'],
              'dropout': dropout}
    
    self._model = RunModel(config)
    self._model.params = jax.tree_util.tree_map(np.array, checkpoint['model_state_dict'])
    self._setup()
    self.set_seed(seed)

    self._num = 1
    self._inputs = {}
    self._tied_lengths = False

  def prep_inputs(
        self,
        traj=None,
        chain=None,
        homooligomer=False,
        fix_pos=None,
        inverse=False,
        rm_aa=None,
        verbose=False,
        **kwargs,
    ):
        """Get inputs from an MDTraj object."""
        if traj is not None:
            traj = prep_from_mdtraj(traj, chain=chain,)
        else:
            raise ValueError(
                "One of 'mdtraj_frame', 'pdb_filename', or 'pdb_string' must be provided."
            )
        atom_idx = tuple(residue_constants.atom_order[k] for k in ["N", "CA", "C", "O"])
        chain_idx = np.concatenate([[n] * l for n, l in enumerate(traj["lengths"])])
        self._lengths = traj["lengths"]
        L = sum(self._lengths)

        self._inputs = {
            "X": traj["batch"]["all_atom_positions"][:, atom_idx], # fine to keep as is
            "mask": traj["batch"]["all_atom_mask"][:, 1],
            "S": traj["batch"]["aatype"],
            "residue_idx": traj["residue_index"],
            "chain_idx": chain_idx,
            "lengths": np.array(self._lengths),
            "bias": np.zeros((L, 20)),
        }

        if rm_aa is not None:
            for aa in rm_aa.split(","):
                self._inputs["bias"][..., aa_order[aa]] -= 1e6

        if fix_pos is not None:
            p = prep_pos(fix_pos, **traj["idx"])["pos"]
            if inverse:
                p = np.delete(np.arange(L), p)
            self._inputs["fix_pos"] = p
            self._inputs["bias"][p] = 1e7 * np.eye(21)[self._inputs["S"]][p, :20]

        if homooligomer:
            #raise NotImplementedError("'homooligomer=True' not yet implemented")
            assert min(self._lengths) == max(self._lengths)
            self._tied_lengths = True
            self._len = self._lengths[0]
        else:
            self._tied_lengths = False
            self._len = sum(self._lengths)

        self.traj = traj

        if verbose:
            print("lengths", self._lengths)
            if "fix_pos" in self._inputs:
                print("the following positions will be fixed:")
                print(self._inputs["fix_pos"])

  def get_af_inputs(self, af):
    '''get inputs from alphafold model'''

    self._lengths = af._lengths
    self._len = af._len

    self._inputs["residue_idx"] = af._inputs["residue_index"]
    self._inputs["chain_idx"]   = af._inputs["asym_id"]
    self._inputs["lengths"]     = np.array(self._lengths)

    # set bias
    L = sum(self._lengths)
    self._inputs["bias"] = np.zeros((L,20))
    self._inputs["bias"][-af._len:] = af._inputs["bias"]
    
    if "offset" in af._inputs:
      self._inputs["offset"] = af._inputs["offset"]

    if "batch" in af._inputs:
      atom_idx = tuple(residue_constants.atom_order[k] for k in ["N","CA","C","O"])
      batch = af._inputs["batch"]
      self._inputs["X"]    = batch["all_atom_positions"][:,atom_idx]
      self._inputs["mask"] = batch["all_atom_mask"][:,1]
      self._inputs["S"]    = batch["aatype"]

    # fix positions
    if af.protocol == "binder":
      p = np.arange(af._target_len)
    else:
      p = af.opt.get("fix_pos",None)
    
    if p is not None:
      self._inputs["fix_pos"] = p
      self._inputs["bias"][p] = 1e7 * np.eye(21)[self._inputs["S"]][p,:20]

    # tie positions
    if af._args["homooligomer"]:
      assert min(self._lengths) == max(self._lengths)
      self._tied_lengths = True
    else:
      self._tied_lengths = False

  def sample(self, temperature=0.1, rescore=False, **kwargs):
    '''Sample one sequence for each conformer'''
    I = copy_dict(self._inputs)
    I.update(kwargs)
    key = I.pop("key",self.key())
    keys = jax.random.split(key,1)
    O = self._sample_conformers(keys, I, temperature, self._tied_lengths)
    if rescore:
      O = self._rescore_parallel(keys, I, O["S"], O["decoding_order"])

    # must squeeze here, unlike regular model
    O = jax.tree_util.tree_map(lambda x: np.squeeze(np.array(x)), O)

    # process outputs to human-readable form
    O.update(self._get_seq(O))
    O.update(self._get_score(I,O))
    return O

  def sample_parallel(self, batch=10, temperature=0.1, rescore=False, **kwargs):
    '''sample new sequence(s) in parallel
    NOT IMPLEMENTED
    
    '''
    if batch != 1:
      raise NotImplementedError("Batched sampling not implemented for conformational ensembles.")
  

  def _get_seq(self, O):
        """one_hot to amino acid sequence (still returns Python strings)"""

        def split_seq(seq_str, lengths, tied_lengths):  # pass lengths and tied_lengths explicitly
            if len(lengths) > 1:
                # This string manipulation cannot be JITted.
                # If this were inside a JITted function, it would be a host callback.
                seq_str = "".join(np.insert(list(seq_str), np.cumsum(lengths[:-1]), "/"))
                if tied_lengths:
                    seq_str = seq_str.split("/")[0]
            return seq_str

        seqs = []
        # Assuming O["S"] is (batch, L, 21) or (L, 21)
        # Convert JAX array to NumPy for iteration and string conversion
        S_numpy = np.array(O["S"].argmax(axis=1)).T # modified axis=1 and transposed for ensemble
        if S_numpy.ndim == 1:
            S_numpy = S_numpy[None, :]  # ensure batch dimension

        for s_np in S_numpy:
            # This part is Python string manipulation
            seq = "".join([order_aa[a_idx] for a_idx in s_np])
            seq = split_seq(seq, self._lengths, self._tied_lengths)  # pass necessary attributes
            seqs.append(seq)
        return {"seq": np.array(seqs)}
        
  def _get_score(self, I, O):
    '''
    logits to score/sequence_recovery
    return {"score":score (L, n_frames), "seqid":seqid (L, n_frames)}
    '''
    # this is reasonably fast, even without jax
    mask = I["mask"].copy()
    if "fix_pos" in I:
      mask[I["fix_pos"]] = 0
    
    mask = np.expand_dims(mask, -1)

    # softmaxes are now mapped over axis=1
    log_q = log_softmax(O["logits"], axis=1)[:,:20,:]
    q = softmax(O["logits"][:,:20,:], axis=1)

    # sums are over axis 0
    if "S" in O:
      S = O["S"][:,:20,:]
      score = -(S * log_q).sum(axis=1)
      seqid = S.argmax(axis=1) == np.expand_dims(self._inputs["S"], -1)
    else:
      score = -(q * log_q).sum(axis=0)
      seqid = np.zeros_like(score)
      
    score = (score * mask).sum(axis=0) / (mask.sum() + 1e-8)
    seqid = (seqid * mask).sum(axis=0) / (mask.sum() + 1e-8)

    return {"score":score, "seqid":seqid}
        


  def _get_score_jax(self, inputs_S, logits, mask, fix_pos=None): # Pass necessary inputs directly
    ''' logits to score/sequence_recovery - JAX compatible version '''


    current_mask = mask
    if fix_pos is not None and fix_pos.shape[0] > 0: # Ensure fix_pos is not empty
        # Ensure mask is a JAX array for .at.set to work
        current_mask = jnp.array(current_mask) # Convert if it's numpy
        current_mask = current_mask.at[fix_pos].set(0)

    # Use jax.nn functions
    log_q = jax.nn.log_softmax(logits, axis=-1)[..., :20]
    q = jax.nn.softmax(logits[..., :20], axis=-1)
    
    S_scored_one_hot = jax.nn.one_hot(inputs_S, num_classes=21)[...,:20] # Assuming inputs_S is integer encoded for the sequence to score
                                                                    # This would be I["S"] from the score() method
    
    score = -(S_scored_one_hot * log_q).sum(-1)

    seqid = (inputs_S == self._inputs["S"])

    masked_score_sum = (score * current_mask).sum(-1)
    masked_seqid_sum = (seqid * current_mask).sum(-1)
    mask_sum = current_mask.sum() + 1e-8

    final_score = masked_score_sum / mask_sum
    final_seqid = masked_seqid_sum / mask_sum
    
    return {"score": final_score, "seqid": final_seqid}


  def score(self, seq_numeric=None, **kwargs): # seq_numeric is an integer array
    '''score sequence - JAX compatible version (mostly)'''
    current_inputs = jax.tree_util.tree_map(jnp.array, self._inputs)

    if seq_numeric is not None:
        # seq_numeric is expected to be an integer array of amino acid indices
        p = jnp.arange(current_inputs["S"].shape[0])
        s_shape_0 = current_inputs["S"].shape[0] # Store shape for JAX tracing

        if self._tied_lengths and seq_numeric.shape[0] == self._lengths[0]:
            # Assuming self._lengths is available and compatible
            # seq_numeric might need tiling if it represents one chain of a homooligomer
            num_repeats = len(self._lengths)
            seq_numeric = jnp.tile(seq_numeric, num_repeats)

        if "fix_pos" in current_inputs and current_inputs["fix_pos"].shape[0] > 0:
            # Ensure shapes are concrete or JAX can trace them
            if seq_numeric.shape[0] == (s_shape_0 - current_inputs["fix_pos"].shape[0]):
                p = jnp.delete(p, current_inputs["fix_pos"], axis=0)
        
        # Update S using .at[].set()
        # Ensure seq_numeric is correctly broadcasted or indexed if p is tricky
        current_inputs["S"] = current_inputs["S"].at[p].set(seq_numeric)

    # Combine kwargs with current_inputs, ensuring JAX types
    for k, v in kwargs.items():
        current_inputs[k] = jnp.asarray(v) if not isinstance(v, jax.Array) else v


    key_to_use = current_inputs.pop("key", self.key()) # self.key() provides a JAX key

    # _score is already JITted and expects JAX-compatible inputs
    # The arguments to _score are X, mask, residue_idx, chain_idx, key, S, bias, decoding_order etc.
    # Ensure all these are present in current_inputs and are JAX arrays.
    
    # Prepare arguments for self._score, ensuring they are all JAX arrays
    score_fn_args = {k: current_inputs[k] for k in [
        'X', 'mask', 'residue_idx', 'chain_idx', 'S', 'bias'
        ] if k in current_inputs}
    
    if "decoding_order" in current_inputs:
        score_fn_args["decoding_order"] = current_inputs["decoding_order"]
    if "fix_pos" in current_inputs: # _score uses fix_pos to adjust decoding_order
         score_fn_args["fix_pos"] = current_inputs["fix_pos"]


    # O will be a dictionary of JAX arrays
    O = self._score(**score_fn_args, key=key_to_use)

    # Call the JAX-compatible _get_score
    # It needs: current_inputs["S"] (the sequence being scored, possibly modified),
    # O["logits"], current_inputs["mask"], and current_inputs.get("fix_pos")
    score_info = self._get_score(
        inputs_S=current_inputs["S"], # This is the S that was actually scored by _score
        logits=O["logits"],
        mask=current_inputs["mask"],
        fix_pos=current_inputs.get("fix_pos")
    )
    O.update(score_info) # O remains a dict of JAX arrays

    # If you need to convert to NumPy arrays for external use, do it here,
    # but the function itself now primarily deals with JAX arrays.
    # For full JAX compatibility of `score` itself (e.g. to JIT it),
    # this conversion should be outside.
    # return jax.tree_map(np.array, O)
    return O # Returns dict of JAX array

  def get_logits(self, **kwargs):
    '''get logits'''
    return self.score(**kwargs)["logits"]

  def get_unconditional_logits(self, **kwargs):
    L = self._inputs["X"].shape[0]
    kwargs["ar_mask"] = np.zeros((L,L))
    return self.score(**kwargs)["logits"]

  def set_seed(self, seed=None):
    np.random.seed(seed=seed)
    self.key = Key(seed=seed).get

  def _setup(self):
        def _score_internal(
            X, mask, residue_idx, chain_idx, key, S, bias, **kwargs
        ):  # Added S and bias
            I = {
                "X": X,
                "mask": mask,
                "residue_idx": residue_idx,
                "chain_idx": chain_idx,
                "S": S,  # Pass S
                "bias": bias,  # Pass bias
            }
            I.update(kwargs)

            if "decoding_order" not in I:
                key, sub_key = jax.random.split(key)
                randn = _randomize_sophie(sub_key, X)
                randn = jnp.where(I["mask"], randn, randn + 1)
                if "fix_pos" in I and I["fix_pos"].shape[0] > 0:  # check if fix_pos is not empty
                    randn = randn.at[I["fix_pos"]].add(-1)
                I["decoding_order"] = randn.argsort()

            # _aa_convert is JAX-compatible
            for k_item in ["S", "bias"]:  # Use k_item to avoid conflict with key
                if k_item in I:
                    I[k_item] = _aa_convert(I[k_item])

            output_dict = self._model.score(self._model.params, key, I)
            output_dict["S"] = _aa_convert(output_dict["S"], rev=True)
            output_dict["logits"] = _aa_convert(output_dict["logits"], rev=True)
            return output_dict

        self._score = jax.jit(_score_internal)

        def _sample_internal(
            X,
            mask,
            residue_idx,
            chain_idx,
            key,
            temperature=0.1,
            tied_lengths=False,
            bias=None,
            **kwargs,
        ):  # added bias
        # single conformer sampling
            I = {
                "X": X,
                "mask": mask,
                "residue_idx": residue_idx,
                "chain_idx": chain_idx,
                "temperature": temperature,
                "bias": bias,  # Pass bias
            }
            I.update(kwargs)

            # define decoding order (as in original _sample)
            if "decoding_order" in I:
                if I["decoding_order"].ndim == 1:
                    I["decoding_order"] = I["decoding_order"][:, None]
            else:
                key, sub_key = jax.random.split(key)
                #randn = jax.random.uniform(sub_key, (I["X"].shape[0],))
                randn = _randomize_sophie(sub_key, X)

                

                randn = jnp.where(I["mask"], randn, randn + 1)
                if "fix_pos" in I and I["fix_pos"].shape[0] > 0:
                    randn = randn.at[I["fix_pos"]].add(-1)
                if tied_lengths:
                    copies = I["lengths"].shape[0]
                    decoding_order_tied = randn.reshape(copies, -1).mean(0).argsort()
                    I["decoding_order"] = (
                        jnp.arange(I["X"].shape[0]).reshape(copies, -1).T[decoding_order_tied]
                    )
                else:
                    I["decoding_order"] = randn.argsort()[:, None]

            # S is not an input to _model.sample, but bias is
            if "S" in I:
                I["S"] = _aa_convert(
                    I["S"]
                )  # If S is somehow passed (e.g. for conditioning, though MPNN typically doesn't)
            if "bias" in I:
                I["bias"] = _aa_convert(I["bias"])

            O_dict = self._model.sample(self._model.params, key, I)
            O_dict["S"] = _aa_convert(O_dict["S"], rev=True)  # This is the sampled S
            O_dict["logits"] = _aa_convert(O_dict["logits"], rev=True)
            return O_dict

        self._sample = jax.jit(_sample_internal, static_argnames=["tied_lengths"])

        # 
        def _vmap_sample_seqs_from_conformers(key, inputs, temperature, tied_lengths):
            inputs_copy = dict(inputs)  # Shallow copy for modification
            inputs_copy.pop("temperature", None)
            inputs_copy.pop("key", None)
            # Ensure 'bias' is correctly handled if it's part of 'inputs'
            f_of_X = jax.jit(
                 partial(self._sample, key=key,
            **{k : v for k,v in inputs_copy.items() if k not in ("X",)},
            temperature=temperature, tied_lengths=tied_lengths), static_argnames=["tied_lengths"]
            )
            # vmap over positions
            return jax.vmap(f_of_X, in_axes=-1, out_axes=-1)(inputs_copy["X"])

        # this is vmap over keys, but there's only one.
        # this is just the easiest way to square with earliest code
        # but might be good to refactor
        fn_vmap_sample_conformers = jax.vmap(_vmap_sample_seqs_from_conformers, in_axes=[0, None, None, None])
        # difference, no jit for now  
        self._sample_conformers = fn_vmap_sample_conformers

        def _vmap_rescore_parallel(key, inputs, S_rescore, decoding_order_rescore):
            inputs_copy = dict(inputs)  # Shallow copy
            inputs_copy.pop("S", None)
            inputs_copy.pop("decoding_order", None)
            inputs_copy.pop("key", None)
            # Ensure 'bias' from original inputs is used, and S_rescore is the new S
            return self._score(
                **inputs_copy, key=key, S=S_rescore, decoding_order=decoding_order_rescore
            )  # Pass S and decoding_order

        fn_vmap_rescore = jax.vmap(_vmap_rescore_parallel, in_axes=[0, None, 0, 0])
        self._rescore_parallel = jax.jit(fn_vmap_rescore)

#######################################################################################

def _aa_convert(x, rev=False):
  mpnn_alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
  af_alphabet =   'ARNDCQEGHILKMFPSTWYVX'
  if x is None:
    return x
  else:
    if rev:
      return x[...,tuple(mpnn_alphabet.index(k) for k in af_alphabet)]
    else:
      x = jax.nn.one_hot(x,21) if jnp.issubdtype(x.dtype, jnp.integer) else x
      if x.shape[-1] == 20:
        x = jnp.pad(x,[[0,0],[0,1]])
      return x[...,tuple(af_alphabet.index(k) for k in mpnn_alphabet)]

unknown_aa_index = aa_order.get('X', 20) # Default index for unknown AAs

def convert_sequence_to_numeric(sequence_str: str,
                                aa_map: dict = aa_order,
                                all_chain_lengths: list = None,
                                is_homooligomer_tied: bool = False) -> jnp.array:
    """
    Converts a protein sequence string into a JAX integer array.

    Args:
        sequence_str: The amino acid sequence string.
                      - For monomers: "ACEG..."
                      - For heteromers (chains separated by '/'): "ACEG.../FGHI..."
                      - For homooligomers where is_homooligomer_tied is True and
                        only one chain's sequence is provided: "ACEG..." (will be tiled).
        aa_map: Dictionary mapping amino acid characters to integers (e.g., aa_order).
        all_chain_lengths: List of lengths of all chains in the complex.
                           Example: [100, 100] for a dimer of length 100 each.
                           Used for homooligomer tiling.
        is_homooligomer_tied: Boolean. If True and sequence_str is for a single
                              chain of a homooligomer, the sequence will be tiled.

    Returns:
        jnp.array: A JAX array of integers representing the full sequence.
    """
    numeric_sequence_list = []

    # Handle homooligomer case where a single chain sequence is provided to be tiled
    if is_homooligomer_tied and \
       all_chain_lengths and \
       len(all_chain_lengths) > 0 and \
       "/" not in sequence_str:
        # Check if the provided sequence string matches the length of one chain
        if len(sequence_str) == all_chain_lengths[0]: # Assuming all chains have the same length
            num_chains = len(all_chain_lengths)
            # Tile the string sequence before converting to numeric
            sequence_str = "/".join([sequence_str] * num_chains)
        # TODO: add a warning or error if the lengths don't match

    # Process chain by chain if '/' is present, otherwise process the whole string
    chains = sequence_str.split('/')
    
    for chain_seq_str in chains:
        for aa_char in chain_seq_str:
            # Use .get(key, default_value) to handle unexpected characters
            numeric_sequence_list.append(aa_map.get(aa_char, unknown_aa_index))

    return jnp.array(numeric_sequence_list, dtype=jnp.int32)

### >:D >:D >:D >:D >:D >:D >:D >:D # >:D >:D >:D >:D >:D >:D >:D >:D # >:D >:D >:D >:D
def _randomize_sophie(key, X_conformer, max_freq=1e9, min_freq=1e3):
    """
    WARNING: EXPERIMENTAL

    Use X_conformer as a natural entropy source to randomize decoding order
    by transforming spatial coordinates with a random-frequency sine wave.
    """
    randfreq = (max_freq - min_freq)*jax.random.uniform(key) + min_freq
    randn = 0.5*(1 + jnp.sin(X_conformer * randfreq).sum(axis=(1,2)))
    return randn