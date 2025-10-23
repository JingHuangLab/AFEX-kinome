# %%

# Make prediction with AF2.
# Zilin Song, 20230820
# Tengyu Xie, 20230913
# Wang, Yuxuan, 20240815
# should have sudo apt-get install hmmer

import os
import sys
script_path = os.path.abspath(__file__)
print(script_path)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)
print("Current working directory:", os.getcwd())
sys.path.insert(0, str(script_dir))  # 插入路径到最前面
import numpy as np
from typing import Tuple
import jax, jax.numpy as jnp, optax
from absl import app, flags
from alphafold.model import config
from alphafold.common import protein
from afexplore_runner import get_afe_runner, AFExploreRunModel
from kinase_state_return_cond_df import identify_state
import pandas as pd

# %% 
# DIR: raw_features as input.
flags.DEFINE_string('rawfeat_dir', None,
                    'Path to directory that stores the raw features.')
# DIR: output.
flags.DEFINE_string('output_dir', None,
                    'Path to a directory that stores all outputs.')
# DIR: data.
flags.DEFINE_string('afparam_dir', None,
                    'Path to directory of supporting data / model parameters.')
# Config: number of optimization steps.
flags.DEFINE_integer('nsteps', 10, 'Number of optimization steps.')
# Config-AF: number of MSA clusters
flags.DEFINE_integer(
    'nclust', 512, 'Number of MSA clusters used for featurization, this number scales linearly with memory usage.')
flags.DEFINE_float('learning_rate', 0.02,
                   'Learning rate for gradient updates.')
# PRESET: models: monomer only.
flags.DEFINE_enum('model_preset', 'monomer', ['monomer', ], 'Choose preset model configuration - the monomer model')

flags.DEFINE_enum('protein_type', 'kinase',
                  ['kinase'],
                  'Choose protein type - kinase'
                  'Loss functions of other proteins are required to be added accordingly.')
flags.DEFINE_enum('target_state', 'active',
                  ['active'],
                  'active refers to DFGin/BLAminus/Saltbridge-in/Activation loop NT/CT-in')
flags.DEFINE_integer('num_success', 10,
                     'The number of accumalated successful samplings.')
flags.DEFINE_float('plddt_target', 0.95, 'threshold of plddt target')
FLAGS = flags.FLAGS

# %% 
# A list of atoms (excluding hydrogen) for each AA type used by AlphaFold2. PDB naming convention.
atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]

phe_restype_atom_dict={'F':'CZ',  'R':'CZ',  'L':'CG',  'P':'CG', 'N':'CG',
                   'M':'CE',  'S':'OG',  'H':'NE2', 'V':'CB', 'A':'CB',
                   'W':'CZ3', 'Y':'OH',  'G':'CA',  'C':'SG', 'D':'CG',
                   'E':'OE1', 'Q':'OE1', 'I':'CD1', 'K':'NZ', 'T':'OG1'}

# DFG conformation data used in Kincore calculation
dfginter={'BABtrans':(-80.20,128.22,-117.47,23.76,-85.16,133.21,181.42)}
dfgout={'BBAminus':(-138.56,-176.12,-144.35,103.66,-82.59,-9.03,290.59)}
dfgin_minus={'BLAminus':(-128.64,178.67,61.15,81.21,-96.89,20.53,289.12),
             'ABAminus':(-111.82,-7.64,-141.55,148.01,-127.79,23.32,296.17),
             'BLBminus':(-134.79,175.48,60.44,65.35,-79.44,145.34,287.56)}
dfgin_plus={'BLAplus':(-119.24,167.71,58.94,34.08,-89.42,-8.54,55.63),
            'BLBplus':(-125.28,172.53,59.98,32.92,-85.51,145.28,49.01)}
dfgin_trans={'BLBtrans':(-106.16,157.24,69.37,21.33,-61.73,134.56,215.23)}

def compute_d1(prediction_result, Phe_num, Phe_restype, Glu4_num):
    phe_atom_type = phe_restype_atom_dict[Phe_restype]
    d1 = jnp.linalg.norm(prediction_result['structure_module']['final_atom_positions'][Phe_num, atom_types.index(phe_atom_type), :] - prediction_result['structure_module']['final_atom_positions'][Glu4_num, atom_types.index('CA'), :])
    return d1

def compute_d2(prediction_result, Phe_num, Phe_restype, Lys_num):
    phe_atom_type = phe_restype_atom_dict[Phe_restype]     
    d2 = jnp.linalg.norm(prediction_result['structure_module']['final_atom_positions'][Phe_num, atom_types.index(phe_atom_type), :] - prediction_result['structure_module']['final_atom_positions'][Lys_num, atom_types.index('CA'), :])
    return d2

def compute_ActLoopNT_dis(prediction_result, XHRD_num, XHRD_restype, DFG6_num, DFG6_restype):          
    dist1 = jnp.linalg.norm(prediction_result['structure_module']['final_atom_positions'][XHRD_num, atom_types.index('N'), :] - prediction_result['structure_module']['final_atom_positions'][DFG6_num, atom_types.index('O'), :])
    dist2 = jnp.linalg.norm(prediction_result['structure_module']['final_atom_positions'][XHRD_num, atom_types.index('O'), :] - prediction_result['structure_module']['final_atom_positions'][DFG6_num, atom_types.index('N'), :])
    mindist = jnp.min(jnp.array([dist1, dist2]))
    return mindist

def compute_ActLoopCT_dis(prediction_result, APE9_num, APE9_restype, Arg_num, Arg_restype):          
    dis = jnp.linalg.norm(prediction_result['structure_module']['final_atom_positions'][APE9_num, atom_types.index('CA'), :] - prediction_result['structure_module']['final_atom_positions'][Arg_num, atom_types.index('O'), :])
    return dis

def compute_saltbridge_dis(prediction_result, Lys_num, Lys_restype, Glu_num, Glu_restype):          
    #Distance Lys-Glu salt bridge over all hydrogen bonding atoms
    #Have to calculate 4 distances in some kinases (Arg-Glu, Arg-Asp) and sometimes 2 (Lys-Glu) and sometimes 1
    Katom1="CB"
    Katom2="CB"
    Eatom1="CB"
    Eatom2="CB"
    
    # Define atom mappings for residues
    lys_atoms = {
        "K": ("NZ", "NZ"),
        "R": ("NH1", "NH2"),
        "Y": ("OH", "OH"),
        "H": ("ND1", "NE2"),
        "Q": ("OE1", "NE2"),
        "N": ("OD1", "ND2"),
        "S": ("OG", "OG"),
        "T": ("OG1", "OG1"),
        "C": ("SG", "SG"),
        "G": ("N", "O")
    }
    glu_atoms = {
        "E": ("OE1", "OE2"),
        "Q": ("OE1", "NE2"),
        "D": ("OD1", "OD2"),
        "N": ("OD1", "ND2"),
        "R": ("NH1", "NH2"),
        "K": ("NZ", "NZ"),
        "H": ("ND1", "NE2"),
        "Y": ("OH", "OH"),
        "S": ("OG", "OG"),
        "T": ("OG1", "OG1"),
        "W": ("NE1", "NE1"),
        "G": ("N", "O")
    }
        # Check if Lys_restype and Glu_restype exist in their respective atom dictionaries
    if Lys_restype not in lys_atoms or Glu_restype not in glu_atoms:
        # Return 0 if either restype is not found in the dictionaries
        return 0.
    
    # Extract the atom types
    Katom1, Katom2 = lys_atoms[Lys_restype]
    Eatom1, Eatom2 = glu_atoms[Glu_restype]
    
    # Calculate distances
    dist1 = jnp.linalg.norm(prediction_result['structure_module']['final_atom_positions'][Lys_num, atom_types.index(Katom1), :] - prediction_result['structure_module']['final_atom_positions'][Glu_num, atom_types.index(Eatom1), :]).astype(float)
    dist2 = jnp.linalg.norm(prediction_result['structure_module']['final_atom_positions'][Lys_num, atom_types.index(Katom1), :] - prediction_result['structure_module']['final_atom_positions'][Glu_num, atom_types.index(Eatom2), :]).astype(float)
    dist3 = jnp.linalg.norm(prediction_result['structure_module']['final_atom_positions'][Lys_num, atom_types.index(Katom2), :] - prediction_result['structure_module']['final_atom_positions'][Glu_num, atom_types.index(Eatom1), :]).astype(float)
    dist4 = jnp.linalg.norm(prediction_result['structure_module']['final_atom_positions'][Lys_num, atom_types.index(Katom2), :] - prediction_result['structure_module']['final_atom_positions'][Glu_num, atom_types.index(Eatom2), :]).astype(float)

    # Calculate minimum distance
    mindist = jnp.min(jnp.array([dist1, dist2, dist3, dist4]))
    return mindist

def calc_dihedral(v1, v2, v3, v4):
    """Calculate the dihedral angle in radians using JAX.
    Args:
        v1, v2, v3, v4: JAX arrays representing four connected points.
    Returns:
        Dihedral angle in radians.
    """
    # Compute three vector groups
    b1 = v2 - v1
    b2 = v3 - v2
    b3 = v4 - v3
    # Compute normal vectors
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    # Normalize normal vectors
    n1 /= jnp.linalg.norm(n1)
    n2 /= jnp.linalg.norm(n2)
    # Compute dihedral angle
    x = jnp.dot(n1, n2)
    y = jnp.dot(jnp.cross(n1, n2), b2 / jnp.linalg.norm(b2))
    return jnp.arctan2(y, x)

def compute_phi(prev_c, curr_n, curr_ca, curr_c):          
    phi = calc_dihedral(prev_c, curr_n, curr_ca, curr_c)
    return jnp.degrees(phi)

def compute_psi(curr_n,curr_ca,curr_c,next_n):
    psi = calc_dihedral(curr_n,curr_ca,curr_c,next_n)
    return jnp.degrees(psi)

def dihedral_cosine_dis(prediction_result, XDFG_num, XDFG_restype, Asp_num, Phe_num, spatial, spatial_cutoff_upper):
    """
    - x1 should be the minimum value among x1, x2, and x3
    - x1 should be less than spatial_cutoff_upper
    """
    results = []
    x_dfg_phi = (compute_phi(jnp.asarray(prediction_result['structure_module']['final_atom_positions'][XDFG_num -1, atom_types.index('C'), :]),
                            jnp.asarray(prediction_result['structure_module']['final_atom_positions'][XDFG_num, atom_types.index('N'), :]),
                            jnp.asarray(prediction_result['structure_module']['final_atom_positions'][XDFG_num, atom_types.index('CA'), :]),
                            jnp.asarray(prediction_result['structure_module']['final_atom_positions'][XDFG_num, atom_types.index('C'), :]))).astype(float)
    x_dfg_psi = (compute_psi(jnp.asarray(prediction_result['structure_module']['final_atom_positions'][XDFG_num, atom_types.index('N'), :]),
                            jnp.asarray(prediction_result['structure_module']['final_atom_positions'][XDFG_num, atom_types.index('CA'), :]),
                            jnp.asarray(prediction_result['structure_module']['final_atom_positions'][XDFG_num, atom_types.index('C'), :]),
                            jnp.asarray(prediction_result['structure_module']['final_atom_positions'][XDFG_num + 1, atom_types.index('N'), :]))).astype(float)
    dfg_asp_phi = (compute_phi(jnp.asarray(prediction_result['structure_module']['final_atom_positions'][Asp_num -1, atom_types.index('C'), :]),
                            jnp.asarray(prediction_result['structure_module']['final_atom_positions'][Asp_num, atom_types.index('N'), :]),
                            jnp.asarray(prediction_result['structure_module']['final_atom_positions'][Asp_num, atom_types.index('CA'), :]),
                            jnp.asarray(prediction_result['structure_module']['final_atom_positions'][Asp_num, atom_types.index('C'), :]))).astype(float)
    dfg_asp_psi = (compute_psi(jnp.asarray(prediction_result['structure_module']['final_atom_positions'][Asp_num, atom_types.index('N'), :]),
                            jnp.asarray(prediction_result['structure_module']['final_atom_positions'][Asp_num, atom_types.index('CA'), :]),
                            jnp.asarray(prediction_result['structure_module']['final_atom_positions'][Asp_num, atom_types.index('C'), :]),
                            jnp.asarray(prediction_result['structure_module']['final_atom_positions'][Asp_num + 1, atom_types.index('N'), :]))).astype(float)
    dfg_phe_phi = (compute_phi(jnp.asarray(prediction_result['structure_module']['final_atom_positions'][Phe_num -1, atom_types.index('C'), :]),
                            jnp.asarray(prediction_result['structure_module']['final_atom_positions'][Phe_num, atom_types.index('N'), :]),
                            jnp.asarray(prediction_result['structure_module']['final_atom_positions'][Phe_num, atom_types.index('CA'), :]),
                            jnp.asarray(prediction_result['structure_module']['final_atom_positions'][Phe_num, atom_types.index('C'), :]))).astype(float)
    dfg_phe_psi = (compute_psi(jnp.asarray(prediction_result['structure_module']['final_atom_positions'][Phe_num, atom_types.index('N'), :]),
                            jnp.asarray(prediction_result['structure_module']['final_atom_positions'][Phe_num, atom_types.index('CA'), :]),
                            jnp.asarray(prediction_result['structure_module']['final_atom_positions'][Phe_num, atom_types.index('C'), :]),
                            jnp.asarray(prediction_result['structure_module']['final_atom_positions'][Phe_num + 1, atom_types.index('N'), :]))).astype(float)
    
    for clusters in spatial:
        cosine_dis=(2/6)*(
            (1-jnp.cos(jnp.radians(x_dfg_phi-float(spatial[clusters][0]))))+
            (1-jnp.cos(jnp.radians(x_dfg_psi-float(spatial[clusters][1]))))+
            (1-jnp.cos(jnp.radians(dfg_asp_phi-float(spatial[clusters][2]))))+
            (1-jnp.cos(jnp.radians(dfg_asp_psi-float(spatial[clusters][3]))))+
            (1-jnp.cos(jnp.radians(dfg_phe_phi-float(spatial[clusters][4]))))+
            (1-jnp.cos(jnp.radians(dfg_phe_psi-float(spatial[clusters][5])))))
        results.append(cosine_dis)

    num_clusters = len(results)
    if num_clusters == 0:
        return 0.0
    
    min_loss = 0.0

    if num_clusters > 1:
        other_results = jnp.array(results[1:])
        differences = results[0] - other_results
        positive_diffs = jnp.maximum(0, differences)
        min_loss = jnp.sum(positive_diffs)
    
    cutoff_loss = jnp.maximum(0, results[0] - spatial_cutoff_upper)
    total_loss = min_loss + cutoff_loss
    return results, total_loss

def compute_Phe_chi1(prediction_result, Phe_num):
    phe_chi1=999.0
    phe_chi1 = calc_dihedral(jnp.asarray(prediction_result['structure_module']['final_atom_positions'][Phe_num, atom_types.index('N'), :]),
                                        jnp.asarray(prediction_result['structure_module']['final_atom_positions'][Phe_num, atom_types.index('CA'), :]),
                                        jnp.asarray(prediction_result['structure_module']['final_atom_positions'][Phe_num, atom_types.index('CB'), :]),
                                        jnp.asarray(prediction_result['structure_module']['final_atom_positions'][Phe_num, atom_types.index('CG'), :]))
    phe_chi1_degrees = jnp.degrees(phe_chi1)
    if phe_chi1_degrees < 0.:
        phe_chi1_degrees += 360.
    return phe_chi1_degrees

def calculate_plddt_loss_fn(x, cutoff_lower, cutoff_upper, step):
    if x == 999:
        return 0.
    else: 
        # # If x is in the range [cutoff_upper, 1], the loss is 0
        loss_inside = jnp.where((x >= cutoff_upper), 0.0, 0.0)
        # If x is less than cutoff_upper, the loss value increases
        loss_outside = jnp.where(x < cutoff_upper, jax.nn.relu(cutoff_upper - x), 0.0)
        # The total loss is the sum of the losses from the two parts outside the interval
        return loss_inside + loss_outside

def calculate_general_distance_loss_fn(x, cutoff_lower, cutoff_upper):
    if x == 999:
        return 0.
    else: 
        # If x is within the range [cutoff_lower, cutoff_upper], the loss is 0.
        loss_inside = jnp.where((x >= cutoff_lower) & (x <= cutoff_upper), 0.0, 0.0)
        # If x is less than cutoff_lower or greater than cutoff_upper, the loss value increases.
        loss_outside_lower = jnp.where(x < cutoff_lower, jax.nn.relu(cutoff_lower - x), 0.0)
        loss_outside_upper = jnp.where(x > cutoff_upper, jax.nn.relu(x - cutoff_upper), 0.0)
        # The total loss is the sum of the losses from the two parts outside the interval.
        return loss_inside + loss_outside_lower + loss_outside_upper

def calculate_ActLoopCT_distance_loss_fn(x, cutoff_lower, cutoff_upper, group, score):
    if x == 999:
        return 0.
    elif (group in ["BUB", "HASP", "PKDCC", "TP53RK", "PAN3", "RNASEL"] or 
          (group == "PEAK" and score < 300)):
        return 0.
    else: 
        # If x is within the range [cutoff_lower, cutoff_upper], the loss is 0.
        loss_inside = jnp.where((x >= cutoff_lower) & (x <= cutoff_upper), 0.0, 0.0)
        # If x is less than cutoff_lower or greater than cutoff_upper, the loss value increases.
        loss_outside_lower = jnp.where(x < cutoff_lower, jax.nn.relu(cutoff_lower - x), 0.0)
        loss_outside_upper = jnp.where(x > cutoff_upper, jax.nn.relu(x - cutoff_upper), 0.0)
        # The total loss is the sum of the losses from the two parts outside the interval.
        return loss_inside + loss_outside_lower + loss_outside_upper

def calculate_SaltBr_distance_loss_fn(x, cutoff_lower, cutoff_upper, KEtype):
    if x == 999:
        return 0.
    elif KEtype in ["KE", "RE", "KD", "RD", "KN", "RN"]:
        # If x is within the range [cutoff_lower, cutoff_upper], the loss is 0.
        loss_inside = jnp.where((x >= cutoff_lower) & (x <= cutoff_upper), 0.0, 0.0)
        # If x is less than cutoff_lower or greater than cutoff_upper, the loss value increases.
        loss_outside_lower = jnp.where(x < cutoff_lower, jax.nn.relu(cutoff_lower - x), 0.0)
        loss_outside_upper = jnp.where(x > cutoff_upper, jax.nn.relu(x - cutoff_upper), 0.0)
        # The total loss is the sum of the losses from the two parts outside the interval.
        return loss_inside + loss_outside_lower + loss_outside_upper
    else: 
        return 0.

def afe_fitting(afe_runner: AFExploreRunModel,
                af_features: dict, n_steps: int,
                learning_rate: float,
                ) -> optax.Params:
    """Fit the AFExplore model."""
    afe_weights = jnp.ones((af_features['msa_feat'].shape[0],
                            af_features['msa_feat'].shape[1],
                            af_features['msa_feat'].shape[2],
                            23, ), )
    
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params=afe_weights)

    #---------------------------------------------------------------------------------------------

    def afe_loss_fn(afe_params: Tuple,
                    af_features: dict,
                    ) -> Tuple[jnp.ndarray, dict]:
        prediction_result = afe_runner.predict(af_features, afe_params, 0)
        
        # Loss from pLDDT
        # copied from ./alphafold/common/confidence.py
        plddt_logits = prediction_result['predicted_lddt']['logits']
        plddt_bin_width = 1./plddt_logits.shape[-1]
        plddt_bin_centers = jnp.arange(
            start=.5*plddt_bin_width, stop=1., step=plddt_bin_width, )
        plddt_ca = jnp.sum(jax.nn.softmax(plddt_logits, axis=-1)
                           * plddt_bin_centers[None, :], axis=-1)
        plddt = jnp.mean(plddt_ca)
        plddt_loss = calculate_plddt_loss_fn(plddt, plddt_cutoff_lower, plddt_cutoff_upper, i)
        
        # actloop_region pLDDT and loss
        plddt_actloop_region = jnp.mean(plddt_ca[int(actloop_start):int(actloop_end)])
        if FLAGS.target_state == 'active':
            plddt_actloop_loss = calculate_plddt_loss_fn(plddt_actloop_region, plddt_cutoff_lower, plddt_cutoff_upper, i)
        else: plddt_actloop_loss = 0.
            
        # constrain d1 dis_phe_rre4 <=11. and d2 dis_phe_lys >=11., for DFGin
        if Glu4_restype != "-" and Phe_restype != "-":
            dis_phe_rre4 = compute_d1(prediction_result, Phe_num, Phe_restype, Glu4_num)
            d1_loss = calculate_general_distance_loss_fn(dis_phe_rre4, d1_cutoff_lower, d1_cutoff_upper)
        else: 
            dis_phe_rre4 = 999.
            d1_loss=0.
        # constrain d2_cutoff_lower=11. d2_cutoff_upper=20.
        if Lys_restype != "-" and Glu_restype != "-":
            dis_phe_lys = compute_d2(prediction_result, Phe_num, Phe_restype, Lys_num)
            d2_loss = calculate_general_distance_loss_fn(dis_phe_lys, d2_cutoff_lower, d2_cutoff_upper)
        else: 
            dis_phe_lys = 999.
            d2_loss=0.

        # contrain chi1
        phe_chi1 = compute_Phe_chi1(prediction_result, Phe_num)
        phe_chi1_loss = jnp.radians(calculate_general_distance_loss_fn(phe_chi1, chi1_cutoff_lower, chi1_cutoff_upper))
        
        # contrain 0. <= Dihedral_dis <= spatial_cutoff
        # spatial_cutoff_upper=0.45
        if FLAGS.target_state == 'active':
            cosine_dis_without_chi1, spatial_cluster_loss = dihedral_cosine_dis(prediction_result, XDFG_num, XDFG_restype, Asp_num, Phe_num, dfgin_minus, spatial_cutoff_upper)

        # for ActLoopNT-in # contrain 0. < DFG6-XHRD-dis <= nt_hbondcutoff
        # nt_hbondcutoff_upper=3.6
        dis_xhrd = compute_ActLoopNT_dis(prediction_result, XHRD_num, XHRD_restype, DFG6_num, DFG6_restype)
        if FLAGS.target_state == 'active':
            ActLoopNT_in_loss = calculate_general_distance_loss_fn(dis_xhrd, nt_hbondcutoff_lower, nt_hbondcutoff_upper)
        else: ActLoopNT_in_loss=0.
        
        # for ActLoopCT-in # contrain 0. < APE9-Arg-dis <= ape9cutoff
        # ape9cutoff_lower=0.
        # ape9cutoff_upper=6.0
        if (group in ["BUB", "HASP", "PKDCC", "TP53RK", "PAN3", "RNASEL"] or 
              (group == "PEAK" and score < 300)):
            dis_ape9 = 999.
            ActLoopCT_in_loss = 0.
        else: 
            dis_ape9 = compute_ActLoopCT_dis(prediction_result, APE9_num, APE9_restype, Arg_num, Arg_restype)
            if FLAGS.target_state == 'active':
                ActLoopCT_in_loss = calculate_ActLoopCT_distance_loss_fn(dis_ape9, ape9cutoff_lower, ape9cutoff_upper, group, score)
            else: ActLoopCT_in_loss = 0.
        
        # for SaltBr-in # contrain 0. < LysNZ-GluOE-dis <= KEcutoff
        # KEcutoff_lower=0.
        # if KEtype == 'KE' or KEtype == 'RE' : KEcutoff_upper=nt_hbondcutoff_upper
        # if KEtype == 'KD' or KEtype == 'RD' : KEcutoff_upper=nt_hbondcutoff_upper+1.5
        # if KEtype == 'KN' or KEtype == 'RN':  KEcutoff_upper=10.0  # no saltbridge if Asn
        dis_saltbridge = compute_saltbridge_dis(prediction_result, Lys_num, Lys_restype, Glu_num, Glu_restype)
        if FLAGS.target_state == 'active':
            SaltBr_in_loss = calculate_SaltBr_distance_loss_fn(dis_saltbridge, KEcutoff_lower, KEcutoff_upper, KEtype)
        else: SaltBr_in_loss = 0.
        
        return plddt_loss * 2 + plddt_actloop_loss * 2 + d1_loss + d2_loss + phe_chi1_loss * 10 + spatial_cluster_loss * 10 + ActLoopNT_in_loss + ActLoopCT_in_loss + SaltBr_in_loss, (prediction_result, jax.lax.stop_gradient(plddt_loss), 
                                                jax.lax.stop_gradient(plddt),
                                                jax.lax.stop_gradient(plddt_actloop_loss),
                                                jax.lax.stop_gradient(plddt_actloop_region),
                                                jax.lax.stop_gradient(d1_loss),
                                                jax.lax.stop_gradient(d2_loss),
                                                jax.lax.stop_gradient(dis_phe_rre4), 
                                                jax.lax.stop_gradient(dis_phe_lys),
                                                jax.lax.stop_gradient(phe_chi1_loss),
                                                jax.lax.stop_gradient(phe_chi1),
                                                jax.lax.stop_gradient(spatial_cluster_loss),
                                                jax.lax.stop_gradient(cosine_dis_without_chi1),
                                                jax.lax.stop_gradient(ActLoopNT_in_loss),
                                                jax.lax.stop_gradient(dis_xhrd),
                                                jax.lax.stop_gradient(ActLoopCT_in_loss),
                                                jax.lax.stop_gradient(dis_ape9),
                                                jax.lax.stop_gradient(SaltBr_in_loss),
                                                jax.lax.stop_gradient(dis_saltbridge))
# ------------------------------------------------------------------------------------------------
    if FLAGS.protein_type == 'kinase':
        # print("FLAGS.protein_type == 'kinase'")
        
        # gets the original location of the hmms file
        hmm_loc = os.path.dirname(os.path.realpath(__file__))+'/Kincore_standalone2/HMMs'
        # print("hmm_loc:", hmm_loc)
        
        prediction_result = afe_runner.predict(af_features, afe_weights, 0)
        # Generate and save the pdb for this prediction
        p = protein.from_prediction(features=af_features,
                                result=prediction_result,
                                b_factors=None,
                                # True for Monomer.
                                remove_leading_feature_dimension=True, )

        AFmodel_path = os.path.join(FLAGS.output_dir, 'afe_model_initial.pdb')
        # print("AFmodel_path:", AFmodel_path)
        
        with open(AFmodel_path, 'w') as f:
                f.write(protein.to_pdb(p))
            
        # recognize residue indices by Kincore
        args_identify_state = (AFmodel_path, hmm_loc, 0)
        conf_df = identify_state(args_identify_state)
        # print("Status successfully identified for:", AFmodel_path)
        
        # save conf_df_initial
        conf_df.to_csv(os.path.join(FLAGS.output_dir, 'conf_df_initial.csv'), index=False)
        
        sys.stdout.flush()
    
    # Initialize statistical values
    count = 0
    loss_records = []
    plddt_loss_records = []
    plddt_records = []
    plddt_actloop_loss_records = []
    plddt_actloop_region_records = []
    d1_loss_records = []
    d2_loss_records = []
    d1_records = []
    d2_records = []
    chi1_loss_records = []
    chi1_records = []
    spatial_cluster_loss_records = []
    dihedral_dis_nochi1_records = []
    ActLoopNT_in_loss_records = []
    dis_xhrd_records = []
    ActLoopCT_in_loss_records = []
    dis_ape9_records = []
    SaltBr_in_loss_records = []
    dis_saltbridge_records = []
    
    # Check target state and set constrains
    # Note that indexing starts from 0, so the 145th residue corresponds to index 144
    actloop_start = int(conf_df.at[0,'Asp_num'] - 1)
    actloop_end = int(conf_df.at[0,'APE_num'] - 1)
    Lys_num = int(conf_df.at[0,'Lys_num'] -1)
    Lys_restype = conf_df.at[0,'Lys_restype']
    Glu_num = int(conf_df.at[0,'Glu_num'] -1)
    Glu_restype = conf_df.at[0,'Glu_restype']
    Glu4_num = int(conf_df.at[0,'Glu4_num'] -1)
    Glu4_restype = conf_df.at[0,'Glu4_restype']
    APE_num = int(conf_df.at[0,'APE_num'] -1)
    APE9_num = int(conf_df.at[0,'APE9_num'] -1)
    APE9_restype = conf_df.at[0,'APE9_restype']
    DFG6_num = int(conf_df.at[0,'DFG6_num'] -1)
    DFG6_restype = conf_df.at[0,'DFG6_restype']
    HPN7_num = int(conf_df.at[0,'HPN7_num'] -1)
    HRD_num = int(conf_df.at[0,'HRD_num'] -1)
    Arg_num = int(conf_df.at[0,'Arg_num'] -1)
    Arg_restype = conf_df.at[0,'Arg_restype']
    XHRD_num = int(conf_df.at[0,'XHRD_num'] -1)
    XHRD_restype = conf_df.at[0,'XHRD_restype']
    XDFG_num = int(conf_df.at[0,'XDFG_num'] -1)
    XDFG_restype = conf_df.at[0,'XDFG_restype']
    Asp_num = int(conf_df.at[0,'Asp_num'] -1)
    Phe_num = int(conf_df.at[0,'Phe_num'] -1)
    Phe_restype = conf_df.at[0,'Phe_restype']
    Hinge1_num = int(conf_df.at[0,'Hinge1_num'] -1)
    Gly_num = int(conf_df.at[0,'Gly_num'] -1)
    group=conf_df.at[0, 'Group']
    plddt_cutoff_lower = FLAGS.plddt_target
    plddt_cutoff_upper = FLAGS.plddt_target
    if FLAGS.target_state == 'active':
        d1_cutoff_lower=0.
        d1_cutoff_upper=11.
        d2_cutoff_lower=11.
        d2_cutoff_upper=20.
        nt_hbondcutoff_lower=0.
        nt_hbondcutoff_upper=3.6
        KEtype = Lys_restype + Glu_restype
        KEcutoff_lower=0.
        KEcutoff_upper=999.
        if KEtype == 'KE' or KEtype == 'RE' : KEcutoff_upper=nt_hbondcutoff_upper
        if KEtype == 'KD' or KEtype == 'RD' : KEcutoff_upper=nt_hbondcutoff_upper+1.5
        if KEtype == 'KN' or KEtype == 'RN' : KEcutoff_upper=10.0  # no saltbridge if Asn
    if FLAGS.target_state == 'active':
        chi1_cutoff_lower=240.
        chi1_cutoff_upper=360.
    spatial_cutoff_lower=0.
    spatial_cutoff_upper=0.45
    ape9cutoff_lower=0.
    score=conf_df.at[0, 'Score']
    if group == "TYR": ape9cutoff_upper=8.0
    else:ape9cutoff_upper=6.0

    for i in range(n_steps):  # One optimization step.
        
        # Prepare for loss function.
        # if FLAGS.protein_type == 'kinase':
        (loss, aux), grads = jax.value_and_grad(afe_loss_fn,
                                        has_aux=True)(afe_weights, af_features)
        # print(f'grads: {grads}')
         
        updates, opt_state = optimizer.update(updates=grads, state=opt_state)
        afe_weights = optax.apply_updates(params=afe_weights, updates=updates)
        
        # records for statistics
        plddt_loss_record = aux[1]
        plddt_record = aux[2]
        plddt_actloop_loss_record = aux[3]
        plddt_actloop_region_record = aux[4]
        d1_loss_record = aux[5]
        d2_loss_record = aux[6]
        d1_record = aux[7]
        d2_record = aux[8]
        chi1_loss_record = aux[9]
        chi1_record = aux[10]
        spatial_cluster_loss_record = aux[11]
        dihedral_dis_nochi1_record = float(aux[12][0])
        ActLoopNT_in_loss_record = aux[13]
        dis_xhrd_record = aux[14]
        ActLoopCT_in_loss_record = aux[15]
        dis_ape9_record = aux[16]
        SaltBr_in_loss_record = aux[17]
        dis_saltbridge_record = aux[18]
        
        loss_records.append(loss)
        plddt_loss_records.append(plddt_loss_record)
        plddt_records.append(plddt_record)
        plddt_actloop_loss_records.append(plddt_actloop_loss_record)
        plddt_actloop_region_records.append(plddt_actloop_region_record)
        d1_loss_records.append(d1_loss_record)
        d2_loss_records.append(d2_loss_record)
        d1_records.append(d1_record)
        d2_records.append(d2_record)
        chi1_loss_records.append(chi1_loss_record)
        chi1_records.append(chi1_record)
        spatial_cluster_loss_records.append(spatial_cluster_loss_record)
        dihedral_dis_nochi1_records.append(dihedral_dis_nochi1_record)
        ActLoopNT_in_loss_records.append(ActLoopNT_in_loss_record)
        dis_xhrd_records.append(dis_xhrd_record)
        ActLoopCT_in_loss_records.append(ActLoopCT_in_loss_record)
        dis_ape9_records.append(dis_ape9_record)
        SaltBr_in_loss_records.append(SaltBr_in_loss_record)
        dis_saltbridge_records.append(dis_saltbridge_record)
        
        print('Step:', i+1, '|total_loss:', loss, '|best loss:', np.min(loss_records),
              '|plddt_loss', plddt_loss_record, '|plddt:', plddt_record, '|best plddt:', np.max(plddt_records),
              '|plddt_actloop_loss', plddt_actloop_loss_record, '|plddt_actloop_region:', plddt_actloop_region_record, '|best plddt_actloop:', np.max(plddt_actloop_region_records),
              '|d1_loss:', d1_loss_record, '|best d1_loss:', np.min(d1_loss_records),
              '|d2_loss:', d2_loss_record, '|best d2_loss:', np.min(d2_loss_records),
              '|d1:', d1_record, '|d2:', d2_record,
              '|chi1_loss:', chi1_loss_record, '|best chi1_loss:', np.min(chi1_loss_records), '|chi1:', chi1_record,
              '|spatial_cluster_loss:', spatial_cluster_loss_record, '|best spatial_cluster_loss:', np.min(spatial_cluster_loss_records),
              '|dihedral_dis_nochi1:', dihedral_dis_nochi1_record,
              '|ActLoopNT_in_loss:', ActLoopNT_in_loss_record, '|best ActLoopNT_in_loss:', np.min(ActLoopNT_in_loss_records),
              '|dis_xhrd:', dis_xhrd_record,
              '|ActLoopCT_in_loss:', ActLoopCT_in_loss_record, '|best ActLoopCT_in_loss:', np.min(ActLoopCT_in_loss_records),
              '|dis_ape9:', dis_ape9_record,
              '|SaltBr_in_loss:', SaltBr_in_loss_record, '|best SaltBr_in_loss:', np.min(SaltBr_in_loss_records),
              '|dis_saltbridge:', dis_saltbridge_record)

        sys.stdout.flush()
        
        metric_df = pd.DataFrame({'total_loss': loss_records, 'plddt_loss': plddt_loss_records, 
                                      'plddt': plddt_records, 'plddt_actloop_loss': plddt_actloop_loss_records,
                                      'plddt_actloop_region': plddt_actloop_region_records,
                                      'd1_loss': d1_loss_records, 'd2_loss': d2_loss_records, 'd1': d1_records, 'd2': d2_records,
                                      'chi1_loss': chi1_loss_records, 'chi1': chi1_records,
                                      'spatial_cluster_loss': spatial_cluster_loss_records, 'dihedral_dis_nochi1': dihedral_dis_nochi1_records,
                                      'ActLoopNT_in_loss': ActLoopNT_in_loss_records, 'dis_xhrd': dis_xhrd_records,
                                      'ActLoopCT_in_loss': ActLoopCT_in_loss_records,'dis_ape9': dis_ape9_records,
                                      'SaltBr_in_loss': SaltBr_in_loss_records, 'dis_saltbridge': dis_saltbridge_records})
    
        metric_df.to_csv(FLAGS.output_dir+'/metrics_' +
                     str(learning_rate)+'.csv', index=None)
        
        # Generate and Save PDB
        p = protein.from_prediction(features=af_features,
                                result=aux[0],
                                b_factors=None,
                                # True for Monomer.
                                remove_leading_feature_dimension=True, )

        with open(os.path.join(FLAGS.output_dir, f'afe_model_{i+1}.pdb'), 'w') as f:
                    f.write(protein.to_pdb(p))
        
        # Check state and break out
        if loss == 0:
            count += 1
            print('target state threshold reached.')
        else: count = 0

        if count > FLAGS.num_success:
            print('target state threshold reached for %s times.' %
                      (FLAGS.num_success))
            break
        else: print(f'target state threshold reached for {count} times.')        
        

def main(argv):
    """AF2 inference."""
    # Sanity checks: Args count.
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    
    print(f'learning rate: {FLAGS.learning_rate}')
    print(f'plddt_target: {FLAGS.plddt_target}')
        
    model_runner = get_afe_runner(afparam_dir=FLAGS.afparam_dir,
                                  model_name=config.MODEL_PRESETS[FLAGS.model_preset][0],
                                  num_cluster=FLAGS.nclust, )

    # Load featurized MSAs.
    raw_feat = np.load(os.path.join(FLAGS.rawfeat_dir, 'features.pkl'), allow_pickle=True)
    # Process to real features. 
    feat = model_runner.process_features(raw_features=raw_feat,
                                         random_seed=123, )

    # feat = jnp.load(os.path.join(FLAGS.rawfeat_dir, 'processed_features.pkl'), allow_pickle=True)
    afe_fitting(afe_runner=model_runner,
                af_features=feat,
                n_steps=FLAGS.nsteps,
                learning_rate=FLAGS.learning_rate)
        
# %% 
if __name__ == '__main__':
    flags.mark_flags_as_required([
        'rawfeat_dir',
        'output_dir',
        'afparam_dir',
        'nsteps',
        'nclust',
        'plddt_target'
    ])
    app.run(main)


