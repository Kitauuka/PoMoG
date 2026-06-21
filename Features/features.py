"""
PoMoG （）。

：
1)  FASTA ；
2)  ESMFold、ProtTrans、DSSP ；
3) ，。

：
- ： argparse ；
- ：；
- ：，。
"""

import os
import argparse
import logging
import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import numpy as np
from tqdm import tqdm
import gc
from Bio import pairwise2
import sys
import subprocess

LOGGER = logging.getLogger("pomog.feature_prep")


DEFAULT_PATHS = {
    # DSSP （）。
    "mkdssp": "./Features/dssp-2.0.4/mkdssp",
    #  ProtT5 （transformers.from_pretrained ）。
    "prot_t5_model": "./models/Prot-T5-XL-U50",
    #  ESMFold checkpoint。ESMFold v1  ESM2 3B  + folding trunk。
    "esmfold_model": "./models/esmfold_3B_v1.pt",
    # ESMFold  ESM2 3B checkpoint。
    "esmfold_esm2_model": "./models/ESM2/esm2_t36_3B_UR50D.pt",
    # ESMFold （ get_esmfold ）。
    "esmfold_script": "./Features/esmfold/esmfold.py",
    # （）。
    "data_root": "./data",
    # ：processed（/）。
    "default_save_stage": "processed",
    # ：pomog。
    "default_task_name": "pomog",
    # （ data_root + stage + task_name ）。
    "save_dir": "./data/processed/pomog",
}


def _ensure_logger(save_dir=None):
    """（ + ）。

    ：
    -  handler，；
    -  save_dir， feature_prep.log。
    """
    if LOGGER.handlers:
        return

    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(save_dir, "feature_prep.log"), encoding="utf-8")
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)


def _log_info(message):
    _ensure_logger()
    LOGGER.info(message)


def _log_warning(message):
    _ensure_logger()
    LOGGER.warning(message)


def _log_error(message):
    _ensure_logger()
    LOGGER.error(message)



def _read_fasta_as_dict(fasta_path):
    """
     FASTA  {protein_id: sequence} 。

    ：
    -  FASTA（）；
    - ；
    -  protein_id ，（）。

    :
        fasta_path: FASTA 。

    :
        dict[str, str]:  ID 。
    """
    protein_dict = {}
    current_id = None
    seq_chunks = []

    with open(fasta_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith('>'):
                if current_id is not None:
                    protein_dict[current_id] = "".join(seq_chunks).strip().upper()
                current_id = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)

    if current_id is not None:
        protein_dict[current_id] = "".join(seq_chunks).strip().upper()

    return protein_dict


def _resolve_save_dir(save_dir, save_stage):
    """
    。

    ：
    1)  --save_dir，；
    2)  data/<save_stage>/<task_name>。

    :
        save_dir: （ None）。
        save_stage: "interim"  "processed"。

    :
        str: 。
    """
    if save_dir:
        return save_dir
    return os.path.join(DEFAULT_PATHS["data_root"], save_stage, DEFAULT_PATHS["default_task_name"])


def _parse_gpu_ids(gpu_ids_arg, gpu_arg):
    """
     GPU ， --gpu（） --gpu_ids（）。

    ：
    -  --gpu_ids（ "0,1,2"）， int；
    -  --gpu（ 0， -1  CPU-only）；
    - ， DataParallel 。

    :
        gpu_ids_arg: 。
        gpu_arg: 。

    :
        list[int]:  GPU 。
    """
    if gpu_ids_arg:
        ids = []
        for token in gpu_ids_arg.split(','):
            token = token.strip()
            if token:
                ids.append(int(token))
        if not ids:
            raise ValueError("--gpu_ids ")
    else:
        ids = [gpu_arg]

    # 。
    seen = set()
    uniq = []
    for gid in ids:
        if gid not in seen:
            uniq.append(gid)
            seen.add(gid)
    return uniq


def check_missing_features(
    protein_ids,
    structure_dir,
    prottrans_dir,
    dssp_dir,
    extra_tensor_dirs=None,
    extra_json_dirs=None,
    require_structure=True,
    require_prottrans=True,
    require_dssp=True,
):
    """
     protein_id ，。

    ：
    - : Structures/{id}.tensor（ require_structure ）
    - ProtTrans: ProtTrans/{id}.tensor（ require_prottrans ）
    - DSSP: DSSP/{id}.tensor（ require_dssp ）
    - extra_tensor_dirs:  `{id}.tensor` （ ESM2_3B）
    - extra_json_dirs:  `{id}.json` （ ESMFoldConfidence）

     ProtTrans （Max/Min）。

    :
        protein_ids:  ID 。
        structure_dir: 。
        prottrans_dir: ProtTrans 。
        dssp_dir: DSSP 。
        extra_tensor_dirs:  tensor 。
        require_structure: 。
        require_prottrans:  ProtTrans 。
        require_dssp:  DSSP 。

    :
        missing_ids:  ID 。
        need_stats:  ProtTrans 。
    """
    missing_ids = []
    extra_tensor_dirs = list(extra_tensor_dirs or [])
    extra_json_dirs = list(extra_json_dirs or [])
    for protein_id in protein_ids:
        required_files = []
        if require_structure:
            required_files.append(os.path.join(structure_dir, f"{protein_id}.tensor"))
        if require_prottrans:
            required_files.append(os.path.join(prottrans_dir, f"{protein_id}.tensor"))
        if require_dssp:
            required_files.append(os.path.join(dssp_dir, f"{protein_id}.tensor"))
        extra_ok = all(os.path.exists(os.path.join(extra_dir, f"{protein_id}.tensor")) for extra_dir in extra_tensor_dirs)
        extra_json_ok = all(os.path.exists(os.path.join(extra_dir, f"{protein_id}.json")) for extra_dir in extra_json_dirs)
        if not (all(os.path.exists(path) for path in required_files) and extra_ok and extra_json_ok):
            missing_ids.append(protein_id)

    stat_max = os.path.join(prottrans_dir, "Max_ProtTrans.npy")
    stat_min = os.path.join(prottrans_dir, "Min_ProtTrans.npy")
    need_stats = require_prottrans and not (os.path.exists(stat_max) and os.path.exists(stat_min))
    return missing_ids, need_stats


def _write_temp_fasta(temp_fasta_path, protein_dict, selected_ids):
    """
     ID  FASTA。

     only-missing ： FASTA，
    。

    :
        temp_fasta_path:  FASTA 。
        protein_dict:  {id: sequence} 。
        selected_ids:  ID 。
    """
    with open(temp_fasta_path, "w", encoding="utf-8") as f:
        for protein_id in selected_ids:
            f.write(f">{protein_id}\n{protein_dict[protein_id]}\n")


def compute_prottrans_normalization_stats(prottrans_dir):
    """
     ProtTrans /。

    ：
    - Max_ProtTrans.npy
    - Min_ProtTrans.npy

    。

    :
        prottrans_dir: ProtTrans 。
    """
    files = [f for f in os.listdir(prottrans_dir) if f.endswith('.tensor')]
    if not files:
        _log_warning("ProtTrans ： .tensor ")
        return

    first = torch.load(os.path.join(prottrans_dir, files[0]))
    first_arr = first.numpy() if hasattr(first, 'numpy') else first
    if first_arr.ndim != 2:
        _log_warning("ProtTrans ：")
        return

    dim = first_arr.shape[1]
    global_max = np.full(dim, -1e9)
    global_min = np.full(dim, 1e9)

    for fname in tqdm(files, desc='ProtTrans Stats'):
        arr_obj = torch.load(os.path.join(prottrans_dir, fname))
        arr = arr_obj.numpy() if hasattr(arr_obj, 'numpy') else arr_obj
        global_max = np.maximum(global_max, np.max(arr, axis=0))
        global_min = np.minimum(global_min, np.min(arr, axis=0))

    np.save(os.path.join(prottrans_dir, 'Max_ProtTrans.npy'), global_max)
    np.save(os.path.join(prottrans_dir, 'Min_ProtTrans.npy'), global_min)
    _log_info(' ProtTrans : Max_ProtTrans.npy / Min_ProtTrans.npy')


def compute_lm_normalization_stats(feature_dir, stat_prefix):
    """
    /。

    ：
    - Max_{stat_prefix}.npy
    - Min_{stat_prefix}.npy
    """
    files = [f for f in os.listdir(feature_dir) if f.endswith('.tensor')]
    if not files:
        _log_warning(f"{stat_prefix} ： .tensor ")
        return

    first = torch.load(os.path.join(feature_dir, files[0]), map_location="cpu")
    first_arr = first.numpy() if hasattr(first, 'numpy') else first
    if first_arr.ndim != 2:
        _log_warning(f"{stat_prefix} ：")
        return

    dim = first_arr.shape[1]
    global_max = np.full(dim, -1e9)
    global_min = np.full(dim, 1e9)

    for fname in tqdm(files, desc=f'{stat_prefix} Stats'):
        arr_obj = torch.load(os.path.join(feature_dir, fname), map_location="cpu")
        arr = arr_obj.numpy() if hasattr(arr_obj, 'numpy') else arr_obj
        global_max = np.maximum(global_max, np.max(arr, axis=0))
        global_min = np.minimum(global_min, np.min(arr, axis=0))

    np.save(os.path.join(feature_dir, f'Max_{stat_prefix}.npy'), global_max)
    np.save(os.path.join(feature_dir, f'Min_{stat_prefix}.npy'), global_min)
    _log_info(f' {stat_prefix} : Max_{stat_prefix}.npy / Min_{stat_prefix}.npy')


# === DSSP  ===

def process_dssp(dssp_file_path):
    """
     .dssp 。

    （ 9 ）：
    -  1 ： ASA（， [0, 1]）；
    -  2~9 ：8  one-hot（H/B/E/G/I/T/S/C）。

    :
        dssp_file_path: DSSP 。

    :
        seq: （）。
        features: numpy.float32 ，shape  [L, 9]。
    """
    aa_type = "ACDEFGHIKLMNPQRSTVWY"
    SS_type = "HBEGITSC"
    # MaxASA （Tien et al., 2013）。
    rASA_std = {
        'A': 129.0, 'R': 274.0, 'N': 195.0, 'D': 193.0, 'C': 167.0,
        'E': 223.0, 'Q': 225.0, 'G': 104.0, 'H': 224.0, 'I': 197.0,
        'L': 201.0, 'K': 236.0, 'M': 224.0, 'F': 240.0, 'P': 159.0,
        'S': 155.0, 'T': 172.0, 'W': 285.0, 'Y': 263.0, 'V': 174.0
    }

    with open(dssp_file_path, "r") as f:
        lines = f.readlines()

    seq = ""
    dssp_feature = []

    p = 0
    #  DSSP （ DSSP ）。
    #  RESIDUE ，："#  RESIDUE AA STRUCTURE ..."。
    while p < len(lines):
        header = lines[p].lstrip()
        if header.startswith("#") and "RESIDUE" in header:
            break
        p += 1

    # ： RESIDUE ， # 。
    if p == len(lines):
        p = 0
        while p < len(lines) and not lines[p].lstrip().startswith("#"):
            p += 1

    if p == len(lines):
        raise ValueError(f" {dssp_file_path}  DSSP  '#'。")

    for i in range(p + 1, len(lines)):
        line = lines[i]
        # 。
        if len(line) < 17 or line[13] == "!" or line[13] == "*":
            continue

        aa = line[13]
        if aa.islower():  # DSSP 。
            aa = 'C'

        if aa not in aa_type:
            continue

        SS = line[16]
        if SS == " ":
            SS = "C"  #  Coil 。

        SS_vec = np.zeros(8)
        ss_idx = SS_type.find(SS)
        if ss_idx != -1:
            SS_vec[ss_idx] = 1

        try:
            ACC = float(line[34:38].strip())
        except ValueError:
            # （），。
            # ： seq， seq  feature 。
            continue

        # ，，。
        seq += aa
        #  MaxASA 。
        ASA = min(1.0, ACC / rASA_std.get(aa, 1.0))
        dssp_feature.append(np.concatenate((np.array([ASA]), SS_vec)))

    return seq, np.array(dssp_feature, dtype=np.float32)


def match_dssp(seq, dssp_matrix, ref_seq):
    """
    （globalxx） DSSP 。

    ：
    - DSSP ， FASTA /；
    -  0 ，。

    :
        seq: DSSP 。
        dssp_matrix:  seq （list  ndarray）。
        ref_seq: FASTA 。

    :
         numpy.float32 ，shape  [len(ref_seq), 9]。
    """
    # pairwise2 ；。
    alignments = pairwise2.align.globalxx(ref_seq, seq)
    if not alignments:
        raise ValueError("，。")

    ref_aligned = alignments[0].seqA
    dssp_aligned = alignments[0].seqB

    padded_item = np.zeros(9, dtype=np.float32)

    new_dssp = []
    dssp_idx = 0
    for aa in dssp_aligned:
        if aa == "-":
            new_dssp.append(padded_item)
        else:
            if dssp_idx < len(dssp_matrix):
                new_dssp.append(dssp_matrix[dssp_idx])
                dssp_idx += 1
            else:  #  DSSP ， 0 。
                new_dssp.append(padded_item)

    matched_dssp = []
    for i in range(len(ref_aligned)):
        if ref_aligned[i] != "-":
            matched_dssp.append(new_dssp[i])

    return np.array(matched_dssp, dtype=np.float32)


def get_dssp(fasta_path, dssp_exec_path, pdb_dir, dssp_dir):
    """
     FASTA  DSSP 。

    ：
    1)  ID  `pdb_dir`  .pdb；
    2)  `mkdssp`  .dssp；
    3)  DSSP  FASTA ；
    4)  `.tensor`（）；
    5)  .dssp，。

    :
        fasta_path: FASTA 。
        dssp_exec_path: mkdssp 。
        pdb_dir: PDB （ `{protein_id}.pdb`）。
        dssp_dir: （ `{protein_id}.tensor`）。
    """
    msg = f"\n---  DSSP  (: {dssp_dir}) ---"
    print(msg)
    _log_info(msg)
    if not os.path.exists(dssp_exec_path):
        raise FileNotFoundError(f"DSSP: {dssp_exec_path}")

    os.makedirs(dssp_dir, exist_ok=True)

    with open(fasta_path, 'r') as f:
        lines = f.readlines()

    #  FASTA  {protein_id: sequence} 。
    protein_dict = {}
    for i in range(0, len(lines), 2):
        if lines[i].startswith('>'):
            protein_id = lines[i][1:].strip()
            seq = lines[i + 1].strip()
            protein_dict[protein_id] = seq

    dssp_success_ids = []
    dssp_failed_ids = []

    for protein_id, ref_seq in tqdm(protein_dict.items(), desc="DSSP calculation"):
        pdb_file = os.path.join(pdb_dir, f"{protein_id}.pdb")
        dssp_file = os.path.join(dssp_dir, f"{protein_id}.dssp")
        tensor_file = os.path.join(dssp_dir, f"{protein_id}.tensor")

        if not os.path.exists(pdb_file):
            warn_msg = f":  {protein_id} PDB，DSSP。"
            print(warn_msg)
            _log_warning(warn_msg)
            dssp_failed_ids.append(protein_id)
            continue

        #  mkdssp  .dssp 。
        command = f"{dssp_exec_path} -i {pdb_file} -o {dssp_file}"
        os.system(command)

        if not os.path.exists(dssp_file):
            warn_msg = f": DSSP {dssp_file} 。"
            print(warn_msg)
            _log_warning(warn_msg)
            dssp_failed_ids.append(protein_id)
            continue

        try:
            dssp_seq, dssp_matrix = process_dssp(dssp_file)

            #  DSSP ，。
            if dssp_seq != ref_seq:
                dssp_matrix = match_dssp(dssp_seq, list(dssp_matrix), ref_seq)

            if len(dssp_matrix) != len(ref_seq):
                warn_msg = f":  {protein_id} DSSP ({len(dssp_matrix)})  ({len(ref_seq)}) 。"
                print(warn_msg)
                _log_warning(warn_msg)
                dssp_failed_ids.append(protein_id)
                continue

            torch.save(torch.tensor(dssp_matrix, dtype=torch.float32), tensor_file)
            dssp_success_ids.append(protein_id)
        except Exception as e:
            err_msg = f" {protein_id} DSSP: {e}"
            print(err_msg)
            _log_error(err_msg)
            dssp_failed_ids.append(protein_id)
        finally:
            #  .dssp 。
            if os.path.exists(dssp_file):
                os.remove(dssp_file)

    print("✅ DSSP ！")
    _log_info("✅ DSSP ！")
    _log_info(f"DSSP:  {len(protein_dict)} ,  {len(dssp_success_ids)} ,  {len(dssp_failed_ids)} ")
    if dssp_success_ids:
        _log_info("DSSPID: " + ", ".join(dssp_success_ids))
    if dssp_failed_ids:
        _log_warning("DSSPID: " + ", ".join(dssp_failed_ids))


# === ProtTrans  ===

def get_prottrans(fasta_path, output_dir, model_path, gpu_ids):
    """
     FASTA  ProtTrans（ProtT5）。

    ：
    - （T5  token ）；
    -  U/Z/O/B  X；
    -  attention mask  padding， token `</s>`；
    -  `.tensor`。

    :
        fasta_path: FASTA 。
        output_dir: 。
        model_path:  ProtTrans 。
        gpu_ids: GPU ；/， CPU。
    """
    msg = f"\n---  ProtTrans  (: {output_dir}) ---"
    print(msg)
    _log_info(msg)
    os.makedirs(output_dir, exist_ok=True)

    protein_dict = {}
    with open(fasta_path, "r") as f:
        lines = f.readlines()
    for i in range(0, len(lines), 2):
        if lines[i].startswith('>'):
            protein_id = lines[i][1:].strip()
            seq = lines[i + 1].strip()
            protein_dict[protein_id] = seq

    print(" ProtTrans ...")
    _log_info(" ProtTrans ...")
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(model_path)
    except Exception as e:
        err_msg = f":  ProtTrans ， {model_path}。: {e}"
        print(err_msg)
        _log_error(err_msg)
        return

    gc.collect()

    available_gpu_ids = [gid for gid in gpu_ids if gid is not None and gid >= 0]
    use_multi_gpu = torch.cuda.is_available() and len(available_gpu_ids) > 1

    if torch.cuda.is_available() and available_gpu_ids:
        device = torch.device(f'cuda:{available_gpu_ids[0]}')
    else:
        device = torch.device('cpu')

    model = model.to(device)
    if use_multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=available_gpu_ids)
    model = model.eval()

    print(f"ProtTrans : {device}")
    _log_info(f"ProtTrans : {device}")
    if use_multi_gpu:
        _log_info(f"ProtTrans GPU: {available_gpu_ids}")

    # ：batch_size=2 ； 1。
    batch_size = 2

    # ，。
    protein_items = list(protein_dict.items())
    prottrans_success_ids = []
    prottrans_failed_ids = []

    for i in tqdm(range(0, len(protein_items), batch_size), desc="ProtTrans Embeddings"):
        batch_items = protein_items[i:i + batch_size]
        batch_ids = [item[0] for item in batch_items]
        batch_seqs = [item[1] for item in batch_items]

        # ：。
        sequences_for_t5 = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in batch_seqs]

        # 。
        ids = tokenizer.batch_encode_plus(sequences_for_t5, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # 。
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)

        embedding = embedding.last_hidden_state.cpu()

        # （ ID ）。
        for j in range(len(batch_ids)):
            seq_len = (attention_mask[j] == 1).sum()
            seq_emd = embedding[j][:seq_len - 1]  #  </s> token。

            tensor_file = os.path.join(output_dir, f"{batch_ids[j]}.tensor")
            try:
                torch.save(seq_emd, tensor_file)
                prottrans_success_ids.append(batch_ids[j])
            except Exception as e:
                prottrans_failed_ids.append(batch_ids[j])
                _log_error(f" ProtTrans  {batch_ids[j]}: {e}")

    print("✅ ProtTrans ！")
    _log_info("✅ ProtTrans ！")
    _log_info(f"ProtTrans:  {len(protein_items)} ,  {len(prottrans_success_ids)} ,  {len(prottrans_failed_ids)} ")
    if prottrans_success_ids:
        _log_info("ProtTransID: " + ", ".join(prottrans_success_ids))
    if prottrans_failed_ids:
        _log_warning("ProtTransID: " + ", ".join(prottrans_failed_ids))


# === ESMFold  ===

def get_pdb_xyz(pdb_lines):
    """
     PDB ， [L, 5, 3] 。

    5 ：
    - ：N, CA, C, O；
    -  R：。

    ：
    - ， CA ；
    - ， 0 。
    """
    current_pos = -1000
    X = []
    current_aa = {}
    zero_residue = [
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    ]

    for line in pdb_lines:
        # “”， 5 。
        if (line.startswith("ATOM") and int(line[22:26].strip()) != current_pos) or line.startswith("TER"):
            if current_aa:
                # 。
                r_group_coords = [current_aa[atom] for atom in current_aa if atom not in ["N", "CA", "C", "O"]]
                if not r_group_coords:
                    r_group_coords = [current_aa.get("CA", np.zeros(3))]  # ， CA。

                r_group_center = np.mean(r_group_coords, axis=0)

                # （ 0）。
                n_coord = current_aa.get("N", np.zeros(3))
                ca_coord = current_aa.get("CA", np.zeros(3))
                c_coord = current_aa.get("C", np.zeros(3))
                o_coord = current_aa.get("O", np.zeros(3))

                X.append([n_coord, ca_coord, c_coord, o_coord, r_group_center])
                current_aa = {}

            if line.startswith("TER"):
                break

            next_pos = int(line[22:26].strip())
            if current_pos != -1000 and next_pos > current_pos + 1:
                for _ in range(current_pos + 1, next_pos):
                    X.append(zero_residue)
            current_pos = next_pos

        if line.startswith("ATOM"):
            atom_name = line[12:16].strip()
            if atom_name in ["N", "CA", "C", "O"] or not atom_name.startswith("H"):
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float32)
                current_aa[atom_name] = xyz

    #  PDB  TER，，
    #  1。
    if current_aa:
        r_group_coords = [current_aa[atom] for atom in current_aa if atom not in ["N", "CA", "C", "O"]]
        if not r_group_coords:
            r_group_coords = [current_aa.get("CA", np.zeros(3))]

        r_group_center = np.mean(r_group_coords, axis=0)
        n_coord = current_aa.get("N", np.zeros(3))
        ca_coord = current_aa.get("CA", np.zeros(3))
        c_coord = current_aa.get("C", np.zeros(3))
        o_coord = current_aa.get("O", np.zeros(3))
        X.append([n_coord, ca_coord, c_coord, o_coord, r_group_center])

    return np.array(X, dtype=np.float32)


def get_esmfold(
    fasta_path,
    output_dir,
    gpu_ids,
    esmfold_chunk_size=64,
    esmfold_cpu_offload=False,
    esmfold_model_path=None,
    esmfold_esm2_model_path=None,
    save_esmfold_esm2=False,
    esmfold_esm2_output_dir=None,
    esmfold_esm2_repr_layer=-1,
    save_esmfold_confidence=False,
    esmfold_confidence_output_dir=None,
    esmfold_pae_output_dir=None,
    esmfold_plddt_output_dir=None,
    esmfold_confidence_tensor_dtype="float16",
):
    """
     ESMFold  3D ， tensor。

    ：
    - `{protein_id}.pdb`：（ESMFold ）；
    - `{protein_id}.tensor`： PDB  [L, 5, 3] 。

    :
        fasta_path: FASTA 。
        output_dir: （ .pdb  .tensor）。
        gpu_ids: ESMFold  GPU ；。
        esmfold_chunk_size: ESMFold axial attention ，。
        esmfold_cpu_offload:  ESMFold CPU offload（ OOM ）。
        esmfold_model_path:  ESMFold checkpoint。
        esmfold_esm2_model_path: ESMFold  ESM2 3B checkpoint。
        save_esmfold_esm2:  ESMFold  ESM2 3B 。
        esmfold_esm2_output_dir: ESM2 3B 。
        esmfold_esm2_repr_layer:  ESM2 ；-1 。
        save_esmfold_confidence:  ESMFold confidence （PAE、pLDDT、pTM ）。
        esmfold_confidence_output_dir: confidence JSON 。
        esmfold_pae_output_dir: PAE tensor 。
        esmfold_plddt_output_dir: residue pLDDT tensor 。
        esmfold_confidence_tensor_dtype: PAE/pLDDT tensor  dtype。
    """
    msg = f"\n---  ESMFold 3D (: {output_dir}) ---"
    print(msg)
    _log_info(msg)
    esmfold_script_path = DEFAULT_PATHS['esmfold_script']

    #  ESMFold 。
    if not os.path.exists(esmfold_script_path):
        raise FileNotFoundError(f"ESMFold: {esmfold_script_path}")

    os.makedirs(output_dir, exist_ok=True)

    protein_dict = _read_fasta_as_dict(fasta_path)
    protein_ids = list(protein_dict.keys())
    available_gpu_ids = [gid for gid in gpu_ids if gid is not None and gid >= 0]

    # ：
    # - ：；
    # - ： protein_id ，。
    if len(available_gpu_ids) <= 1:
        gpu_to_use = available_gpu_ids[0] if available_gpu_ids else -1
        command = (
            f"{sys.executable} {esmfold_script_path} "
            f"-i {fasta_path} "
            f"-o {output_dir} "
            f"--gpu {gpu_to_use} "
            f"--chunk-size {esmfold_chunk_size}"
        )
        if esmfold_model_path:
            command += f" --model-path {esmfold_model_path}"
        if esmfold_esm2_model_path:
            command += f" --esm2-model-path {esmfold_esm2_model_path}"
        if save_esmfold_esm2:
            command += (
                f" --save-esm2-embeddings"
                f" --esm2-output {esmfold_esm2_output_dir}"
                f" --esm2-repr-layer {esmfold_esm2_repr_layer}"
            )
        if save_esmfold_confidence:
            command += (
                f" --save-confidence"
                f" --confidence-output {esmfold_confidence_output_dir}"
                f" --pae-output {esmfold_pae_output_dir}"
                f" --plddt-output {esmfold_plddt_output_dir}"
                f" --confidence-tensor-dtype {esmfold_confidence_tensor_dtype}"
            )
        if esmfold_cpu_offload and gpu_to_use >= 0:
            command += " --cpu-offload"
        print(f": {command}")
        _log_info(f": {command}")
        os.system(command)
    else:
        if esmfold_cpu_offload:
            _log_warning("ESMFold GPU cpu-offload，。")
        _log_info(f"ESMFold GPU: {available_gpu_ids}")
        procs = []
        temp_fastas = []
        gpu_count = len(available_gpu_ids)

        for idx, gpu_id in enumerate(available_gpu_ids):
            shard_ids = protein_ids[idx::gpu_count]
            if not shard_ids:
                continue

            shard_fasta = os.path.join(output_dir, f".tmp_esmfold_gpu{gpu_id}_{idx}.fasta")
            _write_temp_fasta(shard_fasta, protein_dict, shard_ids)
            temp_fastas.append(shard_fasta)

            command = (
                f"{sys.executable} {esmfold_script_path} "
                f"-i {shard_fasta} "
                f"-o {output_dir} "
                f"--gpu {gpu_id} "
                f"--chunk-size {esmfold_chunk_size}"
            )
            if esmfold_model_path:
                command += f" --model-path {esmfold_model_path}"
            if esmfold_esm2_model_path:
                command += f" --esm2-model-path {esmfold_esm2_model_path}"
            if save_esmfold_esm2:
                command += (
                    f" --save-esm2-embeddings"
                    f" --esm2-output {esmfold_esm2_output_dir}"
                    f" --esm2-repr-layer {esmfold_esm2_repr_layer}"
                )
            if save_esmfold_confidence:
                command += (
                    f" --save-confidence"
                    f" --confidence-output {esmfold_confidence_output_dir}"
                    f" --pae-output {esmfold_pae_output_dir}"
                    f" --plddt-output {esmfold_plddt_output_dir}"
                    f" --confidence-tensor-dtype {esmfold_confidence_tensor_dtype}"
                )
            _log_info(f"ESMFold : gpu={gpu_id}, ={len(shard_ids)}")
            procs.append((gpu_id, subprocess.Popen(command, shell=True)))

        for gpu_id, proc in procs:
            return_code = proc.wait()
            if return_code != 0:
                _log_error(f"ESMFold : gpu={gpu_id}, return_code={return_code}")
            else:
                _log_info(f"ESMFold : gpu={gpu_id}")

        failed_gpu_ids = [gpu_id for gpu_id, proc in procs if proc.returncode != 0]
        if failed_gpu_ids:
            raise RuntimeError(f"ESMFold ，GPU: {failed_gpu_ids}")

        for tmp in temp_fastas:
            if os.path.exists(tmp):
                os.remove(tmp)

    #  PDB  tensor。
    print("PDB...")
    esmfold_success_ids = []
    esmfold_failed_ids = []

    for protein_id in tqdm(protein_ids, desc="Processing PDBs"):
        pdb_file_path = os.path.join(output_dir, f"{protein_id}.pdb")
        tensor_file_path = os.path.join(output_dir, f"{protein_id}.tensor")

        if not os.path.exists(pdb_file_path):
            warn_msg = f": {protein_id}.pdb 。"
            print(warn_msg)
            _log_warning(warn_msg)
            esmfold_failed_ids.append(protein_id)
            continue

        with open(pdb_file_path, 'r') as f:
            pdb_lines = f.readlines()

        try:
            coords = get_pdb_xyz(pdb_lines)
            torch.save(torch.from_numpy(coords), tensor_file_path)
            esmfold_success_ids.append(protein_id)
        except Exception as e:
            err_msg = f" {protein_id}.pdb : {e}"
            print(err_msg)
            _log_error(err_msg)
            esmfold_failed_ids.append(protein_id)

    print("✅ ESMFold ！")
    _log_info("✅ ESMFold ！")
    _log_info(f"ESMFold:  {len(protein_ids)} ,  {len(esmfold_success_ids)} ,  {len(esmfold_failed_ids)} ")
    if esmfold_success_ids:
        _log_info("ESMFoldID: " + ", ".join(esmfold_success_ids))
    if esmfold_failed_ids:
        _log_warning("ESMFoldID: " + ", ".join(esmfold_failed_ids))


def main():
    """
    ：（ FASTA ）。

    ：
    -  FASTA ->  ；
    -  PKL/JSON， scripts/convert_data_dict_to_fasta.py  FASTA。

    ：
        ：
            python Features/features.py --fasta xxx.fasta --run_esmfold
         + DSSP：
            python Features/features.py --fasta xxx.fasta --run_esmfold --run_dssp
        ：
            python Features/features.py --fasta xxx.fasta --run_esmfold --run_prottrans --run_dssp
    """
    # ： FASTA 。
    parser = argparse.ArgumentParser(description="PoMoG")
    parser.add_argument("--fasta", type=str, required=True, help="FASTA")
    parser.add_argument("--gpu", type=int, default=0, help="： (: 0  -1 CPU)")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="：GPU， 0,1,2； --gpu")

    # ：。
    parser.add_argument("--run_esmfold", action='store_true', help="ESMFold3D")
    parser.add_argument("--run_prottrans", action='store_true', help="ProtTrans")
    parser.add_argument("--run_dssp", action='store_true', help="DSSP")
    parser.add_argument("--esmfold_chunk_size", type=int, default=64,
                        help="ESMFold chunk size，OOM 32/16")
    parser.add_argument("--esmfold_cpu_offload", action='store_true',
                        help="ESMFold  CPU offload，")
    parser.add_argument("--esmfold_model_path", type=str, default=DEFAULT_PATHS["esmfold_model"],
                        help="ESMFold checkpoint ")
    parser.add_argument("--esmfold_esm2_model_path", type=str, default=DEFAULT_PATHS["esmfold_esm2_model"],
                        help="ESMFold  ESM2 3B checkpoint ")
    parser.add_argument("--run_esmfold_esm2", action="store_true",
                        help=" ESMFold  ESM2 3B ")
    parser.add_argument("--esmfold_esm2_output_dir", type=str, default=None,
                        help="ESMFold  ESM2 3B ； save_dir/ESM2_3B")
    parser.add_argument("--esmfold_esm2_repr_layer", type=int, default=-1,
                        help=" ESMFold  ESM2 ；-1 ")
    parser.add_argument("--esmfold_esm2_stat_prefix", type=str, default="ESM2_3B",
                        help="ESMFold  ESM2 3B ")
    parser.add_argument("--run_esmfold_confidence", action="store_true",
                        help=" ESMFold  PAE、 pLDDT  pTM ")
    parser.add_argument("--esmfold_confidence_output_dir", type=str, default=None,
                        help="ESMFold confidence JSON ； save_dir/ESMFoldConfidence")
    parser.add_argument("--esmfold_pae_output_dir", type=str, default=None,
                        help="ESMFold PAE tensor ； save_dir/ESMFoldPAE")
    parser.add_argument("--esmfold_plddt_output_dir", type=str, default=None,
                        help="ESMFold residue pLDDT tensor ； save_dir/ESMFoldPlddt")
    parser.add_argument("--esmfold_confidence_tensor_dtype", type=str, default="float16",
                        choices=["float16", "float32"],
                        help="PAE/pLDDT  dtype； float16 ")

    # ： save_stage ， save_dir。
    parser.add_argument("--save_dir", type=str, default=None,
                        help="； --save_stage ")
    parser.add_argument("--save_stage", type=str, default=DEFAULT_PATHS["default_save_stage"],
                        choices=["interim", "processed"],
                        help=" --save_dir ， data/<save_stage>/<task>")

    parser.add_argument("--prottrans_model_path", type=str, default=DEFAULT_PATHS["prot_t5_model"],
                        help="ProtTrans")
    parser.add_argument("--dssp_exec_path", type=str, default=DEFAULT_PATHS["mkdssp"],
                        help="mkdssp")
    parser.add_argument("--only_missing", action="store_true",
                        help="")
    parser.add_argument("--compute_stats", action="store_true",
                        help=" ProtTrans ")

    args = parser.parse_args()

    try:
        args.gpu_ids_list = _parse_gpu_ids(args.gpu_ids, args.gpu)
    except Exception as e:
        print(f"invalid GPU options: {e}")
        sys.exit(1)

    args.save_dir = _resolve_save_dir(args.save_dir, args.save_stage)

    if args.run_esmfold_esm2 and not args.run_esmfold:
        print("--run_esmfold_esm2 requires --run_esmfold")
        sys.exit(1)
    if args.run_esmfold_confidence and not args.run_esmfold:
        print("--run_esmfold_confidence requires --run_esmfold")
        sys.exit(1)

    if not (args.run_esmfold or args.run_prottrans or args.run_dssp):
        print("choose at least one feature stage: --run_esmfold, --run_prottrans, or --run_dssp")
        sys.exit(1)

    if not os.path.exists(args.fasta):
        print(f"missing FASTA: {args.fasta}")
        sys.exit(1)

    _ensure_logger(args.save_dir)
    _log_info(f"feature log: {os.path.join(args.save_dir, 'feature_prep.log')}")
    structure_dir = os.path.join(args.save_dir, "Structures")
    prottrans_dir = os.path.join(args.save_dir, "ProtTrans")
    dssp_dir = os.path.join(args.save_dir, "DSSP")
    esmfold_esm2_dir = args.esmfold_esm2_output_dir or os.path.join(args.save_dir, "ESM2_3B")
    esmfold_confidence_dir = args.esmfold_confidence_output_dir or os.path.join(args.save_dir, "ESMFoldConfidence")
    esmfold_pae_dir = args.esmfold_pae_output_dir or os.path.join(args.save_dir, "ESMFoldPAE")
    esmfold_plddt_dir = args.esmfold_plddt_output_dir or os.path.join(args.save_dir, "ESMFoldPlddt")

    print("PoMoG feature generation")
    _log_info("PoMoG feature generation")
    print(f"  - FASTA: {args.fasta}")
    _log_info(f"  - FASTA: {args.fasta}")
    print(f"  - output: {args.save_dir}")
    print(f"  - GPU: {args.gpu_ids_list}")
    _log_info(f"  - output: {args.save_dir}")
    _log_info(f"  - GPU: {args.gpu_ids_list}")
    print(f"  - ESMFold chunk_size: {args.esmfold_chunk_size}")
    _log_info(f"  - ESMFold chunk_size: {args.esmfold_chunk_size}")
    print(f"  - ESMFold cpu_offload: {args.esmfold_cpu_offload}")
    _log_info(f"  - ESMFold cpu_offload: {args.esmfold_cpu_offload}")
    print(f"  - ESMFold checkpoint: {args.esmfold_model_path}")
    _log_info(f"  - ESMFold checkpoint: {args.esmfold_model_path}")
    print(f"  - ESMFold ESM2 checkpoint: {args.esmfold_esm2_model_path}")
    _log_info(f"  - ESMFold ESM2 checkpoint: {args.esmfold_esm2_model_path}")
    print(f"  - : {structure_dir}")
    _log_info(f"  - : {structure_dir}")
    if args.run_esmfold_esm2:
        print(f"  - ESMFold ESM2_3B : {esmfold_esm2_dir}")
        _log_info(f"  - ESMFold ESM2_3B : {esmfold_esm2_dir}")
    if args.run_esmfold_confidence:
        print(f"  - ESMFold confidence : {esmfold_confidence_dir}")
        print(f"  - ESMFold PAE : {esmfold_pae_dir}")
        print(f"  - ESMFold pLDDT : {esmfold_plddt_dir}")
        print(f"  - ESMFold confidence tensor dtype: {args.esmfold_confidence_tensor_dtype}")
        _log_info(f"  - ESMFold confidence : {esmfold_confidence_dir}")
        _log_info(f"  - ESMFold PAE : {esmfold_pae_dir}")
        _log_info(f"  - ESMFold pLDDT : {esmfold_plddt_dir}")
        _log_info(f"  - ESMFold confidence tensor dtype: {args.esmfold_confidence_tensor_dtype}")
    print(f"  - : {prottrans_dir}")
    _log_info(f"  - : {prottrans_dir}")
    print(f"  - DSSP: {dssp_dir}")
    _log_info(f"  - DSSP: {dssp_dir}")
    print("-" * 30)

    target_fasta = args.fasta
    temp_fasta_path = None
    need_stats = False

    if args.only_missing:
        all_protein_dict = _read_fasta_as_dict(target_fasta)
        extra_tensor_dirs = []
        if args.run_esmfold_esm2:
            extra_tensor_dirs.append(esmfold_esm2_dir)
        if args.run_esmfold_confidence:
            extra_tensor_dirs.extend([esmfold_pae_dir, esmfold_plddt_dir])
        missing_ids, need_stats = check_missing_features(
            protein_ids=list(all_protein_dict.keys()),
            structure_dir=structure_dir,
            prottrans_dir=prottrans_dir,
            dssp_dir=dssp_dir,
            extra_tensor_dirs=extra_tensor_dirs,
            extra_json_dirs=[esmfold_confidence_dir] if args.run_esmfold_confidence else None,
            require_structure=args.run_esmfold,
            require_prottrans=args.run_prottrans,
            require_dssp=args.run_dssp,
        )
        _log_info(f":  {len(all_protein_dict)} ,  {len(missing_ids)} ")

        if missing_ids:
            temp_fasta_path = os.path.join(args.save_dir, "temp_missing.fasta")
            _write_temp_fasta(temp_fasta_path, all_protein_dict, missing_ids)
            target_fasta = temp_fasta_path
            _log_info(f"，FASTA: {temp_fasta_path}")
        else:
            _log_info("，")

    try:
        if args.run_esmfold and (not args.only_missing or temp_fasta_path):
            get_esmfold(
                target_fasta,
                structure_dir,
                args.gpu_ids_list,
                esmfold_chunk_size=args.esmfold_chunk_size,
                esmfold_cpu_offload=args.esmfold_cpu_offload,
                esmfold_model_path=args.esmfold_model_path,
                esmfold_esm2_model_path=args.esmfold_esm2_model_path,
                save_esmfold_esm2=args.run_esmfold_esm2,
                esmfold_esm2_output_dir=esmfold_esm2_dir,
                esmfold_esm2_repr_layer=args.esmfold_esm2_repr_layer,
                save_esmfold_confidence=args.run_esmfold_confidence,
                esmfold_confidence_output_dir=esmfold_confidence_dir,
                esmfold_pae_output_dir=esmfold_pae_dir,
                esmfold_plddt_output_dir=esmfold_plddt_dir,
                esmfold_confidence_tensor_dtype=args.esmfold_confidence_tensor_dtype,
            )

        if args.run_prottrans and (not args.only_missing or temp_fasta_path):
            get_prottrans(target_fasta, prottrans_dir, args.prottrans_model_path, args.gpu_ids_list)

        if args.run_dssp and (not args.only_missing or temp_fasta_path):
            # DSSP  PDB ， structure_dir 。
            get_dssp(target_fasta, args.dssp_exec_path, structure_dir, dssp_dir)

        if args.run_prottrans and (args.compute_stats or need_stats):
            compute_prottrans_normalization_stats(prottrans_dir)

        if args.run_esmfold_esm2 and args.compute_stats:
            compute_lm_normalization_stats(esmfold_esm2_dir, args.esmfold_esm2_stat_prefix)

    finally:
        #  only_missing  FASTA。
        if temp_fasta_path and os.path.exists(temp_fasta_path):
            os.remove(temp_fasta_path)

    print("\n！")
    _log_info("！")


if __name__ == '__main__':
    main()
