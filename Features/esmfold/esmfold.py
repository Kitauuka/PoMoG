from pathlib import Path
import sys
import logging
import typing as T
import argparse
import json

import torch

import esm
import esm.esmfold.v1.pretrained as esmfold_pretrained
from esm.data import read_fasta
from timeit import default_timer as timer

"""
obtain the ESMFold-predicted structures
"""

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%y/%m/%d %H:%M:%S",
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

PathLike = T.Union[str, Path]


def _load_local_esm2_t36(model_path: Path):
    model_data = torch.load(str(model_path), map_location="cpu")
    return esm.pretrained.load_model_and_alphabet_core(model_path.stem, model_data, None)


def load_esmfold_model(model_path: T.Optional[Path], esm2_model_path: T.Optional[Path]):
    original_esm2_loader = esm.pretrained.esm2_t36_3B_UR50D

    if esm2_model_path is not None and esm2_model_path.exists():
        logger.info(f"Loading ESM2 t36 3B language model from {esm2_model_path}")

        def _local_esm2_loader():
            return _load_local_esm2_t36(esm2_model_path)

        esm.pretrained.esm2_t36_3B_UR50D = _local_esm2_loader
    elif esm2_model_path is not None:
        raise FileNotFoundError(f"ESM2 model path not found: {esm2_model_path}")

    try:
        if model_path is not None and model_path.exists():
            logger.info(f"Loading ESMFold checkpoint from {model_path}")
            return esmfold_pretrained._load_model(str(model_path))
        if model_path is not None:
            raise FileNotFoundError(f"ESMFold model path not found: {model_path}")
        return esm.pretrained.esmfold_v1()
    finally:
        esm.pretrained.esm2_t36_3B_UR50D = original_esm2_loader


def _resolve_repr_layer(layer: int, num_layers_with_embedding: int) -> int:
    layer_idx = int(layer)
    if layer_idx < 0:
        layer_idx = num_layers_with_embedding + layer_idx
    if layer_idx < 0 or layer_idx >= num_layers_with_embedding:
        raise ValueError(
            f"ESM2 repr layer {layer} out of range for {num_layers_with_embedding} available layers"
        )
    return layer_idx


def enable_esm2_embedding_capture(model, repr_layer: int):
    original_compute = model._compute_language_model_representations

    def _wrapped_compute(esmaa: torch.Tensor) -> torch.Tensor:
        esm_s = original_compute(esmaa)
        layer_idx = _resolve_repr_layer(repr_layer, int(esm_s.shape[2]))
        model._last_esm2_embeddings = esm_s[:, :, layer_idx, :].detach().float().cpu()
        model._last_esm2_repr_layer = layer_idx
        return esm_s

    model._compute_language_model_representations = _wrapped_compute


def save_captured_esm2_embeddings(model, headers, sequences, output_dir: Path):
    embeddings = getattr(model, "_last_esm2_embeddings", None)
    if embeddings is None:
        raise RuntimeError("ESM2 embeddings were requested but no captured embeddings were found")

    output_dir.mkdir(parents=True, exist_ok=True)
    for batch_idx, (header, seq) in enumerate(zip(headers, sequences)):
        embedding = embeddings[batch_idx, : len(seq)].contiguous()
        torch.save(embedding, output_dir / f"{header}.tensor")

    model._last_esm2_embeddings = None


def save_confidence_outputs(
    output,
    headers,
    sequences,
    confidence_output_dir: Path,
    pae_output_dir: Path,
    plddt_output_dir: Path,
    tensor_dtype: torch.dtype,
):
    confidence_output_dir.mkdir(parents=True, exist_ok=True)
    pae_output_dir.mkdir(parents=True, exist_ok=True)
    plddt_output_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, (header, seq) in enumerate(zip(headers, sequences)):
        seq_len = len(seq)
        plddt = output["plddt"][batch_idx, :seq_len, 1].contiguous().to(dtype=tensor_dtype)
        pae = output["predicted_aligned_error"][batch_idx, :seq_len, :seq_len].contiguous().to(dtype=tensor_dtype)
        max_pae = output["max_predicted_aligned_error"]
        if max_pae.ndim > 0:
            max_pae = max_pae[batch_idx]

        torch.save(plddt, plddt_output_dir / f"{header}.tensor")
        torch.save(pae, pae_output_dir / f"{header}.tensor")

        summary = {
            "protein_id": header,
            "length": seq_len,
            "mean_plddt": float(output["mean_plddt"][batch_idx]),
            "ptm": float(output["ptm"][batch_idx]),
            "max_predicted_aligned_error": float(max_pae),
            "plddt_tensor": str(plddt_output_dir / f"{header}.tensor"),
            "pae_tensor": str(pae_output_dir / f"{header}.tensor"),
        }
        with (confidence_output_dir / f"{header}.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)


def _parse_tensor_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return dtype_map[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported confidence tensor dtype: {dtype_name}") from exc




def enable_cpu_offloading(model):
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import enable_wrap, wrap

    torch.distributed.init_process_group(
        backend="nccl", init_method="tcp://localhost:9999", world_size=1, rank=0
    )

    wrapper_kwargs = dict(cpu_offload=CPUOffload(offload_params=True))

    with enable_wrap(wrapper_cls=FullyShardedDataParallel, **wrapper_kwargs):
        for layer_name, layer in model.layers.named_children():
            wrapped_layer = wrap(layer)
            setattr(model.layers, layer_name, wrapped_layer)
        model = wrap(model)

    return model


def init_model_on_gpu_with_cpu_offloading(model):
    model = model.eval()
    model_esm = enable_cpu_offloading(model.esm)
    del model.esm
    model.cuda()
    model.esm = model_esm
    return model


def create_batched_sequence_datasest(
        sequences: T.List[T.Tuple[str, str]], max_tokens_per_batch: int = 1024
) -> T.Generator[T.Tuple[T.List[str], T.List[str]], None, None]:
    batch_headers, batch_sequences, num_tokens = [], [], 0
    for header, seq in sequences:
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences
            batch_headers, batch_sequences, num_tokens = [], [], 0
        batch_headers.append(header)
        batch_sequences.append(seq)
        num_tokens += len(seq)

    yield batch_headers, batch_sequences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--fasta",
        help="Path to input FASTA file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-o", "--pdb", help="Path to output PDB directory", type=Path, required=True
    )
    parser.add_argument('--gpu', type=int, default=0, help='gpu.')
    parser.add_argument(
        "--num-recycles",
        type=int,
        default=None,
        help="Number of recycles to run. Defaults to number used in training (4).",
    )
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=1024,
        help="Maximum number of tokens per gpu forward-pass. This will group shorter sequences together "
             "for batched prediction. Lowering this can help with out of memory issues, if these occur on "
             "short sequences.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunks axial attention computation to reduce memory usage from O(L^2) to O(L). "
             "Equivalent to running a for loop over chunks of of each dimension. Lower values will "
             "result in lower memory usage at the cost of speed. Recommended values: 128, 64, 32. "
             "Default: None."
    )
    parser.add_argument("--cpu-only", help="CPU only", action="store_true")
    parser.add_argument(
        "--cpu-offload", help="Enable CPU offloading", action="store_true"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("./models/esmfold_3B_v1.pt"),
        help="Local ESMFold checkpoint.",
    )
    parser.add_argument(
        "--esm2-model-path",
        type=Path,
        default=Path("./models/ESM2/esm2_t36_3B_UR50D.pt"),
        help="Local ESM2 t36 3B checkpoint used inside ESMFold.",
    )
    parser.add_argument(
        "--save-esm2-embeddings",
        action="store_true",
        help="Save residue-level ESM2 3B embeddings captured during the ESMFold forward pass.",
    )
    parser.add_argument(
        "--esm2-output",
        type=Path,
        default=None,
        help="Output directory for captured ESM2 3B tensors.",
    )
    parser.add_argument(
        "--esm2-repr-layer",
        type=int,
        default=-1,
        help="ESM2 representation layer to save. -1 means the final layer.",
    )
    parser.add_argument(
        "--save-confidence",
        action="store_true",
        help="Save ESMFold confidence outputs: PAE matrix, residue pLDDT, and scalar summary.",
    )
    parser.add_argument(
        "--confidence-output",
        type=Path,
        default=None,
        help="Output directory for ESMFold confidence JSON summaries.",
    )
    parser.add_argument(
        "--pae-output",
        type=Path,
        default=None,
        help="Output directory for predicted aligned error tensors.",
    )
    parser.add_argument(
        "--plddt-output",
        type=Path,
        default=None,
        help="Output directory for residue-level pLDDT tensors.",
    )
    parser.add_argument(
        "--confidence-tensor-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Tensor dtype used when saving PAE and pLDDT.",
    )
    args = parser.parse_args()

    if not args.fasta.exists():
        raise FileNotFoundError(args.fasta)
    if args.save_esm2_embeddings and args.esm2_output is None:
        raise ValueError("--esm2-output is required when --save-esm2-embeddings is set")
    if args.save_confidence:
        missing_confidence_args = [
            name
            for name, value in [
                ("--confidence-output", args.confidence_output),
                ("--pae-output", args.pae_output),
                ("--plddt-output", args.plddt_output),
            ]
            if value is None
        ]
        if missing_confidence_args:
            raise ValueError(", ".join(missing_confidence_args) + " are required when --save-confidence is set")

    args.pdb.mkdir(exist_ok=True)
    confidence_tensor_dtype = _parse_tensor_dtype(args.confidence_tensor_dtype)

    # Read fasta and sort sequences by length
    logger.info(f"Reading sequences from {args.fasta}")
    all_sequences = sorted(
        read_fasta(args.fasta), key=lambda header_seq: len(header_seq[1])
    )
    logger.info(f"Loaded {len(all_sequences)} sequences from {args.fasta}")

    logger.info("Loading model")
    model = load_esmfold_model(args.model_path, args.esm2_model_path)
    model = model.eval()
    model.set_chunk_size(args.chunk_size)
    if args.save_esm2_embeddings:
        enable_esm2_embedding_capture(model, args.esm2_repr_layer)

    # [BUG FIX]  GPU
    device = torch.device(f'cuda:{args.gpu}')

    if args.cpu_only:
        model.cpu()
    elif args.cpu_offload:
        print('cpu offload')
        model = init_model_on_gpu_with_cpu_offloading(model)
    else:
        # [BUG FIX] ， cuda:0
        model = model.to(device)

    logger.info(f"Starting Predictions on {device}")

    batched_sequences = create_batched_sequence_datasest(
        all_sequences, args.max_tokens_per_batch
    )

    num_completed = 0
    num_sequences = len(all_sequences)
    for headers, sequences in batched_sequences:
        start = timer()
        try:
            output = model.infer(sequences, num_recycles=args.num_recycles)
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                if len(sequences) > 1:
                    logger.info(
                        f"Failed (CUDA out of memory) to predict batch of size {len(sequences)}. "
                        "Try lowering `--max-tokens-per-batch`."
                    )
                else:
                    logger.info(
                        f"Failed (CUDA out of memory) on sequence {headers[0]} of length {len(sequences[0])}."
                    )

                continue
            raise

        output = {key: value.cpu() for key, value in output.items()}
        if args.save_esm2_embeddings:
            save_captured_esm2_embeddings(model, headers, sequences, args.esm2_output)
        if args.save_confidence:
            save_confidence_outputs(
                output,
                headers,
                sequences,
                args.confidence_output,
                args.pae_output,
                args.plddt_output,
                confidence_tensor_dtype,
            )
        try:
            pdbs = model.output_to_pdb(output)
        except:
            print('not done !!')
        tottime = timer() - start
        time_string = f"{tottime / len(headers):0.1f}s"
        if len(sequences) > 1:
            time_string = time_string + f" (amortized, batch size {len(sequences)})"
        for header, seq, pdb_string, mean_plddt, ptm in zip(
                headers, sequences, pdbs, output["mean_plddt"], output["ptm"]
        ):
            output_file = args.pdb / f"{header}.pdb"
            output_file.write_text(pdb_string)
            num_completed += 1
            logger.info(
                f"Predicted structure for {header} with length {len(seq)}, pLDDT {mean_plddt:0.1f}, "
                f"pTM {ptm:0.3f} in {time_string}. "
                f"{num_completed} / {num_sequences} completed."
            )
