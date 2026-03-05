#!/usr/bin/env python3
"""
CLASP Vertex AI training job helper.

Packages the QLoRA training job and submits it to Vertex AI
using Google Cloud's serverless training infrastructure.

Usage:
    # Set up auth first:
    #   gcloud auth application-default login
    #   gcloud config set project YOUR_PROJECT_ID

    python vertex_train.py --project YOUR_PROJECT --region us-central1
    python vertex_train.py --project YOUR_PROJECT --staging-bucket gs://YOUR_BUCKET
"""
import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("clasp.vertex")

# Default training container image (NVIDIA GPU-optimized)
TRAIN_IMAGE = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4:latest"


def create_custom_job(
    project: str,
    region: str,
    staging_bucket: str,
    display_name: str = "clasp-qlora-cosmos-reason2",
    machine_type: str = "a2-highgpu-1g",  # 1x A100 40GB
    accelerator_type: str = "NVIDIA_TESLA_A100",
    accelerator_count: int = 1,
    sft_data_gcs: str = "",
    base_model: str = "Qwen/Qwen3-VL-8B-Instruct",
    epochs: int = 3,
    lr: float = 2e-4,
    lora_r: int = 32,
    lora_alpha: int = 64,
    max_seq_len: int = 1024,
    batch_size: int = 4,
    grad_accum: int = 4,
):
    """Submit a QLoRA training CustomJob to Vertex AI."""
    try:
        from google.cloud import aiplatform
    except ImportError:
        log.error("google-cloud-aiplatform not installed. Run: pip install google-cloud-aiplatform")
        return None

    aiplatform.init(
        project=project,
        location=region,
        staging_bucket=staging_bucket,
    )

    # Training args passed as command-line flags to our training script
    train_args = [
        "--model", base_model,
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--lora-r", str(lora_r),
        "--lora-alpha", str(lora_alpha),
        "--max-seq-len", str(max_seq_len),
        "--batch-size", str(batch_size),
        "--grad-accum", str(grad_accum),
        "--output", "/gcs/output/clasp-lora-checkpoint",
    ]

    if sft_data_gcs:
        train_args.extend(["--data", sft_data_gcs])

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": machine_type,
                "accelerator_type": accelerator_type,
                "accelerator_count": accelerator_count,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": TRAIN_IMAGE,
                "command": ["python3", "train_qlora.py"],
                "args": train_args,
                "env": [
                    {"name": "WANDB_DISABLED", "value": "true"},
                    {"name": "HF_HOME", "value": "/tmp/hf_cache"},
                ],
            },
        }
    ]

    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=f"{staging_bucket}/clasp-training-output",
    )

    log.info("Submitting Vertex AI CustomJob: %s", display_name)
    log.info("  Machine: %s + %dx %s", machine_type, accelerator_count, accelerator_type)
    log.info("  Model: %s", base_model)
    log.info("  Epochs: %d, LR: %s, LoRA r=%d alpha=%d", epochs, lr, lora_r, lora_alpha)
    log.info("  Staging: %s", staging_bucket)

    job.run(
        service_account=None,  # uses default compute SA
        sync=False,  # don't block — return immediately
    )

    log.info("Job submitted! Resource name: %s", job.resource_name)
    log.info("Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=%s", project)
    return job


def create_from_local_container(
    project: str,
    region: str,
    staging_bucket: str,
    image_uri: str,
    display_name: str = "clasp-qlora-cosmos-reason2-custom",
    machine_type: str = "a2-highgpu-1g",
    accelerator_type: str = "NVIDIA_TESLA_A100",
    accelerator_count: int = 1,
):
    """Submit using a custom Docker image (pushed to Artifact Registry)."""
    try:
        from google.cloud import aiplatform
    except ImportError:
        log.error("google-cloud-aiplatform not installed")
        return None

    aiplatform.init(project=project, location=region, staging_bucket=staging_bucket)

    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=image_uri,
    )

    log.info("Submitting custom container job: %s", display_name)
    log.info("  Image: %s", image_uri)

    model = job.run(
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        base_output_dir=f"{staging_bucket}/clasp-training-output",
        service_account=None,
        sync=False,
    )

    log.info("Job submitted! Resource name: %s", job.resource_name)
    return job


def main():
    parser = argparse.ArgumentParser(description="CLASP Vertex AI training job helper")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument("--staging-bucket", required=True,
                        help="GCS bucket for staging (gs://your-bucket)")
    parser.add_argument("--sft-data", default="",
                        help="GCS path to SFT JSONL (gs://bucket/path/sft_dataset.openai.jsonl)")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Base model for fine-tuning")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--machine", default="a2-highgpu-1g",
                        help="Vertex AI machine type")
    parser.add_argument("--gpu", default="NVIDIA_TESLA_A100",
                        help="Accelerator type")
    parser.add_argument("--gpu-count", type=int, default=1)

    # Custom container mode
    parser.add_argument("--custom-image", default="",
                        help="Custom Docker image URI (skips default container)")

    args = parser.parse_args()

    if args.custom_image:
        create_from_local_container(
            project=args.project,
            region=args.region,
            staging_bucket=args.staging_bucket,
            image_uri=args.custom_image,
            machine_type=args.machine,
            accelerator_type=args.gpu,
            accelerator_count=args.gpu_count,
        )
    else:
        create_custom_job(
            project=args.project,
            region=args.region,
            staging_bucket=args.staging_bucket,
            sft_data_gcs=args.sft_data,
            base_model=args.model,
            epochs=args.epochs,
            lr=args.lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            machine_type=args.machine,
            accelerator_type=args.gpu,
            accelerator_count=args.gpu_count,
        )


if __name__ == "__main__":
    main()
