# MNIST Multi-Cloud Inference Endpoint

Deploy the same MNIST digit-classification model as a Triton Inference Server endpoint on **AWS SageMaker** and **Alibaba Cloud PAI-EAS**, and hit either one from a shared client.

A small CNN is trained locally on MNIST, exported to ONNX, packaged in Triton model repository format, and uploaded to object storage (S3 or OSS). Infrastructure for each cloud is defined as code: AWS CDK for AWS, ROS CDK for Alibaba Cloud. Both endpoints speak the identical Triton V2 JSON protocol, so the client only differs in URL and auth header.

## Setup

Requires Python >= 3.10. Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Optional extras:

```bash
uv sync --extra dev        # pytest, hypothesis
uv sync --extra aws        # boto3
uv sync --extra alicloud   # oss2
```

The infra directories are separate projects with their own dependencies:

```bash
cd infra_aws      && uv sync   # aws-cdk-lib, constructs
cd infra_alicloud && uv sync   # ros-cdk-core, ros-cdk-oss, ros-cdk-pai
```

### Credentials

Both clouds read credentials from the environment. Copy your values into a `.env` at the project root (gitignored):

| Variable | Used by |
|----------|---------|
| `OSS_BUCKET` | `scripts/upload_model_alicloud.py` |
| `ALICLOUD_REGION` | `scripts/upload_model_alicloud.py` |
| `ALIBABA_CLOUD_ACCESS_KEY_ID` | OSS upload, ROS CDK |
| `ALIBABA_CLOUD_ACCESS_KEY_SECRET` | OSS upload, ROS CDK |
| `ALICLOUD_ENDPOINT_URL` | `scripts/test_endpoint_alicloud.py` |
| `ALICLOUD_API_TOKEN` | `scripts/test_endpoint_alicloud.py` |

AWS uses the standard credential chain (`aws configure` / `AWS_PROFILE`).

For Alibaba Cloud, install the CLI and ROS CDK toolkit:

```bash
brew install aliyun-cli
aliyun configure                      # RAM user AccessKey, region, lang
npm install -g @alicloud/ros-cdk-cli
```

OSS, PAI (including EAS), and ROS all need to be activated once in the Alibaba Cloud console before first deploy.

## Workflow

### 1. Train the model

```bash
uv run python model/train.py
```

Downloads MNIST into `data/`, trains one epoch, and writes `mnist_model.pt`. Sanity-check it with:

```bash
uv run python model/test_predict_sample.py   # random sample from the test set
uv run python model/test_predict.py          # your own PNG
```

### 2. Package and upload

**AWS** — converts to ONNX, validates, builds the Triton repo, tars it, uploads to S3:

```bash
uv run python scripts/package_model.py
```

Note: the S3 bucket name is currently hardcoded in that script. Update it to match your `MnistStorageStack` output.

**Alibaba Cloud** — uploads the committed `model/triton_repo/` tree to OSS as loose files under `models/`:

```bash
uv run python scripts/upload_model_alicloud.py
```

Run the OSS upload **before** deploying the ROS stacks, since `EasStack` references the model path in OSS.

### 3. Deploy

**AWS** (region `eu-west-1`, from `infra_aws/`):

```bash
cdk deploy MnistStorageStack
cdk deploy MnistSageMakerStack
```

**Alibaba Cloud** (region `cn-hangzhou`, from `infra_alicloud/`):

```bash
ros-cdk synth
ros-cdk deploy MnistStorageStack
ros-cdk deploy MnistEasStack
ros-cdk deploy MnistAccessStack
```

Both endpoints require GPU instances because the Triton serving container has no CPU-only variant. Defaults are `ml.g4dn.xlarge` on AWS and `ecs.gn6i-c4g1.xlarge` (spot) on Alibaba Cloud.

### 4. Test the endpoint

```bash
uv run python scripts/test_endpoint_aws.py        # via boto3 invoke_endpoint
uv run python scripts/test_endpoint_alicloud.py   # via HTTPS + Authorization token
```

Both send a synthetic "7" and print the predicted digit, confidence, and latency.

### 5. Draw digits interactively

```bash
uv run python src/app/main.py
```

A tkinter canvas — draw with the mouse, Enter to predict, Escape to clear. It currently posts to a hardcoded `http://localhost:8000` Triton URL, so it only works against a local server. Wiring it to the cloud providers is the open work in the `multi-cloud-minimal` spec.

## Project Structure

```
.
├── pyproject.toml                   # Root project — shared model/client code
├── uv.lock                          # Locked dependency versions
├── .python-version                  # Pinned Python version for uv
├── .env                             # Credentials (gitignored)
├── src/
│   ├── config.py                    # PackagerConfig, DeployerConfig
│   ├── exceptions.py                # PipelineError hierarchy
│   ├── model_packager.py            # MNISTNet + packaging pipeline
│   ├── app/
│   │   ├── main.py                  # tkinter DigitCanvas drawing client
│   │   └── preprocessing.py         # Canvas → 784-float tensor
│   └── inference/
│       ├── models.py                # UnifiedResponse, ProviderConfig
│       ├── errors.py                # ErrorCategory, InferenceError, categorize_error
│       ├── validation.py            # Shape/range validation for inference input
│       └── providers/
│           ├── base.py              # ProviderBackend ABC
│           ├── aws.py               # AWSBackend (x-api-key auth)
│           └── alicloud.py          # AlicloudBackend (Authorization token)
├── model/
│   ├── train.py                     # Train MNISTNet on MNIST
│   ├── test_predict.py              # Predict from a local PNG
│   ├── test_predict_sample.py       # Predict a random test-set sample
│   ├── mnist_model.pt               # Trained weights
│   ├── mnist_model.onnx(.data)      # Exported ONNX
│   └── triton_repo/mnist/           # Triton model repository
│       ├── config.pbtxt             # Triton model config
│       └── 1/model.onnx(.data)      # Versioned model files
├── scripts/
│   ├── package_model.py             # Full pipeline → S3
│   ├── upload_model_alicloud.py     # triton_repo/ → OSS
│   ├── test_endpoint_aws.py         # Invoke SageMaker endpoint
│   └── test_endpoint_alicloud.py    # Invoke PAI-EAS endpoint
├── infra_aws/                       # AWS CDK app
│   ├── app.py                       # Entry point, stack wiring, cost tags
│   ├── cdk.json                     # Context: model_key, instance_type
│   └── stacks/
│       ├── storage_stack.py         # S3 bucket
│       ├── sagemaker_stack.py       # IAM role, model, endpoint config, endpoint
│       └── api_stack.py             # Placeholder (not implemented)
├── infra_alicloud/                  # ROS CDK app
│   ├── app.py                       # Entry point, stack wiring
│   ├── config.py                    # AlicloudDeployConfig, COMMON_TAGS
│   ├── cdk.json                     # Context: model_key, instance_type, region
│   └── stacks/
│       ├── storage_stack.py         # OSS bucket
│       ├── eas_stack.py             # PAI-EAS Triton service
│       └── access_stack.py          # Token + public endpoint outputs
├── tests/                           # pytest suite
├── test_ali/                        # Throwaway `ros-cdk init` scaffold
├── design_decisions/                # Architecture trade-off write-ups
└── .kiro/
    ├── hooks/                       # Agent hooks (formatting, README sync)
    ├── steering/aws-agent-rules.md  # AWS conventions for the agent
    └── specs/                       # Requirements → design → tasks
```

## File Descriptions

### Shared code (`src/`)

| File | Purpose |
|------|---------|
| `config.py` | `PackagerConfig` (local model path, S3 bucket/prefix, ONNX opset) and `DeployerConfig` (endpoint name, instance type, region, model name). |
| `exceptions.py` | `PipelineError` base with `ModelLoadError`, `ConversionError`, `ValidationError`, `UploadError`, `DeploymentError`. |
| `model_packager.py` | `MNISTNet` (2 conv + 2 FC CNN) and `ModelPackager`, which runs load → ONNX export → validate → Triton repo → tar.gz → S3 upload with retries. |
| `app/main.py` | `DigitCanvas` tkinter app: 280x280 drawing surface, preprocesses to 28x28, posts a Triton V2 payload, shows digit + softmax confidence. Currently hardcoded to localhost. |
| `app/preprocessing.py` | `preprocess_canvas()` — LANCZOS resize to 28x28, normalize to [0,1], flatten to 784 floats. |

### Inference client (`src/inference/`)

| File | Purpose |
|------|---------|
| `models.py` | `UnifiedResponse` (digit, confidence, probabilities, provider, latency) and `ProviderConfig` (provider id, endpoint URL, region, credential key). |
| `errors.py` | `ErrorCategory` enum, `InferenceError`, and `categorize_error()` which maps status codes and exception types to provider-agnostic categories without leaking URLs or tokens. |
| `validation.py` | `validate_input()` — accepts `(28,28)`, `(1,28,28)`, or `(784,)` arrays/lists, enforces the [0,1] range, returns a flat float list. |
| `providers/base.py` | `ProviderBackend` ABC: `send_request()`, `health_check()`, `provider_name`. |
| `providers/aws.py` | `AWSBackend` — Triton V2 over HTTPS with `x-api-key`, softmax-normalizes logits. |
| `providers/alicloud.py` | `AlicloudBackend` — same protocol with an `Authorization` token header. |

The provider abstraction is built but not yet wired into the tkinter app.

### AWS infrastructure (`infra_aws/`)

| File | Purpose |
|------|---------|
| `app.py` | CDK app for `eu-west-1`. Instantiates the three stacks, passes bucket/endpoint references between them, and applies cost-tracking tags (`project`, `environment`, `managed-by`, `team`, plus per-stack `Component`). |
| `cdk.json` | Runs `app.py` from the root venv; sets `model_key` and `instance_type` context. |
| `stacks/storage_stack.py` | S3 bucket with SSE-S3, `RemovalPolicy.DESTROY`, and auto-delete objects (dev/test convenience). Exports name and ARN. |
| `stacks/sagemaker_stack.py` | SageMaker execution role with scoped S3 read, `CfnModel` pointing at the Triton 24.05 GPU DLC, endpoint config, and endpoint. `validate_instance_type()` rejects anything outside `ml.g4dn/g5/g6/p3/p4d`. |
| `stacks/api_stack.py` | Placeholder. Accepts the endpoint name but creates no API Gateway resources yet. |

### Alibaba Cloud infrastructure (`infra_alicloud/`)

| File | Purpose |
|------|---------|
| `app.py` | ROS CDK app. Reads `model_key`, `instance_type`, `region` from context and wires `StorageStack` → `EasStack` → `AccessStack`. |
| `config.py` | `AlicloudDeployConfig` (region, GPU instance, spot flag, replica bounds, target QPS, OSS bucket/prefix), allowed GPU families, valid regions, and `COMMON_TAGS`. |
| `cdk.json` | Runs `app.py` from the local `.venv`; sets model key, instance type, and `cn-hangzhou`. |
| `stacks/storage_stack.py` | OSS bucket `mnist-model-artifacts-alicloud` with AES256 encryption, a 7-day abort-incomplete-multipart lifecycle rule, and force-delete for cleanup. |
| `stacks/eas_stack.py` | `ALIYUN::PAI::Service` running the PAI Triton 23.05 image, spot GPU instance by default, QPS-based autoscaling (target 10, 1–3 replicas). Validates against `ecs.gn6i/gn6v/gn7i/gn7e`. Outputs service name and endpoint URL. |
| `stacks/access_stack.py` | Documents PAI-EAS auth: declares a no-echo `AccessToken` parameter and exports the token and public HTTPS endpoint as ROS outputs. |

### Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `package_model.py` | Runs `ModelPackager.run()` end to end and prints the resulting S3 URI. Bucket name is hardcoded. |
| `upload_model_alicloud.py` | Walks `model/triton_repo/`, uploads each file to OSS under `models/` preserving structure, skipping dotfiles. Retries 3x with a 1s delay; loads config from `.env`. |
| `test_endpoint_aws.py` | Builds a synthetic "7", invokes the SageMaker endpoint via `boto3`, prints the argmax and raw scores. |
| `test_endpoint_alicloud.py` | Same input, posted over HTTPS to PAI-EAS with an `Authorization` header. Prints digit, confidence, and round-trip time. |

### Model assets (`model/`)

| File | Purpose |
|------|---------|
| `train.py` | Trains `MNISTNet` for one pass over MNIST with Adam and cross-entropy, saves `mnist_model.pt`. |
| `test_predict.py` | Loads a PNG, preprocesses it, and prints the prediction. |
| `test_predict_sample.py` | Picks a random MNIST test image and verifies the prediction against its label. |
| `triton_repo/mnist/config.pbtxt` | Triton config: `onnxruntime_onnx` platform, max batch 8, input `input` FP32 `[1,28,28]`, output `output` FP32 `[10]`. |

### Tests (`tests/`)

| File | Purpose |
|------|---------|
| `test_model_packager.py` | ONNX conversion (valid/invalid models, opset config) and ONNX validation. |
| `test_model_packager_repo.py` | Triton repo layout creation and tar.gz packaging. |
| `test_model_packager_upload.py` | S3 upload retry behavior (success after retries, failure after 3 attempts, delay timing) and `run()` orchestration order. |
| `test_cdk_stacks.py` | `aws_cdk.assertions.Template` checks that the synthesized AWS templates contain the expected resources and properties. |

No tests currently cover `src/inference`, `src/app`, or the ROS CDK stacks.

Parts of the suite are currently red because the test modules lag behind refactors — see [Running Tests](#running-tests).

### Design decisions (`design_decisions/`)

| File | Purpose |
|------|---------|
| `architecture-options.md` | Compares Lambda, Lambda + Provisioned Concurrency, SageMaker Serverless, and SageMaker Real-Time, with cost/latency trade-offs. |
| `model-format-comparison.md` | Compares ONNX vs CatBoost binary vs JSON, and container options (Triton, pre-built, extended, BYOC). |
| `aws-vs-alicloud-architecture.md` | Maps AWS services to Alibaba Cloud equivalents, diagrams both architectures, explains why they differ, and records the decision to skip Alibaba Cloud API Gateway for now. |

### Specs (`.kiro/specs/`)

| Spec | Purpose |
|------|---------|
| `mnist-inference-endpoint/` | The original AWS-only feature: train, convert, package, upload, deploy to SageMaker with Triton. |
| `multi-cloud-inference/` | Extends to Alibaba Cloud PAI-EAS plus a desktop app for one-click provider switching behind a unified interface. |
| `multi-cloud-minimal/` | Narrower take: add AWS/Alibaba endpoint support to the existing tkinter app via a JSON config or env vars. Not yet implemented. |

### `test_ali/`

A blank `ros-cdk init` scaffold kept around as a reference for ROS CDK project layout. Not part of the deployment.

## Agent Hooks

This project uses Kiro agent hooks (`.kiro/hooks/`) to automate development workflows.

### Format on Create — `format-on-create.json`

Formats Python files with [ruff](https://docs.astral.sh/ruff/) whenever the agent creates a new `.py` file.

- **Trigger:** `PostFileCreate` | **Matcher:** `\.(py)$`
- **Command:** `uvx ruff format "$KIRO_FILE_PATH"`

### Format on Save — `format-on-save.json`

Formats Python files with ruff whenever a `.py` file is saved.

- **Trigger:** `PostFileSave` | **Matcher:** `\.(py)$`
- **Command:** `uvx ruff format "$KIRO_FILE_PATH"`

### Update README on Save — `update-readme-on-save.json`

An agent-type hook that prompts Kiro to review and update the README after every file save, keeping documentation in sync with code changes.

- **Trigger:** `PostFileSave`
- **Action type:** `agent`

## Steering

`.kiro/steering/aws-agent-rules.md` sets AWS conventions for the agent: prefer infrastructure-as-code over CLI mutations, verify uncertain AWS details against docs, follow Well-Architected principles, avoid em dashes in resource names, and never read secret values directly out of Secrets Manager.

## Running Tests

```bash
uv run pytest tests/
```

Scope to `tests/` — a bare `uv run pytest` also collects `test_ali/`, whose scaffold tests import `ros_cdk_core` and fail in the root venv.

Known failures, all from tests drifting behind source refactors rather than broken behavior:

- `tests/test_model_packager.py` fails to import `DownloadError`, which was renamed to `ModelLoadError` when the pipeline switched from downloading a model to loading a local one.
- Several tests pass `model_source_url` to `PackagerConfig`, which now takes `model_path`.
- One `test_cdk_stacks.py` case expects `ml.c5.large` to be accepted, but `validate_instance_type()` now requires a GPU family.

Updating those tests is outstanding work.
