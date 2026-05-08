#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# SAM3 Garment Segmentation Server — Setup & Start Script
# ═══════════════════════════════════════════════════════════════════
#
# Usage:
#   PORT=8000 bash setup.sh              # Full setup + start
#   PORT=8000 bash setup.sh --start      # Start server (skip setup)
#   bash setup.sh --setup-only           # Setup only (no start)
#   bash setup.sh --help                 # Show help
#
# Environment Variables:
#   PORT          — Server port (default: 8000)
#   HF_TOKEN      — HuggingFace token for gated model access (default: hardcoded)
#   CUDA_VISIBLE_DEVICES — GPU selection (default: 0)
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

log()   { echo -e "${GREEN}[SAM3]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
header(){ echo -e "\n${CYAN}${BOLD}═══ $* ═══${NC}\n"; }

# ── Config ────────────────────────────────────────────────────────
PORT="${PORT:-8000}"
HF_TOKEN="${HF_TOKEN:-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}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="${SCRIPT_DIR}/server"
SAM3_DIR="${SCRIPT_DIR}/sam3_repo"
VENV_DIR="${SCRIPT_DIR}/venv"

# ── Help ──────────────────────────────────────────────────────────
show_help() {
    echo -e "${BOLD}SAM3 Garment Segmentation Server${NC}"
    echo ""
    echo "Usage:"
    echo "  PORT=8000 bash setup.sh              Full setup + start server"
    echo "  PORT=8000 bash setup.sh --start       Start server only (skip setup)"
    echo "  bash setup.sh --setup-only            Install deps + download model only"
    echo "  bash setup.sh --help                  Show this help"
    echo ""
    echo "Environment Variables:"
    echo "  PORT                   Server port (default: 8000)"
    echo "  HF_TOKEN               HuggingFace access token (for gated model)"
    echo "  CUDA_VISIBLE_DEVICES   GPU to use (default: 0)"
    echo ""
    echo "Prerequisites:"
    echo "  - Request access to facebook/sam3 at https://huggingface.co/facebook/sam3"
    echo "  - NVIDIA GPU with CUDA 12.6+"
    echo "  - Python 3.12+ (or will use system python3)"
    exit 0
}

# ── Parse Args ────────────────────────────────────────────────────
START_ONLY=false
SETUP_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --start)      START_ONLY=true ;;
        --setup-only) SETUP_ONLY=true ;;
        --help|-h)    show_help ;;
        *)            warn "Unknown argument: $arg" ;;
    esac
done

# ── Utility Functions ─────────────────────────────────────────────
check_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        log "GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | while read -r line; do
            log "  → $line"
        done
    else
        warn "nvidia-smi not found — GPU may not be available"
    fi
}

setup_hf_auth() {
    if [ -n "${HF_TOKEN:-}" ]; then
        log "Authenticating with HuggingFace using HF_TOKEN..."
        huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || \
            python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')" 2>/dev/null || \
            warn "HF auth failed — model download may fail if not already cached"
    else
        # Check if already logged in
        if python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
            log "Already authenticated with HuggingFace"
        else
            warn "HF_TOKEN not set and not logged in."
            warn "Set HF_TOKEN=<your_token> or run: huggingface-cli login"
            warn "You need access to https://huggingface.co/facebook/sam3"
        fi
    fi
}

# ══════════════════════════════════════════════════════════════════
#  SETUP PHASE
# ══════════════════════════════════════════════════════════════════
run_setup() {
    header "System Check"
    check_gpu

    header "Installing System Dependencies"
    # Install python3-venv if not available (common on Vast.ai)
    if ! python3 -m venv --help &>/dev/null 2>&1; then
        log "Installing python3-venv..."
        apt-get update -qq && apt-get install -y -qq python3-venv 2>/dev/null || \
            warn "Could not install python3-venv (may need sudo)"
    fi

    header "Setting Up Python Environment"
    if [ ! -d "$VENV_DIR" ]; then
        log "Creating virtual environment at $VENV_DIR..."
        python3 -m venv "$VENV_DIR" || {
            warn "venv creation failed, will use system python"
            VENV_DIR=""
        }
    fi

    if [ -n "$VENV_DIR" ] && [ -d "$VENV_DIR" ]; then
        source "$VENV_DIR/bin/activate"
        log "Activated venv: $VENV_DIR"
    fi

    # Upgrade pip
    pip install --upgrade pip setuptools wheel -q

    header "Installing PyTorch"
    # Check if torch is already installed with CUDA
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
        log "PyTorch ${TORCH_VER} with CUDA already installed ✓"
    else
        log "Installing PyTorch with CUDA..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 -q || \
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q || \
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q || {
            err "Failed to install PyTorch with CUDA"
            exit 1
        }
        log "PyTorch installed ✓"
    fi

    header "Installing SAM 3"
    # Clone and install the official sam3 repo
    if [ ! -d "$SAM3_DIR" ]; then
        log "Cloning facebookresearch/sam3..."
        git clone https://github.com/facebookresearch/sam3.git "$SAM3_DIR"
    else
        log "SAM3 repo already exists, pulling latest..."
        cd "$SAM3_DIR" && git pull && cd "$SCRIPT_DIR"
    fi

    log "Installing sam3 package..."
    cd "$SAM3_DIR"
    pip install -e . -q
    cd "$SCRIPT_DIR"
    log "SAM 3 package installed ✓"

    # Install optional speedup deps (non-fatal)
    log "Installing optional speedups (flash-attn, einops)..."
    pip install einops ninja -q 2>/dev/null || true
    pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128 -q 2>/dev/null || \
        warn "flash-attn-3 not installed (optional)"

    header "Installing Server Dependencies"
    pip install \
        fastapi \
        uvicorn[standard] \
        python-multipart \
        scipy \
        Pillow \
        numpy \
        huggingface_hub \
        transformers \
        accelerate \
        -q
    log "Server dependencies installed ✓"

    header "HuggingFace Authentication"
    setup_hf_auth

    header "Pre-downloading SAM 3 Model Weights"
    log "HF_HOME=${HF_HOME}"
    mkdir -p "$HF_HOME"
    log "Triggering model download (this may take a few minutes)..."
    python3 -c "
from sam3.model_builder import build_sam3_image_model
print('[SAM3] Downloading model weights...')
model = build_sam3_image_model()
print('[SAM3] Model downloaded successfully ✓')
" || warn "Pre-download had issues — model will download on first server start"

    log ""
    log "════════════════════════════════════════════════"
    log "  Setup complete! ✓"
    log "  Start server with: PORT=${PORT} bash setup.sh --start"
    log "════════════════════════════════════════════════"
}

# ══════════════════════════════════════════════════════════════════
#  START PHASE
# ══════════════════════════════════════════════════════════════════
start_server() {
    header "Starting SAM3 Server on port ${PORT}"

    # Activate venv if exists
    if [ -d "$VENV_DIR" ]; then
        source "$VENV_DIR/bin/activate"
        log "Activated venv ✓"
    fi

    # Set CUDA device
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

    # HF auth (in case token is set but not persisted)
    if [ -n "${HF_TOKEN:-}" ]; then
        export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
        export HF_TOKEN="$HF_TOKEN"
    fi

    log "Server directory: ${SERVER_DIR}"
    log "Port: ${PORT}"
    log ""
    log "Endpoints:"
    log "  POST /segment          — Segment garment → green blur mask"
    log "  POST /segment/raw-mask — Raw binary mask (debug)"
    log "  GET  /health           — Health check"
    log "  GET  /config           — View current config"
    log "  POST /config/prompts   — Update text prompts"
    log "  POST /config/mask      — Update mask settings"
    log "  POST /config/reset     — Reset to defaults"
    log ""

    cd "$SERVER_DIR"
    exec uvicorn main:app \
        --host 0.0.0.0 \
        --port "$PORT" \
        --workers 1 \
        --timeout-keep-alive 300 \
        --log-level info
}

# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if $START_ONLY; then
    start_server
elif $SETUP_ONLY; then
    run_setup
else
    run_setup
    start_server
fi
