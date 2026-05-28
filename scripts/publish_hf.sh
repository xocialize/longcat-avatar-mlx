#!/usr/bin/env bash
# Publish both MLX variants to mlx-community.
#
# Staged so you can validate auth + repo creation against tiny model-card
# uploads BEFORE committing to the ~89 GB weight upload. Run from any
# terminal session — `hf upload` is resumable on interrupt.
#
# Prereqs:
#   - `hf auth whoami` should show membership in `mlx-community`
#   - Conversion output present at /Users/dustinnielson/DEV_INT/longcat-avatar-mlx-weights/
#   - This repo's `docs/model-cards/*.md` already copied into the variant dirs

set -euo pipefail

WEIGHTS_ROOT="${WEIGHTS_ROOT:-/Users/dustinnielson/DEV_INT/longcat-avatar-mlx-weights}"
BASE_DIR="$WEIGHTS_ROOT/LongCat-Video-Avatar-1.5-bf16"
MERGED_DIR="$WEIGHTS_ROOT/LongCat-Video-Avatar-1.5-bf16-dmd-merged"
Q4_DIR="$WEIGHTS_ROOT/LongCat-Video-Avatar-1.5-q4-dmd-merged"
Q8_DIR="$WEIGHTS_ROOT/LongCat-Video-Avatar-1.5-q8-dmd-merged"

BASE_REPO="mlx-community/LongCat-Video-Avatar-1.5-bf16"
MERGED_REPO="mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged"
Q4_REPO="mlx-community/LongCat-Video-Avatar-1.5-q4-dmd-merged"
Q8_REPO="mlx-community/LongCat-Video-Avatar-1.5-q8-dmd-merged"

case "${1:-help}" in
  stage1|cards)
    # Tiny upload of just the model card + small configs. Creates the repos
    # via `hf repos create --exist-ok` (HF CLI 1.x dropped `upload --create-repo`).
    echo "=== Stage 1: model cards + configs (validates auth, creates repos) ==="
    echo "[1/2] $BASE_REPO"
    hf repos create "$BASE_REPO" --type model --exist-ok
    hf upload "$BASE_REPO" "$BASE_DIR/README.md" "README.md" --repo-type model
    hf upload "$BASE_REPO" "$BASE_DIR/pipeline_config.json" "pipeline_config.json" \
        --repo-type model
    echo ""
    echo "[2/2] $MERGED_REPO"
    hf repos create "$MERGED_REPO" --type model --exist-ok
    hf upload "$MERGED_REPO" "$MERGED_DIR/README.md" "README.md" --repo-type model
    hf upload "$MERGED_REPO" "$MERGED_DIR/pipeline_config.json" "pipeline_config.json" \
        --repo-type model
    echo ""
    echo "Stage 1 complete. Check the cards at:"
    echo "  https://huggingface.co/$BASE_REPO"
    echo "  https://huggingface.co/$MERGED_REPO"
    echo ""
    echo "If both look good, run: $0 stage2"
    ;;

  stage2|weights)
    # Full directory upload. Each file is content-addressed, so interrupted
    # uploads are resumable — re-running this skips already-uploaded files.
    echo "=== Stage 2: weight uploads (~89 GB total, resumable) ==="
    echo ""
    echo "[1/2] $BASE_REPO  (~46 GB)"
    hf upload "$BASE_REPO" "$BASE_DIR" "." --repo-type model
    echo ""
    echo "[2/2] $MERGED_REPO  (~43 GB)"
    hf upload "$MERGED_REPO" "$MERGED_DIR" "." --repo-type model
    echo ""
    echo "Done. Verify the file lists at:"
    echo "  https://huggingface.co/$BASE_REPO/tree/main"
    echo "  https://huggingface.co/$MERGED_REPO/tree/main"
    ;;

  merged)
    # Upload only the merged variant (typical workflow: ship recommended first)
    echo "=== Upload merged variant only ==="
    hf repos create "$MERGED_REPO" --type model --exist-ok
    hf upload "$MERGED_REPO" "$MERGED_DIR/README.md" "README.md" --repo-type model
    hf upload "$MERGED_REPO" "$MERGED_DIR" "." --repo-type model
    ;;

  base)
    echo "=== Upload base variant only ==="
    hf repos create "$BASE_REPO" --type model --exist-ok
    hf upload "$BASE_REPO" "$BASE_DIR/README.md" "README.md" --repo-type model
    hf upload "$BASE_REPO" "$BASE_DIR" "." --repo-type model
    ;;

  q4)
    echo "=== Upload q4-dmd-merged variant ==="
    hf repos create "$Q4_REPO" --type model --exist-ok
    hf upload "$Q4_REPO" "$Q4_DIR/README.md" "README.md" --repo-type model
    hf upload "$Q4_REPO" "$Q4_DIR" "." --repo-type model
    ;;

  q8)
    echo "=== Upload q8-dmd-merged variant ==="
    hf repos create "$Q8_REPO" --type model --exist-ok
    hf upload "$Q8_REPO" "$Q8_DIR/README.md" "README.md" --repo-type model
    hf upload "$Q8_REPO" "$Q8_DIR" "." --repo-type model
    ;;

  quant)
    # Both quantized variants in sequence. `hf repos create --exist-ok` is
    # idempotent; `hf upload` is content-addressed and resumable on interrupt.
    echo "=== Upload q4 + q8 quantized variants ==="
    echo "[1/2] $Q4_REPO  (~24 GB)"
    hf repos create "$Q4_REPO" --type model --exist-ok
    hf upload "$Q4_REPO" "$Q4_DIR/README.md" "README.md" --repo-type model
    hf upload "$Q4_REPO" "$Q4_DIR" "." --repo-type model
    echo ""
    echo "[2/2] $Q8_REPO  (~31 GB)"
    hf repos create "$Q8_REPO" --type model --exist-ok
    hf upload "$Q8_REPO" "$Q8_DIR/README.md" "README.md" --repo-type model
    hf upload "$Q8_REPO" "$Q8_DIR" "." --repo-type model
    echo ""
    echo "Done. Verify the file lists at:"
    echo "  https://huggingface.co/$Q4_REPO/tree/main"
    echo "  https://huggingface.co/$Q8_REPO/tree/main"
    ;;

  help|*)
    cat <<EOF
Usage:
    $0 stage1     # push just model cards + configs for bf16 variants (~1 min)
    $0 stage2     # push all bf16 weights (~89 GB, hours, resumable)
    $0 merged     # full upload of just the bf16-dmd-merged variant
    $0 base       # full upload of just the base bf16 variant
    $0 q4         # full upload of q4-dmd-merged variant (~24 GB)
    $0 q8         # full upload of q8-dmd-merged variant (~31 GB)
    $0 quant      # full upload of BOTH q4 + q8 variants in sequence (~55 GB)

Typical workflow (initial release — already done):
    $0 stage1           # validate everything works, model cards live on HF
    $0 stage2           # commit to the long bf16 weight upload

Or skip the staging and just push the recommended variant:
    $0 merged

Quant variants (added later):
    $0 quant            # uploads q4 and q8 sequentially with auto repo-create

Env overrides:
    WEIGHTS_ROOT=$WEIGHTS_ROOT

Auth check:
    hf auth whoami      # should list orgs=mlx-community
EOF
    ;;
esac
