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

BASE_REPO="mlx-community/LongCat-Video-Avatar-1.5-bf16"
MERGED_REPO="mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged"

case "${1:-help}" in
  stage1|cards)
    # Tiny upload of just the model card + small configs. Creates the repos
    # if they don't exist (hf upload --create-repo). Validates auth.
    echo "=== Stage 1: model cards + configs (validates auth, creates repos) ==="
    echo "[1/2] $BASE_REPO"
    hf upload "$BASE_REPO" "$BASE_DIR/README.md" "README.md" \
        --repo-type model --create-repo
    hf upload "$BASE_REPO" "$BASE_DIR/pipeline_config.json" "pipeline_config.json" \
        --repo-type model
    echo ""
    echo "[2/2] $MERGED_REPO"
    hf upload "$MERGED_REPO" "$MERGED_DIR/README.md" "README.md" \
        --repo-type model --create-repo
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
    hf upload "$MERGED_REPO" "$MERGED_DIR/README.md" "README.md" \
        --repo-type model --create-repo
    hf upload "$MERGED_REPO" "$MERGED_DIR" "." --repo-type model
    ;;

  base)
    echo "=== Upload base variant only ==="
    hf upload "$BASE_REPO" "$BASE_DIR/README.md" "README.md" \
        --repo-type model --create-repo
    hf upload "$BASE_REPO" "$BASE_DIR" "." --repo-type model
    ;;

  help|*)
    cat <<EOF
Usage:
    $0 stage1     # push just model cards + configs (creates repos, validates auth, ~1 min)
    $0 stage2     # push all weights (~89 GB, hours, resumable)
    $0 merged     # full upload of just the DMD-merged variant
    $0 base       # full upload of just the base variant

Typical workflow:
    $0 stage1           # validate everything works, model cards live on HF
    # eyeball the cards on huggingface.co/...
    $0 stage2           # commit to the long weight upload

Or skip the staging and just push the recommended variant:
    $0 merged

Env overrides:
    WEIGHTS_ROOT=$WEIGHTS_ROOT

Auth check:
    hf auth whoami      # should list orgs=mlx-community
EOF
    ;;
esac
