#!/usr/bin/env bash
# Bench the already-running production instance with the 4-cell profile.
#
# Usage: ./bench-4cell.sh [recipe] [output-dir]
# Defaults: recipe/production.yaml, benchmarks/
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "${here}")"
recipe="${1:-${repo}/recipe/production.yaml}"
outdir="${2:-${repo}/benchmarks}"

if [[ ! -f "${recipe}" ]]; then
  echo "recipe not found: ${recipe}" >&2
  exit 2
fi
if [[ ! -f "${repo}/benchmarks/ab-4cell.yaml" ]]; then
  echo "profile not found: ${repo}/benchmarks/ab-4cell.yaml" >&2
  exit 2
fi

mkdir -p "${outdir}"
stamp="$(date +%Y%m%d-%H%M)"

# --skip-run: bench the live instance, do not restart it.
# --no-stop: leave it serving afterwards.
# --fresh:   the bench id ignores the recipe, so never reuse another boot's state.
sparkrun benchmark perf "${recipe}" \
  --skip-run --no-stop --fresh \
  --profile "${repo}/benchmarks/ab-4cell.yaml" \
  --output "${outdir}/ab-${stamp}.yaml"
