#!/usr/bin/env bash
# Verify the production lane's boot gates. Run this ON the head node (node-0)
# after /health returns 200 and the Engram pack has finished.
#
# Usage: ./verify-gates.sh
# Env:   EXPECT_CERT_HEAD=1 if you intentionally run the CERT head.
set -uo pipefail

expect_cert="${EXPECT_CERT_HEAD:-0}"
pid="$(pgrep -f 'sglang::scheduler' | head -1 || true)"
if [[ -z "${pid}" ]]; then
  echo "FAIL: no sglang::scheduler process found on this node" >&2
  exit 2
fi

log="/proc/${pid}/root/tmp/sparkrun_serve.log"
if [[ -r "${log}" ]]; then
  readlog() { grep -a "$@" "${log}"; }
  countlog() { grep -ac "$@" "${log}"; }
elif sudo -n test -e "${log}" 2>/dev/null; then
  readlog() { sudo -n grep -a "$@" "${log}"; }
  countlog() { sudo -n grep -ac "$@" "${log}"; }
else
  echo "FAIL: ${log} does not exist (or is not readable with passwordless sudo)" >&2
  exit 2
fi

fail=0
check_eq() {
  local label="$1" got="$2" want="$3"
  if [[ "${got}" -eq "${want}" ]]; then
    printf 'PASS  %-42s %s\n' "${label}" "${got}"
  else
    printf 'FAIL  %-42s got %s, want %s\n' "${label}" "${got}" "${want}"
    fail=1
  fi
}
check_ge() {
  local label="$1" got="$2" want="$3"
  if [[ "${got}" -ge "${want}" ]]; then
    printf 'PASS  %-42s %s\n' "${label}" "${got}"
  else
    printf 'FAIL  %-42s got %s, want >= %s\n' "${label}" "${got}" "${want}"
    fail=1
  fi
}

echo "scheduler pid ${pid}; log ${log}"
echo

# The split must have run: one line for wqkv_a, one for engram.wkv.
check_ge "replicated_split layers ON" "$(countlog 'split ON, window GEMMs (no pad)')" 2
check_eq "replicated_split slice failures" "$(countlog 'slice failed on rank 0')" 0
check_ge "compact gather (roce) ON" "$(countlog 'compact gather (roce) bit-identical, ON')" 1
check_ge "moe_b12x_next armed" "$(countlog '\[moe_b12x_next\] armed')" 1
check_ge "RoCEnante world=4" "$(countlog 'RoCEnante ready: world=4')" 1

if [[ "${expect_cert}" == "0" ]]; then
  check_eq "CERT head armed (expected absent)" "$(countlog 'CERT head armed')" 0
fi

cap_line="$(readlog 'max_total_num_tokens=.*context_len' | tail -1 || true)"
if [[ -n "${cap_line}" ]]; then
  printf 'INFO  %s\n' "${cap_line}"
else
  printf 'FAIL  %-42s not found\n' "capacity line"
  fail=1
fi

echo
if [[ "${fail}" -ne 0 ]]; then
  echo "One or more gates failed; do not benchmark this instance."
  exit 1
fi
echo "All gates passed."
