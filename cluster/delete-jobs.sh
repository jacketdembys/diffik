#!/bin/bash
# Delete DiffIK k8s jobs.
# Usage:
#   bash cluster/delete-jobs.sh            # architecture-sweep jobs only (diffik-arch-*)
#   bash cluster/delete-jobs.sh all        # ALL diffik jobs (k8s-app=diffik-job)
#   bash cluster/delete-jobs.sh model      # model-size-sweep jobs (diffik-model-*)
#   bash cluster/delete-jobs.sh <prefix>   # any jobs whose name starts with <prefix>
set -euo pipefail

sel="${1:-arch}"

if [ "${sel}" = "all" ]; then
  kubectl delete jobs -l k8s-app=diffik-job
  exit 0
fi

case "${sel}" in
  arch)  pattern="diffik-arch-" ;;
  model) pattern="diffik-model-" ;;
  *)     pattern="${sel}" ;;
esac

jobs=$(kubectl get jobs -o name | grep -- "${pattern}" || true)
if [ -z "${jobs}" ]; then
  echo "no jobs matching '${pattern}'"
  exit 0
fi
echo "deleting jobs matching '${pattern}':"
echo "${jobs}"
echo "${jobs}" | xargs kubectl delete
