#!/bin/bash
# Delete all DiffIK jobs (mirrors delete-jobs-for-ik.sh).
kubectl delete jobs -l k8s-app=diffik-job
