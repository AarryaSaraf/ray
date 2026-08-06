#!/usr/bin/env bash
# The 15-second checks, as a script you run rather than something you source.
#
# `source common.sh && check_env && check_s3` also works now, but running it is
# safer: nothing it does can touch the state of your interactive shell.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_env
check_s3
echo "preflight OK -- ${NUM_CPUS} CPUs, ${OBJECT_STORE_MB} MiB object store, ${S3_ROOT}"
