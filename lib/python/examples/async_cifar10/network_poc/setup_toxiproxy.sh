#!/usr/bin/env bash
set -euo pipefail

# 1. Require an active conda env (so binaries land somewhere durable + on PATH).
if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "activate the flame conda env first (conda activate my_flame_env)" >&2
  exit 1
fi

# 2. Ensure go is present; install via conda if missing (no sudo needed).
if ! command -v go >/dev/null 2>&1; then
  echo "installing go via conda-forge..."
  conda install -y -c conda-forge go
fi

# 3. Build toxiproxy-server / toxiproxy-cli into the conda env's bin/.
export GOBIN="$CONDA_PREFIX/bin"
TOXIPROXY_VERSION="v2.12.0"
go install "github.com/Shopify/toxiproxy/v2/cmd/server@${TOXIPROXY_VERSION}"
go install "github.com/Shopify/toxiproxy/v2/cmd/cli@${TOXIPROXY_VERSION}"
mv "$GOBIN/server" "$GOBIN/toxiproxy-server"
mv "$GOBIN/cli" "$GOBIN/toxiproxy-cli"

# 4. Sanity check.
toxiproxy-server --version
toxiproxy-cli --version
echo "toxiproxy installed into $GOBIN"