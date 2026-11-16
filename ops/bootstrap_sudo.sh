#!/usr/bin/env bash
# One-time Phase 1 system setup. Run with: sudo bash ~/dcfg/ops/bootstrap_sudo.sh
set -euo pipefail
mkdir -p -m 755 /etc/apt/keyrings
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg -o /etc/apt/keyrings/githubcli-archive-keyring.gpg
chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list
apt update
apt install -y build-essential cmake git ffmpeg unzip curl libgl1 libglib2.0-0 python3-dev gh
echo "bootstrap OK"
