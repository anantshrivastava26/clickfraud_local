#!/bin/bash
# Ship the working tree to the EC2 host and rebuild there.
#   EC2_HOST=1.2.3.4 SSH_KEY=~/.ssh/clickfraud-monitor-key.pem deploy/ec2/deploy.sh
# Stop on the first error (-e) or unset variable (-u), and fail a pipeline if any stage fails.
set -euo pipefail
# Work from the repository root, wherever the script is called from.
cd "$(dirname "$0")/../.."
# Fail fast, with a clear message, if the target host was not given.
: "${EC2_HOST:?set EC2_HOST}"
# Key pair of the instance. Never commit it: .gitignore blocks *.pem.
SSH_KEY=${SSH_KEY:-~/.ssh/clickfraud-monitor-key.pem}
# accept-new trusts the host's key on first connection and pins it afterwards, so a later change of
# key is rejected.
SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new ec2-user@$EC2_HOST"

# Stream a tarball of the three deployable directories over SSH, leaving out local build output and
# caches, and unpack it on the host.
tar czf - --exclude=node_modules --exclude=.venv --exclude=dist --exclude=__pycache__ --exclude=.git \
  backend frontend deploy | $SSH "mkdir -p /opt/clickfraud-monitor && tar xzf - -C /opt/clickfraud-monitor"
# Then run the same script CI uses. With BACKEND_IMAGE and FRONTEND_IMAGE unset it builds the images
# from the files just uploaded.
$SSH "bash /opt/clickfraud-monitor/deploy/ec2/run.sh"
