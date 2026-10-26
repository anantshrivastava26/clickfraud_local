#!/bin/bash
# Ship the working tree to the EC2 host and rebuild there.
#   EC2_HOST=1.2.3.4 SSH_KEY=~/.ssh/clickfraud-monitor-key.pem deploy/ec2/deploy.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${EC2_HOST:?set EC2_HOST}"
SSH_KEY=${SSH_KEY:-~/.ssh/clickfraud-monitor-key.pem}
SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new ec2-user@$EC2_HOST"

tar czf - --exclude=node_modules --exclude=.venv --exclude=dist --exclude=__pycache__ --exclude=.git \
  backend frontend deploy | $SSH "mkdir -p /opt/clickfraud-monitor && tar xzf - -C /opt/clickfraud-monitor"
$SSH "bash /opt/clickfraud-monitor/deploy/ec2/run.sh"
