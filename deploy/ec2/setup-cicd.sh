#!/bin/bash
# One-time bootstrap for the CI/CD pipeline in .github/workflows/ci-cd.yml.
#
# GitHub can't create its own way into AWS, so someone with admin credentials runs
# this once. After that every push to main deploys through OIDC with no stored keys.
#
#   1. Deploys infra/github-deploy-role.yaml as stack clickfraud-monitor-github-deploy
#      (ECR repos, EC2 instance role, GitHub deploy role)
#   2. Attaches the instance role to the EC2 host so SSM and ECR pulls work
#   3. Creates the GitHub "production" environment and sets the secret/variables the workflow reads
# Stop on the first error (-e) or unset variable (-u), and fail a pipeline if any stage fails.
set -euo pipefail
# Git Bash on Windows rewrites arguments that look like Unix paths; switch that off so aws and gh
# receive them unchanged.
export MSYS_NO_PATHCONV=1
cd "$(dirname "$0")/../.."

# The defaults describe the current deployment. Override them through environment variables to
# target another region, instance, URL or repository.
REGION=${AWS_REGION:-ap-south-1}
INSTANCE_ID=${EC2_INSTANCE_ID:-i-027f73ab6453b85b9}
APP_URL=${APP_URL:-http://3.6.185.10}
GH_REPO=${GH_REPO:-anantshrivastava26/clickfraud_local}
STACK=clickfraud-monitor-github-deploy

echo "== CloudFormation: $STACK"
# Create or update the stack. CAPABILITY_NAMED_IAM acknowledges that it creates IAM roles with fixed
# names, and --no-fail-on-empty-changeset makes re-running the script a no-op instead of an error.
aws cloudformation deploy \
  --region "$REGION" \
  --stack-name "$STACK" \
  --template-file infra/github-deploy-role.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides GitHubRepo="$GH_REPO" InstanceId="$INSTANCE_ID" \
  --tags Project=clickfraud-monitor \
  --no-fail-on-empty-changeset

# Read one output value of the stack by key.
output() {
  aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}
ROLE_ARN=$(output DeployRoleArn)
PROFILE=$(output InstanceProfileName)

echo "== Attach $PROFILE to $INSTANCE_ID"
# Attach the profile only if the instance has none yet, so re-running is safe (an instance can hold
# one profile).
if [ "$(aws ec2 describe-iam-instance-profile-associations --region "$REGION" \
      --filters Name=instance-id,Values="$INSTANCE_ID" --query 'length(IamInstanceProfileAssociations)' --output text)" = "0" ]; then
  aws ec2 associate-iam-instance-profile --region "$REGION" --instance-id "$INSTANCE_ID" \
    --iam-instance-profile Name="$PROFILE" >/dev/null
fi

echo "== GitHub: production environment, secret and variables"
# The deploy role trusts only the `production` environment (see infra/github-deploy-role.yaml), so
# the workflow's deploy job has to run in it. The role ARN is stored as a secret and the other
# values as plain variables.
gh api -X PUT "repos/$GH_REPO/environments/production" >/dev/null
gh secret set AWS_DEPLOY_ROLE_ARN --repo "$GH_REPO" --body "$ROLE_ARN"
gh variable set AWS_REGION --repo "$GH_REPO" --body "$REGION"
gh variable set EC2_INSTANCE_ID --repo "$GH_REPO" --body "$INSTANCE_ID"
gh variable set APP_URL --repo "$GH_REPO" --body "$APP_URL"

echo
echo "Done. The SSM agent picks up the instance role within a few minutes. Check with:"
echo "  aws ssm describe-instance-information --region $REGION --filters Key=InstanceIds,Values=$INSTANCE_ID"
echo "Optional approval gate: GitHub > Settings > Environments > production > Required reviewers."
