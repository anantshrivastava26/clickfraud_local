#!/bin/bash
# One-time AWS + GitHub setup for the CI/CD pipeline in .github/workflows/ci-cd.yml.
# Creates (all named clickfraud-monitor-*):
#   - ECR repos clickfraud-monitor/backend and clickfraud-monitor/frontend (keep last 10 images)
#   - EC2 role + instance profile: SSM agent access and ECR pull, attached to the instance
#   - GitHub OIDC deploy role, trusted only for pushes to main of this repo:
#     push to the two ECR repos, send SSM commands to this one instance
#   - GitHub repo variables the workflow reads
# Reuses the account's existing token.actions.githubusercontent.com OIDC provider.
set -euo pipefail
export MSYS_NO_PATHCONV=1

REGION=${AWS_REGION:-ap-south-1}
INSTANCE_ID=${EC2_INSTANCE_ID:-i-027f73ab6453b85b9}
APP_URL=${APP_URL:-http://3.6.185.10}
GH_REPO=${GH_REPO:-anantshrivastava26/clickfraud_local}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
TAG='Key=Project,Value=clickfraud-monitor'

echo "== ECR"
for r in backend frontend; do
  aws ecr describe-repositories --region "$REGION" --repository-names "clickfraud-monitor/$r" >/dev/null 2>&1 ||
    aws ecr create-repository --region "$REGION" --repository-name "clickfraud-monitor/$r" \
      --image-scanning-configuration scanOnPush=true --tags "$TAG" >/dev/null
  aws ecr put-lifecycle-policy --region "$REGION" --repository-name "clickfraud-monitor/$r" --lifecycle-policy-text \
    '{"rules":[{"rulePriority":1,"description":"keep last 10","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":10},"action":{"type":"expire"}}]}' >/dev/null
done

echo "== EC2 instance role"
if ! aws iam get-role --role-name clickfraud-monitor-ec2-role >/dev/null 2>&1; then
  aws iam create-role --role-name clickfraud-monitor-ec2-role --tags "$TAG" --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
fi
aws iam attach-role-policy --role-name clickfraud-monitor-ec2-role --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam attach-role-policy --role-name clickfraud-monitor-ec2-role --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
if ! aws iam get-instance-profile --instance-profile-name clickfraud-monitor-ec2-profile >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name clickfraud-monitor-ec2-profile --tags "$TAG" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name clickfraud-monitor-ec2-profile --role-name clickfraud-monitor-ec2-role
  sleep 10 # instance profiles take a moment to become usable
fi
if [ "$(aws ec2 describe-iam-instance-profile-associations --region "$REGION" --filters Name=instance-id,Values="$INSTANCE_ID" --query 'length(IamInstanceProfileAssociations)' --output text)" = "0" ]; then
  aws ec2 associate-iam-instance-profile --region "$REGION" --instance-id "$INSTANCE_ID" \
    --iam-instance-profile Name=clickfraud-monitor-ec2-profile >/dev/null
fi

echo "== GitHub deploy role"
OIDC="arn:aws:iam::$ACCOUNT:oidc-provider/token.actions.githubusercontent.com"
TRUST=$(cat <<EOF
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Federated":"$OIDC"},"Action":"sts:AssumeRoleWithWebIdentity",
 "Condition":{"StringEquals":{"token.actions.githubusercontent.com:aud":"sts.amazonaws.com",
 "token.actions.githubusercontent.com:sub":"repo:$GH_REPO:ref:refs/heads/main"}}}]}
EOF
)
POLICY=$(cat <<EOF
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":"ecr:GetAuthorizationToken","Resource":"*"},
 {"Effect":"Allow","Action":["ecr:BatchCheckLayerAvailability","ecr:InitiateLayerUpload","ecr:UploadLayerPart",
   "ecr:CompleteLayerUpload","ecr:PutImage","ecr:BatchGetImage","ecr:GetDownloadUrlForLayer"],
  "Resource":["arn:aws:ecr:$REGION:$ACCOUNT:repository/clickfraud-monitor/backend","arn:aws:ecr:$REGION:$ACCOUNT:repository/clickfraud-monitor/frontend"]},
 {"Effect":"Allow","Action":"ssm:SendCommand","Resource":["arn:aws:ec2:$REGION:$ACCOUNT:instance/$INSTANCE_ID",
   "arn:aws:ssm:$REGION::document/AWS-RunShellScript"]},
 {"Effect":"Allow","Action":["ssm:GetCommandInvocation","ssm:ListCommandInvocations"],"Resource":"*"}]}
EOF
)
if aws iam get-role --role-name clickfraud-monitor-github-deploy >/dev/null 2>&1; then
  aws iam update-assume-role-policy --role-name clickfraud-monitor-github-deploy --policy-document "$TRUST"
else
  aws iam create-role --role-name clickfraud-monitor-github-deploy --tags "$TAG" --assume-role-policy-document "$TRUST" >/dev/null
fi
aws iam put-role-policy --role-name clickfraud-monitor-github-deploy --policy-name clickfraud-monitor-deploy --policy-document "$POLICY"
ROLE_ARN=$(aws iam get-role --role-name clickfraud-monitor-github-deploy --query Role.Arn --output text)

echo "== GitHub repo variables"
gh variable set AWS_ROLE_ARN --repo "$GH_REPO" --body "$ROLE_ARN"
gh variable set AWS_REGION --repo "$GH_REPO" --body "$REGION"
gh variable set EC2_INSTANCE_ID --repo "$GH_REPO" --body "$INSTANCE_ID"
gh variable set APP_URL --repo "$GH_REPO" --body "$APP_URL"

echo "Done. The SSM agent picks up the new instance role within a few minutes; check with:"
echo "  aws ssm describe-instance-information --filters Key=InstanceIds,Values=$INSTANCE_ID"
