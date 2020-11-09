#/bin/bash
if [ -z "$CI_PIPELINE_ID" ]; then
  CI_PIPELINE_ID=0
fi

BRANCH=`git rev-parse --abbrev-ref HEAD`

printf "branch\tpipeline\n"
printf "$BRANCH\t$CI_PIPELINE_ID\n"
