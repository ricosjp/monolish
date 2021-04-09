# Contribution guide {#contribution}

If you want to contribute to monolish, create an [Issue](https://github.com/ricosjp/monolish/issues) and create a pull request for the Issue.
Before resolving the WIP of the pull request, you need to do the following two things

- Apply clang-format with `make format` command.
- Record the major fixes and the URL of the pull request in `CHANGELOG.md`.

The `make format` command will automatically apply clang-format to all git added files.

## How to pull request

monolish is tested on the RICOS gitlab server using gitlab CI since Github Actions do not provide GPUs.

Currently, the gitlab server in RICOS is internal, so the pull request needs to be approved by someone in RICOS.

### For contributer (without RICOS co. Ltd.)
1. Create an issue on github
2. Submit a pull request from github

We are mirroring github from gitlab.
However, gitlab is not mirrored from github.

So we don't resolve push and pull requests on github.
After RICOS members move PR branches to internal GitLab by hand, PR branches will be tested on gitlab-CI.
It is merged to master in gitlab in the company and mirrored to github.

If the test fails, we share the result of gitlab CI on github issue.

We want contributors to share their build and test environment on the issue, e.g. `MKL`, `OSS`, `NVIDIA`, etc.

### For RICOS member
1. create an issue on github
2. git checkout -b [branch].
3. (edit file...)
4. git push gitlab [branch]
5. make a merge request from gitlab
6. someone in RICOS approves it
7. (The master branch in github is mirrored in the master branch in gitlab, so when it's merged into master, it's automatically reflected in github.)
8. manually close the github issue with a link to the commit hash
