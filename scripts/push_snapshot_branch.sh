#!/usr/bin/env bash
set -euo pipefail

BRANCH="${1:?branch name is required}"
SOURCE_DIR="${2:?source directory is required}"

if [[ -z "${GITHUB_TOKEN:-}" || -z "${GITHUB_REPOSITORY:-}" ]]; then
  echo "GITHUB_TOKEN and GITHUB_REPOSITORY are required" >&2
  exit 2
fi
if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source directory does not exist: $SOURCE_DIR" >&2
  exit 2
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
cp -a "$SOURCE_DIR"/. "$TMP_DIR"/
cd "$TMP_DIR"
git init -b "$BRANCH"
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add -A
git commit -m "Update $BRANCH snapshot"
git remote add origin "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
git push --force origin "$BRANCH"
