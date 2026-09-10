#!/usr/bin/env bash
# git credential helper: hand back the gh token for github.com
case "$1" in
  get)
    echo "username=x-access-token"
    echo "password=$(gh auth token)"
    ;;
  store|erase)
    exit 0
    ;;
esac
