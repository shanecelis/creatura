#!/bin/bash
# debugger.sh
#
# Informed by this answer[1].
#
# [1]: https://stackoverflow.com/questions/52685869/is-there-a-way-to-directly-run-the-program-built-by-cargo-in-gdb-or-lldb

if [[ -z "$DEBUG" ]]; then
    exec "$@"
else
    exec rust-lldb "$@"
fi
