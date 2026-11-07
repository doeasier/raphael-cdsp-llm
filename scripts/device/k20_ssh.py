#!/usr/bin/env python3
"""Reusable SSH/SFTP helper for the K20 Pro (Redmi K20 Pro, Ubuntu rootfs).

Connection: USB-NCM 172.16.42.1 (root / 1234). Tries the local SSH key first,
then falls back to password auth. Used by the cDSP GEMM deploy + verification
scripts so they don't depend on sshpass (not available on this Windows host).
"""
import os
import sys
import paramiko

HOST = os.environ.get("K20_HOST", "172.16.42.1")
PORT = int(os.environ.get("K20_PORT", "22"))
USER = os.environ.get("K20_USER", "root")
PASS = os.environ.get("K20_PASS", "1234")
KEY  = os.path.expanduser("~/.ssh/id_rsa")


def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # try key first
    try:
        client.connect(HOST, port=PORT, username=USER, key_filename=KEY,
                       timeout=8, look_for_keys=False, allow_agent=False)
        return client
    except Exception:
        pass
    # fall back to password
    client.connect(HOST, port=PORT, username=USER, password=PASS,
                   timeout=8, look_for_keys=False, allow_agent=False)
    return client


def run(client, cmd, timeout=120):
    """Run a command, return (stdout, stderr, exit_status)."""
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    status = stdout.channel.recv_exit_status()
    return out, err, status


def put(client, local, remote):
    """SFTP put a single file."""
    sftp = client.open_sftp()
    try:
        sftp.put(local, remote)
    finally:
        sftp.close()


def put_dir(client, local_dir, remote_dir):
    """Recursively SFTP-put a directory (preserving structure)."""
    sftp = client.open_sftp()
    try:
        for root, _dirs, files in os.walk(local_dir):
            rel = os.path.relpath(root, local_dir)
            rdir = os.path.normpath(os.path.join(remote_dir, rel)).replace("\\", "/")
            try:
                sftp.stat(rdir)
            except IOError:
                sftp.mkdir(rdir)
            for f in files:
                sftp.put(os.path.join(root, f),
                         os.path.normpath(os.path.join(rdir, f)).replace("\\", "/"))
    finally:
        sftp.close()


def put_exec(client, local, remote, timeout=120):
    """Transfer a file to the device over an SSH exec channel using base64.

    The SFTP subsystem on this device's dropbear/openssh is flaky (sporadic
    ENOENT on write-open even when the dir exists), so we pipe base64 through a
    normal shell command instead, which is reliable.
    """
    import base64
    with open(local, "rb") as fh:
        data = fh.read()
    b64 = base64.b64encode(data).decode("ascii").replace("\n", "")
    # Single-quote safe: base64 alphabet contains no single quotes.
    cmd = "mkdir -p $(dirname %s) && printf '%%s' '%s' | base64 -d > %s" % (
        remote, b64, remote)
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    err = stderr.read().decode("utf-8", "replace")
    status = stdout.channel.recv_exit_status()
    if status != 0:
        raise RuntimeError("put_exec failed for %s: %s" % (remote, err))
    return status


if __name__ == "__main__":
    c = connect()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "uname -a"
    out, err, st = run(c, cmd)
    sys.stdout.write(out)
    if err:
        sys.stderr.write(err)
    c.close()
    sys.exit(st)
