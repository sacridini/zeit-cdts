"""What the operating system needs: no sleep while tiles are processed, and the firewall
rule that lets helpers reach the coordinator."""

import contextlib
import shutil
import subprocess
import sys

from . import _group


@contextlib.contextmanager
def keep_awake():
    """Keep the computer from sleeping (the screen may still turn off) while in the block."""
    proc = None
    if sys.platform == "win32":
        try:
            import ctypes

            # ES_CONTINUOUS | ES_SYSTEM_REQUIRED, for this thread until reset.
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except Exception:
            pass
    else:
        import os

        tool = None
        if sys.platform == "darwin" and shutil.which("caffeinate"):
            tool = ["caffeinate", "-i", "-w", str(os.getpid())]
        elif shutil.which("systemd-inhibit"):
            tool = ["systemd-inhibit", "--what=sleep", "--why=zeit is processing tiles", "--mode=block",
                    "sleep", "infinity"]
        if tool:
            try:
                proc = subprocess.Popen(tool, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                        stdin=subprocess.DEVNULL)
            except OSError:
                proc = None
    try:
        yield
    finally:
        if sys.platform == "win32":
            try:
                import ctypes

                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            except Exception:
                pass
        if proc is not None:
            proc.terminate()


def firewall_commands(tcp_port: int = _group.TCP_PORT, udp_port: int = _group.UDP_PORT):
    """The netsh commands of the Windows rules: inbound, from the local subnet only."""
    rules = []
    for proto, port in (("TCP", tcp_port), ("UDP", udp_port)):
        name = f"zeit-tiles-{proto}-{port}"
        rules.append(["netsh", "advfirewall", "firewall", "delete", "rule", f"name={name}"])
        rules.append(["netsh", "advfirewall", "firewall", "add", "rule", f"name={name}", "dir=in", "action=allow",
                      f"protocol={proto}", f"localport={port}", "remoteip=localsubnet", "profile=any"])
    return rules


def open_firewall(tcp_port: int = _group.TCP_PORT, udp_port: int = _group.UDP_PORT) -> str:
    """Create the rules on Windows (asking for administrator rights); elsewhere, say what to do."""
    if sys.platform != "win32":
        if sys.platform == "darwin":
            return ("macOS asks on the first run whether Python may accept incoming connections: allow it. "
                    "Nothing else is needed.")
        return (f"If a firewall is active, allow TCP {tcp_port} and UDP {udp_port} from the local network, e.g.\n"
                f"  sudo ufw allow from 192.168.1.0/24 to any port {tcp_port} proto tcp\n"
                f"  sudo ufw allow from 192.168.1.0/24 to any port {udp_port} proto udp\n"
                "(with your network's subnet)")
    listing = "\n".join("  " + " ".join(cmd) for cmd in firewall_commands(tcp_port, udp_port) if cmd[3] == "add")
    # One elevation prompt for the four commands.
    script = " & ".join(" ".join(cmd) for cmd in firewall_commands(tcp_port, udp_port))
    ps = f"Start-Process -FilePath cmd.exe -Verb RunAs -Wait -WindowStyle Hidden -ArgumentList '/c {script}'"
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    except OSError as e:
        return f"Could not run PowerShell ({e}). Run these in an administrator terminal:\n{listing}"
    check = subprocess.run(["netsh", "advfirewall", "firewall", "show", "rule", f"name=zeit-tiles-TCP-{tcp_port}"],
                           capture_output=True, text=True)
    if check.returncode != 0:
        return ("The rules were not created (administrator rights refused?). Run these in an administrator "
                f"terminal:\n{listing}")
    return (f"Firewall rules created: TCP {tcp_port} and UDP {udp_port}, from the local subnet only. "
            "Helpers can now reach this computer.")
