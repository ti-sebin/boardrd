# SPDX-License-Identifier: GPL-2.0
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""
initrd builder: assemble a staging tree and pack it as a cpio archive.

Steps:
  1. build_busybox()       — cross-compile static busybox
  2. create_staging_dir()  — skeleton dirs + /dev/console
  3. install_busybox()     — copy binary + create applet symlinks
  4. copy_modules()        — copy .ko files preserving kernel/ hierarchy
  5. run_depmod()          — regenerate modules.dep/alias for the staged set
  6. write_modules_conf()  — /etc/initrd-modules.conf (module list + settle_ms)
  7. write_init_script()   — generate /init from template
  8. pack_cpio()           — produce compressed cpio archive
"""

import logging
import os
import shutil
import subprocess
import tempfile

from .paths import get_templates_dir

log = logging.getLogger(__name__)

_TEMPLATE_PATH = os.path.join(get_templates_dir(), 'init.sh.tmpl')

# Busybox applets that must be installed as symlinks
_REQUIRED_APPLETS = [
    # Core init
    'sh', 'bash', 'mount', 'umount', 'switch_root', 'findfs',
    # Network
    'udhcpc', 'ip', 'timeout',
    # Module management
    'modprobe', 'insmod', 'rmmod', 'lsmod',
    # Filesystem / device
    'mkdir', 'mknod', 'ls',
    # Text / parsing
    'echo', 'cat', 'grep', 'cut', 'sleep', 'tr', 'od', 'sed',
    # Diagnostics
    'dmesg', 'ps', 'free',
]

_COMPRESSION_CMDS = {
    'gzip':  ['gzip', '-9', '-n'],
    'xz':    ['xz', '-9'],
    'zstd':  ['zstd', '-19', '-T0'],
    'none':  None,
}


def _busybox_make_vars(cross_compile, arch, use_llvm, musl_root=None):
    """
    Return (env, extra_make_args) for the busybox make invocation.

    GCC mode  (use_llvm=False):
        CROSS_COMPILE=aarch64-linux-gnu-   ARCH=arm64
        Compiler: $(CROSS_COMPILE)gcc

    LLVM + GCC sysroot (use_llvm=True, musl_root=None):
        CC=clang --target=aarch64-linux-gnu --sysroot=<gcc-sysroot>
        Sysroot auto-detected via $(cross_compile)gcc -print-sysroot.

    LLVM + musl (use_llvm=True, musl_root=/path/to/aarch64-linux-musl-cross):
        CC=clang --target=aarch64-linux-musl --sysroot=$MUSL_ROOT/aarch64-linux-musl
        LDFLAGS=-fuse-ld=lld -L$MUSL_ROOT/aarch64-linux-musl/lib
        CROSS_COMPILE not set (musl toolchain is self-contained).
        Download: wget https://musl.cc/aarch64-linux-musl-cross.tgz
    """
    env = os.environ.copy()
    extra = []

    if arch:
        env['ARCH'] = arch

    if not use_llvm:
        if cross_compile:
            env['CROSS_COMPILE'] = cross_compile
        return env, extra

    # --- LLVM + musl ---
    if musl_root:
        # Derive arch prefix from cross_compile (e.g. aarch64-none-linux-gnu-)
        # or arch (arm64 → aarch64).
        arch_cpu = (cross_compile or '').split('-')[0] or {
            'arm64': 'aarch64', 'arm': 'arm', 'riscv': 'riscv64',
        }.get(arch or '', 'aarch64')
        musl_target  = f'{arch_cpu}-linux-musl'
        musl_sysroot = os.path.join(musl_root, musl_target)
        musl_lib     = os.path.join(musl_sysroot, 'lib')
        # --gcc-toolchain tells clang where to find GCC CRT files (crtbeginT.o,
        # crtend.o, libgcc.a) which live under lib/gcc/<target>/<version>/.
        # This is separate from --sysroot (which covers libc/libm headers+libs).
        cc = (f'clang --target={musl_target}'
              f' --sysroot={musl_sysroot}'
              f' --gcc-toolchain={musl_root}')
        ldflags = f'-fuse-ld=lld -L{musl_lib}'

        log.debug("musl build: target=%s sysroot=%s", musl_target, musl_sysroot)
        extra = [
            f'CC={cc}',
            f'LDFLAGS={ldflags}',
            'AR=llvm-ar',
            'NM=llvm-nm',
            'STRIP=llvm-strip',
            'OBJCOPY=llvm-objcopy',
            'OBJDUMP=llvm-objdump',
            'HOSTCC=clang',
        ]
        # musl toolchain is self-contained — no CROSS_COMPILE needed
        return env, extra

    # --- LLVM + GCC sysroot ---
    clang_target = cross_compile.rstrip('-') if cross_compile else None

    # clang needs --sysroot to find the target's libc/libgcc for static
    # linking. Without it, clang silently produces a host-architecture binary.
    sysroot = ''
    if clang_target and cross_compile:
        try:
            sysroot = subprocess.check_output(
                [f'{cross_compile}gcc', '-print-sysroot'],
                stderr=subprocess.DEVNULL).decode().strip()
            log.debug("clang sysroot: %s", sysroot)
        except (subprocess.CalledProcessError, FileNotFoundError):
            log.warning("Could not detect sysroot from %sgcc — "
                        "LLVM build may produce wrong-architecture binary; "
                        "consider --musl-root for a self-contained toolchain",
                        cross_compile)

    cc = f'clang --target={clang_target}' if clang_target else 'clang'
    if sysroot:
        cc += f' --sysroot={sysroot}'

    extra = [
        f'CC={cc}',
        'AR=llvm-ar',
        'NM=llvm-nm',
        'STRIP=llvm-strip',
        'OBJCOPY=llvm-objcopy',
        'OBJDUMP=llvm-objdump',
        'HOSTCC=clang',
    ]
    if cross_compile:
        env['CROSS_COMPILE'] = cross_compile

    log.debug("LLVM build: CC=%s", cc)
    return env, extra


def build_busybox(busybox_src, busybox_config, build_dir,
                  cross_compile=None, arch=None, use_llvm=False,
                  musl_root=None):
    """
    Build a static busybox binary.

    Uses allnoconfig + KCONFIG_ALLCONFIG to start from all-disabled then
    enable only the applets listed in busybox_config.

    Args:
        cross_compile: GCC tool prefix (e.g. 'aarch64-linux-gnu-') or
                       LLVM target triple prefix (same value, used as
                       --target= when use_llvm=True).
        use_llvm:      Use clang + llvm-* tools instead of GCC toolchain.

    Returns path to the busybox binary.
    """
    os.makedirs(build_dir, exist_ok=True)

    env, extra_vars = _busybox_make_vars(cross_compile, arch, use_llvm,
                                         musl_root=musl_root)
    base_make = ['make', '-C', busybox_src, f'O={build_dir}'] + extra_vars

    log.info("Building busybox (%s) in %s ...",
             'LLVM' if use_llvm else 'GCC', build_dir)

    # Copy our config as the base .config, then run olddefconfig to
    # satisfy any Kconfig dependencies that allnoconfig+KCONFIG_ALLCONFIG
    # would silently drop (causing applets to disappear at runtime).
    import shutil
    os.makedirs(build_dir, exist_ok=True)
    shutil.copy2(busybox_config, os.path.join(build_dir, '.config'))
    # oldconfig resolves any unresolved symbols using defaults.
    # Pipe 'yes ""' to auto-answer all interactive prompts with default.
    yes_proc = subprocess.Popen(['yes', ''], stdout=subprocess.PIPE)
    subprocess.run(
        base_make + ['oldconfig'],
        env=env, check=True,
        stdin=yes_proc.stdout,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    yes_proc.terminate()

    ncpus = os.cpu_count() or 1
    subprocess.run(
        base_make + [f'-j{ncpus}', 'CONFIG_STATIC=y'],
        env=env, check=True
    )

    binary = os.path.join(build_dir, 'busybox')
    if not os.path.exists(binary):
        raise FileNotFoundError(
            f"busybox binary not found at {binary} after build"
        )

    # Verify architecture — clang without a valid sysroot silently
    # produces a host-architecture binary instead of failing.
    if arch:
        _ARCH_STRINGS = {
            'arm64':  ('aarch64', 'ARM aarch64'),
            'arm':    ('ARM,',),
            'riscv':  ('RISC-V',),
            'x86_64': ('x86-64',),
        }
        expected = _ARCH_STRINGS.get(arch)
        if expected:
            try:
                file_out = subprocess.check_output(
                    ['file', binary]).decode()
                if not any(s in file_out for s in expected):
                    raise RuntimeError(
                        f"busybox binary has wrong architecture.\n"
                        f"  Expected: {arch} {expected}\n"
                        f"  Got: {file_out.strip()}\n"
                        f"  For LLVM builds, ensure {cross_compile}gcc is on "
                        f"PATH so the aarch64 sysroot can be detected.")
            except FileNotFoundError:
                pass  # 'file' command not available — skip check

    log.info("busybox built: %s", binary)
    return binary


_UDHCPC_SCRIPT = """\
#!/bin/sh
# Minimal udhcpc default script for boardrd initrd
case "$1" in
    bound|renew)
        ip addr flush dev "$interface" 2>/dev/null
        ip addr add "$ip/$mask" dev "$interface"
        [ -n "$router" ] && ip route add default via "$router" dev "$interface"
        ;;
    deconfig)
        ip addr flush dev "$interface" 2>/dev/null
        ;;
esac
"""


def create_staging_dir(staging):
    """Create the initrd directory skeleton."""
    dirs = [
        'proc', 'sys', 'dev', 'mnt/root', 'etc',
        'bin', 'sbin',
        'usr/share/udhcpc',
        'lib/modules',
    ]
    for d in dirs:
        os.makedirs(os.path.join(staging, d), exist_ok=True)

    script_path = os.path.join(staging, 'usr', 'share', 'udhcpc',
                               'default.script')
    with open(script_path, 'w') as f:
        f.write(_UDHCPC_SCRIPT)
    os.chmod(script_path, 0o755)

    # /dev/console is injected directly into the cpio archive as a device
    # node entry (see _cpio_device_entry) — no os.mknod() needed (avoids
    # requiring root privileges on the build host).

    log.debug("Staging skeleton created at %s", staging)


def install_busybox(staging, busybox_bin):
    """Copy busybox to /bin/busybox and create applet symlinks.

    Layout:
      /bin/busybox        — the binary
      /bin/<applet>  -> busybox          (relative, same dir)
      /sbin/<applet> -> ../bin/busybox   (relative, one level up)
    """
    bin_dir  = os.path.join(staging, 'bin')
    sbin_dir = os.path.join(staging, 'sbin')

    dest = os.path.join(bin_dir, 'busybox')
    shutil.copy2(busybox_bin, dest)
    os.chmod(dest, 0o755)

    sbin_applets = {'switch_root', 'modprobe', 'ip'}
    for applet in _REQUIRED_APPLETS:
        if applet in sbin_applets:
            link_path = os.path.join(sbin_dir, applet)
            target = '../bin/busybox'
        else:
            link_path = os.path.join(bin_dir, applet)
            target = 'busybox'
        if not os.path.exists(link_path):
            os.symlink(target, link_path)

    log.debug("busybox installed with %d applet symlinks", len(_REQUIRED_APPLETS))


def copy_modules(staging, modules_dir, ko_relative_paths):
    """
    Copy .ko files into staging, preserving the kernel/ directory hierarchy.

    modules_dir: path to lib/modules/<KERNELRELEASE>/
    ko_relative_paths: iterable of paths relative to modules_dir
                       (e.g. 'kernel/drivers/mmc/host/sdhci_am654.ko')
    """
    kernelrelease = os.path.basename(modules_dir)
    dest_base = os.path.join(staging, 'lib', 'modules', kernelrelease)
    os.makedirs(dest_base, exist_ok=True)

    copied = 0
    for rel_path in ko_relative_paths:
        src = os.path.join(modules_dir, rel_path)
        if not os.path.exists(src):
            log.warning("Module file not found, skipping: %s", src)
            continue
        dst = os.path.join(dest_base, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    log.info("Copied %d .ko files to staging", copied)
    return kernelrelease


def _find_depmod():
    """Locate depmod binary — it lives in /sbin on many distros, not on PATH."""
    import shutil
    depmod = shutil.which('depmod')
    if depmod:
        return depmod
    for candidate in ('/sbin/depmod', '/usr/sbin/depmod',
                      '/usr/local/sbin/depmod'):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "depmod not found on PATH or in /sbin. "
        "Install it with:  apt-get install kmod  OR  dnf install kmod")


def run_depmod(staging, kernelrelease, system_map=None):
    """
    Run depmod against the staged module tree.
    Generates modules.dep, modules.alias, etc. for the staged subset.

    system_map: path to System.map from the kernel build. When provided,
    depmod knows which symbols are supplied by the built-in kernel and
    does not mark them as missing dependencies for the staged modules.
    Without it, modules with dependencies on built-in kernel symbols may
    fail to load at runtime due to a spurious 'Unknown symbol' error.
    """
    cmd = [_find_depmod(), '-a', '-b', staging]
    if system_map and os.path.exists(system_map):
        cmd += ['-F', system_map]
        log.debug("depmod using System.map: %s", system_map)
    else:
        log.warning("No System.map provided to depmod — modules with "
                    "built-in kernel symbol dependencies may fail to load")
    cmd.append(kernelrelease)
    subprocess.run(cmd, check=True)
    log.debug("depmod completed for %s", kernelrelease)


def write_modules_conf(staging, mode_to_modules, mode_to_settle_ms, resolver=None):
    """
    Write /etc/initrd-modules.conf.

    Format:
        <mode>.settle_ms=<ms>
        <mode>:<module_name>

    Modules are written in dependency order (deps before dependents) so each
    modprobe call finds its dependencies already loaded. If resolver is None,
    falls back to alphabetical order.
    """
    conf_path = os.path.join(staging, 'etc', 'initrd-modules.conf')
    lines = [
        '# boardrd generated — do not edit',
        '# settle_ms: Pass 1 → Pass 2 delay per workflow',
    ]
    for mode, ms in sorted(mode_to_settle_ms.items()):
        lines.append(f'{mode}.settle_ms={ms}')
    lines.append('')
    for mode, modules in sorted(mode_to_modules.items()):
        if resolver is not None:
            ordered = resolver.topo_sort(modules)
        else:
            ordered = sorted(modules)
        for mod in ordered:
            lines.append(f'{mode}:{mod}')

    with open(conf_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log.debug("Wrote %s", conf_path)


def write_init_script(staging, boards, workflows):
    """
    Generate /init from the template, substituting board and workflow lists.
    """
    with open(_TEMPLATE_PATH) as f:
        content = f.read()

    board_str = ', '.join(boards)
    workflow_str = ' '.join(workflows)

    content = content.replace('@BOARDS@', board_str)
    content = content.replace('@WORKFLOWS@', workflow_str)

    init_path = os.path.join(staging, 'init')
    with open(init_path, 'w') as f:
        f.write(content)
    os.chmod(init_path, 0o755)
    log.debug("Wrote /init")


def _cpio_device_entry(path, major, minor, mode=0o600):
    """
    Return raw bytes for a cpio newc entry representing a character device.
    No root privileges required — the device node is encoded entirely in
    the archive header without touching the filesystem.

    path:  archive path (e.g. 'dev/console')
    major/minor: device numbers
    mode:  file permission bits (default 0o600)
    """
    name = path.encode() + b'\x00'
    namesize = len(name)
    # S_IFCHR = 0o020000
    full_mode = 0o020000 | mode

    header = (
        b'070701'                          # magic
        + f'{0:08x}'.encode()              # ino
        + f'{full_mode:08x}'.encode()      # mode
        + f'{0:08x}'.encode()              # uid
        + f'{0:08x}'.encode()              # gid
        + f'{1:08x}'.encode()              # nlink
        + f'{0:08x}'.encode()              # mtime
        + f'{0:08x}'.encode()              # filesize
        + f'{0:08x}'.encode()              # devmajor (containing device)
        + f'{0:08x}'.encode()              # devminor
        + f'{major:08x}'.encode()          # rdevmajor
        + f'{minor:08x}'.encode()          # rdevminor
        + f'{namesize:08x}'.encode()       # namesize
        + f'{0:08x}'.encode()              # check
    )
    # Pad header + name to 4-byte boundary
    data = header + name
    pad = (4 - (len(data) % 4)) % 4
    return data + b'\x00' * pad


def _cpio_trailer():
    """Return the cpio TRAILER!!! entry that terminates the archive."""
    name = b'TRAILER!!!\x00'
    namesize = len(name)
    header = (
        b'070701'
        + f'{0:08x}'.encode()   # ino
        + f'{0:08x}'.encode()   # mode
        + f'{0:08x}'.encode()   # uid
        + f'{0:08x}'.encode()   # gid
        + f'{1:08x}'.encode()   # nlink
        + f'{0:08x}'.encode()   # mtime
        + f'{0:08x}'.encode()   # filesize
        + f'{0:08x}'.encode()   # devmajor
        + f'{0:08x}'.encode()   # devminor
        + f'{0:08x}'.encode()   # rdevmajor
        + f'{0:08x}'.encode()   # rdevminor
        + f'{namesize:08x}'.encode()
        + f'{0:08x}'.encode()   # check
    )
    data = header + name
    pad = (4 - (len(data) % 4)) % 4
    return data + b'\x00' * pad


def pack_cpio(staging, output_path, compression='gzip'):
    """
    Pack the staging directory into a cpio archive.

    compression: 'gzip' | 'xz' | 'zstd' | 'none'
    """
    if compression not in _COMPRESSION_CMDS:
        raise ValueError(
            f"Unknown compression '{compression}'. "
            f"Valid: {list(_COMPRESSION_CMDS)}"
        )

    compress_cmd = _COMPRESSION_CMDS[compression]

    # Determine output path / suffix
    if compression != 'none' and not output_path.endswith(
            ('.gz', '.xz', '.zst')):
        suffixes = {'gzip': '.gz', 'xz': '.xz', 'zstd': '.zst'}
        output_path += suffixes[compression]

    log.info("Packing initrd → %s (compression=%s)", output_path, compression)

    # Device nodes that cannot be created on disk without root are injected
    # directly as cpio header entries prepended to the archive.
    device_entries = _cpio_device_entry('dev/console', 5, 1, mode=0o600)

    # find . | cpio -o -H newc | [compress] > output
    find_proc = subprocess.Popen(
        ['find', '.', '-print0'],
        cwd=staging,
        stdout=subprocess.PIPE
    )
    # Collect staging-dir cpio into memory, then concatenate with device
    # entries. Two valid cpio archives concatenated = one valid archive;
    # the kernel's initramfs extractor processes them in sequence.
    cpio_proc = subprocess.Popen(
        ['cpio', '--null', '-o', '-H', 'newc'],
        cwd=staging,
        stdin=find_proc.stdout,
        stdout=subprocess.PIPE,
    )
    assert find_proc.stdout is not None
    find_proc.stdout.close()
    staging_cpio, _ = cpio_proc.communicate()
    find_proc.wait()

    if cpio_proc.returncode != 0:
        raise subprocess.CalledProcessError(cpio_proc.returncode, 'cpio')

    # Prepend device node entries (device_entries already has its own
    # TRAILER-less structure; staging_cpio ends with its own TRAILER).
    combined = device_entries + staging_cpio

    if compress_cmd:
        compress_proc = subprocess.Popen(
            compress_cmd,
            stdin=subprocess.PIPE,
            stdout=open(output_path, 'wb'),
        )
        assert compress_proc.stdin is not None
        compress_proc.stdin.write(combined)
        compress_proc.stdin.close()
        compress_proc.wait()
        if compress_proc.returncode != 0:
            raise subprocess.CalledProcessError(
                compress_proc.returncode, compress_cmd)
    else:
        with open(output_path, 'wb') as out_f:
            out_f.write(combined)

    size = os.path.getsize(output_path)
    log.info("initrd written: %s (%.1f MiB)", output_path, size / 1048576)
    return output_path


def build_initrd(modules_dir, ko_paths, mode_to_modules, mode_to_settle_ms,
                 boards, workflows, output_path,
                 busybox_bin=None, busybox_src=None, busybox_config=None,
                 busybox_build_dir=None, cross_compile=None, arch=None,
                 use_llvm=False, compression='gzip', system_map=None,
                 resolver=None, musl_root=None):
    """
    Full pipeline: staging tree → cpio archive.

    Either busybox_bin (pre-built) or busybox_src+busybox_config must be given.
    """
    _bb_ctx = None
    if busybox_bin is None:
        if busybox_src is None or busybox_config is None:
            raise ValueError(
                "Either busybox_bin or (busybox_src + busybox_config) required"
            )
        if busybox_build_dir:
            # Persistent dir: user opted in for incremental rebuilds
            os.makedirs(busybox_build_dir, exist_ok=True)
            build_dir = busybox_build_dir
        else:
            # Temp dir: cleaned up after the cpio is packed
            _bb_ctx = tempfile.TemporaryDirectory(prefix='boardrd_bb_')
            build_dir = _bb_ctx.name
        busybox_bin = build_busybox(
            busybox_src, busybox_config, build_dir,
            cross_compile, arch, use_llvm,
            musl_root=musl_root)

    with tempfile.TemporaryDirectory(prefix='boardrd_staging_') as staging:
        create_staging_dir(staging)
        install_busybox(staging, busybox_bin)
        kernelrelease = copy_modules(staging, modules_dir, ko_paths)
        run_depmod(staging, kernelrelease, system_map)
        write_modules_conf(staging, mode_to_modules, mode_to_settle_ms,
                           resolver=resolver)
        write_init_script(staging, boards, workflows)
        pack_cpio(staging, output_path, compression)

    if _bb_ctx is not None:
        _bb_ctx.cleanup()

    return output_path
