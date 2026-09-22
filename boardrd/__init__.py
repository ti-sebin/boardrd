#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""
boardrd — board-specific initrd builder for embedded Linux systems.

Builds a minimal initramfs containing exactly the kernel modules needed
to boot a specific set of boards across multiple boot modes.

Usage:
    boardrd.py --config boards/arm64/ti/am62x-sk.yaml \\
               --modules-dir /path/to/lib/modules/6.x.y \\
               --output am62x-initrd.cpio.gz

    boardrd.py --generate-workflows \\
               --dtb arch/arm64/boot/dts/ti/k3-am625-sk.dtb \\
               --workflow mmc \\
               --modules-dir /path/to/lib/modules/6.x.y
"""

import argparse
import logging
import os
import sys

import yaml

from .lib.dt_analyzer import get_compatibles
from .lib.mod_resolver import ModResolver
from .lib.workflow import load_workflow, generate_workflow_yaml
from .lib.initrd_builder import build_initrd
from .lib.paths import get_busybox_src, get_busybox_config, get_workflows_dir

log = logging.getLogger('boardrd')

_DEFAULT_BUSYBOX_SRC    = get_busybox_src()
_DEFAULT_BUSYBOX_CONFIG = get_busybox_config()
_DEFAULT_WORKFLOW_DIR   = get_workflows_dir()


# --------------------------------------------------------------------------
# Config loading (YAML baseline, CLI overrides)
# --------------------------------------------------------------------------

def _normalise_workflows(workflows, board_boot_nodes=None):
    """
    Normalise the workflows list to a list of dicts with 'name' and
    optional 'boot_nodes'.

    Accepts mixed input:
      - Plain string:      'mmc'
        → {name: mmc, boot_nodes: <board_boot_nodes or None>}
      - Dict without boot_nodes:  {name: mmc}
        → {name: mmc, boot_nodes: <board_boot_nodes or None>}
      - Dict with boot_nodes:     {name: nfs, boot_nodes: [ethernet0]}
        → unchanged

    board_boot_nodes: legacy top-level boot_nodes from the board entry,
                      used as fallback when a workflow has none of its own.
    """
    result = []
    for wf in (workflows or []):
        if isinstance(wf, str):
            entry = {'name': wf}
        else:
            entry = dict(wf)
        # Inherit board-level boot_nodes only if workflow has none
        if 'boot_nodes' not in entry and board_boot_nodes:
            entry['boot_nodes'] = board_boot_nodes
        result.append(entry)
    return result


def _normalise_board_yaml(cfg):
    """
    Normalise board YAML or main config to internal format.

    Board YAML (flat):
        name: am62x-sk
        dtbs: [...]
        workflows:
          - name: mmc
            boot_nodes: [mmc0, mmc1]
          - name: nfs
            boot_nodes: [ethernet0]

    Main config (boards-list):
        boards:
          - name: am62x-sk
            dtbs: [...]
            workflows: [{name: mmc, boot_nodes: [...]}, ...]

    Workflows may also be plain strings for backward compatibility.
    Board-level boot_nodes (legacy) are inherited by workflows that
    don't specify their own.
    """
    if 'dtbs' in cfg and 'boards' not in cfg:
        board_boot_nodes = cfg.pop('boot_nodes', None)
        raw_workflows = cfg.pop('workflows', [])
        board_entry = {
            'name': cfg.get('name', 'board'),
            'dtbs': cfg.pop('dtbs'),
            'workflows': _normalise_workflows(raw_workflows, board_boot_nodes),
        }
        if not raw_workflows:
            log.warning("No workflows: in board config — nothing to build")
        cfg['boards'] = [board_entry]
    else:
        # Normalise workflows inside each board entry
        for board in cfg.get('boards', []):
            board_boot_nodes = board.pop('boot_nodes', None)
            board['workflows'] = _normalise_workflows(
                board.get('workflows', []), board_boot_nodes)
    return cfg


def _resolve_dtb_paths(cfg, kernel_dir):
    """Prepend kernel_dir to DTB paths that are not already absolute."""
    if not kernel_dir:
        return
    for board in cfg.get('boards', []):
        resolved = []
        for dtb in board.get('dtbs', []):
            if not os.path.isabs(dtb):
                resolved.append(os.path.join(kernel_dir, dtb))
            else:
                resolved.append(dtb)
        board['dtbs'] = resolved


def load_config(args):
    """
    Merge YAML config file (if given) with CLI arguments.
    CLI values take priority over YAML values.
    --dtb and --workflow on CLI replace (not append) the YAML lists.
    Board YAMLs passed directly to --config are normalised automatically.
    DTB paths are resolved relative to --kernel-dir if given.
    """
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        cfg = _normalise_board_yaml(cfg)

    # CLI overrides — each replaces the YAML value if provided
    def override(key, cli_val):
        if cli_val is not None:
            cfg[key] = cli_val

    override('modules_dir',    args.modules_dir)
    override('kernel_dir',     args.kernel_dir)
    override('system_map',     args.system_map)
    override('output',         args.output)
    override('compression',    args.compression)
    # --busybox can be a pre-built binary (file) or a build cache dir
    if args.busybox:
        p = args.busybox
        if os.path.isfile(p):
            cfg['busybox'] = p           # pre-built binary
        else:
            cfg['busybox_build_dir'] = p  # build cache directory
    override('busybox_src', args.busybox_src)
    override('musl_root',   args.musl_root)
    override('cross_compile',  args.cross_compile)
    override('arch',           args.arch)
    if args.llvm or args.musl_root:
        cfg['use_llvm'] = True
    override('workflow_dir',   args.workflow_dir)

    # List overrides: --dtb replaces boards entirely; --workflow replaces workflows
    if args.dtb:
        cfg['boards'] = [{'name': os.path.basename(d), 'dtbs': [d]}
                         for d in args.dtb]
    if args.workflow:
        cfg['workflows'] = args.workflow
    if args.boot_nodes:
        for board in cfg.get('boards', []):
            board['boot_nodes'] = args.boot_nodes

    # Resolve DTB paths relative to kernel_dir
    _resolve_dtb_paths(cfg, cfg.get('kernel_dir'))

    return cfg


def _find_tool(*names, extra_paths=()):
    """Return the path to the first found tool, or None."""
    import shutil
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    for name in names:
        for d in extra_paths:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    return None


def _check_required_tools(cfg):
    """
    Check all external tools needed for a build.
    Returns a list of error strings (empty = all good).
    Reports every missing tool at once rather than failing on the first.
    """
    SBIN = ('/sbin', '/usr/sbin', '/usr/local/sbin')
    errs = []

    def need(tool, pkg, *aliases, extra_paths=SBIN):
        if not _find_tool(tool, *aliases, extra_paths=extra_paths):
            errs.append(f"'{tool}' not found.  Install:  "
                        f"apt-get install {pkg}   OR   dnf install {pkg}")

    # Always required
    need('cpio',   'cpio')
    need('find',   'findutils')
    need('depmod', 'kmod', extra_paths=SBIN)
    need('make',   'make')

    # Compression (only the selected one)
    comp = cfg.get('compression', 'gzip')
    if comp == 'gzip':
        need('gzip', 'gzip')
    elif comp == 'xz':
        need('xz', 'xz-utils')
    elif comp == 'zstd':
        need('zstd', 'zstd')

    # Busybox build tools
    building_busybox = (cfg.get('busybox_src') or _DEFAULT_BUSYBOX_SRC) and \
                       not cfg.get('busybox')
    if building_busybox:
        need('make', 'make')
        use_llvm  = cfg.get('use_llvm', False)
        musl_root = cfg.get('musl_root')

        if musl_root or use_llvm:
            need('clang',        'clang')
            need('llvm-ar',      'llvm',   extra_paths=SBIN)
            need('llvm-nm',      'llvm',   extra_paths=SBIN)
            need('llvm-strip',   'llvm',   extra_paths=SBIN)
            need('llvm-objcopy', 'llvm',   extra_paths=SBIN)
            need('llvm-objdump', 'llvm',   extra_paths=SBIN)
            if musl_root or use_llvm:
                need('ld.lld',   'lld',    extra_paths=SBIN)
        else:
            cross = cfg.get('cross_compile', '')
            if cross:
                need(f'{cross}gcc', f'gcc-{cross.rstrip("-")}')
            else:
                need('gcc', 'gcc')

    return errs


def validate_config(cfg, build=True):
    """Validate configuration. Set build=False to skip busybox checks (--list-modules)."""
    missing = []
    errors = []

    if not cfg.get('modules_dir'):
        missing.append('--modules-dir / modules_dir')
    elif not os.path.isdir(cfg['modules_dir']):
        errors.append(f"modules_dir not found: {cfg['modules_dir']}\n"
                      f"  Run 'make modules_install INSTALL_MOD_PATH=...' first")

    if not cfg.get('boards'):
        missing.append('--dtb / boards')

    has_any_workflow = any(
        b.get('workflows') for b in cfg.get('boards', [])
    ) or cfg.get('workflows')
    if not has_any_workflow:
        missing.append('--workflow / workflows')

    # Normalise modules_dir and auto-detect versioned subdir
    if cfg.get('modules_dir'):
        mdir = os.path.normpath(cfg['modules_dir'])
        if not os.path.exists(os.path.join(mdir, 'modules.alias')):
            subdirs = sorted(
                d for d in os.listdir(mdir)
                if os.path.isdir(os.path.join(mdir, d))
                and os.path.exists(os.path.join(mdir, d, 'modules.alias'))
            )
            if len(subdirs) == 1:
                mdir = os.path.join(mdir, subdirs[0])
                log.info("Auto-selected kernel release: %s", subdirs[0])
            elif len(subdirs) > 1:
                errors.append(
                    f"Multiple kernel releases in {mdir}: "
                    f"{', '.join(subdirs)}\n"
                    f"  Use --modules-dir {os.path.join(mdir, subdirs[-1])}"
                )
        cfg['modules_dir'] = mdir

    if build:
        busybox = cfg.get('busybox')
        if busybox and not os.path.isfile(busybox):
            errors.append(f"busybox binary not found: {busybox}\n"
                          f"  Pass --busybox /path/to/builddir (will build there) "
                          f"or --busybox /path/to/busybox-binary")

        # Check busybox.config is accessible (catches stale installs)
        from .lib.paths import get_busybox_config
        bbc = get_busybox_config()
        if not os.path.isfile(bbc):
            errors.append(f"busybox.config not found: {bbc}\n"
                          f"  Re-install boardrd: pip install --no-cache-dir .")

    if build:
        errors.extend(_check_required_tools(cfg))

    system_map = cfg.get('system_map')
    if system_map and not os.path.isfile(system_map):
        log.warning("System.map not found: %s — depmod will run without it",
                    system_map)
        cfg['system_map'] = None

    if missing:
        log.error("Missing required config: %s", ', '.join(missing))
        sys.exit(1)
    if errors:
        for e in errors:
            log.error(e)
        sys.exit(1)


# --------------------------------------------------------------------------
# Build pipeline
# --------------------------------------------------------------------------

def run_build(cfg):
    modules_dir = cfg['modules_dir']
    kernelrelease = os.path.basename(modules_dir)
    resolver = ModResolver(modules_dir)
    workflow_dir = cfg.get('workflow_dir', _DEFAULT_WORKFLOW_DIR)

    log.info("Kernel release: %s", kernelrelease)

    # Per-mode module sets and settle times
    mode_to_modules = {}    # mode → set of module names
    mode_to_settle_ms = {}  # mode → int

    # Collect all workflow names across all boards (+ top-level list)
    all_wf_names = set(cfg.get('workflows', []))
    for board in cfg['boards']:
        for wf in board.get('workflows', []):
            all_wf_names.add(wf['name'])

    # --- Workflow anchor modules (generic stack, same for all boards) ---
    for wf_name in sorted(all_wf_names):
        wf = load_workflow(wf_name, workflow_dir)
        anchor_mods = resolver.resolve_anchors(wf.anchors)
        mode_to_modules[wf_name] = anchor_mods
        mode_to_settle_ms[wf_name] = wf.settle_ms
        log.info("Workflow %-8s: %d anchors → %d modules (settle %dms)",
                 wf_name, len(wf.anchors), len(anchor_mods), wf.settle_ms)

    # --- DTB-derived modules, scoped per (board, workflow) ---
    for board in cfg['boards']:
        dtbs = board.get('dtbs', [])
        board_name = board.get('name', dtbs[0] if dtbs else 'unknown')
        board_wf_list = board.get('workflows', [
            {'name': n} for n in cfg.get('workflows', [])
        ])

        log.info("Board: %s", board_name)

        for wf_entry in board_wf_list:
            wf_name = wf_entry['name']
            boot_nodes = wf_entry.get('boot_nodes') or board.get('boot_nodes')

            log.info("  Workflow: %-8s  boot_nodes: %s",
                     wf_name, boot_nodes or '(all enabled nodes)')

            all_compatibles = set()
            for dtb in dtbs:
                compatibles = get_compatibles(dtb, boot_nodes)
                all_compatibles.update(compatibles)

            dt_modules = resolver.resolve_compatibles(all_compatibles)
            log.info("    %d compatible strings → %d DT modules",
                     len(all_compatibles), len(dt_modules))

            if wf_name not in mode_to_modules:
                mode_to_modules[wf_name] = set()
            mode_to_modules[wf_name].update(dt_modules)

            extra = wf_entry.get('extra_modules', [])
            if extra:
                extra_mods = resolver.resolve_anchors(extra)
                log.info("    %d extra_modules → %d modules: %s",
                         len(extra), len(extra_mods), ', '.join(sorted(extra)))
                mode_to_modules[wf_name].update(extra_mods)

    # --- Resolve .ko paths for all modules ---
    all_module_names = set()
    for mods in mode_to_modules.values():
        all_module_names.update(mods)

    ko_paths = []
    missing = []
    for name in sorted(all_module_names):
        path = resolver.ko_path(name)
        if path:
            ko_paths.append(path)
        else:
            missing.append(name)

    if missing:
        builtin = [m for m in missing if resolver.is_builtin(m)]
        truly_missing = [m for m in missing if not resolver.is_builtin(m)]
        if builtin:
            log.debug("Built-in (=y, no .ko needed): %s", ', '.join(builtin))
        if truly_missing:
            log.warning("Not found as module or built-in: %s",
                        ', '.join(truly_missing))

    log.info("Total: %d modules → %d .ko files to bundle",
             len(all_module_names), len(ko_paths))

    # --- Build initrd ---
    output = cfg.get('output', 'initrd.cpio.gz')
    compression = cfg.get('compression', 'gzip')
    busybox_build = cfg.get('busybox_build_dir')
    # If build dir given and binary exists inside it, use it directly
    busybox_bin = cfg.get('busybox')
    if busybox_bin is None and busybox_build:
        candidate = os.path.join(busybox_build, 'busybox')
        if os.path.isfile(candidate):
            busybox_bin = candidate
    busybox_src = cfg.get('busybox_src', _DEFAULT_BUSYBOX_SRC)
    busybox_cfg = cfg.get('busybox_config', _DEFAULT_BUSYBOX_CONFIG)

    # Fail early with a useful message if we can't build or use busybox
    if busybox_bin is None and (busybox_src is None or not os.path.isdir(busybox_src)):
        hint = (f"  --busybox {busybox_build}/busybox not found and no source tree available.\n"
                f"  Options:\n"
                f"    1. Provide a pre-built binary:  --busybox /path/to/busybox\n"
                f"    2. Build from source:           --busybox-src /path/to/busybox-src\n"
                f"       Clone with: git clone https://github.com/vda-linux/busybox_mirror.git "
                f"{busybox_build or '/tmp/busybox-src'}  "
                f"(see https://busybox.net/source.html)")
        if busybox_build:
            log.error(hint)
        else:
            log.error("No busybox binary or source provided.\n%s", hint)
        sys.exit(1)

    board_names = [b.get('name', b['dtbs'][0] if b.get('dtbs') else 'board')
                   for b in cfg['boards']]

    build_initrd(
        modules_dir=modules_dir,
        ko_paths=ko_paths,
        mode_to_modules=mode_to_modules,  # resolver does topo sort
        resolver=resolver,
        mode_to_settle_ms=mode_to_settle_ms,
        boards=board_names,
        workflows=sorted(all_wf_names),
        output_path=output,
        busybox_bin=busybox_bin,
        busybox_src=busybox_src,
        busybox_config=busybox_cfg,
        busybox_build_dir=busybox_build,
        cross_compile=cfg.get('cross_compile'),
        arch=cfg.get('arch'),
        use_llvm=cfg.get('use_llvm', False),
        compression=compression,
        system_map=cfg.get('system_map'),
        musl_root=cfg.get('musl_root'),
    )

    log.info("Done: %s", output)


def run_generate_workflows(cfg, args):
    """Generate workflow YAML suggestions from a sample DTB."""
    if not args.dtb:
        log.error("--dtb required for --generate-workflows")
        sys.exit(1)

    modules_dir = cfg.get('modules_dir')
    if not modules_dir:
        log.error("--modules-dir required for --generate-workflows")
        sys.exit(1)

    resolver = ModResolver(modules_dir)
    workflow_dir = cfg.get('workflow_dir', _DEFAULT_WORKFLOW_DIR)

    for dtb in args.dtb:
        boot_nodes = args.boot_nodes or None
        compatibles = get_compatibles(dtb, boot_nodes)
        dt_modules = resolver.resolve_compatibles(compatibles)
        log.info("DTB %s: %d DT-matched modules", dtb, len(dt_modules))

        for wf_name in (args.workflow or ['mmc', 'nfs', 'ospi', 'usb', 'ufs']):
            suggestions = generate_workflow_yaml(
                wf_name, dt_modules, resolver, workflow_dir)
            log.info("Generated %s.yaml: %d suggested anchors",
                     wf_name, len(suggestions))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog='boardrd',
        description='Board-specific initrd builder for embedded Linux systems.',
    )
    p.add_argument('--config', metavar='FILE',
                   help='YAML config file (CLI args override YAML values)')
    p.add_argument('--modules-dir', metavar='DIR',
                   help='Path to lib/modules/<KERNELRELEASE>/')
    p.add_argument('--kernel-dir', metavar='DIR',
                   help='Kernel source/build tree root. DTB paths in board '
                        'configs are resolved relative to this directory.')
    p.add_argument('--system-map', metavar='FILE',
                   help='Path to System.map from the kernel build. Passed to '
                        'depmod so it knows which symbols the built-in kernel '
                        'provides, preventing spurious dependency failures.')
    p.add_argument('--dtb', metavar='FILE', action='append',
                   help='DTB file (repeatable; replaces YAML boards list)')
    p.add_argument('--boot-nodes', metavar='PATH', nargs='+', dest='boot_nodes',
                   help='DT node path(s) for boot device (space-separated, '
                        'matches boot_nodes: in board YAML)')
    p.add_argument('--workflow', metavar='NAME', action='append',
                   help='Boot workflow (mmc/nfs/ospi/usb/ufs; replaces YAML list)')
    p.add_argument('--workflow-dir', metavar='DIR',
                   help=f'Workflow YAML directory (default: {_DEFAULT_WORKFLOW_DIR})')
    p.add_argument('--output', metavar='FILE',
                   help='Output initrd filename (default: initrd.cpio.gz)')
    p.add_argument('--compression',
                   choices=['gzip', 'xz', 'zstd', 'none'], default=None,
                   help='Compression format (default: gzip)')
    p.add_argument('--busybox', metavar='PATH',
                   help='Pre-built busybox binary (FILE) OR persistent build '
                        'cache directory (DIR). If a directory, boardrd looks '
                        'for DIR/busybox and builds there if not found.')
    p.add_argument('--busybox-src', metavar='DIR',
                   help='busybox source tree for building '
                        f'(default: {_DEFAULT_BUSYBOX_SRC or "none — clone busybox manually"})')
    p.add_argument('--cross-compile', metavar='PREFIX',
                   help='Cross-compiler prefix for GCC (e.g. aarch64-linux-gnu-) '
                        'or target triple for LLVM (same value, e.g. aarch64-linux-gnu-)')
    p.add_argument('--arch', metavar='ARCH',
                   help='Target architecture (e.g. arm64)')
    p.add_argument('--llvm', action='store_true', default=False,
                   help='Use LLVM/clang toolchain instead of GCC. '
                        '--cross-compile value becomes the clang --target= triple.')
    p.add_argument('--musl-root', metavar='DIR',
                   help='Path to a musl cross-compilation toolchain directory '
                        '(e.g. aarch64-linux-musl-cross/). Implies --llvm. '
                        'Download: wget https://musl.cc/aarch64-linux-musl-cross.tgz && '
                        'tar -xf aarch64-linux-musl-cross.tgz. '
                        'Sets CC=clang --target=aarch64-linux-musl --sysroot=DIR/aarch64-linux-musl '
                        'and LDFLAGS=-fuse-ld=lld.')
    p.add_argument('--generate-workflows', action='store_true',
                   help='Generate workflow YAML suggestions from DTB analysis')
    p.add_argument('--list-modules', action='store_true',
                   help='Print module list and exit (dry run)')
    p.add_argument('-v', '--verbose', action='store_true',
                   help='Enable verbose logging')
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s: %(message)s',
    )

    cfg = load_config(args)

    if args.generate_workflows:
        run_generate_workflows(cfg, args)
        return

    validate_config(cfg, build=not args.list_modules)

    if args.list_modules:
        # Dry run: print module list without building
        _dry_run_list(cfg)
        return

    run_build(cfg)


def _dry_run_list(cfg):
    modules_dir = cfg['modules_dir']
    resolver = ModResolver(modules_dir)
    workflow_dir = cfg.get('workflow_dir', _DEFAULT_WORKFLOW_DIR)

    all_modules = set()

    all_wf_names = set(cfg.get('workflows', []))
    for board in cfg['boards']:
        for wf in board.get('workflows', []):
            all_wf_names.add(wf['name'])

    for wf_name in all_wf_names:
        wf = load_workflow(wf_name, workflow_dir)
        all_modules.update(resolver.resolve_anchors(wf.anchors))

    for board in cfg['boards']:
        dtbs = board.get('dtbs', [])
        board_wf_list = board.get('workflows', [
            {'name': n} for n in cfg.get('workflows', [])
        ])
        for wf_entry in board_wf_list:
            boot_nodes = wf_entry.get('boot_nodes') or board.get('boot_nodes')
            for dtb in dtbs:
                compatibles = get_compatibles(dtb, boot_nodes)
                all_modules.update(resolver.resolve_compatibles(compatibles))
            extra = wf_entry.get('extra_modules', [])
            if extra:
                all_modules.update(resolver.resolve_anchors(extra))

    for name in sorted(all_modules):
        path = resolver.ko_path(name)
        if path:
            label = path
        elif resolver.is_builtin(name):
            label = '(built-in =y)'
        else:
            label = '(NOT FOUND)'
        print(f"{name:40s}  {label}")
    print(f"\nTotal: {len(all_modules)} modules")


if __name__ == '__main__':
    main()
