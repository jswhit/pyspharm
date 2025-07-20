import subprocess
import os
import sys
import shutil
from pathlib import Path
import datetime


def print_header(message):
    print("\n" + "="*70)
    print(f"===== {message}")
    print("="*70)


def run_command(command, cwd=None, env=None, check=True, capture_output=True, shell=False):
    print(
        f"\nExecuting: {' '.join(command) if isinstance(command, list) else command}")
    if cwd:
        print(f"  In directory: {cwd}")
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    try:
        if capture_output:
            process = subprocess.run(command, cwd=cwd, env=process_env,
                                     capture_output=True, text=True, check=check,
                                     encoding='utf-8', errors='replace', shell=shell)
            if process.stdout:
                print("STDOUT:\n", process.stdout)
            if process.stderr:
                print("STDERR/Progress:\n", process.stderr)
        else:
            process = subprocess.run(command, cwd=cwd, env=process_env,
                                     check=check, shell=shell)
        print(
            f"Command '{' '.join(command) if isinstance(command, list) else command}' executed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(
            f"Command execution error: {' '.join(command) if isinstance(command, list) else command}")
        print("Return code:", e.returncode)
        if hasattr(e, 'stdout') and e.stdout:
            print("STDOUT:\n", e.stdout)
        if hasattr(e, 'stderr') and e.stderr:
            print("STDERR:\n", e.stderr)
        return False
    except FileNotFoundError:
        print(
            f"Error: Command '{command[0] if isinstance(command, list) else command.split()[0]}' not found.")
        print(f"   Please ensure it is installed and in your PATH, or the Conda environment is properly activated.")
        return False
    except Exception as e:
        print(f"Unexpected error while executing command: {e}")
        return False


def attempt_install_build_tools():
    print_header("Attempting to install build tools with Conda")
    conda_executable = shutil.which("conda")
    if not conda_executable:
        print("[X] Conda executable not found. Cannot install tools automatically.")
        print("    Please run manually in your activated Conda environment:")
        print("    conda install -c conda-forge m2w64-toolchain meson ninja")
        return False
    print(f"Using Conda executable: {conda_executable}")
    install_command = [
        conda_executable, "install", "-y",
        "-c", "conda-forge",
        "m2w64-toolchain", "meson", "ninja"
    ]
    print("Attempting to run the following command to install tools (this may take some time):")
    print(f"  {' '.join(install_command)}")
    if run_command(install_command, capture_output=False):
        print("[OK] Conda install command finished. Please check output for success.")
        return True
    else:
        print("[X] Conda install command failed. Please check the error above.")
        return False


def check_tools_and_environment(auto_install=False):
    print_header("Checking required tools and environment")
    tools = ["gcc", "gfortran", "meson", "ninja"]
    missing_tools = []
    for tool in tools:
        tool_path = shutil.which(tool)
        if tool_path:
            print(f"[OK] {tool} found (path: {tool_path}).")
        else:
            print(f"[X] {tool} not found.")
            missing_tools.append(tool)
    if missing_tools:
        print(
            f"\nWarning: The following build tools are missing: {', '.join(missing_tools)}")
        if auto_install:
            user_consent = input(
                "The script can try to install them with Conda. Continue? (y/n): ").strip().lower()
            if user_consent == 'y':
                if attempt_install_build_tools():
                    print("\nRechecking tools...")
                    all_found_after_install = True
                    for tool_check_again in tools:
                        if shutil.which(tool_check_again):
                            print(f"[OK] {tool_check_again} found.")
                        else:
                            print(f"[X] {tool_check_again} still not found.")
                            all_found_after_install = False
                    if not all_found_after_install:
                        print(
                            "One or more tools are still missing after attempted installation. Please check Conda installation manually.")
                        return False
                    print(
                        "[OK] All initially checked tools found after attempted installation.")
                else:
                    return False
            else:
                print(
                    "User chose not to auto-install. Please install missing tools manually and retry.")
                print("  conda install -c conda-forge m2w64-toolchain meson ninja")
                return False
        else:
            print(
                "Please run the following command in your activated Conda environment to install them:")
            print("  conda install -c conda-forge m2w64-toolchain meson ninja")
            return False
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        print("\n[!] Warning: CONDA_PREFIX environment variable is not set.")
        print(
            f"    Current Python interpreter (sys.executable): {sys.executable}")
        print(f"    Current install target (sys.prefix): {sys.prefix}")
        print("    It is strongly recommended to run this script in the activated target Conda environment for correct installation and environment detection.")
    else:
        print(f"[OK] Script appears to be running in a Conda environment.")
        print(f"    Conda environment path (CONDA_PREFIX): {conda_prefix}")
        if Path(sys.prefix).resolve() != Path(conda_prefix).resolve():
            print(
                f"    [!] Note: Python install path (sys.prefix) '{Path(sys.prefix).resolve()}'")
            print(
                f"        is different from Conda environment path (CONDA_PREFIX) '{Path(conda_prefix).resolve()}'.")
            print(
                f"        pyspharm will be installed to the location specified by sys.prefix ('{Path(sys.prefix).resolve()}').")
            print(f"        Please ensure this is your intended Conda environment.")
        else:
            print(
                f"    Install target (sys.prefix) matches Conda environment: {Path(sys.prefix).resolve()}")
    return True


def main():
    current_utc_time_str = "2025-05-13 15:16:54"
    current_user_login = "SQYQianYe"
    print_header(
        f"pyspharm automated build and install script (native CPU optimization)")
    print(f"Script author: {current_user_login} (with Gemini assistance)")
    print(f"Current user: {current_user_login}")
    print(f"Script start time (UTC): {current_utc_time_str}")
    if not check_tools_and_environment(auto_install=True):
        print("\nRequired tools or environment check failed. Script aborted.")
        sys.exit(1)
    while True:
        pyspharm_source_dir_str = input(
            "\nEnter the full path to the extracted 'pyspharm-master' (or similar) source code folder: ").strip()
        pyspharm_source_dir = Path(pyspharm_source_dir_str).resolve()
        if pyspharm_source_dir.is_dir() and (pyspharm_source_dir / "meson.build").exists():
            print(f"Using pyspharm source directory: {pyspharm_source_dir}")
            break
        else:
            print(
                f"Error: Path '{pyspharm_source_dir_str}' is not a valid pyspharm source directory (meson.build file not found or path invalid). Please try again.")
    install_target_path = Path(sys.prefix).resolve()
    print(f"Will install to target path (sys.prefix): {install_target_path}")
    build_dir_name = "build_pyspharm_native_optimized"
    print(f"Build directory will be: {pyspharm_source_dir / build_dir_name}")
    compile_env_vars = {
        "CC": shutil.which("gcc") or "gcc",
        "FC": shutil.which("gfortran") or "gfortran",
        "FFLAGS": "-O3 -march=native -funroll-loops",
        "CFLAGS": "-O3 -march=native"
    }
    print(f"\nSetting environment variables for compilation (including -march=native):")
    for key, value in compile_env_vars.items():
        print(f"  {key}={value}")
    print_header(
        f"Step 1: Configure pyspharm with Meson (build directory: {build_dir_name})")
    meson_executable = shutil.which("meson") or "meson"
    meson_setup_cmd = [
        meson_executable,
        "setup",
        "--wipe",
        "--buildtype=release",
        "--prefix", str(install_target_path),
        build_dir_name
    ]
    if not run_command(meson_setup_cmd, cwd=pyspharm_source_dir, env=compile_env_vars):
        print("\nMeson setup failed. Script aborted.")
        sys.exit(1)
    print_header("Step 2: Build pyspharm with Meson/Ninja")
    meson_compile_cmd = [meson_executable,
                         "compile", "-C", build_dir_name, "-v"]
    if not run_command(meson_compile_cmd, cwd=pyspharm_source_dir, env=compile_env_vars, capture_output=False):
        print("\nMeson compile failed. Script aborted.")
        sys.exit(1)
    print_header("Step 3: Install pyspharm with Meson")
    meson_install_cmd = [meson_executable, "install", "-C", build_dir_name]
    if not run_command(meson_install_cmd, cwd=pyspharm_source_dir, env=compile_env_vars):
        print("\nMeson install failed. Script aborted.")
        sys.exit(1)
    print_header("Native CPU-optimized installation attempt complete")
    print(f"\npyspharm build and installation process (with -march=native optimization) has been executed.")
    if sys.platform == "win32":
        spharm_install_subpath = Path("Lib") / "site-packages" / "spharm"
    else:
        py_version_short = f"python{sys.version_info.major}.{sys.version_info.minor}"
        spharm_install_subpath = Path(
            "lib") / py_version_short / "site-packages" / "spharm"
    expected_spharm_path = install_target_path / spharm_install_subpath
    print(
        f"It should be installed in your target environment, for example under '{expected_spharm_path}'.")
    print("\nPlease verify in a new Python interpreter (with the target environment activated) using:")
    print("  import spharm")
    print("  print(spharm.__version__)")
    print("  print(spharm.__file__)")
    print("  and run your performance test code to compare results.")
    print("\nReminder: Libraries compiled with -march=native may not run on computers with different CPUs.")
    print_header("Next steps: Install windspharm (if needed)")
    print("If you need the windspharm library (which depends on pyspharm):")
    print("Since pyspharm was manually built and installed with Meson, the standard `pip install windspharm`")
    print("may fail because it cannot find pyspharm (or tries to download and build an old version from PyPI).")
    print("\nThe recommended way to install windspharm is to use the `--no-deps` option to prevent pip from reinstalling dependencies:")
    print("In your activated Conda environment, run:")
    print("  pip install --no-deps windspharm")
    print("\nThis will only install the windspharm package itself, which should be able to find the spharm module we just built.")
    print("After installation, you can try `import windspharm` in Python to verify.")


if __name__ == "__main__":
    main()
