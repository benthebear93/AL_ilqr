from pathlib import Path
import numpy as np


def save_outputs(x_sol, u_sol, base_name="car"):
    """
    Save x_sol and u_sol under <project_root>/data as .npy and .npz.
    """
    here = Path(__file__).resolve().parent
    data_dir = (here.parent / "data").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    x_np = np.array(x_sol)
    u_np = np.array(u_sol)

    x_path = data_dir / "x_sol.npy"
    u_path = data_dir / "u_sol.npy"
    np.save(x_path, x_np)
    np.save(u_path, u_np)

    np.savez(data_dir / f"{base_name}_solution.npz", x_sol=x_np, u_sol=u_np)

    print(f"Saved x to: {x_path}")
    print(f"Saved u to: {u_path}")
    print(f"Saved bundle: {data_dir / f'{base_name}_solution.npz'}")
    return x_path, u_path


def load_x_sol_from(data_dir: Path) -> np.ndarray:
    npz_path = data_dir / "car_solution.npz"
    if npz_path.exists():
        with np.load(npz_path) as Z:
            if "x_sol" not in Z:
                raise KeyError(f"'x_sol' key not found in {npz_path}")
            return Z["x_sol"]

    npy_path = data_dir / "x_sol.npy"
    if npy_path.exists():
        return np.load(npy_path)

    raise FileNotFoundError(
        "Could not find x_sol.\n" f"Tried:\n  - {npz_path}\n  - {npy_path}"
    )
