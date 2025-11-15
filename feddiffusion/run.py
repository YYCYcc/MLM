# run.py
import os, multiprocessing as mp, time, sys
from utils.common import load_cfg, set_seed
from fed.server import FedServer
from fed.client import Client

# —— 子进程目标函数 —— #
def server_proc(cfg, queues):
    set_seed(cfg["seed"] + 1000)
    srv = FedServer(cfg, queues)
    srv.run()

def client_proc(cid, cfg, queues, root_dir):  # ←← 这里需要 root_dir
    set_seed(cfg["seed"] + cid)
    cli = Client(cid, cfg, queues, root_dir)  # ←← 传 root_dir
    cli.run()

def launch():
    cfg = load_cfg("/home/yyc/data/feddiffusion/cfg/base.yaml")
    set_seed(cfg["seed"])

    # 建议使用 spawn，避免 CUDA 在 fork 后崩溃
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # 统一项目根目录（包含 data/clients_* 等）
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # ←← root_dir 计算
    print(f"[Launcher] ROOT_DIR={ROOT_DIR}", flush=True)

    mgr = mp.Manager()
    queues = {"to_srv": mgr.Queue()}
    for i in range(cfg["data"]["n_clients"]):
        queues[f"to_cli_{i}"] = mgr.Queue()

    procs = []

    # Server
    ps = mp.Process(target=server_proc, args=(cfg, queues), daemon=False)
    ps.start(); procs.append(ps)

    # Clients （注意把 ROOT_DIR 也传进去）
    for cid in range(cfg["data"]["n_clients"]):
        pc = mp.Process(target=client_proc, args=(cid, cfg, queues, ROOT_DIR), daemon=False)
        pc.start(); procs.append(pc)

    # 主进程等待
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("\n[Launcher] KeyboardInterrupt, terminating ...", flush=True)
        for p in procs:
            if p.is_alive():
                p.terminate()

if __name__ == "__main__":
    launch()
