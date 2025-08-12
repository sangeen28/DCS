import argparse
from tqdm import trange
from torch.utils.data import DataLoader

from dcs.data.mnist import load_mnist, MNISTNet
from dcs.data.data_loader import partition_dirichlet, make_client_loaders, device
from dcs.fl.federated_learning import FederatedClient, FederatedServer, FLConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--clients', type=int, default=10)
    ap.add_argument('--select', type=int, default=5)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=1)
    args = ap.parse_args()

    dev = device()

    train_ds, test_ds = load_mnist()
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    idxs = partition_dirichlet(len(train_ds), args.clients, alpha=0.5)
    loaders = make_client_loaders(train_ds, idxs, batch_size=args.batch)

    base_model = MNISTNet()
    clients = {
        cid: FederatedClient(cid, base_model, loaders[cid], device=str(dev))
        for cid in loaders.keys()
    }
    server = FederatedServer(base_model, clients, FLConfig(
        rounds=args.rounds, local_epochs=args.epochs, clients_per_round=args.select, device=str(dev)
    ))

    _, acc0 = server.evaluate(test_loader)
    print(f"Round -1: acc={acc0:.4f}")

    for r in trange(args.rounds, desc="FL Rounds"):
        out = server.run_round(r, test_loader)
        print(out)

    print("History keys:", list(server.history.keys()))
    print("Final acc:", server.history['acc'][-1])

if __name__ == '__main__':
    main()
