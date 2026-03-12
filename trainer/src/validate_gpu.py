"""
Passo 2: Validação de GPU — ROCm + PyTorch + SpikingJelly
Executar dentro do container snn-trainer:
    docker compose --profile full run --rm snn-trainer python -m src.validate_gpu
Ou directamente no host com o ambiente conda/uv correcto.
"""
import sys


def main():
    print("=== Validação GPU — SNN Organismo ===\n")

    # 1. PyTorch + ROCm
    try:
        import torch
        print(f"PyTorch versão : {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"CUDA disponível: {cuda_ok}")
        if cuda_ok:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                mem  = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {name} ({mem:.1f} GB)")
            device = torch.device("cuda")
            t = torch.randn(1000, 1000, device=device)
            result = (t @ t).mean().item()
            print(f"  Teste tensor GPU: OK (mean={result:.4f})")
        else:
            print("  AVISO: GPU não detectada — a rodar em CPU")
            device = torch.device("cpu")
        print()
    except ImportError as e:
        print(f"ERRO — PyTorch não instalado: {e}")
        sys.exit(1)

    # 2. SpikingJelly
    try:
        import spikingjelly
        from spikingjelly.activation_based import neuron, functional, layer
        sj_ver = getattr(spikingjelly, "__version__", "0.0.0.0.14")
        print(f"SpikingJelly versão: {sj_ver}")

        # Forward pass básico numa LIF
        lif = neuron.LIFNode(tau=20.0).to(device)
        x = torch.rand(8, 18, device=device)           # T=8, 18 inputs
        spikes = lif(x)
        functional.reset_net(lif)
        print(f"  LIF forward pass: OK (shape={tuple(spikes.shape)}, "
              f"spikes={spikes.sum().item():.0f})")
        print()
    except ImportError as e:
        print(f"ERRO — SpikingJelly não instalado: {e}")
        sys.exit(1)

    # 3. NumPy
    try:
        import numpy as np
        world = np.zeros((200, 200), dtype=np.uint8)
        world[10, 10] = 1
        print(f"NumPy versão    : {np.__version__}")
        print(f"  Mundo 200×200 : OK (sum={world.sum()})")
        print()
    except ImportError as e:
        print(f"ERRO — NumPy não instalado: {e}")
        sys.exit(1)

    # Resumo
    status = "GPU (ROCm)" if torch.cuda.is_available() else "CPU"
    print(f"=== Resultado: TUDO OK — a correr em {status} ===")


if __name__ == "__main__":
    main()
