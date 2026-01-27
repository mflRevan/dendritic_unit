
import torch
from dendritic_unit.model.transformer import DendriticTransformer
from dendritic_unit.core.layers import DendriticLayerSiLU_Template, DendriticLayerSiLU_FFN

def test_sanity():
    print("Initializing DendriticTransformer...")
    model = DendriticTransformer(
        vocab_size=1000,
        seq_length=128,
        dim=256,
        num_heads=4,
        num_layers=2,
        use_dendritic=True,
        use_dendritic_ffn=False # Test GLU variant first
    ).cuda()
    
    print("Model initialized.")
    x = torch.randint(0, 1000, (2, 128)).cuda()
    
    print("Forward pass...")
    out = model(x)
    print(f"Output shape: {out.shape}")
    
    print("Backward pass...")
    loss = out.mean()
    loss.backward()
    print("Backward pass successful.")

    # Test FFN variant
    print("\nTesting FFN Variant...")
    model_ffn = DendriticTransformer(
        vocab_size=1000,
        seq_length=128,
        dim=256,
        num_heads=4,
        num_layers=2,
        use_dendritic=True,
        use_dendritic_ffn=True # Test FFN variant
    ).cuda()
    out = model_ffn(x)
    out.mean().backward()
    print("FFN Variant successful.")

if __name__ == "__main__":
    test_sanity()
