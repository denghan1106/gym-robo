import os
import torch

def save_model(actor, critic, action_dim, obs_dim, directory="checkpoint", filename="ppo"):
    os.makedirs(directory, exist_ok=True)
    checkpoint = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "action_dim": action_dim,
        "obs_dim": obs_dim,
    }
    torch.save(checkpoint, os.path.join(directory, f"{filename}_all.pth"))
    print(f"Model (with meta info) saved to {directory}/")

def load_model(actor, critic, directory="checkpoint", filename="ppo"):
    checkpoint_path = os.path.join(directory, f"{filename}_all.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # 检查action_dim
    model_action_dim = actor.mean.out_features if hasattr(actor, 'mean') else None
    ckpt_action_dim = checkpoint.get("action_dim", None)
    if model_action_dim is not None and ckpt_action_dim is not None and model_action_dim != ckpt_action_dim:
        raise ValueError(f"action_dim不一致: checkpoint={ckpt_action_dim} 当前模型={model_action_dim}")
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    print(f"Model loaded from {checkpoint_path} (action_dim={ckpt_action_dim}, obs_dim={checkpoint.get('obs_dim')})")
