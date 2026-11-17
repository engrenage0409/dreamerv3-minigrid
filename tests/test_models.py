import torch
from dreamer import WorldModel


def test_world_model_shapes():
    model = WorldModel(action_dim=7)
    images = torch.rand(2, 4, 3, 7, 7)
    actions = torch.nn.functional.one_hot(torch.randint(0, 7, (2, 4)), 7).float()
    out = model(images, actions)
    assert out["image_logits"].shape == images.shape
    assert out["reward"].shape == (2, 4)
    assert out["continue_logits"].shape == (2, 4)
