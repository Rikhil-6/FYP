"""Tiny end-to-end runtime check. Synthetic inputs; no CLEAR-100 download."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from evaluate import evaluate
from metrics import binary_metrics, topk_map_agreement
from model import TemporalViT
from run_store import RunStore, manual_snapshot
from siamese import ViTTwinRouter


class TinyImages(Dataset):
    def __init__(self, year):
        self.year = year

    def __len__(self):
        return 2

    def __getitem__(self, index):
        image = torch.rand((3, 224, 224), generator=torch.Generator().manual_seed(10 + index))
        return image, torch.tensor(index), torch.tensor([0, self.year - 2002, index, 1])


def one_step(model, device):
    model.train()
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.backbone.encoder.layer[-1].parameters():
        parameter.requires_grad = True
    image, label, _ = TinyImages(2012)[0]
    logits, saliency, inverse = model(image[None].to(device), return_counterfactual=True)
    loss = F.cross_entropy(logits, label[None].to(device))
    if inverse is not None:
        loss = loss + 0.1 * F.relu(0.02 - (logits[0, label] - inverse[0, label]))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-5)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert logits.shape == (1, 100)
    assert saliency.shape == (1, 256)
    assert torch.isfinite(loss)
    return round(loss.detach().item(), 4)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    loaders = {name: DataLoader(TinyImages(year), batch_size=1)
               for name, year in (("id_val", 2011), ("test", 2014))}
    losses = {}
    xai = None
    for method in ("baseline", "xai"):
        model = TemporalViT(method).to(device)
        losses[method] = one_step(model, device)
        report, records = evaluate(model, loaders, year_column=1, device=device,
                                   splits=("id_val", "test"))
        assert report["test"]["all"]["count"] == 2
        assert len(records) == 4
        if method == "xai":
            xai = model
        del model
    newer = TemporalViT("xai").to(device)
    newer.load_state_dict(xai.state_dict())
    with torch.no_grad():
        newer.saliency_head[-1].bias.add_(0.1)
    router = ViTTwinRouter(xai.eval(), newer.eval())
    image, _, _ = TinyImages(2014)[0]
    comparison = router.compare(image[None].to(device))
    prediction, flag, route = router.route(comparison, {"global": 0.0})
    forced = {"old_logits": torch.tensor([[4.0, 0.0]], device=device),
              "new_logits": torch.tensor([[0.0, 4.0]], device=device),
              "score": torch.tensor([0.5], device=device),
              "old_evidence": torch.tensor([0.1], device=device),
              "new_evidence": torch.tensor([0.3], device=device)}
    forced_prediction, forced_flag, forced_route = router.route(forced, {"global": 0.2})
    assert bool(forced_flag[0]) and bool(forced_route[0]) and int(forced_prediction[0]) == 1
    forced["new_evidence"] = torch.tensor([0.05], device=device)
    blocked_prediction, _, blocked_route = router.route(forced, {"global": 0.2})
    assert not bool(blocked_route[0]) and int(blocked_prediction[0]) == 0
    overlap = topk_map_agreement(comparison["old_map"][0].detach().cpu().numpy(),
                                 comparison["new_map"][0].detach().cpu().numpy())
    trigger = binary_metrics([False, True], [False, bool(flag[0])])
    assert prediction.shape == (1,)
    assert 0 <= overlap["iou"] <= 1
    store = RunStore(Path("runs"), "smoke")
    store.log({"event": "smoke", "losses": losses})
    store.save_model(newer, "best", 1, "synthetic_macro_f1", report["test"]["all"]["macro_f1"])
    snapshot = manual_snapshot(store.path, "verified")
    assert snapshot.exists()
    print(json.dumps({"status": "passed", "device": device, "losses": losses,
                      "metrics": {"macro_f1": report["test"]["all"]["macro_f1"],
                                  "macro_iou": report["test"]["all"]["macro_iou"]},
                      "saliency_agreement_iou": overlap["iou"],
                      "trigger_f1": trigger["f1"], "routed": bool(route[0]),
                      "run": str(store.path)}, indent=2))


if __name__ == "__main__":
    main()
