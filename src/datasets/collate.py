import torch

from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]) -> dict[str, torch.Tensor | list]:
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    result = {}

    for key in ("audio", "text_encoded"):
        result[key] = pad_sequence([item[key].squeeze(0) for item in dataset_items], batch_first=True)

    for key in ("text", "audio_path"):
        result[key] = [item[key] for item in dataset_items]

    for key, shape_idx in (("text_encoded", 1), ("spectrogram", 2)):
        result[f"{key}_length"] = torch.tensor([item[key].shape[shape_idx] for item in dataset_items])

    result["spectrogram"] = pad_sequence([item["spectrogram"].squeeze(0).permute(1, 0) for item in dataset_items], batch_first=True).permute(0, 2, 1)

    return result
