import torch
import torch.nn as nn
from torch import Tensor


class BlockRNN(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_state_dim: int,
            dropout_p: float,
    ) -> None:
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_state_dim,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=True,
        )

        self.relu = nn.ReLU()

    def forward(self, input: Tensor, input_lengths: Tensor) -> Tensor:
        length = input.size(0)

        input = self.relu(self.batch_norm(input.transpose(1, 2))) 

        outputs = nn.utils.rnn.pack_padded_sequence(input.transpose(1, 2), input_lengths.cpu(), enforce_sorted=False)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(self.rnn(outputs)[0], total_length=length)

        return outputs


class ConvBlock(nn.Module):
    def __init__(self, sequential: nn.Sequential) -> None:
        super().__init__()
        self.sequential = sequential

    def forward(self, input: Tensor, seq_lengths: Tensor) -> tuple[Tensor, Tensor]:
        output = input

        for module in self.sequential:
            output = module(output)
            mask = self._create_mask(output, seq_lengths)

            if output.is_cuda:
                mask = mask.cuda()
            if isinstance(module, nn.Conv2d):
                seq_lengths = ((seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1).float() / float(module.stride[1])).int() + 1

            output = torch.where(mask, torch.tensor(0, dtype=output.dtype, device=output.device), output)

        return output, seq_lengths
    
    def _create_mask(self, input: Tensor, seq_lengths: Tensor) -> Tensor:
        batch_size, _, _, _ = input.size()
        mask = torch.zeros_like(input, dtype=bool)

        for idx in range(batch_size):
            length = seq_lengths[idx].item()
            if input.size(2) > length:
                mask[idx, :, :, length:] = 1

        return mask


class DeepSpeech2(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        num_rnn_layers: int,
        rnn_hidden_dim: int,
        in_channels: int,
        out_channels: int,
        dropout_p: float,
    ) -> None:
        super().__init__()
        self.conv = ConvBlock(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Hardtanh(0, 20),
                nn.Conv2d(out_channels, out_channels, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5), bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Hardtanh(0, 20),
            )
        )

        rnn_output_size = rnn_hidden_dim * 2
        first_rnn_layer = [BlockRNN(
            input_size=1024,
            hidden_state_dim=rnn_hidden_dim,
            dropout_p=dropout_p,
        )]
        other_rnn_layers = [
            BlockRNN(
                input_size=rnn_output_size,
                hidden_state_dim=rnn_hidden_dim,
                dropout_p=dropout_p,
            )
            for _ in range(num_rnn_layers - 1)
        ]

        first_rnn_layer.extend(other_rnn_layers)
        self.rnn_layers = nn.ModuleList(first_rnn_layer)

        self.fc = nn.Sequential(
            nn.LayerNorm(rnn_output_size),
            nn.Linear(rnn_output_size, n_tokens, bias=False),
        )

    def forward(self, spectrogram: Tensor, spectrogram_length: Tensor,  **batch) -> dict[str, Tensor]:
        spectrogram = spectrogram.unsqueeze(1)
        outputs, output_lengths = self.conv(spectrogram, spectrogram_length)
        batch_size, _, _, seq_lengths = outputs.size()
        outputs = outputs.transpose(1, 3)
        outputs = outputs.reshape(batch_size, seq_lengths, -1)
        outputs = outputs.permute(1, 0, 2).contiguous()

        for rnn_layer in self.rnn_layers:
            outputs = rnn_layer(outputs, output_lengths)

        outputs = self.fc(outputs.transpose(0, 1)).log_softmax(dim=-1)

        return {"log_probs": outputs, "log_probs_length": output_lengths}
