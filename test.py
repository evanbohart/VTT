import torch

def WER(
    predicted,
    targets,
):
    assert predicted.ndim == 2
    assert targets.ndim == 2

    assert predicted.shape == targets.shape

    total_wer = 0.0

    for i in range(predicted.size(0)):
        predicted_unpadded = predicted[i][predicted[i] != 0]
        targets_unpadded = targets[i][targets[i] != 0]

        m = len(predicted_unpadded)
        n = len(targets_unpadded)

        if (n == 0) and (m > 0):
            return 1.0
        elif (n == 0):
            return 0.0

        dp = torch.zeros(m+1, n+1, dtype=torch.long, device=device)

        for j in range(m+1):
            dp[j, 0] = j
        for j in range(n+1):
            dp[0, j] = j

        for j in range(1, m+1):
            for k in range(1, n+1):
                if predicted_unpadded[j-1] == targets_unpadded[k-1]:
                    dp[j, k] = dp[j-1][k-1]
                else:
                    dp[j, k] = 1 + min(
                        dp[j-1][k],
                        dp[j][k-1],
                        dp[j-1][k-1]
                    )
        total_wer += dp[m][n] / n

    return total_wer / predicted.size(0)


def test(
    accum_steps,
    loader,
    model,
    device
):
    batches = len(loader)
    total_wer = 0.0

    for batch_num, batch in enumerate(loader, 1):
        (
            encoder_x_batch,
            _,
            src_padding_mask_batch,
            _,
            targets_batch
        ) = batch

        tgt_causal_mask = torch.triu(
            torch.ones(
                model.decoder_seq_len, model.decoder_seq_len, dtype=torch.bool, device=device
            ),
            diagonal=1
        )

        batch_size = targets_batch.size(0)
        decoder_seq_len = targets_batch.size(1)

        decoder_x_batch = torch.zeros(
            batch_size, decoder_seq_len, dtype=torch.long, device=device
        )

        decoder_x_batch[:, 0] = 1

        tgt_padding_mask_batch = (torch.arange(decoder_seq_len) >= 1).to(device)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        ffs = 0

        while not finished.all():
            with torch.no_grad():
                logits = model(
                    encoder_x_batch,
                    decoder_x_batch,
                    src_padding_mask_batch,
                    tgt_padding_mask_batch,
                    tgt_causal_mask
                )

                predicted = torch.argmax(logits, dim=1)

                decoder_x_batch[~finished, ffs+1] = predicted[~finished]
                tgt_padding_mask_batch[~finished, ffs+1] = False

                finished |= (predicted == 2)

                ffs += 1

        batch_wer = WER(encoder_x_batch, targets)

        print(f"Batch {batch_num} | WER {batch_wer:.4f}")

        total_wer += batch_wer

    print(f"Average WER {(total_wer/batches):.4f}")
