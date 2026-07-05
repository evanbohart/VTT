import torch

def train(
    epochs,
    accum_steps,
    loader,
    model,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device
):
    tgt_causal_mask = torch.triu(
        torch.ones(
            model.decoder_seq_len, model.decoder_seq_len, dtype=torch.bool, device=device
        ),
        diagonal=1
    )

    batches = len(loader)
    assert batches % accum_steps == 0

    for i in range(epochs):
        optimizer.zero_grad()

        running_loss = 0.0

        for batch_num, batch in enumerate(loader, 1):
            (
                encoder_x_batch,
                decoder_x_batch,
                src_padding_mask_batch,
                tgt_padding_mask_batch,
                targets_batch
            ) = batch

            with torch.cuda.amp.autocast():
                logits = model(
                    encoder_x_batch.to(device),
                    decoder_x_batch.to(device),
                    src_padding_mask_batch.to(device),
                    tgt_padding_mask_batch.to(device),
                    tgt_causal_mask
                )

                loss = criterion(
                    logits.view(-1, model.vocab_size),
                    targets_batch.to(device).view(-1)
                )

            running_loss += loss.item()

            scaler.scale(loss/accum_steps).backward()

            if batch_num % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()

                scheduler.step()

                print(
                    f"Epoch: {i+1} | Batch: {batch_num} | "
                    f"Loss: {running_loss:.4f}"
                )

    torch.save(model.state_dict(), "model.pt")
