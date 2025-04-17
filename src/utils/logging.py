def log_metrics(wandb, epoch, train_loss, train_accuracy, val_accuracy):
    if wandb:
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy
        })
