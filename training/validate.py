import torch

def validate(model, loss_fn1, loss_fn2, device, val_loader):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_rev, y_risk, y_churn = y_batch[:,0], y_batch[:,1], y_batch[:,2]

            pred_rev, pred_risk, pred_churn = model(X_batch)

            loss_rev = loss_fn1(pred_rev, y_rev)
            loss_risk = loss_fn1(pred_risk, y_risk)
            loss_churn = loss_fn2(pred_churn, y_churn)

            loss = loss_rev + loss_risk + loss_churn
            val_loss += loss.item()

    return val_loss / len(val_loader)
