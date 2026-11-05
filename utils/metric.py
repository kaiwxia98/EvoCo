import numpy as np
import torch

def MAE_torch(prediction: torch.Tensor, target: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        mask = ~torch.isclose(
            target,
            torch.tensor(null_val, device=target.device, dtype=target.dtype).expand_as(target),
            atol=5e-5, rtol=0.0,
        )
    mask = mask.float()
    mask = mask / torch.mean(mask).clamp_min(1e-6)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(prediction - target) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')

def masked_mape_np(y_true, y_pred, null_val=np.nan, den_floor=0.0, extra_mask=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        if extra_mask is not None:
            mask = mask * extra_mask.astype("float32")
        m = float(mask.mean())
        if m <= 0.0:
            return 0.0
        mask /= m
        denom = np.abs(y_true)
        if den_floor and den_floor > 0.0:
            denom = np.maximum(denom, den_floor)
        mape = np.abs((y_pred - y_true) / denom)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def restore_eval_cms(ground_truth, prediction, args):

    gt = np.asarray(ground_truth, dtype=np.float64)
    pr = np.asarray(prediction, dtype=np.float64)
    extra = None
    if bool(getattr(args, "eval_expm1", False)):
        gt = np.expm1(np.clip(gt, -1.0, 20.0))
        pr = np.expm1(np.clip(pr, -1.0, 20.0))
        pr = np.maximum(pr, 0.0)
        min_cms = float(getattr(args, "mape_min_cms", 1.0) or 0.0)
        extra = gt >= min_cms
        med = getattr(args, "station_median_cms", None)
        if med is not None and gt.ndim >= 2:
            med = np.asarray(med, dtype=np.float64).reshape(-1)
            n = gt.shape[1]
            if med.size >= n:
                thr = np.maximum(min_cms, med[:n]).reshape(1, n, *([1] * (gt.ndim - 2)))
                extra = gt >= thr
        if not getattr(args, "_logged_eval_expm1", False):
            vars(args)["_logged_eval_expm1"] = True
            logger = getattr(args, "logger", None)
            if logger is not None:
                logger.info("[eval] expm1 restore to cms; MAPE on Q>=max(%.3g, station median)", min_cms)
    scale = float(getattr(args, "eval_y_scale", 1.0) or 1.0)
    if abs(scale - 1.0) > 1e-12:
        gt = gt * scale
        pr = pr * scale
        if not getattr(args, "_logged_eval_y_scale", False):
            vars(args)["_logged_eval_y_scale"] = True
            logger = getattr(args, "logger", None)
            if logger is not None:
                logger.info("[eval] y scale x%g (MAE/RMSE in percent if 100)", scale)
    return gt, pr, extra

def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))

def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))

def cal_metric_stbp(ground_truth, prediction, args):

    args.logger.info("[*] year {}, testing".format(args.year))
    mape_floor = float(getattr(args, "mape_den_floor", 0.0) or 0.0)
    ground_truth, prediction, extra = restore_eval_cms(ground_truth, prediction, args)
    mae_list, rmse_list, mape_list = [], [], []
    n_steps = int(ground_truth.shape[2])
    for t in range(1, n_steps + 1):
        gt_t = ground_truth[:, :, t - 1:t]
        pred_t = prediction[:, :, t - 1:t]
        extra_t = None if extra is None else extra[:, :, t - 1:t]
        mae = masked_mae_np(gt_t, pred_t, 0)
        rmse = masked_mse_np(gt_t, pred_t, 0) ** 0.5
        mape = masked_mape_np(gt_t, pred_t, 0, den_floor=mape_floor, extra_mask=extra_t)
        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)
        if t in (3, 6, 12):
            args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(t, mae, rmse, mape))
            args.result[str(t)][" MAE"][args.year] = mae
            args.result[str(t)]["MAPE"][args.year] = mape
            args.result[str(t)]["RMSE"][args.year] = rmse
    args.result["Avg"][" MAE"][args.year] = np.mean(mae_list)
    args.result["Avg"]["RMSE"][args.year] = np.mean(rmse_list)
    args.result["Avg"]["MAPE"][args.year] = np.mean(mape_list)
    args.logger.info("T:Avg\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(
        np.mean(mae_list), np.mean(rmse_list), np.mean(mape_list)))

def cal_metric(ground_truth, prediction, args):

    protocol = str(getattr(args, "metric_protocol", "prefix") or "prefix").lower()
    if protocol in ("stbp", "single", "single_step"):
        return cal_metric_stbp(ground_truth, prediction, args)
    args.logger.info("[*] year {}, testing".format(args.year))
    mape_floor = float(getattr(args, "mape_den_floor", 0.0) or 0.0)
    ground_truth, prediction, extra = restore_eval_cms(ground_truth, prediction, args)
    mae_list, rmse_list, mape_list = [], [], []
    for i in range(1, 13):
        extra_i = None if extra is None else extra[:, :, :i]
        mae = masked_mae_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        rmse = masked_mse_np(ground_truth[:, :, :i], prediction[:, :, :i], 0) ** 0.5
        mape = masked_mape_np(ground_truth[:, :, :i], prediction[:, :, :i], 0, den_floor=mape_floor, extra_mask=extra_i)
        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)
        if i == 3 or i == 6 or i == 12:
            args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i, mae, rmse, mape))
            args.result[str(i)][" MAE"][args.year] = mae
            args.result[str(i)]["MAPE"][args.year] = mape
            args.result[str(i)]["RMSE"][args.year] = rmse
    args.result["Avg"][" MAE"][args.year] = np.mean(mae_list)
    args.result["Avg"]["RMSE"][args.year] = np.mean(rmse_list)
    args.result["Avg"]["MAPE"][args.year] = np.mean(mape_list)
    args.logger.info("T:Avg\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(
        np.mean(mae_list), np.mean(rmse_list), np.mean(mape_list)))
