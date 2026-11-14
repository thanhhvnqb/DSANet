from utils.run_paths import prepare_run
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import DSANet
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.ucf_detectionMAP import getDetectionMAP as dmAP
from utils.training_log import save_evaluation_log
import ucf_option

def refine_scores_hierarchical(logits_mlp: torch.Tensor,
                               logits_align: torch.Tensor,
                               temp: float = 5.0) -> torch.Tensor:
    epsilon = 1e-12
    total_abnormal_prob = torch.sigmoid(logits_mlp / temp) 
    total_normal_prob = 1.0 - total_abnormal_prob           
    p_align = F.softmax(logits_align / temp, dim=1)
    p_align_abnormal_only = p_align[:, 1:]
    sum_p_align_abnormal = p_align_abnormal_only.sum(dim=1, keepdim=True)
    abnormal_distribution = p_align_abnormal_only / (sum_p_align_abnormal + epsilon)
    final_abnormal_probs = total_abnormal_prob * abnormal_distribution
    final_probabilities = torch.cat([total_normal_prob, final_abnormal_probs], dim=1)
    return final_probabilities

def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, DNP_use, device, args, return_details=False):
    
    model.to(device)
    model.eval()

    element_logits2_stack = []
    coarse_video_scores = []
    alignment_video_scores = []
    abnormal_scores = []
    abnormal_gt = []
    abnormal_videos = 0
    gt = np.asarray(gt).reshape(-1)
    gt_offset = 0

    with torch.no_grad():
        for item in testdataloader:
            visual = item[0].squeeze(0)
            length = item[2]

            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            if DNP_use:
                _, logits1, logits2, _, _, _ = model(visual, padding_mask, prompt_text, lengths, DNP_use)
            else:
                _, logits1, logits2, _, _ = model(visual, padding_mask, prompt_text, lengths, DNP_use)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])

            optimized_probs = refine_scores_hierarchical(logits1[0:len_cur], logits2[0:len_cur], args.temp)
            coarse_scores = torch.sigmoid(logits1[:len_cur].squeeze(-1))
            alignment_scores = 1 - F.softmax(logits2[:len_cur], dim=-1)[:, 0]
            coarse_frame_scores = np.repeat(coarse_scores.cpu().numpy(), 16)
            gt_video = gt[gt_offset:gt_offset + len(coarse_frame_scores)]
            if len(gt_video) != len(coarse_frame_scores):
                raise ValueError(f"Prediction/GT length mismatch for video {item[3][0]}")
            gt_offset += len(coarse_frame_scores)
            coarse_video_scores.append(coarse_frame_scores)
            alignment_video_scores.append(np.repeat(alignment_scores.cpu().numpy(), 16))
            if item[4][0] != "Normal":
                abnormal_videos += 1
                abnormal_scores.append(coarse_frame_scores)
                abnormal_gt.append(gt_video)

            element_logits2 = optimized_probs.detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

    if gt_offset != len(gt):
        raise ValueError(f"Per-video frame scores cover {gt_offset} frames, but ground truth has {len(gt)} frames.")
    if not abnormal_gt:
        raise ValueError("No abnormal videos were found in the evaluation set.")
    abnormal_gt = np.concatenate(abnormal_gt)
    abnormal_scores = np.concatenate(abnormal_scores)
    if not np.all(np.isin(abnormal_gt, (0, 1))):
        raise ValueError("Abnormal-video ground truth must contain only 0 and 1.")
    abnormal_frames = int(np.count_nonzero(abnormal_gt))
    normal_frames = len(abnormal_gt) - abnormal_frames
    if not abnormal_frames or not normal_frames:
        raise ValueError("Abnormal-video ground truth must contain both normal and abnormal frames to calculate Ano-AUC.")

    coarse_frame_scores = np.concatenate(coarse_video_scores)
    alignment_frame_scores = np.concatenate(alignment_video_scores)
    ROC1 = roc_auc_score(gt, coarse_frame_scores)
    AP1 = average_precision_score(gt, coarse_frame_scores)
    alignment_auc = roc_auc_score(gt, alignment_frame_scores)
    alignment_ap = average_precision_score(gt, alignment_frame_scores)
    ano_auc = roc_auc_score(abnormal_gt, abnormal_scores)
    ano_ap = average_precision_score(abnormal_gt, abnormal_scores)
    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    print("=" * 60)
    print("UCF-Crime Evaluation")
    print("=" * 60)
    print("[Paper / checkpoint metric]")
    print(f"{'AUC':<31}: {ROC1 * 100:.2f}%")
    print("\n[Fine-grained anomaly detection]")
    for threshold, score in zip(iou, dmap):
        print(f'{f"mAP@{threshold:.1f}":<31}: {score:.2f}%')
    print(f"{'Average mAP':<31}: {np.mean(dmap):.2f}%")
    print("\n[Additional research metrics]")
    print(f"{'Ano-AUC (abnormal videos only)':<31}: {ano_auc * 100:.2f}%")
    print(f"{'Ano-AP  (abnormal videos only)':<31}: {ano_ap * 100:.2f}%")
    print("\n[Auxiliary branch diagnostics]")
    print(f"{'C-branch AUC':<31}: {ROC1 * 100:.2f}%")
    print(f"{'C-branch AP':<31}: {AP1 * 100:.2f}%")
    print(f"{'A-branch AUC':<31}: {alignment_auc * 100:.2f}%")
    print(f"{'A-branch AP':<31}: {alignment_ap * 100:.2f}%")
    print("=" * 60)
    print(f"UCF abnormal-only evaluation: abnormal_videos={abnormal_videos}, "
          f"frames={len(abnormal_gt)}, abnormal_frames={abnormal_frames}, "
          f"normal_frames={normal_frames}")
    print("=" * 60, flush=True)

    metrics = {
        "frame_level_roc_auc": float(ROC1),
        "frame_level_average_precision": float(AP1),
        "alignment_frame_level_roc_auc": float(alignment_auc),
        "alignment_frame_level_average_precision": float(alignment_ap),
        "abnormal_video_frame_roc_auc": float(ano_auc),
        "abnormal_video_frame_average_precision": float(ano_ap),
        "fine_grained_map": {
            f"iou_{threshold:.1f}": float(score)
            for threshold, score in zip(iou, dmap)
        },
        "average_fine_grained_map": float(np.mean(dmap)),
        "counts": {
            "abnormal_videos": int(abnormal_videos),
            "abnormal_video_frames": int(len(abnormal_gt)),
            "abnormal_frames": int(abnormal_frames),
            "normal_frames": int(normal_frames),
        },
    }
    if return_details:
        return ROC1, AP1, metrics
    return ROC1, AP1


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    prepare_run(args, "ucf", training=False)

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})

    testdataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = DSANet(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, args, device)
    model_param = torch.load(args.model_path)
    model.load_state_dict(model_param)

    _, _, metrics = test(
        model, testdataloader, args.visual_length, prompt_text, gt, gtsegments,
        gtlabels, args.DNP_use, device, args, return_details=True,
    )
    log_path = save_evaluation_log(
        "ucf",
        args,
        device,
        selection_metric="frame_level_roc_auc",
        selection_value=metrics["frame_level_roc_auc"],
        metrics=metrics,
    )
    print(f"Evaluation JSON saved to {log_path}")
