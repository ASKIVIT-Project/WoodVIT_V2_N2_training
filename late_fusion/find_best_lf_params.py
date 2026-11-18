"""List the best late-fusion parameter sets of a split (top lists for the manual choice).

Reads every evaluated folder under RESULTS_ROOT/Late_Fusion_noEF/<SPLIT>/ (Late_Fusion/ if
config_fusion.py names an EF model) and writes to its
top_metrics/: all_metrics_per_folder.csv, one top list each for segm mAP, nMCC (classified) and
C-OSA (descending) and for M-OSA (ascending), and combined_top<n>_per_folder.csv. The instance
metrics are taken at IOU_THRESHOLD and CONF_THRESHOLD (the operating point of the paper).

Run:  python late_fusion/find_best_lf_params.py
"""
from config_fusion import MODEL_FOLDERS, RESULTS_ROOT
from woodvit_rcnn.fusion import fusion_roles, write_top_lists

SPLIT = "val"
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.8
TOP_N = 3


if __name__ == "__main__":
    out_dir, tops = write_top_lists(RESULTS_ROOT, SPLIT, IOU_THRESHOLD, CONF_THRESHOLD, TOP_N,
                                    roles=fusion_roles(MODEL_FOLDERS))
    for metric, df in tops.items():
        print(f"\nTop {TOP_N} by {metric}:")
        print(df[["folder", "segm_mAP", "nMCC_classified", "C_OSA", "M_OSA"]].round(3).to_string(index=False))
    print(f"\nSaved to {out_dir}")
