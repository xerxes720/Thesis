# Run with:
# powershell -ExecutionPolicy Bypass -File .\run_all.ps1

$ErrorActionPreference = "Stop"

# -----------------------------
# 0) Rebuild decision tree bundle (YOU REQUESTED THIS AT THE BEGINNING)
# -----------------------------
#python -m education_framework.scripts.build_decision_tree `
#  --csv education_framework/data/algebra_2005_2006_train.txt `
#  --out education_framework/data/kdd_bundle.joblib `
#  --n_topics 7 --max_depth 7 --min_leaf 50 --ema_alpha 0.2 --seed 0 `
#  --cluster_mode behavior `
#  --action_mode discover --n_actions 5 `
#  --action_features hints,incorrects,duration,opp `
#  --action_min_cluster_frac 0.06 --action_label_mode schema `
#  --leaf_shrinkage_prior 10 `
#  --topic_gain_mode balanced_primary `
#  --quality_eps_frac 0.05 --quality_eps_min 0.0001

# -----------------------------
# 1) Core experiment settings
# -----------------------------
$bundle   = "education_framework/data/kdd_bundle.joblib"
#$bundle   = "education_framework/data/kdd_bundle_BIN75.joblib"

# Suggested: enough for your curves to stabilize without going insane
$episodes = 1000
$maxSteps = 400

# Use 3 seeds for defensibility if you can afford it.
# For quick iteration keep just @(23).
$seeds    = @(23)

# Logging
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$logPath = "logs\run_all_tutee_focus_$ts.log"
Start-Transcript -Path $logPath -Append | Out-Null

function Run-Job {
  param(
    [string]$tag,
    [int]$seed,
    [string[]]$argsList
  )

  Write-Host ""
  Write-Host "--------------------------------------------------"
  Write-Host "RUN: $tag | seed=$seed | episodes=$episodes | max_steps=$maxSteps"
  Write-Host "--------------------------------------------------"

  $baseArgs = @(
    "-m", "education_framework.main",
    "--bundle", $bundle,
    "--episodes", $episodes,
    "--max_steps", $maxSteps,
    "--seed", $seed,
    "--run_tag", $tag,
    "--log_ll_action_effects",
    "--log_ll_agreement",
    "--log_ll_per_topic"
  )

  python @baseArgs @argsList
}

try {

  foreach ($seed in $seeds) {

    # -----------------------------
    # 0) Flat baseline (for your first two plots)
    # -----------------------------
#    Run-Job -tag "flat_baseline" -seed $seed -argsList @(
#      "--arch", "flat"
#    )
    # -----------------------------
    # A) "Paper implementation" anchor runs (NO TUTEE)
    # -----------------------------

#    Run-Job -tag "paper_single" -seed $seed -argsList @(
#      "--arch", "hrl",
#      "--ll_mode", "single",
#      "--share_mode", "off"
#    )

#     Paper baseline: HRL, multi LL, no experience sharing
#    Run-Job -tag "paper_multi_no_es" -seed $seed -argsList @(
#      "--arch", "hrl",
#      "--ll_mode", "multi",
#      "--share_mode", "off"
#    )

#    Run-Job -tag "paper_multi_mutual" -seed $seed -argsList @(
#      "--arch", "hrl",
#      "--ll_mode", "multi",
#      "--experience_sharing",
#      "--share_mode", "mutual"
#    )
    # Paper ES: HRL, multi LL, experience sharing (weighted_cka)
    # NOTE: I intentionally DO NOT enable peer_gate_action_effects here,
    # because that is YOUR extension, not the original paper’s mechanism.
#    Run-Job -tag "paper_multi_weighted_cka" -seed $seed -argsList @(
#      "--arch", "hrl",
#      "--ll_mode", "multi",
#      "--experience_sharing",
#      "--share_mode", "weighted_cka"
#    )

    # -----------------------------
    # B) Your extension: add tutee, hold everything else fixed
    # -----------------------------

    # Tutee vs no-tutee under the NO-ES backbone
#    Run-Job -tag "tutee_no_es" -seed $seed -argsList @(
#      "--arch", "hrl",
#      "--ll_mode", "multi",
#      "--share_mode", "off",
#      "--use_tutee"
#    )

    # Tutee vs no-tutee under the ES backbone (weighted_cka)
#    Run-Job -tag "tutee_weighted_cka" -seed $seed -argsList @(
#      "--arch", "hrl",
#      "--ll_mode", "multi",
#      "--experience_sharing",
#      "--share_mode", "weighted_cka",
#      "--use_tutee"
#    )

    # -----------------------------
    # C) Tutee Control runs:
    # -----------------------------

    Run-Job -tag "tutee_weighted_cka" -seed $seed -argsList @(
    "--arch", "hrl",
    "--ll_mode", "multi",
    "--experience_sharing",
    "--share_mode", "weighted_cka",
    "--use_tutee",
    "--tutee_ll_policy", "learned",
    "--post_eval_tutee_swap",
    "--post_eval_episodes", "200"
    )

    Run-Job -tag "tutee_controlA_randLL_all" -seed $seed -argsList @(
    "--arch", "hrl",
    "--ll_mode", "multi",
    "--experience_sharing",
    "--share_mode", "weighted_cka",
    "--use_tutee",
    "--tutee_ll_policy", "random_all",
    "--tutee_disable_ll_training"
    )
    Run-Job -tag "tutee_controlA_randLL_ready" -seed $seed -argsList @(
    "--arch", "hrl",
    "--ll_mode", "multi",
    "--experience_sharing",
    "--share_mode", "weighted_cka",
    "--use_tutee",
    "--tutee_ll_policy", "random_allowed",
    "--tutee_disable_ll_training"
    )
#     Random tutee ll action
#    Run-Job -tag "tutee_controlA_randLL_all" -seed $seed -argsList @(
#      "--arch", "hrl",
#      "--ll_mode", "multi",
#      "--experience_sharing",
#      "--share_mode", "weighted_cka",
#      "--use_tutee",
#      "--tutee_ll_policy", "random_all"
#      "--tutee_disable_ll_training"
#    )
#    Run-Job -tag "tutee_controlA_randLL_ready" -seed $seed -argsList @(
#      "--arch", "hrl",
#      "--ll_mode", "multi",
#      "--experience_sharing",
#      "--share_mode", "weighted_cka",
#      "--use_tutee",
#      "--tutee_ll_policy", "random_allowed"
#      "--tutee_disable_ll_training"
#    )

    # -----------------------------
    # Optional (only if you want it): show that YOUR peer-gate is a stability fix
    # This is NOT a tutee hyperparameter ablation; it's a "why ES behaves" ablation.
    # If you want to keep the ablation set minimal, leave this commented.
    # -----------------------------
    # Run-Job -tag "paper_multi_weighted_cka_gateAE" -seed $seed -argsList @(
    #   "--arch", "hrl",
    #   "--ll_mode", "multi",
    #   "--experience_sharing",
    #   "--share_mode", "weighted_cka",
    #   "--peer_gate_action_effects",
    #   "--peer_gate_topk", "2"
    # )
    #
    # Run-Job -tag "tutee_weighted_cka_gateAE" -seed $seed -argsList @(
    #   "--arch", "hrl",
    #   "--ll_mode", "multi",
    #   "--experience_sharing",
    #   "--share_mode", "weighted_cka",
    #   "--peer_gate_action_effects",
    #   "--peer_gate_topk", "2",
    #   "--use_tutee"
    # )

  }

  Write-Host ""
  Write-Host "ALL TUTEE-FOCUSED RUNS DONE."

}
finally {
  Stop-Transcript | Out-Null
}
