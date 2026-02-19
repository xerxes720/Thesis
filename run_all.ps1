# Run with:
# powershell -ExecutionPolicy Bypass -File .\run_all.ps1

#$ErrorActionPreference = "Stop"

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
#python -m education_framework.scripts.build_decision_tree
#--csv education_framework/data/algebra_2005_2006_train.txt
#--out education_framework/data/kdd_bundle_BIN75.joblib
#--n_topics 5 --ema_alpha 0.2 --seed 0
#--cluster_mode behavior
#--action_mode discover --n_actions 5
#--action_features hints,incorrects,duration,opp
#--action_label_mode schema
#--topic_gain_mode balanced_primary
#--quality_eps_frac 0.05 --quality_eps_min 0.0001
#--topic_gain_mode_bins linear
$ErrorActionPreference = 'Stop'

# -----------------------------------------------------------------------------
# Central runner for: (1) paper re-implementation plots, (2) tutee + controls
#
# Trainer:
#   python -m education_framework.main
# CSV naming:
#   <run_tag>__seed=<seed>.csv
# Output:
#   education_framework/runs
# -----------------------------------------------------------------------------

$bundle = 'education_framework/data/kdd_bundle.joblib'
$episodes = 2000
$maxSteps = 300

$runsDir = 'education_framework/runs'
$env:RUNS_DIR = $runsDir

# Use >=3 seeds for defensibility (committee-friendly). Add more if you can.
$seeds = @(0)

function Run-Train
{
    param(
        [string]$tag,
        [int]$seed,
        [string]$arch = 'hrl',
        [string]$llMode = 'multi',
        [switch]$experienceSharing,
        [string]$shareMode = 'weighted_cka',
        [switch]$useTutee,
        [string]$tuteeLLPolicy = 'learned',
        [switch]$tuteeDisableLLTraining,

        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$extraArgs
    )

    $arguments = @(
        '--bundle', $bundle,
        '--episodes', $episodes,
        '--max_steps', $maxSteps,
        '--seed', $seed,
        '--arch', $arch,
        '--ll_mode', $llMode,
        '--run_tag', $tag
    )

    if ($experienceSharing)
    {
        $arguments += @('--experience_sharing', '--share_mode', $shareMode)
    }

    if ($useTutee)
    {
        $arguments += @('--use_tutee', '--tutee_ll_policy', $tuteeLLPolicy)
        if ($tuteeDisableLLTraining)
        {
            $arguments += @('--tutee_disable_ll_training')
        }
    }

    if ($extraArgs)
    {
        $arguments += $extraArgs
    }

    Write-Host "`n--------------------------------------------------" -ForegroundColor Cyan
    Write-Host "RUN: $tag | seed=$seed | arch=$arch | ll_mode=$llMode | ES=$( $experienceSharing.IsPresent )($shareMode) | tutee=$( $useTutee.IsPresent )($tuteeLLPolicy)" -ForegroundColor Cyan
    Write-Host "--------------------------------------------------" -ForegroundColor Cyan

    python -m education_framework.main @arguments
}


# -----------------------------------------------------------------------------
# 1) Paper re-implementation runs
#    Plot 1: flat vs HRL multi (no ES)
#    Plot 2: flat vs HRL multi (no ES)
#    Plot 3: HRL single vs HRL multi (no ES) vs HRL multi mutual ES vs HRL multi CFA/wCKA ES
# -----------------------------------------------------------------------------
foreach ($seed in $seeds) {
#    # flat baseline (paper exp1)
#    Run-Train -tag 'flat_baseline' -seed $seed -arch 'flat' -llMode 'single'
#
##     HRL multi (no ES)
#    Run-Train -tag 'paper_multi_no_es' -seed $seed -arch 'hrl' -llMode 'multi'
#
#    # HRL single (shared LL)
#    Run-Train -tag 'paper_single' -seed $seed -arch 'hrl' -llMode 'single'
##
#    # HRL multi + mutual ES (paper ES baseline)
#    Run-Train -tag 'paper_multi_mutual' -seed $seed -arch 'hrl' -llMode 'multi' -experienceSharing -shareMode 'mutual'
#
#    # HRL multi + "CFA" ES (your weighted_cka / wCKA implementation)
#    Run-Train -tag 'paper_multi_weighted_cka' -seed $seed -arch 'hrl' -llMode 'multi' -experienceSharing -shareMode 'weighted_cka' --save_run_diagnostics
}


# -----------------------------------------------------------------------------
# 2) Tutee main + controls (defensibility)
# -----------------------------------------------------------------------------
foreach ($seed in $seeds)
{
    # HRL multi + "CFA" ES (your weighted_cka / wCKA implementation)
#    Run-Train -tag 'paper_multi_weighted_cka_forget' -seed $seed -arch 'hrl' -llMode 'multi' -experienceSharing -shareMode 'weighted_cka' --enable_forgetting --forget_rate 5e-5 --forget_floor 0.25 --retention_decay 0.9995

    # +Tutee (no ES)
#    Run-Train -tag 'tutee_no_es' -seed $seed -arch 'hrl' -llMode 'multi' -useTutee -tuteeLLPolicy 'learned' -shareMode 'off' --post_eval_tutee_swap --post_eval_episodes 200 --save_run_diagnostics

    # +Tutee + ES (CFA/wCKA)
    Run-Train -tag 'tutee_weighted_cka' -seed $seed -arch 'hrl' -llMode 'multi' -useTutee  -tuteeLLPolicy 'learned' -experienceSharing -shareMode 'weighted_cka' --post_eval_tutee_swap --post_eval_episodes 200 --save_run_diagnostics

    # Control A: HL can choose tutee, but tutee LL is random (ALL actions) and we disable its LL training
    Run-Train -tag 'tutee_controlA_randLL_all' -seed $seed -arch 'hrl' -llMode 'multi' -useTutee  -tuteeLLPolicy 'random_all' -experienceSharing -shareMode 'weighted_cka' -tuteeDisableLLTraining --save_run_diagnostics

    # Control A (variant): random only among "ready"/allowed tutee actions
    Run-Train -tag 'tutee_controlA_randLL_ready' -seed $seed -arch 'hrl' -llMode 'multi' -useTutee -tuteeLLPolicy 'random_allowed' -experienceSharing -shareMode 'weighted_cka' -tuteeDisableLLTraining --save_run_diagnostics
}


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
#python education_framework/plots/plot_compare.py --runs_dir $runsDir --preset paper_reimpl --expected_seeds ($seeds -join ',') --no_show
#python education_framework/plots/plot_compare.py --runs_dir $runsDir --preset tutee_defense --expected_seeds ($seeds -join ',') --no_show

Write-Host "`nAll done. Figures are under: $runsDir/figs/" -ForegroundColor Green