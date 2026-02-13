# Run with
# powershell -ExecutionPolicy Bypass -File .\run_all.ps1

$ErrorActionPreference = "Stop"

$bundle   = "education_framework/data/kdd_bundle.joblib"
$episodes = 1000
$maxSteps = 300
$seeds    = @(23) #23 48

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$logPath = "logs\run_all_$ts.log"

# Log everything (console + file)
"Logging to $logPath"
Start-Transcript -Path $logPath -Append | Out-Null

try {
#     python -m education_framework.scripts.build_decision_tree `
#       --csv education_framework/data/algebra_2005_2006_train.txt `
#       --out $bundle `
#       --n_topics 7 `
#       --max_depth 7 `
#       --min_leaf 50 `
#       --ema_alpha 0.2 `
#       --seed 0 `
#       --cluster_mode behavior
     #python -m education_framework.scripts.build_decision_tree --csv education_framework/data/algebra_2005_2006_train.txt --out education_framework/data/kdd_bundle.joblib --n_topics 7 --max_depth 7 --min_leaf 50 --ema_alpha 0.2 --seed 0 --cluster_mode cooccur --action_label_mode cluster --cluster_max_kcs 800 --cluster_min_kc_freq 20
#python -m education_framework.scripts.build_decision_tree   --csv education_framework/data/algebra_2005_2006_train.txt   --out education_framework/data/kdd_bundle.joblib   --n_topics 7 --max_depth 7 --min_leaf 50 --ema_alpha 0.2 --seed 0   --cluster_mode behavior -
#-action_mode discover --n_actions 5   --action_features hints,incorrects,duration,opp   --action_min_cluster_frac 0.06 --action_label_mode schema   --leaf_shrinkage_prior 10   --topic_gain_mode balanced_primary   --quality_eps_frac 0.05 --quality_eps_min 0.0001
    foreach ($seed in $seeds) {
        Write-Host "=============================="
        Write-Host "Running SEED=$seed"
        Write-Host "=============================="

        python -m education_framework.main `
          --bundle $bundle --episodes $episodes --max_steps $maxSteps `
          --arch flat --ll_mode single --share_mode off `
          --seed $seed --run_tag "flat_single"

#        python -m education_framework.main `
#          --bundle $bundle --episodes $episodes --max_steps $maxSteps `
#          --arch hrl --ll_mode single --share_mode off `
#          --seed $seed --run_tag "single_ll" --log_ll_action_effects
##
        python -m education_framework.main `
          --bundle $bundle --episodes $episodes --max_steps $maxSteps `
          --share_mode off `
          --seed $seed --run_tag "multi_no_es" --log_ll_action_effects --log_ll_agreement --log_ll_per_topic
###
#         python -m education_framework.main `
#           --bundle $bundle --episodes $episodes --max_steps $maxSteps `
#           --experience_sharing `
#           --seed $seed --run_tag "multi_weighted_cka" --log_ll_action_effects --log_ll_agreement --log_ll_per_topic

###
#        python -m education_framework.main `
#          --bundle $bundle --episodes $episodes --max_steps $maxSteps `
#          --experience_sharing --share_mode mutual `
#          --seed $seed --run_tag "multi_mutual"
#
#         python -m education_framework.main `
#           --bundle $bundle --episodes $episodes --max_steps $maxSteps `
#           --share_mode off --use_tutee `
#           --seed $seed --run_tag "tutee_no_es"
#
#         python -m education_framework.main `
#           --bundle $bundle --episodes $episodes --max_steps $maxSteps `
#           --experience_sharing --use_tutee `
#           --seed $seed --run_tag "tutee_weighted_cka"
    }

    Write-Host "ALL RUNS DONE."
}
finally {
    Stop-Transcript | Out-Null
}
