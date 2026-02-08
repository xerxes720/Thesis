# Run with
# powershell -ExecutionPolicy Bypass -File .\run_all.ps1

$ErrorActionPreference = "Stop"

$bundle   = "education_framework/data/kdd_bundle.joblib"
$episodes = 800
$maxSteps = 450
$seeds    = @(0) #23 48

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
#       --seed 0

    foreach ($seed in $seeds) {
        Write-Host "=============================="
        Write-Host "Running SEED=$seed"
        Write-Host "=============================="

#        python -m education_framework.main `
#          --bundle $bundle --episodes $episodes --max_steps $maxSteps `
#          --arch flat --ll_mode single --share_mode off `
#          --seed $seed --run_tag "flat_single"
#
        python -m education_framework.main `
          --bundle $bundle --episodes $episodes --max_steps $maxSteps `
          --arch hrl --ll_mode single --share_mode off `
          --seed $seed --run_tag "single_ll"

        python -m education_framework.main `
          --bundle $bundle --episodes $episodes --max_steps $maxSteps `
          --share_mode off `
          --seed $seed --run_tag "multi_no_es"

#         python -m education_framework.main `
#           --bundle $bundle --episodes $episodes --max_steps $maxSteps `
#           --experience_sharing `
#           --seed $seed --run_tag "multi_weighted_cka"
##
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
