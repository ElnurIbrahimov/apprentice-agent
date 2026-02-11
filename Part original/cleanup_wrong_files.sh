#!/bin/bash
# FLUXMIND CLEANUP SCRIPT
# Removes all files created in the WRONG direction (Jan 26-29, 2026)
# Run from project directory: bash cleanup_wrong_files.sh

echo "=========================================="
echo "FLUXMIND CLEANUP - Removing Wrong Files"
echo "=========================================="

# Wrong v0.79-v0.86 documentation
rm -f FluxMind_Overview_v079.md
rm -f FluxMind_Overview_v080.md
rm -f FluxMind_Roadmap_v081.md
rm -f FluxMind_Roadmap_v082.md
rm -f FluxMind_Roadmap_v086.md
rm -f FluxMind_v080_CHANGELOG.md
rm -f FluxMind_v080_Documentation.md
rm -f FluxMind_v081_CHANGELOG.md
rm -f FluxMind_v081_Documentation.md
rm -f FluxMind_v081_Documentation.docx
rm -f FluxMind_v081_MetaLearning_Breakthrough.md
rm -f FluxMind_v081_QuickReference.md
rm -f FluxMind_v081_Results.md
rm -f FluxMind_v082_CHANGELOG.md
rm -f FluxMind_v082_DSL_Analysis.md
rm -f FluxMind_v082_Documentation.md
rm -f FluxMind_v082_Documentation.docx
rm -f FluxMind_v082_Phase1_Analysis.md
rm -f FluxMind_v082_QuickReference.md
rm -f FluxMind_v082_Results.md
rm -f FluxMind_v082_v086_Documentation.md
rm -f FluxMind_v086_API.md
rm -f FluxMind_v086_Architecture.md
rm -f FluxMind_v086_CHANGELOG.md
rm -f FluxMind_v086_Documentation.md
rm -f FluxMind_v086_QuickReference.md
rm -f FluxMind_v086_TechSpec.md
rm -f FluxMind_v086_TrainingGuide.md

echo "[✓] Removed wrong v0.79-v0.86 documentation"

# Wrong Python files
rm -f train_hybrid_v080.py
rm -f run_long_training.py
rm -f train_massive.py
rm -f evaluate_v082_balanced.py
rm -f train_v086_ensemble.py
rm -f hybrid_fluxmind.py
rm -f hybrid_fluxmind_v080.py
rm -f train_hybrid.py
rm -f train_hybrid_v2.py

echo "[✓] Removed wrong Python files"

# Wrong "MetaFluxMind" files (completely wrong direction)
rm -f METAFLUXMIND_DOCUMENTATION.md
rm -f MetaFluxMind_API_REFERENCE.md
rm -f MetaFluxMind_CHANGELOG.md
rm -f MetaFluxMind_CHANGELOG_v0771.md
rm -f MetaFluxMind_DIAGNOSIS.md
rm -f MetaFluxMind_FIX_SUMMARY.md
rm -f MetaFluxMind_README.md
rm -f MetaFluxMind_README_v0771.md
rm -f MetaFluxMind_TECHNICAL_SPEC.md
rm -f MetaFluxMind_v0772_ARCHITECTURE.md
rm -f meta_fluxmind.py
rm -f meta_fluxmind_v0771.py
rm -f meta_fluxmind_v0772.py
rm -f meta_fluxmind_v078.py
rm -f train_meta.py
rm -f train_meta_v0771.py
rm -f train_meta_v0772.py
rm -f train_meta_v0781.py

echo "[✓] Removed wrong MetaFluxMind files"

# Questionable files that may also be wrong
rm -f fluxmind_core.py
rm -f fluxmind_tool.py
rm -f train_fluxmind.py
rm -f API_REFERENCE.md
rm -f CHANGELOG.md
rm -f INTEGRATION_GUIDE.md
rm -f README.md
rm -f TECHNICAL_SPEC.md
rm -f TRAINING_GUIDE.md

echo "[✓] Removed questionable files"

echo ""
echo "=========================================="
echo "CLEANUP COMPLETE"
echo "=========================================="
echo ""
echo "FILES THAT REMAIN (Real FluxMind):"
echo ""
ls -la *.md *.py 2>/dev/null
