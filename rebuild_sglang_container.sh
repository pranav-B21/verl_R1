#!/bin/bash
# Script to rebuild the SGLang container with TLS fixes

cd /work/09585/shijunli4527/vista/Project/verl_R1

echo "========================================="
echo "Building TLS-fixed SGLang container"
echo "========================================="
echo ""
echo "Base: nvcr.io/nvidia/sglang:25.10-py3"
echo "Output: sglang_25.10-py3-tls-fixed.sif"
echo ""
echo "This will take 15-30 minutes..."
echo ""

# Load required modules
module reset
module load nvidia/25.5 cuda/12.9 gcc/15
module load tacc-apptainer

# Build the container
apptainer build \
    --fakeroot \
    sglang_25.10-py3-tls-fixed.sif \
    sglang_25.10-py3-tls-fixed.def

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "SUCCESS! Container built successfully"
    echo "========================================="
    echo ""
    echo "Container location: sglang_25.10-py3-tls-fixed.sif"
    echo "Container size:"
    ls -lh sglang_25.10-py3-tls-fixed.sif
    echo ""
    echo "To use the fixed container:"
    echo "  1. Update your scripts to use: sglang_25.10-py3-tls-fixed.sif"
    echo "  2. Or backup and replace the original:"
    echo "     mv sglang_25.10-py3.sif sglang_25.10-py3.sif.bak"
    echo "     ln -s sglang_25.10-py3-tls-fixed.sif sglang_25.10-py3.sif"
    echo ""
    echo "Test the container with:"
    echo "  apptainer exec --nv sglang_25.10-py3-tls-fixed.sif python3 -c 'import sglang; print(sglang.__version__)'"
else
    echo ""
    echo "========================================="
    echo "ERROR: Container build failed"
    echo "========================================="
    echo ""
    echo "Common issues:"
    echo "  1. Insufficient disk space (needs ~10GB free)"
    echo "  2. Network connectivity to nvcr.io"
    echo "  3. Fakeroot permissions issue"
    echo ""
    echo "To check disk space:"
    echo "  df -h ."
    echo ""
    echo "To retry without fakeroot (if you have sudo):"
    echo "  sudo apptainer build sglang_25.10-py3-tls-fixed.sif sglang_25.10-py3-tls-fixed.def"
    exit 1
fi
