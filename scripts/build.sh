#!/usr/bin/env bash
# Build the Tesseract images of this repository.
#
#   bash scripts/build.sh                 # all components
#   bash scripts/build.sh vane_fea        # one component
#
# The solver packages are installed into the images as wheels built from their own repositories,
# expected next to this one (override with TURBOAI_SRC / VANEFEA_SRC):
#   ../TurboAI          -> turboai   (cascade CFD with OpenFOAM)
#   ../vane-thermomech  -> vanefea   (CAD / mesh / CalculiX FEA)
# The cascade CFD image also needs the compiled OpenFOAM extension libturboaiFvOptions.so
# (TurboAI: scripts/build_openfoam_libs.sh), copied from $FOAM_USER_LIBBIN.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
TURBOAI_SRC=${TURBOAI_SRC:-../TurboAI}
VANEFEA_SRC=${VANEFEA_SRC:-../vane-thermomech}
PY=${PYTHON:-python3}

wheel() {   # wheel <source> <component>
    mkdir -p "tesseracts/$2/wheels"
    rm -f "tesseracts/$2/wheels/"*.whl
    "$PY" -m pip wheel --no-deps -q "$1" -w "tesseracts/$2/wheels"
}

build_vane_fea() {
    wheel "$VANEFEA_SRC" vane_fea
    tesseract build tesseracts/vane_fea
}
build_cascade_cfd() {
    wheel "$TURBOAI_SRC" cascade_cfd
    mkdir -p tesseracts/cascade_cfd/lib
    cp "${FOAM_USER_LIBBIN:-$HOME/OpenFOAM/$USER-v2606/platforms/linux64GccDPInt32Opt/lib}/libturboaiFvOptions.so" \
        tesseracts/cascade_cfd/lib/
    tesseract build tesseracts/cascade_cfd
}
build_vane_surrogate() {
    tesseract build tesseracts/vane_surrogate
}

targets=("$@")
[ ${#targets[@]} -gt 0 ] || targets=(vane_surrogate vane_fea cascade_cfd)
for t in "${targets[@]}"; do
    echo "=== $t ==="
    "build_$t"
done
