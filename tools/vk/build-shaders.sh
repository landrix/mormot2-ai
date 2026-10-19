#!/usr/bin/env bash
# LandrixAI - compiles tools/vk/*.comp to SPIR-V and writes them as Pascal
# constants into src/infer/mormot.ai.infer.vulkan.spv.inc (checked in: the
# build and the run time need no shader compiler)
# - needs glslangValidator (package glslang-tools; without root:
#   apt-get download glslang-tools && dpkg -x glslang-tools_*.deb ~/.local/opt/vk/root)
set -eu
cd "$(dirname "$0")"
GLSLANG=${GLSLANG:-$(command -v glslangValidator || echo ~/.local/opt/vk/root/usr/bin/glslangValidator)}
OUT=../../src/infer/mormot.ai.infer.vulkan.spv.inc
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
ARGS=()
for f in *.comp; do
  n=${f%.comp}
  # matvec8.comp, attndec.comp: only the builds below
  [ "$n" = matvec8 ] && continue
  [ "$n" = attndec ] && continue
  # Vulkan 1.0 SPIR-V: every driver takes it; -Os keeps the constants small
  "$GLSLANG" -V --target-env vulkan1.0 -Os -o "$TMP/$n.spv" "$f" > "$TMP/$n.log" ||
    { cat "$TMP/$n.log"; exit 1; }
  ARGS+=("SPV_$(echo "$n" | tr a-z A-Z)=$TMP/$n.spv")
done
# variants: linear8.comp with subgroup reductions, for Vulkan 1.1 devices
"$GLSLANG" -V --target-env vulkan1.1 -Os -DSUBGROUP -o "$TMP/linear8_sg.spv" linear8.comp   > "$TMP/linear8_sg.log" || { cat "$TMP/linear8_sg.log"; exit 1; }
ARGS+=("SPV_LINEAR8_SG=$TMP/linear8_sg.spv")
# linear8.comp for the experts chosen on the device (-DIDS), plain and subgroup
"$GLSLANG" -V --target-env vulkan1.0 -Os -DIDS -o "$TMP/linear8_ids.spv" linear8.comp > "$TMP/linear8_ids.log" || { cat "$TMP/linear8_ids.log"; exit 1; }
"$GLSLANG" -V --target-env vulkan1.1 -Os -DSUBGROUP -DIDS -o "$TMP/linear8_ids_sg.spv" linear8.comp > "$TMP/linear8_ids_sg.log" || { cat "$TMP/linear8_ids_sg.log"; exit 1; }
ARGS+=("SPV_LINEAR8_IDS=$TMP/linear8_ids.spv" "SPV_LINEAR8_IDS_SG=$TMP/linear8_ids_sg.spv")
# attndec.comp: subgroup arithmetic (Vulkan 1.1), a float or a half-float cache
"$GLSLANG" -V --target-env vulkan1.1 -Os -o "$TMP/attndec.spv" attndec.comp > "$TMP/attndec.log" || { cat "$TMP/attndec.log"; exit 1; }
"$GLSLANG" -V --target-env vulkan1.1 -Os -DKVF16 -o "$TMP/attndec_f16.spv" attndec.comp > "$TMP/attndec_f16.log" || { cat "$TMP/attndec_f16.log"; exit 1; }
ARGS+=("SPV_ATTNDEC=$TMP/attndec.spv" "SPV_ATTNDEC_F16=$TMP/attndec_f16.spv")
# variants: the KV cache in half floats (TQwen3Vulkan.KvF16)
for n in qkrope attention flash; do
  "$GLSLANG" -V --target-env vulkan1.0 -Os -DKVF16 -o "$TMP/${n}_f16.spv" $n.comp > "$TMP/${n}_f16.log" || { cat "$TMP/${n}_f16.log"; exit 1; }
  ARGS+=("SPV_$(echo "$n" | tr a-z A-Z)_F16=$TMP/${n}_f16.spv")
done
# matvec8.comp once per weight type (2 Q8_0, 3 Q4_K, 4 Q6_K), with and without
# subgroup reductions
for t in 2:Q8 3:Q4K 4:Q6K; do
  d=${t%%:*}
  n=${t#*:}
  "$GLSLANG" -V --target-env vulkan1.0 -Os -DDTYPE=${d}u -o "$TMP/mv_$n.spv" matvec8.comp > "$TMP/mv_$n.log" || { cat "$TMP/mv_$n.log"; exit 1; }
  "$GLSLANG" -V --target-env vulkan1.1 -Os -DSUBGROUP -DDTYPE=${d}u -o "$TMP/mv_${n}_sg.spv" matvec8.comp > "$TMP/mv_${n}_sg.log" || { cat "$TMP/mv_${n}_sg.log"; exit 1; }
  ARGS+=("SPV_MATVEC8_$n=$TMP/mv_$n.spv" "SPV_MATVEC8_${n}_SG=$TMP/mv_${n}_sg.spv")
done
python3 spv2pas.py "$OUT" "${ARGS[@]}"
echo "written: $OUT (${#ARGS[@]} shaders)"
