#!/bin/bash

INPUT_MODEL=${1:-"wav2letter_pruned_int8.tflite"}
OUTPUT_HEADER=${2:-"wav2letter_pruned_int8.h"}


if [ ! -f "$INPUT_MODEL" ]; then
    echo "Error: Input file '$INPUT_MODEL' not found."
    exit 1
fi


mkdir -p "$(dirname "$OUTPUT_HEADER")"


GUARD_NAME=$(basename "$INPUT_MODEL" | tr '[:lower:]' '[:upper:]' | sed 's/[^A-Z0-9]/_/g')_H

echo "Generating $OUTPUT_HEADER from $INPUT_MODEL..."


echo "#ifndef $GUARD_NAME" > "$OUTPUT_HEADER"
echo "#define $GUARD_NAME" >> "$OUTPUT_HEADER"
echo "" >> "$OUTPUT_HEADER"


xxd -i "$INPUT_MODEL" | \
sed 's/^unsigned char/static const unsigned char/g' | \
sed 's/\[\]/[] __attribute__((aligned(16)))/g' | \
sed 's/^unsigned int/static const unsigned int/g' >> "$OUTPUT_HEADER"


echo "" >> "$OUTPUT_HEADER"
echo "#endif // $GUARD_NAME" >> "$OUTPUT_HEADER"

echo "Success! Header created at: $OUTPUT_HEADER"