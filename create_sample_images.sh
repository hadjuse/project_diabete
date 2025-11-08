#!/bin/bash
# Create a small sample of images for Streamlit Cloud deployment

SAMPLE_DIR="pages/utils/images/1/DRIMDB_sample"
FULL_DIR="pages/utils/images/1/DRIMDB"

# Create sample directories
mkdir -p "$SAMPLE_DIR/Good"
mkdir -p "$SAMPLE_DIR/Bad"

echo "Creating sample dataset with 10 images from each category..."

# Copy 10 random good images
find "$FULL_DIR/Good" -type f \( -name "*.jpg" -o -name "*.png" \) | shuf -n 10 | while read file; do
    cp "$file" "$SAMPLE_DIR/Good/"
done

# Copy 10 random bad images
find "$FULL_DIR/Bad" -type f \( -name "*.jpg" -o -name "*.png" \) | shuf -n 10 | while read file; do
    cp "$file" "$SAMPLE_DIR/Bad/"
done

echo "✅ Sample dataset created at: $SAMPLE_DIR"
echo "Good images: $(ls "$SAMPLE_DIR/Good" | wc -l)"
echo "Bad images: $(ls "$SAMPLE_DIR/Bad" | wc -l)"
echo "Total size: $(du -sh "$SAMPLE_DIR" | cut -f1)"
