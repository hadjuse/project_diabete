# Streamlit Cloud Configuration

## Required Secrets
Add these in your Streamlit Cloud app settings (Settings → Secrets):

```toml
PATH_FOLDER_IMAGES = "pages/utils/images/1/DRIMDB_sample"
MODEL_PATH = "model/drimdb_model.pth"
```

## Optional Advanced Settings
If you need to customize upload limits or other settings on Streamlit Cloud:

Go to: **App Settings → Advanced settings** and add:

```toml
[server]
maxUploadSize = 10
enableCORS = true

[browser]
gatherUsageStats = false
```

## Troubleshooting Mobile Upload Issues

If you get "Network Error" when uploading from mobile:

1. **Check file size**: Maximum 10MB per file
2. **Check image format**: Use JPG/PNG (avoid HEIC on iPhone)
3. **Try compressing**: Use a photo compression app before upload
4. **Check connection**: Ensure stable internet connection
5. **Try different browser**: Safari vs Chrome on iOS

### Converting HEIC to JPG on iPhone:
- Use the "Shortcuts" app to convert HEIC → JPG
- Or email the photo to yourself (auto-converts)
- Or use apps like "HEIC to JPG Converter"
