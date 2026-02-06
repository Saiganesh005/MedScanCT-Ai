<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>
 
# Run and deploy your AI Studio app
 
This contains everything you need to run your app locally.
 
View your app in AI Studio: https://ai.studio/apps/drive/1zRjnz9CpkcEEbXI8WjixAZqj_bhjT1ix
 
## Run Locally
 
**Prerequisites:**  Node.js
 
 
1. Install dependencies:
    `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
    `npm run dev`

## Download the COVIDxCT dataset

Use the helper script to download the Kaggle dataset into `data/covidxct`:

```bash
python scripts/download_covidxct.py
```

The script uses `kagglehub` and copies the cached download into the repository. Ensure you are authenticated with Kaggle and have the package installed (`pip install kagglehub`).
 
