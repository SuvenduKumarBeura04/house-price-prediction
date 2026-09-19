# Data folder

This folder is meant to hold:

- `train.csv`
- `test.csv`
- `data_description.txt`
- `sample_submission.csv`

These files are **not included** in this project (Kaggle competition rules require you
to download them yourself after accepting the rules).

## How to get them

1. Go to: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
2. Sign in / create a free Kaggle account
3. Click "I understand and accept" on the competition rules (required before download)
4. Download the files and place them in this folder

## Or via Kaggle API

```bash
pip install kaggle
kaggle competitions download -c house-prices-advanced-regression-techniques -p data/
```

(Requires a `kaggle.json` API token — generate it from your Kaggle account settings under
"Create New API Token", then place it in `~/.kaggle/kaggle.json` on Mac/Linux or
`C:\Users\<you>\.kaggle\kaggle.json` on Windows.)
