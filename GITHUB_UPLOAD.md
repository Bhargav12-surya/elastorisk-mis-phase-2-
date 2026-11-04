# Upload this project to GitHub

The folder has Git installed but was not a Git repository when this guide was prepared. The updated `.gitignore` keeps the raw dataset, prepared cache, environments, `.env` files, binary checkpoints/maps, duplicate handoff package, and machine-specific helpers out of a normal `git add .`.

Included material: the standalone program, README, licensing/references, the original modular source and educational notebooks referenced by the README, completed full-run CSV/JSON results and PNG/PDF figures, the full-field panel/provenance, and selected README-linked verification records. The raw data and checkpoints remain on your laptop. Some provenance files contain local paths; those paths describe the original experiment and do not give a clone access to your computer.

## 1. Create an empty repository

Open <https://github.com/new> and choose a repository name, for example `ultrasound-elastography-mis`. Choose the visibility you want. Leave the README, license and gitignore initialization options unchecked because this project already has those files.

Copy the new repository's HTTPS URL. It will look like:

```text
https://github.com/YOUR_USERNAME/ultrasound-elastography-mis.git
```

## 2. Prepare the local commit

In PowerShell:

```powershell
Set-Location -LiteralPath D:\MIS
git init -b main
git add .
git status --short
git diff --cached --stat
```

Review the staged list. It should contain source, documentation and selected results. It should not contain `.env`, `.venv`, `.venv-mis`, raw `data/phantom` or `data/simulation` files, `artifacts/mis_full_cache`, `.pt` checkpoints, or `.npz` maps. Do not use `git add -f` to override the ignore rules for these files.

Check the important ignore rules if desired:

```powershell
git check-ignore .env .venv-mis/Scripts/python.exe data/phantom/Data_1.mat artifacts/mis_full_cache/manifest.json reports/mis_full_training/best.pt
```

Then create the commit:

```powershell
git commit -m "Add MIS elastography project and experimental results"
```

If Git asks for your author identity, set your own name and the email you want attached to this repository, then retry the commit. You can use the private commit email shown in your GitHub account settings. These commands affect only this repository:

```powershell
git config user.name "YOUR NAME"
git config user.email "YOUR GITHUB COMMIT EMAIL"
```

## 3. Connect and push

Replace the URL below with your actual empty repository URL:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/ultrasound-elastography-mis.git
git remote -v
git push -u origin main
```

Complete GitHub authentication if prompted. Keep credentials out of commands, source files and chat messages. If the chosen repository already contains commits, inspect/fetch that history before pushing; do not use a force push to bypass the mismatch.

## 4. What someone cloning the repository can do

They can inspect the source and saved numerical results, view the PNG/PDF figures, and download the repository to open `reports/mis_full_training/report.html` locally. GitHub's ordinary file view does not serve that HTML as an interactive website.

Training and full-field inference need the separately obtained dataset and a suitable Python/PyTorch environment. The saved report does not require retraining. See the main README for setup, data assumptions and commands. `requirements-lock.txt` belongs to the older modular project; the main README and the completed run's environment record describe the working standalone GPU setup.

The project demonstrates self-supervised elastography and probabilistic quality-risk analysis. There are no pathology-linked tumor grades in the supplied data; the repository's claims should retain that boundary.

## Official instructions

- [Adding locally hosted code to GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
- [Ignoring files](https://docs.github.com/en/get-started/git-basics/ignoring-files)

No GitHub repository was created and no upload was performed while preparing this guide.
