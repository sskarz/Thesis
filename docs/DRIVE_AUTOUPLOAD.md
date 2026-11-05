# Thesis PDF auto-upload to Google Drive

The workflow `.github/workflows/thesis-drive-upload.yml` ("Thesis PDF to Drive") runs on every push to `main` that changes `thesis/Thesis_CSULA_Format.pdf`, and on demand from the Actions tab.

It uploads the committed PDF; it does not rebuild it in CI, so Drive always gets exactly the file on `main`. It overwrites the existing Drive file `Thesis_CSULA_Format.pdf` (id `1a80RKhDKND6hQ-isJn_AoXkoLIIhOVUH`) in place with the Drive API (`files.update`), so the file's id, link and sharing stay the same. It then checks that Drive's MD5 matches the local file, and skips the upload if Drive already has that exact file.

## One-time setup

1. **Pick a Google Cloud project.** Open https://console.cloud.google.com, then create a project (any name, e.g. `thesis-drive-sync`) or reuse one.
2. **Enable the Drive API.** In that project: APIs & Services → Library → "Google Drive API" → Enable.
3. **Create a service account and key.**
   - IAM & Admin → Service Accounts → Create service account. Any name; skip the optional role and user steps.
   - Open the new account, go to Keys → Add key → Create new key → JSON. A `.json` file downloads.
   - Copy the account's email (it ends in `.iam.gserviceaccount.com`).
4. **Add the key to the repo.** GitHub → sskarz/Thesis → Settings → Secrets and variables → Actions → New repository secret.
   - Name: `GDRIVE_SA_KEY`.
   - Value: paste the whole contents of the `.json` file.
   - Then delete the downloaded `.json` file from your computer.
5. **Share the Drive file with the service account.** In Drive, open the Share dialog for `Thesis_CSULA_Format.pdf`, add the service account email as **Editor**, and untick "Notify people". Share only this one file.
6. **Merge the PR, then test.** Actions → "Thesis PDF to Drive" → Run workflow (branch `main`). The log should say "Replaced Thesis_CSULA_Format.pdf …", or "already matches" if Drive is already current. After that it runs by itself whenever the PDF on `main` changes.

## Notes

- To point it at a different Drive file, set a repository variable `GDRIVE_THESIS_FILE_ID` (Settings → Secrets and variables → Actions → Variables). That overrides the default id.
- If key creation in step 3 is blocked ("service account key creation is disabled"), the project is under an organization policy. Use a project in your personal account instead.
- To revoke access, remove the service account from the file's sharing, or delete its key in Google Cloud.
- This covers only the CSULA thesis PDF. The short paper (`Thesis_Paper.pdf`) keeps its existing Drive sync routine.
