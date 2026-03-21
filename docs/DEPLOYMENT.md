# ClutchAI Deployment Guide

This guide covers deploying ClutchAI to Google Cloud Run, connecting a custom domain (www.clutchai.app), and configuring secrets from Google Cloud Secret Manager.

## Prerequisites

- `gcloud` CLI installed and authenticated
- Domain `clutchai.app` (or your domain) registered and manageable
- `.env` file with all required credentials

---

## 1. Upload Secrets to Secret Manager

Before deploying, upload your environment variables to Google Cloud Secret Manager:

```bash
./scripts/gcloud/update_secrets.sh
# Or with a specific env file:
./scripts/gcloud/update_secrets.sh .env.prod
```

This creates secrets in Secret Manager with the same names as your env var keys (e.g., `OPENAI_API_KEY`, `YAHOO_CLIENT_ID`).

---

## 2. Grant Cloud Run Access to Secrets

The Cloud Run service account needs permission to read secrets. Run this **once** (or after creating new secrets):

```bash
# Get your project ID and number
PROJECT_ID=$(gcloud config get-value project)
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# Grant Secret Manager access to the default Compute Engine service account
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"
```

---

## 3. Deploy to Cloud Run

```bash
./scripts/gcloud/deploy.sh
```

The deploy script:
- Builds and deploys from source
- Injects secrets from Secret Manager as environment variables
- Sets `YAHOO_REDIRECT_URI=https://www.clutchai.app` for production OAuth

**Important:** Ensure the secrets referenced in `deploy.sh` exist in Secret Manager. If you get "secret not found" errors, run `update_secrets.sh` first or add the missing secrets manually.

---

## 4. Connect Custom Domain (www.clutchai.app)

Cloud Run requires **separate mappings** for the root domain and the www subdomain.

### Step 4a: Map the domain in Cloud Run

1. Go to [Cloud Run Console](https://console.cloud.google.com/run)
2. Click your **clutchai** service
3. Click **Manage custom domains**
4. Click **Add mapping**
5. Select your service: `clutchai` (us-central1)
6. Enter domain: `www.clutchai.app`
7. Click **Continue**
8. (Optional) Repeat for `clutchai.app` if you want the root domain too

### Step 4b: Verify domain ownership

If prompted, add the verification records to your DNS:

- **TXT record**: Add the record shown (e.g., `google-site-verification=...`) at your DNS provider
- Wait for verification (can take a few minutes to hours)

### Step 4c: Add DNS records

After verification, Cloud Run will show the required DNS records. Typically:

| Type | Name | Value |
|------|------|-------|
| CNAME | www | ghs.googlehosted.com (or the value Cloud Run shows) |

For **Cloud Run custom domains**, the exact records depend on your setup. Check the Cloud Run "Manage custom domains" page for the specific CNAME or A records to add.

**Note:** If using Google Domains or Cloud Domains, domain verification may be automatic.

### Step 4d: Wait for DNS propagation

DNS changes can take 24–48 hours to propagate. Use `dig www.clutchai.app` to verify.

---

## 5. Make Yahoo OAuth work on Cloud Run (required for Yahoo features)

**Why it works locally but not on Cloud Run:** Locally, when tokens are missing the Yahoo OAuth library can open a browser or prompt you for a code. On Cloud Run there is no terminal (no stdin) and no browser, so it raises `EOFError: EOF when reading a line` and fails. The fix is to **pre-authenticate once** and give Cloud Run the saved tokens so it never needs to prompt.

### 5a. Add production redirect URI in Yahoo

1. Go to [Yahoo Developer Console](https://developer.yahoo.com/apps/)
2. Edit your ClutchAI app
3. Add `https://www.clutchai.app` to **Redirect URI(s)** (no trailing slash)
4. Save

The deploy script sets `YAHOO_REDIRECT_URI=https://www.clutchai.app` for production.

### 5b. Get tokens once (local OAuth)

1. **Temporarily** set production redirect in your local `.env`:
   ```bash
   YAHOO_REDIRECT_URI=https://www.clutchai.app
   ```
2. Run the app locally and complete Yahoo sign-in when prompted:
   ```bash
   streamlit run app/streamlit_app.py
   ```
   Use the same Yahoo account that should access the app in production. After you sign in, yfpy will write token variables to your `.env`.
3. Build the single token JSON that Cloud Run needs:
   ```bash
   python scripts/gcloud/build_yahoo_token_json.py
   ```
   This prints one line of JSON. If you see "Missing required keys", complete the OAuth flow in the app first so `.env` gets the token vars.

### 5c. Create the secret and redeploy

1. Create the secret (paste the JSON output from the previous step):
   ```bash
   # Replace <paste JSON here> with the single line from build_yahoo_token_json.py
   echo -n '<paste JSON here>' | gcloud secrets create YAHOO_ACCESS_TOKEN_JSON --data-file=- --replication-policy=automatic
   ```
   If the secret already exists, add a new version:
   ```bash
   echo -n '<paste JSON here>' | gcloud secrets versions add YAHOO_ACCESS_TOKEN_JSON --data-file=-
   ```
2. Grant Cloud Run access to the secret (if you haven’t already):
   ```bash
   ./scripts/gcloud/grant_secrets_access.sh
   ```
3. Redeploy. The deploy script already injects `YAHOO_ACCESS_TOKEN_JSON` from Secret Manager:
   ```bash
   ./scripts/gcloud/deploy.sh
   ```

After this, Yahoo OAuth on Cloud Run uses the stored tokens (and refreshes them when needed) and no longer requires stdin or a browser.

---

## Where to check for issues

Use these to debug deployment and runtime errors:

### 1. Cloud Run logs (CLI)

Stream or read recent logs:

```bash
# Recent logs (last 50 entries)
gcloud run services logs read clutchai --region us-central1 --limit 50

# Stream logs live
gcloud run services logs tail clutchai --region us-central1
```

### 2. Cloud Run logs (Console)

1. Go to [Cloud Run](https://console.cloud.google.com/run)
2. Click the **clutchai** service
3. Open the **Logs** tab to see request logs, errors, and stdout/stderr (including Python tracebacks and your app’s logger output)

Filter by severity (e.g. Error) or search for exception messages (e.g. `EOF`, `Failed to initialize`).

### 3. Application logging

The app uses the `logger` module; in production it logs to stderr. Cloud Run captures stderr in the service logs, so you’ll see `logger.error(...)` and `logger.debug(...)` (if debug is on) in the Cloud Run Logs tab. Use `CLUTCHAI_DEBUG=true` or the app’s debug setting for more detail.

---

## Troubleshooting

### "Failed to initialize Multi-Agent System: EOF when reading a line" / "Agent not initialized"

**Cause:** The Yahoo OAuth library tries to do **interactive** auth (e.g. `input()` for a code or open a browser) when tokens are missing. On Cloud Run there is no stdin and no browser, so that raises `EOFError: EOF when reading a line`.

**Fix:** Pre-save Yahoo tokens and inject them on Cloud Run. Follow **Section 5 (Make Yahoo OAuth work on Cloud Run)** above: run OAuth once locally with `YAHOO_REDIRECT_URI=https://www.clutchai.app`, run `python scripts/gcloud/build_yahoo_token_json.py`, create the `YAHOO_ACCESS_TOKEN_JSON` secret, then redeploy. After that, Yahoo features work on Cloud Run without interactive login.

### "Secret not found" or "Permission denied" on secrets

1. Run `./scripts/gcloud/update_secrets.sh` to create/update secrets
2. Run the IAM binding command in Step 2 to grant the service account access
3. Redeploy

### Domain not loading / 404

1. Confirm the domain mapping in Cloud Run (Manage custom domains)
2. Check DNS records at your registrar
3. Wait for DNS propagation (up to 48 hours)

### OAuth redirect fails at www.clutchai.app

1. Ensure `YAHOO_REDIRECT_URI` is set to `https://www.clutchai.app` in the Cloud Run service
2. Add the exact redirect URI to your Yahoo app's allowed list
3. No trailing slash—use `https://www.clutchai.app` not `https://www.clutchai.app/`

### App starts but features don't work

Check Cloud Run logs for missing env vars:

```bash
gcloud run services logs read clutchai --region us-central1 --limit 50
```

Ensure all required secrets exist and are referenced in `deploy.sh`.
