# Yahoo Fantasy Sports API Setup Guide

## Step 1: Create Yahoo Developer Account

1. Go to [Yahoo Developer Console](https://developer.yahoo.com/)
2. Sign in with your Yahoo account
3. If you don't have a Yahoo account, create one first

## Step 2: Create a Fantasy Sports Application

1. Navigate to [Fantasy Sports API](https://developer.yahoo.com/fantasysports/)
2. Click "Create App" or "Get Started"
3. Fill out the application form:
   - **App Name**: ClutchAI (or your preferred name)
   - **App Description**: AI-powered fantasy sports assistant
   - **App Type**: Web Application
   - **Redirect URI**: `https://localhost:8080` (Yahoo requires HTTPS, but this can be adjusted based on your setup)

## Step 3: Get Your Credentials

After creating the app, you'll receive:
- **Client ID** (Consumer Key)
- **Client Secret** (Consumer Secret)

## Step 4: Configure Your Environment

1. Copy the `env.example` file to `.env`:
   ```bash
   cp env.example .env
   ```

2. Edit the `.env` file and replace the placeholder values:
   ```
   YAHOO_CLIENT_ID=your_actual_client_id_here
   YAHOO_CLIENT_SECRET=your_actual_client_secret_here
   YAHOO_REDIRECT_URI=https://localhost:8080
   YAHOO_LEAGUE_ID=your_league_id (optional, defaults to 58930)
   OPENAI_API_KEY=your_openai_api_key
   ```

## Step 5: Test Your Setup

Test your Yahoo API credentials by running the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Then enter your credentials in the sidebar and test with a query like "Show my team roster".

## Important Notes

- **Redirect URI**: Must exactly match what you configured in Yahoo Developer Console (if using OAuth flow)
- **Rate Limits**: Yahoo API has rate limits (usually 100 requests per hour)
- **Permissions**: Your app will need to request specific fantasy sports permissions
- **Testing**: Start with a test league before using production data
- **League ID**: You can find your league ID in the Yahoo Fantasy Sports URL when viewing your league
- **Client ID/Secret**: These are used for OAuth authentication with Yahoo's API

## Troubleshooting

- **Invalid Client ID**: Double-check your Client ID from Yahoo Developer Console
- **Redirect URI Mismatch**: Ensure the URI in your app matches exactly
- **Rate Limit Exceeded**: Wait before making more requests
- **Permission Denied**: Check that your app has the necessary fantasy sports permissions

## Next Steps

Once your Yahoo API credentials are set up:
1. Test the connection by running the Streamlit app: `streamlit run app/streamlit_app.py`
2. Enter your credentials in the sidebar
3. Configure ChromaDB for vector storage (see `docs/CHROMADB_SETUP.md`)
4. Add resources to your vectorstore (see `docs/VECTORSTORE_MANAGEMENT.md`)
5. Start asking questions about your fantasy league!
