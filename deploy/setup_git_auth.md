# Fix Git Pull on Private Repo

The repo was made private, so the server can no longer pull code.
You need to create a GitHub token and save it on the server.

## Step 1: Create a GitHub Token

1. Open https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Give it a name like `aura-server`
4. Set expiration (90 days is fine — you'll need to redo this when it expires)
5. Check the **`repo`** box (this gives access to private repos)
6. Click **"Generate token"** at the bottom
7. **Copy the token immediately** — you can't see it again after leaving the page

## Step 2: Set the Token on the Server

SSH into the server, then run this command (replace `YOUR_TOKEN` with the token you copied):

```bash
sudo -u aura git -C /opt/aura remote set-url origin https://YOUR_TOKEN@github.com/ElnurIbrahimov/apprentice-agent.git
```

## Step 3: Test It

```bash
sudo -u aura git -C /opt/aura fetch
```

If it prints nothing or shows branch info, it worked.
If it says "Authentication failed", double-check the token.

## Step 4: Update as Normal

```bash
sudo bash /opt/aura/deploy/update_server.sh
```

## When the Token Expires

You'll see an auth error during `update_server.sh`. Just repeat Steps 1-3 with a new token.
