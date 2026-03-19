# Telegram Bot Setup for Aura

## Step 1: Create the bot

1. Open Telegram on your phone or desktop
2. Search for **@BotFather** and open a chat with it
3. Send `/newbot`
4. BotFather asks for a **display name** — type whatever you want (e.g. `Aura`)
5. BotFather asks for a **username** — must end in `bot` (e.g. `elnur_aura_bot`)
6. BotFather gives you a token like `7123456789:AAH...` — copy it

## Step 2: Get your Telegram user ID

1. Search for **@userinfobot** on Telegram and open a chat
2. Send `/start` — it replies with your numeric user ID (e.g. `123456789`)
3. Copy this number

## Step 3: Set the token on the server

SSH into the server and run:

```bash
ssh root@89.167.107.134

cd /opt/aura

# Set bot token (replace YOUR_TOKEN with the token from BotFather)
sed -i 's|^TELEGRAM_BOT_TOKEN=.*|TELEGRAM_BOT_TOKEN=YOUR_TOKEN|' .env

# Set allowed users (replace YOUR_USER_ID with your numeric ID)
sed -i 's|^TELEGRAM_ALLOWED_USERS=.*|TELEGRAM_ALLOWED_USERS=YOUR_USER_ID|' .env

# Set yourself as admin
sed -i 's|^TELEGRAM_ADMIN_USERS=.*|TELEGRAM_ADMIN_USERS=YOUR_USER_ID|' .env
```

## Step 4: Restart

```bash
systemctl restart aura-telegram
```

If aura-telegram is not a separate service, restart the main one:

```bash
systemctl restart aura
```

## Step 5: Test

1. Open Telegram and find your bot by its username
2. Send `/start` or just say "hello"
3. If it responds, you're done

## Troubleshooting

Check logs:
```bash
journalctl -u aura-telegram -f --no-pager
# or
journalctl -u aura -f --no-pager | grep -i telegram
```

Verify token is set:
```bash
grep TELEGRAM .env
```
