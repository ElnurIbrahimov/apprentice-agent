# AURA Messaging Setup Guide

Connect AURA to Telegram and WhatsApp for mobile access.

## Quick Start: Telegram (Recommended)

Telegram is the easiest to set up - no additional servers required.

### Step 1: Create a Telegram Bot

1. Open Telegram and search for `@BotFather`
2. Send `/newbot`
3. Choose a name (e.g., "My AURA")
4. Choose a username (must end in `bot`, e.g., `my_aura_bot`)
5. Copy the token BotFather gives you

### Step 2: Configure AURA

**Option A - Environment variable:**
```bash
export TELEGRAM_BOT_TOKEN="your_token_here"
```

**Option B - Create `.env` file in project root:**
```
TELEGRAM_BOT_TOKEN=your_token_here
```

**Option C - Edit config directly:**
Edit `aura/messaging/config.py`:
```python
TELEGRAM_CONFIG = {
    "telegram_token": "your_token_here",
    ...
}
```

### Step 3: Install Dependencies

```bash
pip install python-telegram-bot>=20.0
```

### Step 4: Run the Bot

```bash
python run_telegram.py
```

### Step 5: Test It

1. Open Telegram
2. Search for your bot by username
3. Send `/start`
4. Chat normally!

### Bot Commands

- `/start` - Start conversation
- `/status` - See AURA status
- `/mood` - Check AURA's mood
- `/memory` - What AURA remembers
- `/help` - Show help

### Tips

- Just chat normally - AURA responds naturally
- Say "remember this:" to save something important
- AURA uses the same memory across desktop and mobile
- Fast-path responses work (greetings, emotional shares)

---

## WhatsApp Setup (Advanced)

WhatsApp requires a Node.js bridge server because WhatsApp Web uses a different protocol.

### Step 1: Install Node.js

Make sure you have Node.js 16+ installed:
```bash
node --version  # Should be 16.x or higher
```

### Step 2: Install Bridge Dependencies

```bash
cd aura/messaging/baileys_bridge
npm install
```

### Step 3: Start the Bridge Server

```bash
node server.js
```

The server will display a QR code.

### Step 4: Connect WhatsApp

1. Open WhatsApp on your phone
2. Go to Settings → Linked Devices → Link a Device
3. Scan the QR code displayed in terminal

### Step 5: Enable WhatsApp in Config

Edit `aura/messaging/config.py`:
```python
MESSAGING_CONFIG = {
    "enable_whatsapp": True,
    ...
}
```

Or set environment variable:
```bash
export ENABLE_WHATSAPP=true
```

### Step 6: Install Python Dependencies

```bash
pip install websockets
```

### Step 7: Run AURA with Messaging

```bash
python run_messaging.py
```

---

## Running Both Platforms

To run both Telegram and WhatsApp:

1. Start the WhatsApp bridge: `node aura/messaging/baileys_bridge/server.js`
2. Run AURA messaging: `python run_messaging.py`

Or use the main entry point:
```bash
python main.py --messaging
```

---

## Configuration Options

### Telegram Config (`aura/messaging/config.py`)

```python
TELEGRAM_CONFIG = {
    # Bot token from @BotFather
    "telegram_token": "YOUR_TOKEN",

    # Whitelist (empty = allow all)
    "allowed_users": ["123456789"],

    # Admin users (can use admin commands)
    "admin_users": ["123456789"],

    # Proactive messaging
    "proactive_enabled": True,
    "quiet_hours_start": 22,  # 10 PM
    "quiet_hours_end": 8,     # 8 AM
}
```

### WhatsApp Config

```python
WHATSAPP_CONFIG = {
    # Bridge WebSocket URL
    "websocket_url": "ws://localhost:3001",

    # Whitelist (empty = allow all)
    "allowed_numbers": ["+1234567890"],
}
```

### Getting Your Telegram User ID

Message `@userinfobot` on Telegram - it will reply with your user ID.

---

## Troubleshooting

### "Telegram token not configured"
- Set `TELEGRAM_BOT_TOKEN` environment variable
- Or edit the config file directly

### "python-telegram-bot not installed"
```bash
pip install python-telegram-bot>=20.0
```

### "Failed to connect to WhatsApp bridge"
- Make sure the Node.js bridge is running: `node server.js`
- Check that port 3001 is available
- Try: `lsof -i :3001` to see what's using the port

### Bot not responding
- Check terminal for error messages
- Make sure your user ID is in `allowed_users` (or leave empty to allow all)
- Try `/start` command first

### WhatsApp QR code expired
- QR codes expire after ~60 seconds
- Just wait for a new one to appear
- If it keeps expiring, delete the session folder and restart

### WhatsApp disconnected
- This happens sometimes - the bridge will auto-reconnect
- If it keeps disconnecting, delete session and re-scan QR

---

## Architecture

```
┌─────────────────────────────────────────┐
│           AURA CORE ENGINE              │
│  (emotional, memory, fast-path, etc.)   │
└─────────────────────────────────────────┘
                    ↑
                    │
        ┌───────────┴───────────┐
        │    MESSAGE ROUTER     │
        └───────────┬───────────┘
                    │
      ┌─────────────┼─────────────┐
      ↓             ↓             ↓
┌──────────┐  ┌──────────┐  ┌──────────┐
│ DESKTOP  │  │ TELEGRAM │  │ WHATSAPP │
│ (local)  │  │ Bot API  │  │ (Bridge) │
└──────────┘  └──────────┘  └──────────┘
```

All platforms share the same AURA memory and state.

---

## Files Created

```
aura/messaging/
├── __init__.py           # Module exports
├── base_platform.py      # Abstract base class
├── config.py             # Configuration
├── router.py             # Central message router
├── telegram_bot.py       # Telegram integration
├── whatsapp_bot.py       # WhatsApp integration
└── baileys_bridge/
    ├── server.js         # Node.js WhatsApp bridge
    └── package.json      # Node dependencies

run_telegram.py           # Quick start for Telegram
```

---

## Security Notes

- **Never commit tokens to git** - use environment variables or .env file
- **Use allowed_users** to restrict who can use your bot
- **WhatsApp sessions are stored locally** - keep them private
- The bot has access to everything AURA knows

---

## Success Criteria

- [ ] `python run_telegram.py` starts without errors
- [ ] Sending "hey" returns a warm greeting
- [ ] Sending "I got the job!" returns genuine excitement
- [ ] Sending "remember this: X" stores and confirms
- [ ] `/status`, `/mood`, `/memory` commands work
- [ ] Bot feels like AURA, not a generic chatbot
